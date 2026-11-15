// SPDX-License-Identifier: Apache-2.0
// threadpool.h - a fixed pool with a blocking parallel_for, built for many short calls.
//
// The routed-expert path calls parallel_for a few times per MoE layer, ~200 times per
// decode step, so wake-up latency matters more than throughput: a condition-variable
// wake costs 5-20 us, which would add milliseconds per step. Workers therefore spin on
// the job generation for a short while before sleeping, and the caller works too.
//
// Each call owns its own Job (counters on the caller's stack). A worker joins a job only
// by taking a reference under the mutex, and the caller returns only after unpublishing
// the job and seeing every reference dropped, so a late worker can never run one job's
// function on another job's indices.
//
// Nested calls (parallel_for from inside a task) run serially on the calling thread.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <fstream>
#include <string>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define FULLM_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define FULLM_PAUSE() asm volatile("yield")
#else
#define FULLM_PAUSE() std::this_thread::yield()
#endif

namespace fullm {

// Which CPUs a decode step's ops run on, on a hybrid CPU (Linux). A decode step's ops are
// memory-bound and short (hundreds of parallel_for calls of a few microseconds): past the threads
// that fill the memory bus every extra one costs synchronisation, and an efficiency core can be a
// straggler. Prefill is compute-bound and wants every thread. Qwen3-0.6B int8 decode on an
// i9-14900HX (8 P-cores with SMT, 16 E-cores, one DDR5 channel), tok/s, by the CPUs allowed:
//
//     allowed                     P-cores only   every allowed thread
//     8P + 16E (the chip, 32)         44.6            36       (24 physical: 41; 8P+8E: 44)
//     8P (16 with SMT)                43              45
//     6P + 12E                        40              41.5
//     4P + 8E                         31-36           41
//     3P + 12E                        25              40
//     2P + 4E                         17-22           34
//
// So decode goes to the performance cores only when at least kPerfOnlyMinCores of them are
// allowed (they fill the bus by themselves and the rest would only add synchronisation); with
// fewer, every allowed thread runs everything, unpinned, as on a CPU of one kind.
//
// Hybrid = the CPUs fall into two speed classes, by cpu_capacity (arm64 big.LITTLE) or else by
// cpuinfo_max_freq (Intel P/E: 5.6 against 4.1 GHz on the i9-14900HX). The fast class is every CPU within 15%
// of the fastest: Intel's favoured cores (Turbo Boost Max 3.0) sit 2-4% above the other P-cores,
// E-cores ~25% below, big.LITTLE's little cores at <= 60% of the big ones' capacity.
//
// **Only CPUs the process may use.** The speed classes are the machine's; the CPUs chosen from
// them are the process's (its affinity: taskset, numactl, a cgroup cpuset). A pool that pinned to
// the machine's P-cores moved a process confined to the E-cores (`taskset -c 16-31`) onto
// P-cores another process had been given.
constexpr size_t kPerfOnlyMinCores = 8;

struct CpuInfo {
    int cpu = 0;
    long capacity = 0;          // sysfs cpu_capacity (0: absent)
    long max_freq = 0;          // sysfs cpufreq/cpuinfo_max_freq, kHz (0: absent)
    std::vector<int> siblings;  // the core's CPUs (topology/core_cpus_list; empty: this one only)
};
struct PerfTopology {
    std::vector<int> pin;                 // one allowed CPU per performance core that has one
    std::vector<int> caller;              // every allowed CPU of those cores (their SMT siblings too)
    std::vector<std::vector<int>> cores;  // each such core's allowed CPUs (cores[i][0] == pin[i])
};

// The selection, without sysfs: `allowed` empty means every CPU. Empty when the CPUs are of one
// class, when no fast core has an allowed CPU, or when there are fewer than two CPUs.
inline PerfTopology select_perf_cpus(const std::vector<CpuInfo> &cpus, const std::vector<int> &allowed) {
    PerfTopology out;
    if (cpus.size() < 2) return out;
    auto ok = [&](int c) { return allowed.empty() || std::find(allowed.begin(), allowed.end(), c) != allowed.end(); };
    for (int key = 0; key < 2; ++key) {
        auto val = [&](const CpuInfo &c) { return key == 0 ? c.capacity : c.max_freq; };
        long top = 0;
        bool readable = true;
        for (const CpuInfo &c : cpus) {
            if (val(c) <= 0) readable = false;
            top = std::max(top, val(c));
        }
        if (!readable) continue;  // not this key
        const long cut = top - top * 15 / 100;
        size_t fast = 0;
        for (const CpuInfo &c : cpus) fast += val(c) >= cut;
        if (fast == cpus.size()) continue;  // one class by this key
        std::vector<int> seen;              // cores done, by their first CPU
        for (const CpuInfo &c : cpus) {
            if (val(c) < cut) continue;
            const std::vector<int> sib = c.siblings.empty() ? std::vector<int>{c.cpu} : c.siblings;
            if (std::find(seen.begin(), seen.end(), sib.front()) != seen.end()) continue;
            seen.push_back(sib.front());
            std::vector<int> core;
            for (int k : sib)
                if (ok(k)) core.push_back(k), out.caller.push_back(k);
            if (core.empty()) continue;
            out.pin.push_back(core.front());
            out.cores.push_back(std::move(core));
        }
        break;
    }
    return out;
}

#if defined(__linux__)
namespace cpu_sysfs {
inline std::string read(const std::string &path) {
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    return s;
}
// a sysfs CPU list: "0-31", "0-3,8-11", "16"
inline std::vector<int> parse_list(const std::string &l) {
    std::vector<int> cpus;
    for (size_t i = 0; i < l.size();) {
        int a = std::atoi(l.c_str() + i), b = a;
        size_t j = l.find_first_of(",-", i);
        if (j != std::string::npos && l[j] == '-') {
            b = std::atoi(l.c_str() + j + 1);
            j = l.find(',', j);
        }
        for (int c = a; c <= b; ++c) cpus.push_back(c);
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return cpus;
}
inline std::string cpu_path(int c, const char *name) { return "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/" + name; }
}  // namespace cpu_sysfs
#endif

// The CPUs this process may run on: the affinity of the thread that first asks, which is the
// first pool's creator (the main thread, before any pool has pinned it), so it is what taskset,
// numactl or a cgroup cpuset gave the process. Read once. Empty when unknown (not Linux).
inline const std::vector<int> &allowed_cpus() {
    static const std::vector<int> v = [] {
        std::vector<int> out;
#if defined(__linux__)
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof set, &set) == 0)
            for (int c = 0; c < CPU_SETSIZE; ++c)
                if (CPU_ISSET(c, &set)) out.push_back(c);
#endif
        return out;
    }();
    return v;
}

inline const PerfTopology &perf_topology() {
    static const PerfTopology v = [] {
#if defined(__linux__)
        using cpu_sysfs::read, cpu_sysfs::cpu_path;
        std::vector<CpuInfo> cpus;
        for (int c : cpu_sysfs::parse_list(read("/sys/devices/system/cpu/online"))) {
            CpuInfo i;
            i.cpu = c;
            i.capacity = std::atol(read(cpu_path(c, "cpu_capacity")).c_str());
            i.max_freq = std::atol(read(cpu_path(c, "cpufreq/cpuinfo_max_freq")).c_str());
            i.siblings = cpu_sysfs::parse_list(read(cpu_path(c, "topology/core_cpus_list")));
            cpus.push_back(std::move(i));
        }
        return select_perf_cpus(cpus, allowed_cpus());
#else
        return PerfTopology{};
#endif
    }();
    return v;
}

// The CPUs the process may use (all online ones where that is unknown), except on Apple silicon:
// its performance cores only. On a Linux hybrid CPU the pool holds every allowed CPU (prefill is
// compute-bound and the efficiency cores add half as much again) and runs its perf_only jobs on
// the performance cores among them (ThreadPool).
inline int default_threads() {
#if defined(__APPLE__)
    int perf = 0;
    size_t len = sizeof(perf);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &len, nullptr, 0) == 0 && perf > 0) return perf;
#endif
    if (!allowed_cpus().empty()) return static_cast<int>(allowed_cpus().size());
    return static_cast<int>(std::thread::hardware_concurrency());
}

class ThreadPool {
public:
    // threads: 0 = default_threads(). On a hybrid CPU with at least kPerfOnlyMinCores performance
    // cores the process may use, workers 1..P-1 are pinned one per performance core and, with the
    // caller (pinned to those cores on its first perf_only job, where it takes the one left free),
    // run the perf_only jobs; the other workers keep the process's affinity and sit those out.
    // Every CPU pinned to is one the process was allowed (perf_topology). FULLM_CPU_PIN=0: never
    // (every job on every thread, nothing pinned); =1: whenever two or more performance cores are
    // allowed, however few; unset: the rule above.
    explicit ThreadPool(int threads) {
        int n = threads > 0 ? threads : default_threads();
        if (n < 1) n = 1;
        const char *pe = std::getenv("FULLM_CPU_PIN");
        const int mode = pe && *pe ? (*pe == '0' ? 0 : 1) : -1;  // off, forced, auto
        const std::vector<int> &pc = perf_topology().pin;
        const size_t need = mode == 1 ? 2 : kPerfOnlyMinCores;
        // perf workers: the performance cores but one (the caller's), within the pool
        if (mode != 0 && pc.size() >= need) perf_workers_ = std::min<int>(static_cast<int>(pc.size()) - 1, n - 1);
        // the caller's CPUs: the performance cores no worker is pinned to (core 0, and any past
        // the pool's workers), so it never shares a core with a pinned worker
        const auto &cores = perf_topology().cores;
        for (size_t c = 0; perf_workers_ > 0 && c < cores.size(); ++c)
            if (c == 0 || c > static_cast<size_t>(perf_workers_))
                caller_cpus_.insert(caller_cpus_.end(), cores[c].begin(), cores[c].end());
        for (int i = 1; i < n; ++i) {
            workers_.emplace_back([this, i] { worker(i); });
#if defined(__linux__)
            if (i <= perf_workers_) {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(pc[static_cast<size_t>(i)], &set);
                pthread_setaffinity_np(workers_.back().native_handle(), sizeof set, &set);
            }
#endif
        }
    }

    ~ThreadPool() {
        unpin_caller();
        {
            std::lock_guard<std::mutex> g(mu_);
            stop_ = true;
            gen_.fetch_add(1, std::memory_order_release);
            gen_all_.fetch_add(1, std::memory_order_release);
        }
        cv_perf_.notify_all();
        cv_all_.notify_all();
        for (auto &t : workers_) t.join();
    }

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    int size() const { return static_cast<int>(workers_.size()) + 1; }
    // the threads a job runs on: all, or (perf_only on a hybrid CPU) the performance cores
    int size(bool perf_only) const { return perf_only && perf_workers_ > 0 ? perf_workers_ + 1 : size(); }

    // f(i) for every i in [0, n); returns when all are done. f must not throw. perf_only: on the
    // performance cores where the CPU is hybrid (memory-bound work of a decode step).
    void parallel_for(int64_t n, const std::function<void(int64_t)> &f, bool perf_only = false) {
        if (n <= 0) return;
        if (n == 1 || workers_.empty() || in_task()) {
            for (int64_t i = 0; i < n; ++i) f(i);
            return;
        }
        std::lock_guard<std::mutex> one_job(caller_mu_);
        Job job;
        job.f = &f;
        job.total = n;
        job.perf_only = perf_only && perf_workers_ > 0;
        if (job.perf_only) pin_caller();
        // perf workers watch gen_, the others gen_all_: a perf_only job neither wakes the others nor
        // has them spin (on the performance cores' SMT siblings and the caller's core: that halved a
        // decode step), so they sleep through a decode step and wake for its first wide job
        {
            std::lock_guard<std::mutex> g(mu_);
            job_ = &job;
            gen_.fetch_add(1, std::memory_order_release);
            if (!job.perf_only) gen_all_.fetch_add(1, std::memory_order_release);
        }
        cv_perf_.notify_all();
        if (!job.perf_only) cv_all_.notify_all();
        run(job);
        while (job.done.load(std::memory_order_acquire) < n) FULLM_PAUSE();
        {
            std::lock_guard<std::mutex> g(mu_);
            job_ = nullptr;
        }
        while (job.refs.load(std::memory_order_acquire) != 0) FULLM_PAUSE();
    }

private:
    // The caller works in every job and runs the serial code between them (the executor's and the
    // engine's); left to the scheduler on a hybrid CPU it lands on an efficiency core while the
    // performance cores look busy with the pinned workers, and a decode step then runs at the
    // efficiency core's pace. A thread's first perf_only job in this pool pins it to the performance
    // cores that have no pinned worker (caller_cpus_: core 0's CPUs, SMT siblings included, so
    // several callers can share it), never beside a worker on its core.
    //
    // The pin is undone, not left behind: a thread keeps its affinity from before its first pin
    // (CallerPin, shared by the pools that pinned it) and gets it back when the last such pool is
    // destroyed on that thread. A pool destroyed on another thread cannot reach the caller's
    // thread-local state (or know that the thread still exists): that caller stays pinned until it
    // exits. Threads a pinned thread creates inherit its pinned affinity (Linux semantics).
#if defined(__linux__)
    struct CallerPin {
        cpu_set_t saved;               // the thread's affinity before its first pin
        std::vector<uint64_t> seen;    // the pools whose perf_only jobs it has run (tried once each)
        std::vector<uint64_t> pinned;  // those of them alive that pinned it (saved restored at none)
    };
    static CallerPin &caller_pin() {
        thread_local CallerPin p;
        return p;
    }
#endif
    void pin_caller() {
#if defined(__linux__)
        CallerPin &cp = caller_pin();
        if (std::find(cp.seen.begin(), cp.seen.end(), id_) != cp.seen.end()) return;
        cp.seen.push_back(id_);
        const std::vector<int> &cpus = caller_cpus_.empty() ? perf_topology().caller : caller_cpus_;
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int k : cpus)
            if (k >= 0 && k < CPU_SETSIZE) CPU_SET(k, &set);
        if (CPU_COUNT(&set) == 0) return;
        if (cp.pinned.empty() && pthread_getaffinity_np(pthread_self(), sizeof cp.saved, &cp.saved) != 0) return;
        if (pthread_setaffinity_np(pthread_self(), sizeof set, &set) != 0) return;
        cp.pinned.push_back(id_);
#endif
    }
    // this thread's pin by this pool, undone (its affinity from before restored by the last pool)
    void unpin_caller() {
#if defined(__linux__)
        CallerPin &cp = caller_pin();
        cp.seen.erase(std::remove(cp.seen.begin(), cp.seen.end(), id_), cp.seen.end());
        auto it = std::find(cp.pinned.begin(), cp.pinned.end(), id_);
        if (it == cp.pinned.end()) return;
        cp.pinned.erase(it);
        if (cp.pinned.empty()) pthread_setaffinity_np(pthread_self(), sizeof cp.saved, &cp.saved);
#endif
    }
    static uint64_t next_id() {
        static std::atomic<uint64_t> n{1};
        return n.fetch_add(1, std::memory_order_relaxed);
    }

    struct Job {
        const std::function<void(int64_t)> *f = nullptr;
        int64_t total = 0;
        bool perf_only = false;  // only the perf workers (and the caller) take part
        std::atomic<int64_t> next{0}, done{0};
        std::atomic<int> refs{0};
    };

    static bool &in_task() {
        thread_local bool flag = false;
        return flag;
    }

    static void run(Job &j) {
        in_task() = true;
        for (;;) {
            int64_t i = j.next.fetch_add(1, std::memory_order_relaxed);
            if (i >= j.total) break;
            (*j.f)(i);
            j.done.fetch_add(1, std::memory_order_acq_rel);
        }
        in_task() = false;
    }

    void worker(int idx) {
        const bool perf = idx <= perf_workers_;
        std::atomic<uint64_t> &gen = perf ? gen_ : gen_all_;
        std::condition_variable &cv = perf ? cv_perf_ : cv_all_;
        uint64_t seen = gen.load(std::memory_order_acquire);
        for (;;) {
            int spins = 0;
            while (gen.load(std::memory_order_acquire) == seen && spins < 20000) {
                FULLM_PAUSE();
                ++spins;
            }
            Job *j = nullptr;
            {
                std::unique_lock<std::mutex> lk(mu_);
                // stop_ too: a pool destroyed before this thread first ran bumped gen before
                // `seen` was read, so the bump alone would never wake it (a hang in join)
                cv.wait(lk, [&] { return stop_ || gen.load(std::memory_order_acquire) != seen; });
                seen = gen.load(std::memory_order_acquire);
                if (stop_) return;
                j = job_;
                // a perf_only job posted after the one this worker was woken for: not its job
                if (j && j->perf_only && !perf) j = nullptr;
                if (j) j->refs.fetch_add(1, std::memory_order_acq_rel);
            }
            if (j) {
                run(*j);
                j->refs.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }

    std::vector<std::thread> workers_;
    int perf_workers_ = 0;  // workers 1..perf_workers_ are pinned to performance cores
    const uint64_t id_ = next_id();  // this pool, for CallerPin (an address could be reused)
    std::vector<int> caller_cpus_;   // where a caller is pinned for perf_only jobs (see pin_caller)
    std::mutex mu_, caller_mu_;
    std::condition_variable cv_perf_, cv_all_;      // the perf workers' and the other workers'
    std::atomic<uint64_t> gen_{0}, gen_all_{0};  // every job; the jobs open to every worker
    Job *job_ = nullptr;  // guarded by mu_
    bool stop_ = false;   // guarded by mu_
};

}  // namespace fullm
