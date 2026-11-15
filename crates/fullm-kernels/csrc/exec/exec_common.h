// SPDX-License-Identifier: Apache-2.0
// exec_common.h - what every model family's executor shares: the interface the C ABI
// (fullm_exec.h) drives, and the per-step workspace arena.
#pragma once

#include <algorithm>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "backend.h"
#include "expert_pool.h"
#include "fullm_exec.h"

namespace fullm {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

inline size_t dt_size(DType d) { return d == DType::F32 ? 4 : 2; }

// A bump allocator over one backend allocation; reset per sub-block.
class Arena {
public:
    void init(Backend *b, size_t bytes) {
        b_ = b;
        cap_ = bytes;
        base_ = static_cast<char *>(b->alloc(bytes));
    }
    ~Arena() {
        if (b_ && base_) b_->release(base_);
    }
    void reset(size_t to = 0) { used_ = to; }
    size_t mark() const { return used_; }
    template <typename T>
    T *take(size_t n) {
        const size_t bytes = (n * sizeof(T) + 255) & ~size_t(255);
        if (used_ + bytes > cap_) throw std::runtime_error("workspace arena exhausted (sizing bug)");
        T *p = reinterpret_cast<T *>(base_ + used_);
        used_ += bytes;
        return p;
    }

private:
    Backend *b_ = nullptr;
    char *base_ = nullptr;
    size_t cap_ = 0, used_ = 0;
};

// The prefix cache's host tier: pinned host pages, each a copy of one cache page of every
// paged pool, and the copies between them and the pools. Every copy call is a ticket (1, 2,
// ...; 0 means none); tickets complete in order, on the backend's copy stream.
//
// A worker thread issues the copies: a load of a few thousand pages fills the driver's command
// queue, and issuing then blocks until the copy engines drain it (measured on the H100: up to
// 65 ms a load, 1.4 s over a 16-session run, the steps idle meanwhile). The caller's thread only
// queues the job, with an event marking the main stream at that point (the copies start after
// it), so the worker never touches the main stream, which CUDA graphs capture.
class HostPages {
public:
    // pools: every paged pool ([n_pages][seg_bytes] each), in the order a host page lays them out
    void init(Backend *b, std::vector<void *> pools, size_t seg_bytes, int64_t n_pages) {
        b_ = b;
        pools_ = std::move(pools);
        seg_ = seg_bytes;
        n_dev_ = n_pages;
    }
    ~HostPages() {
        if (!b_) return;
        drain();
        {
            std::lock_guard<std::mutex> g(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        for (void *p : slabs_) b_->host_release(p);
    }
    // wait for every copy queued or in flight (before the pools or the host pages go)
    void drain() noexcept {
        if (!b_) return;
        try {
            wait(last_);
        } catch (...) {
        }
    }
    size_t page_bytes() const { return pools_.size() * seg_; }
    // n host pages, pinned in slabs of about 1 GiB
    void reserve(int64_t n) {
        if (!b_ || page_bytes() == 0) throw std::runtime_error("this executor has no paged cache to back with host pages");
        if (!ptrs_.empty()) throw std::runtime_error("host pages are already reserved");
        if (n <= 0) throw std::runtime_error("the host page count must be positive");
        const int64_t per = std::max<int64_t>(1, int64_t((size_t(1) << 30) / page_bytes()));
        for (int64_t first = 0; first < n; first += per) {
            const int64_t k = std::min(per, n - first);
            char *slab = static_cast<char *>(b_->host_alloc(size_t(k) * page_bytes()));
            slabs_.push_back(slab);
            for (int64_t i = 0; i < k; ++i) ptrs_.push_back(slab + size_t(i) * page_bytes());
        }
        worker_ = std::thread([this] { work(); });
    }
    uint64_t copy(bool to_host, const int32_t *dev, const int32_t *host, int64_t n) {
        if (n < 0) throw std::runtime_error("negative page count");
        if (n == 0) return last_;
        Job j;
        j.to_host = to_host;
        j.dev.assign(dev, dev + n);
        j.host.resize(n);
        for (int64_t i = 0; i < n; ++i) {
            if (dev[i] < 0 || dev[i] >= n_dev_) throw std::runtime_error("cache page out of range");
            if (host[i] < 0 || size_t(host[i]) >= ptrs_.size()) throw std::runtime_error("host page out of range");
            j.host[i] = ptrs_[host[i]];
        }
        j.after = b_->record_event();
        {
            std::lock_guard<std::mutex> g(mu_);
            rethrow();
            j.t = ++last_;
            todo_.push_back(std::move(j));
        }
        cv_.notify_all();
        return last_;
    }
    bool done(uint64_t t) {
        std::lock_guard<std::mutex> g(mu_);
        return done_locked(t);
    }
    // the main stream's later ops run after copy t (the host waits only until it is issued)
    void order(uint64_t t) {
        std::unique_lock<std::mutex> l(mu_);
        if (done_locked(t)) return;
        cv_.wait(l, [&] { return issued_ >= t || !error_.empty(); });
        rethrow();
        if (!done_locked(t)) b_->stream_wait(pending(t));
    }
    void wait(uint64_t t) {
        std::unique_lock<std::mutex> l(mu_);
        if (done_locked(t)) return;
        cv_.wait(l, [&] { return issued_ >= t || !error_.empty(); });
        rethrow();
        if (done_locked(t)) return;
        void *ev = pending(t);
        l.unlock();  // only this thread retires events: ev stays valid
        b_->wait_event(ev);
        l.lock();
        done_locked(t);
    }

private:
    struct Job {
        uint64_t t = 0;
        bool to_host = false;
        std::vector<int32_t> dev;
        std::vector<void *> host;
        void *after = nullptr;  // the main stream when the job was queued
    };
    void work() {
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> l(mu_);
                cv_.wait(l, [&] { return stop_ || !todo_.empty(); });
                if (todo_.empty()) return;
                j = std::move(todo_.front());
                todo_.pop_front();
            }
            void *ev = nullptr;
            std::string err;
            try {
                b_->copy_pages(j.to_host, pools_.data(), int(pools_.size()), seg_, j.dev.data(), j.host.data(),
                               int64_t(j.dev.size()), j.after);
                ev = b_->record_copy_event();
                b_->release_event(j.after);
            } catch (const std::exception &e) {
                err = e.what();
            }
            {
                std::lock_guard<std::mutex> g(mu_);
                if (!err.empty() && error_.empty()) error_ = "host cache copy: " + err;
                flight_.emplace_back(j.t, ev);
                issued_ = j.t;
            }
            cv_.notify_all();
        }
    }
    // retire the completed copies (in order); whether t is among them
    bool done_locked(uint64_t t) {
        while (!flight_.empty() && b_->event_done(flight_.front().second)) {
            done_ = flight_.front().first;
            b_->release_event(flight_.front().second);
            flight_.pop_front();
        }
        return t <= done_;
    }
    // the event of the first issued copy at or after t (completing it completes t)
    void *pending(uint64_t t) const {
        for (auto &c : flight_)
            if (c.first >= t) return c.second;
        throw std::runtime_error("copy ticket was never issued");
    }
    void rethrow() const {
        if (!error_.empty()) throw std::runtime_error(error_);
    }
    Backend *b_ = nullptr;
    std::vector<void *> pools_, slabs_, ptrs_;
    size_t seg_ = 0;
    int64_t n_dev_ = 0;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Job> todo_;                             // queued, not yet issued
    std::deque<std::pair<uint64_t, void *>> flight_;   // issued, with the event that completes them
    uint64_t last_ = 0, issued_ = 0, done_ = 0;
    std::string error_;
    bool stop_ = false;
    std::thread worker_;
};

static_assert(sizeof(SampleRow) == sizeof(sk_sample_row) && offsetof(SampleRow, seed) == offsetof(sk_sample_row, seed) &&
                  offsetof(SampleRow, step) == offsetof(sk_sample_row, step) &&
                  offsetof(SampleRow, mask) == offsetof(sk_sample_row, mask) &&
                  offsetof(SampleRow, min_p_cut) == offsetof(sk_sample_row, min_p_cut) &&
                  offsetof(SampleRow, adj_off) == offsetof(sk_sample_row, adj_off) &&
                  offsetof(SampleRow, logprobs) == offsetof(sk_sample_row, logprobs),
              "SampleRow is sk_sample_row");
static_assert(sizeof(LogitAdj) == sizeof(sk_logit_adj) && offsetof(LogitAdj, sub) == offsetof(sk_logit_adj, sub),
              "LogitAdj is sk_logit_adj");
static_assert(kMaxTopLogprobs == SK_MAX_TOP_LOGPROBS, "one bound on top log-probabilities");

// A device buffer that grows on demand (released with its executor).
struct DevBuf {
    void *p = nullptr;
    size_t n = 0;
    void *ensure(Backend *b, size_t bytes) {
        if (bytes <= n) return p;
        b->sync();  // nothing queued may still read the old one
        if (p) b->release(p);
        p = nullptr, n = 0;
        p = b->alloc(bytes);
        n = bytes;
        return p;
    }
    void release(Backend *b) {
        if (p) b->release(p);
        p = nullptr, n = 0;
    }
};

// What a sampling step needs beyond its rows: the masks, the adjustments, the logits as the
// model gave them (for log-probabilities) and the log-probabilities out.
struct SampleScratch {
    DevBuf masks, adj, raw, lp;
    void release(Backend *b) { masks.release(b), adj.release(b), raw.release(b), lp.release(b); }
};

// The step's sampling inputs checked against its rows: how many masks there are; whether any row
// asks for log-probabilities.
inline int32_t sample_checks(const sk_sample_row *sample, int64_t R, const sk_sample_io *io, int64_t vocab, bool &lp) {
    const uint32_t *masks = io ? io->masks : nullptr;
    const int64_t words = io ? io->mask_words : 0, n_adj = io && io->adj ? io->n_adj : 0;
    int32_t k = 0;
    lp = false;
    for (int64_t r = 0; r < R; ++r) {
        const sk_sample_row &s = sample[r];
        if (s.mask < 0) throw std::runtime_error("negative sampling mask index");
        k = std::max(k, s.mask);
        if (s.adj_n < 0 || s.adj_off < 0 || (s.adj_n > 0 && int64_t(s.adj_off) + s.adj_n > n_adj))
            throw std::runtime_error("a row's logit adjustments are outside the step's");
        if (s.logprobs < 0 || s.logprobs > kMaxTopLogprobs + 1)
            throw std::runtime_error("log-probabilities: at most " + std::to_string(kMaxTopLogprobs) + " alternatives");
        if (!(s.min_p_cut <= 0.0f)) throw std::runtime_error("min-p cut must be <= 0");
        lp |= s.logprobs > 0;
    }
    if (k > 0 && (!masks || words * 32 < vocab))
        throw std::runtime_error("sampling masks missing or narrower than the vocabulary");
    for (int64_t j = 0; j < n_adj; ++j) {
        const sk_logit_adj &a = io->adj[j];
        if (a.token < 0 || a.token >= vocab || !std::isfinite(a.bias) || !(a.rep > 0.0f) || !std::isfinite(a.rep) ||
            std::isnan(a.sub) || a.sub == -INFINITY)
            throw std::runtime_error("invalid logit adjustment (token " + std::to_string(a.token) + ")");
    }
    if (lp && (!io || !io->logprobs || !io->top_ids)) throw std::runtime_error("log-probabilities asked with no buffers");
    return k;
}

// A step's tokens into tokens_out: each of the R logits rows' argmax, or with sample (host) its
// draw, on the device. io (sk_sample_io): masks then adjustments applied to the logits in place
// (a copy first when a row asks for log-probabilities, which are of the model's logits), and the
// log-probabilities out. d_ids: R ids then the non-finite flag; d_sample: room for R rows.
inline void step_tokens(Backend *b, float *logits, int64_t R, int64_t vocab, const sk_sample_row *sample,
                        const sk_sample_io *io, int32_t *d_ids, SampleRow *d_sample, SampleScratch &sc, int32_t *h_ids,
                        int32_t *tokens_out) {
    b->zero(d_ids, size_t(R + 1) * 4);
    bool lp = false;
    float *lpd = nullptr;
    int32_t *idd = nullptr;
    if (sample) {
        const int32_t k = sample_checks(sample, R, io, vocab, lp);
        b->upload(d_sample, sample, size_t(R) * sizeof(SampleRow));
        const float *raw = logits;
        if (lp) {
            float *copy = static_cast<float *>(sc.raw.ensure(b, size_t(R) * vocab * 4));
            b->copy(copy, logits, size_t(R) * vocab * 4);
            raw = copy;
        }
        if (k > 0) {
            const size_t bytes = size_t(k) * size_t(io->mask_words) * 4;
            auto *dm = static_cast<uint32_t *>(sc.masks.ensure(b, bytes));
            b->upload(dm, io->masks, bytes);
            b->mask_rows(logits, d_ids + R, R, vocab, d_sample, dm, io->mask_words);
        }
        if (io && io->adj && io->n_adj > 0) {
            const size_t bytes = size_t(io->n_adj) * sizeof(LogitAdj);
            auto *da = static_cast<LogitAdj *>(sc.adj.ensure(b, bytes));
            b->upload(da, io->adj, bytes);
            b->adjust_rows(logits, R, vocab, d_sample, da);
        }
        b->sample_rows(d_ids, d_ids + R, logits, R, vocab, d_sample);
        if (lp) {
            void *buf = sc.lp.ensure(b, size_t(R) * (2 * kMaxTopLogprobs + 1) * 4);
            lpd = static_cast<float *>(buf);
            idd = reinterpret_cast<int32_t *>(lpd + R * (kMaxTopLogprobs + 1));
            b->logprob_rows(lpd, idd, raw, R, vocab, d_sample, d_ids);
        }
    } else {
        b->argmax_rows(d_ids, d_ids + R, logits, R, vocab);
    }
    b->download(h_ids, d_ids, size_t(R + 1) * 4);
    if (lp) {
        b->download(io->logprobs, lpd, size_t(R) * (kMaxTopLogprobs + 1) * 4);
        b->download(io->top_ids, idd, size_t(R) * kMaxTopLogprobs * 4);
    }
    b->sync();
    if (h_ids[R]) throw std::runtime_error("non-finite logits");
    std::memcpy(tokens_out, h_ids, size_t(R) * 4);
}

// One model family's executor, as the C ABI sees it.
class ExecIface {
public:
    virtual ~ExecIface() = default;
    virtual void add_matrix(const std::string &name, int32_t fmt, int64_t rows, int64_t cols, const void *data,
                            const float *scale) = 0;
    virtual void add_vector(const std::string &name, const float *data, int64_t n) = 0;
    virtual void finalize() = 0;
    // logits_out: [n_rows][vocab]; or, with tokens_out, only each row's argmax, or with sample
    // (host, n_rows) each row's draw, with io's masks, adjustments and log-probabilities
    // (sk_exec_step_sample).
    virtual void step(const sk_step_plan &pl, float *logits_out, sk_step_stats *st, int32_t *tokens_out,
                      const sk_sample_row *sample = nullptr, const sk_sample_io *io = nullptr) = 0;
    virtual size_t slot_bytes() const = 0;
    virtual void save_slot(int32_t slot, void *dst) = 0;
    virtual void load_slot(int32_t slot, const void *src) = 0;
    virtual const char *describe() = 0;
    // routed experts: MoE families only
    virtual void set_expert(int32_t, int32_t, const ExpertPtrs &) { no_experts(); }
    virtual void reserve_gpu_experts(int32_t, size_t) { no_experts(); }
    virtual void set_gpu_expert(int32_t, int32_t, const void *, const float *, const void *, const float *,
                                const void *, const float *) {
        no_experts();
    }
    virtual size_t pool_bytes() const { return 0; }
    // LoRA adapters, unmerged (sk_exec_reserve_lora ...): dense executors only
    virtual void reserve_lora(int32_t, int32_t, uint32_t) { throw std::runtime_error("this executor has no LoRA adapters"); }
    virtual void set_lora(int32_t, int32_t, int32_t, const uint16_t *, size_t, const uint16_t *, size_t, int32_t, float) {
        throw std::runtime_error("this executor has no LoRA adapters");
    }
    virtual void clear_lora(int32_t) { throw std::runtime_error("this executor has no LoRA adapters"); }
    virtual void set_seq_lora(int32_t, int32_t) { throw std::runtime_error("this executor has no LoRA adapters"); }
    // the prefix cache's host tier (HostPages): page_bytes 0 = not supported
    virtual size_t page_bytes() const { return 0; }
    virtual void reserve_host_pages(int64_t) { throw std::runtime_error("this model has no host page tier"); }
    // EAGLE-3 draft heads: dense decoders only
    virtual void set_eagle(const sk_eagle_dims &) { throw std::runtime_error("this model takes no EAGLE-3 head"); }
    virtual void draft(const sk_draft_plan &, int32_t, int32_t *) {
        throw std::runtime_error("this model takes no EAGLE-3 head");
    }
    virtual uint64_t copy_pages(bool, const int32_t *, const int32_t *, int64_t) {
        throw std::runtime_error("this model has no host page tier");
    }
    virtual bool copy_done(uint64_t) { return true; }
    virtual void copy_order(uint64_t) {}
    virtual void copy_wait(uint64_t) {}
    // the quantised-expert cache file (after reserve_gpu_experts): 1 = replaying it (set_gpu_expert
    // then ignores its data), 0 = recording it, -1 = no cache for this executor or format
    virtual int expert_cache(const char *) { return -1; }

private:
    [[noreturn]] static void no_experts() { throw std::runtime_error("this model has no routed experts"); }
};

std::unique_ptr<ExecIface> make_k3_exec(const sk_k3_dims &d, const sk_exec_opts &o);
std::unique_ptr<ExecIface> make_dense_exec(const sk_dense_dims &d, const sk_exec_opts &o);

}  // namespace fullm
