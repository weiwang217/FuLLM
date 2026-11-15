// SPDX-License-Identifier: Apache-2.0
//! Cache pages and the prefix cache: a radix tree of full pages kept after their sequences end
//! and shared, read-only, by every later sequence that starts with the same tokens.
//!
//! A node is one full page: the `page` tokens it holds, keyed under its parent. A node's path
//! from the root spells the prefix whose cache its page holds, so a page is shared only between
//! sequences that agree on every token up to its end (the cache of a position depends on the
//! tokens up to it and nothing else). Pages enter the tree as soon as a sequence fills them, so
//! concurrent requests with a common prefix share it too, and are never written again: sequences
//! only append, and a hit leaves at least the prompt's last token to compute, in a page of its
//! own. When two sequences fill the same page (both started before either finished it), the
//! later one switches to the tree's copy and frees its own, as SGLang does, so its continuation
//! still enters the tree.
//!
//! REFERENCES. Every live sequence holds the nodes of one path from the root, so a child never
//! has more references than its parent and an unreferenced node heads an unreferenced subtree.
//! Unreferenced device pages count as free: taking a page when the free list is empty evicts the
//! least recently released one. The engine's admission (each request's expected pages reserved,
//! shared pages counted once) therefore knows exactly what a step can take.
//!
//! HOST TIER (`with_host`). The tree then lives in host memory and device pages cache it: every
//! node owns a host page (its id), filled from the device when the node enters the tree (write
//! through), and holds a device page while it is resident. Evicting from the device drops only
//! the device copy, of any unreferenced resident node; evicting from the host removes a node, an
//! unreferenced leaf (with its device copy, if any). A lookup stops at the first node that is not
//! resident; `peek` and `load` carry on past it, loading such nodes into fresh device pages. The
//! caller copies the pages (`Insert::Added`, `load`); the pool only says which. Without a host
//! tier a node's id is its device page and evicting from the device removes it (a leaf).
//!
//! NEXT-TOKEN KEYS (`key_next_token`). A model with an EAGLE-3 head keeps, at each position, the
//! head's cache of that position read with the token after it: a page's last position depends
//! on the first token of the next page. Its keys then hold that token too (`page + 1` tokens), so
//! a page is shared only by sequences that agree on it as well, and enters the tree once the
//! sequence holds it.

use std::collections::{BTreeSet, HashMap};

struct Node {
    parent: Option<u32>,
    key: Box<[u32]>,
    children: HashMap<Box<[u32]>, u32>,
    /// The device page holding its cache while resident (always, without a host tier).
    dev: Option<u32>,
    /// Live sequences holding this page (resident nodes only).
    refs: u32,
    /// When the last reference went (the eviction order).
    released: u64,
}

/// Which eviction sets a node is in, and whether it is an idle device page (see `class`).
#[derive(Clone, Copy, Default)]
struct Class {
    released: u64,
    leaf: bool,
    resident: bool,
    idle: bool,
}

/// What `insert` did with a sequence's full page.
#[derive(Debug, PartialEq, Eq)]
pub enum Insert {
    /// The page is in the tree now; with a host tier, the host page to copy it to (none when
    /// the host already held that prefix and the page became its device copy).
    Added(Option<u32>),
    /// Another sequence filled the same page first: use this one instead (the caller frees its
    /// own).
    Shared(u32),
    /// Every host page belongs to a referenced node: the page stays the caller's alone.
    Full,
}

pub struct PagePool {
    page: usize,
    /// Tokens of each key past its page's own (`key_next_token`: 1).
    extra: usize,
    /// Free device pages.
    free: Vec<u32>,
    /// By node id: the host page with a host tier, else the device page.
    nodes: Vec<Option<Node>>,
    /// By device page: the node it holds, when in the tree.
    owner: Vec<Option<u32>>,
    roots: HashMap<Box<[u32]>, u32>,
    /// Unreferenced leaves, oldest release first: the nodes that can leave the tree.
    leaves: BTreeSet<(u64, u32)>,
    /// With a host tier: unreferenced resident nodes, oldest release first (device copies that
    /// can go while the node stays).
    resident: BTreeSet<(u64, u32)>,
    /// With a host tier: the host pages no node uses.
    host_free: Vec<u32>,
    host: bool,
    /// Nodes in the tree; resident ones; resident ones no sequence references.
    n_nodes: usize,
    n_resident: usize,
    idle: usize,
    tick: u64,
    enabled: bool,
}

impl PagePool {
    /// `n_pages` device pages of `page` positions; with `enabled` false, a plain free list.
    pub fn new(n_pages: usize, page: usize, enabled: bool) -> Self {
        Self::build(n_pages, 0, page, enabled)
    }

    /// A prefix cache of `n_host` host pages (more than `n_pages`) cached by `n_pages` device
    /// pages.
    pub fn with_host(n_pages: usize, n_host: usize, page: usize) -> Self {
        assert!(n_host > n_pages, "the host tier holds every cached page: it must be larger than the device's");
        Self::build(n_pages, n_host, page, true)
    }

    fn build(n_pages: usize, n_host: usize, page: usize, enabled: bool) -> Self {
        let host = n_host > 0;
        PagePool {
            page,
            extra: 0,
            free: (0..n_pages as u32).rev().collect(),
            nodes: (0..if host { n_host } else { n_pages }).map(|_| None).collect(),
            owner: vec![None; n_pages],
            roots: HashMap::new(),
            leaves: BTreeSet::new(),
            resident: BTreeSet::new(),
            host_free: (0..n_host as u32).rev().collect(),
            host,
            n_nodes: 0,
            n_resident: 0,
            idle: 0,
            tick: 0,
            enabled,
        }
    }

    /// Keys hold the token after their page too (see the module comment). Before any insert.
    pub fn key_next_token(&mut self) {
        assert_eq!(self.n_nodes, 0, "key_next_token on a tree that holds pages");
        self.extra = 1;
    }

    /// Tokens of each key: the page's, and the next one with `key_next_token`.
    pub fn key_len(&self) -> usize {
        self.page + self.extra
    }

    /// The keys of `tokens`' whole pages in turn (each with the token after it, if keys hold one).
    fn keys<'a>(&self, tokens: &'a [u32]) -> impl Iterator<Item = &'a [u32]> + 'a {
        let (page, extra) = (self.page, self.extra);
        let n = tokens.len().saturating_sub(extra) / page;
        (0..n).map(move |i| &tokens[i * page..(i + 1) * page + extra])
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn host_tier(&self) -> bool {
        self.host
    }

    /// Device pages a sequence can take now: the free ones and the unreferenced cached ones.
    pub fn available(&self) -> usize {
        self.free.len() + self.idle
    }

    /// Device pages held by the prefix cache, referenced or not.
    pub fn cached(&self) -> usize {
        self.n_resident
    }

    /// Pages the host tier holds (every node in the tree); 0 without one.
    pub fn host_cached(&self) -> usize {
        if self.host {
            self.n_nodes
        } else {
            0
        }
    }

    /// A device page for a sequence's own use; evicts from the cache when nothing is free.
    pub fn take(&mut self) -> Option<u32> {
        if let Some(pg) = self.free.pop() {
            return Some(pg);
        }
        if self.host {
            let &(_, id) = self.resident.first()?;
            let old = self.class(id);
            let pg = self.node_mut(id).dev.take().expect("a resident node");
            self.owner[pg as usize] = None;
            self.n_resident -= 1;
            self.reclass(id, old);
            Some(pg)
        } else {
            let &(_, id) = self.leaves.first()?;
            self.remove(id)
        }
    }

    /// Return a device page that is not in the tree.
    pub fn give_back(&mut self, pg: u32) {
        debug_assert!(self.owner[pg as usize].is_none());
        self.free.push(pg);
    }

    /// The device pages of the longest resident prefix of `tokens` in whole pages, now
    /// referenced by the caller (who releases them with `release`).
    pub fn lookup(&mut self, tokens: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        if !self.enabled {
            return out;
        }
        let mut at = None;
        for key in self.keys(tokens) {
            let Some(id) = self.child(at, key) else { break };
            let Some(pg) = self.node(id).dev else { break };
            self.hold(id);
            out.push(pg);
            at = Some(id);
        }
        out
    }

    /// How many whole pages of `tokens` the tree holds right after `after` (the caller's last
    /// held page; `None` at the root), resident or not. Changes nothing.
    pub fn peek(&self, after: Option<u32>, tokens: &[u32]) -> usize {
        if !self.enabled {
            return 0;
        }
        let mut at = after.map(|pg| self.owner_of(pg));
        let mut n = 0;
        for key in self.keys(tokens) {
            let Some(id) = self.child(at, key) else { break };
            n += 1;
            at = Some(id);
        }
        n
    }

    /// Hold up to `n` more pages of `tokens` right after `after`, as `peek` counts them, giving
    /// each node that is not resident a device page (taken here; stops when none is left).
    /// Returns the held device pages, each with the host page to fill it from when it was loaded.
    pub fn load(&mut self, after: Option<u32>, tokens: &[u32], n: usize) -> Vec<(u32, Option<u32>)> {
        let mut out = Vec::new();
        if !self.enabled {
            return out;
        }
        let mut at = after.map(|pg| self.owner_of(pg));
        for key in self.keys(tokens).take(n) {
            let Some(id) = self.child(at, key) else { break };
            let from = match self.node(id).dev {
                Some(_) => None,
                None => {
                    // taking cannot drop a node, only device copies of unreferenced ones
                    let Some(pg) = self.take() else { break };
                    self.make_resident(id, pg);
                    Some(id)
                }
            };
            self.hold(id);
            out.push((self.node(id).dev.expect("resident"), from));
            at = Some(id);
        }
        out
    }

    /// Enter the caller's full device page `pg` holding `key` under `parent` (the caller's
    /// previous page, held by it; `None` at the root). The page that now holds this prefix is
    /// referenced by the caller: `pg` unless `Insert::Shared` says otherwise.
    pub fn insert(&mut self, parent: Option<u32>, key: &[u32], pg: u32) -> Insert {
        debug_assert!(self.enabled && key.len() == self.key_len() && self.owner[pg as usize].is_none());
        let at = parent.map(|p| self.owner_of(p));
        debug_assert!(at.is_none_or(|p| self.node(p).refs > 0));
        if let Some(id) = self.child(at, key) {
            if let Some(have) = self.node(id).dev {
                self.hold(id);
                return Insert::Shared(have);
            }
            // only the host holds it: this page, the same bits, becomes its device copy
            self.make_resident(id, pg);
            self.hold(id);
            return Insert::Added(None);
        }
        let id = if self.host {
            match self.host_page() {
                Some(h) => h,
                None => return Insert::Full,
            }
        } else {
            pg
        };
        let siblings = match at {
            None => &mut self.roots,
            Some(p) => &mut self.node_mut(p).children,
        };
        siblings.insert(key.into(), id);
        // (a referenced parent is in no eviction set, so it needs no reclassing)
        self.nodes[id as usize] =
            Some(Node { parent: at, key: key.into(), children: HashMap::new(), dev: Some(pg), refs: 1, released: 0 });
        self.owner[pg as usize] = Some(id);
        self.n_nodes += 1;
        self.n_resident += 1;
        Insert::Added(self.host.then_some(id))
    }

    /// Drop every cached page no sequence references, on both tiers. Returns how many.
    pub fn flush(&mut self) -> usize {
        let mut n = 0;
        while let Some(&(_, id)) = self.leaves.first() {
            if let Some(pg) = self.remove(id) {
                self.free.push(pg);
            }
            n += 1;
        }
        n
    }

    /// Drop a sequence's references to its cached device pages (a path from the root).
    pub fn release(&mut self, pages: &[u32]) {
        self.tick += 1;
        for &pg in pages.iter().rev() {
            let id = self.owner_of(pg);
            let (old, tick) = (self.class(id), self.tick);
            let n = self.node_mut(id);
            n.refs -= 1;
            if n.refs == 0 {
                n.released = tick;
            }
            self.reclass(id, old);
        }
    }

    fn hold(&mut self, id: u32) {
        let old = self.class(id);
        self.node_mut(id).refs += 1;
        self.reclass(id, old);
    }

    fn make_resident(&mut self, id: u32, pg: u32) {
        let old = self.class(id);
        self.node_mut(id).dev = Some(pg);
        self.owner[pg as usize] = Some(id);
        self.n_resident += 1;
        self.reclass(id, old);
    }

    /// A host page for a new node: a free one, or the least recently released leaf's.
    fn host_page(&mut self) -> Option<u32> {
        if self.host_free.is_empty() {
            let &(_, id) = self.leaves.first()?;
            if let Some(pg) = self.remove(id) {
                self.free.push(pg);
            }
        }
        self.host_free.pop()
    }

    /// Take an unreferenced leaf out of the tree. Returns its device page, now owned by the
    /// caller, when it was resident.
    fn remove(&mut self, id: u32) -> Option<u32> {
        let old = self.class(id);
        debug_assert!(old.leaf);
        self.unclass(id, old);
        let n = self.nodes[id as usize].take().expect("a node");
        self.n_nodes -= 1;
        match n.parent {
            None => {
                self.roots.remove(&n.key);
            }
            Some(p) => {
                let old = self.class(p);
                self.node_mut(p).children.remove(&n.key);
                self.reclass(p, old);
            }
        }
        if self.host {
            self.host_free.push(id);
        }
        let pg = n.dev?;
        self.owner[pg as usize] = None;
        self.n_resident -= 1;
        Some(pg)
    }

    /// Unreferenced nodes are evictable: leaves leave the tree, and with a host tier resident
    /// nodes can drop their device copy. Unreferenced resident nodes are idle device pages.
    fn class(&self, id: u32) -> Class {
        let n = self.node(id);
        let unref = n.refs == 0;
        Class {
            released: n.released,
            leaf: unref && n.children.is_empty(),
            resident: self.host && unref && n.dev.is_some(),
            idle: unref && n.dev.is_some(),
        }
    }

    /// Move node `id` from the sets of its class before a change (`old`) to those of its class now.
    fn reclass(&mut self, id: u32, old: Class) {
        self.unclass(id, old);
        let new = self.class(id);
        if new.leaf {
            self.leaves.insert((new.released, id));
        }
        if new.resident {
            self.resident.insert((new.released, id));
        }
        self.idle += new.idle as usize;
    }

    fn unclass(&mut self, id: u32, c: Class) {
        if c.leaf {
            self.leaves.remove(&(c.released, id));
        }
        if c.resident {
            self.resident.remove(&(c.released, id));
        }
        self.idle -= c.idle as usize;
    }

    fn child(&self, at: Option<u32>, key: &[u32]) -> Option<u32> {
        match at {
            None => self.roots.get(key),
            Some(id) => self.node(id).children.get(key),
        }
        .copied()
    }

    fn owner_of(&self, pg: u32) -> u32 {
        self.owner[pg as usize].expect("a cached page")
    }

    fn node(&self, id: u32) -> &Node {
        self.nodes[id as usize].as_ref().expect("a node in the tree")
    }

    fn node_mut(&mut self, id: u32) -> &mut Node {
        self.nodes[id as usize].as_mut().expect("a node in the tree")
    }

    /// Every structural invariant, or a panic naming the first broken one.
    #[cfg(test)]
    fn check(&self) {
        use std::collections::BTreeSet as Set;
        let ids: Vec<u32> = (0..self.nodes.len() as u32).filter(|&i| self.nodes[i as usize].is_some()).collect();
        assert_eq!(ids.len(), self.n_nodes, "node count");
        let (mut idle, mut leaves, mut resident, mut owned) = (0, Set::new(), Set::new(), Set::new());
        for &id in &ids {
            let n = self.node(id);
            assert_eq!(n.key.len(), self.key_len(), "key length");
            let siblings = match n.parent {
                None => &self.roots,
                Some(p) => {
                    let parent = self.nodes[p as usize].as_ref().expect("parent in the tree");
                    assert!(n.refs <= parent.refs, "a child held more than its parent");
                    &parent.children
                }
            };
            assert_eq!(siblings.get(&n.key), Some(&id), "parent does not point back");
            for (k, &c) in &n.children {
                assert_eq!(self.node(c).parent, Some(id), "child's parent");
                assert_eq!(&self.node(c).key, k, "child's key");
            }
            assert!(n.refs == 0 || n.dev.is_some(), "a referenced node is not resident");
            if let Some(pg) = n.dev {
                assert!(owned.insert(pg), "a device page in two nodes");
                assert_eq!(self.owner[pg as usize], Some(id), "owner of a device page");
                assert!(self.host || pg == id, "without a host tier a node is its device page");
            } else {
                assert!(self.host, "a node without a device page and no host tier");
            }
            if n.refs == 0 {
                if n.children.is_empty() {
                    leaves.insert((n.released, id));
                }
                if n.dev.is_some() {
                    idle += 1;
                    if self.host {
                        resident.insert((n.released, id));
                    }
                }
            }
        }
        let owners = self.owner.iter().filter(|o| o.is_some()).count();
        assert_eq!((owned.len(), owners), (self.n_resident, self.n_resident), "resident count");
        let free: Set<u32> = self.free.iter().copied().collect();
        assert_eq!(free.len(), self.free.len(), "a page twice on the free list");
        assert!(free.is_disjoint(&owned), "a cached page on the free list");
        if self.host {
            let hf: Set<u32> = self.host_free.iter().copied().collect();
            assert_eq!(hf.len(), self.host_free.len(), "a host page twice on its free list");
            assert!(ids.iter().all(|i| !hf.contains(i)), "a node's host page on the free list");
            assert_eq!(hf.len() + ids.len(), self.nodes.len(), "host pages lost");
        }
        assert!(self.roots.values().all(|&r| self.node(r).parent.is_none()), "root entry with a parent");
        assert_eq!(idle, self.idle, "idle count");
        assert_eq!(leaves, self.leaves.iter().copied().collect::<Set<_>>(), "leaves = the unreferenced leaves");
        assert_eq!(resident, self.resident.iter().copied().collect::<Set<_>>(), "resident = unreferenced resident");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sequence's view, as `Paged` keeps it: its pages, the leading ones in the tree.
    struct Seq {
        pages: Vec<u32>,
        shared: usize,
    }

    fn fill(pool: &mut PagePool, toks: &[u32]) -> Seq {
        let held = pool.lookup(&toks[..toks.len() / pool.page * pool.page]);
        let mut s = Seq { shared: held.len(), pages: held };
        while s.pages.len() * pool.page < toks.len() {
            let pg = pool.take().expect("a page");
            s.pages.push(pg);
            let i = s.pages.len() - 1;
            if (i + 1) * pool.page <= toks.len() {
                let parent = i.checked_sub(1).map(|j| s.pages[j]);
                match pool.insert(parent, &toks[i * pool.page..(i + 1) * pool.page], pg) {
                    Insert::Shared(have) => {
                        pool.give_back(pg);
                        s.pages[i] = have;
                    }
                    Insert::Added(_) => {}
                    Insert::Full => panic!("host tier full"),
                }
                s.shared += 1;
            }
        }
        s
    }

    fn drop_seq(pool: &mut PagePool, s: Seq) {
        pool.release(&s.pages[..s.shared]);
        for &pg in &s.pages[s.shared..] {
            pool.give_back(pg);
        }
    }

    #[test]
    fn shares_whole_pages_of_a_common_prefix() {
        let mut pool = PagePool::new(16, 4, true);
        let a = fill(&mut pool, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!((a.pages.len(), a.shared), (3, 2));
        // the same first page, a different second
        let held = pool.lookup(&[1, 2, 3, 4, 5, 6, 7, 0]);
        assert_eq!(held, vec![a.pages[0]]);
        pool.release(&held);
        let held = pool.lookup(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(held, a.pages[..2]);
        pool.release(&held);
        assert!(pool.lookup(&[2, 2, 3, 4]).is_empty());
        assert_eq!(pool.available(), 13);
        drop_seq(&mut pool, a);
        assert_eq!((pool.available(), pool.cached(), pool.idle), (16, 2, 2));
        pool.check();
    }

    #[test]
    fn next_token_keys_share_a_page_only_with_the_token_after_it() {
        let mut pool = PagePool::new(8, 4, true);
        pool.key_next_token();
        // a sequence holding 1..=9: its two full pages enter with the token after each
        let (p0, p1) = (pool.take().unwrap(), pool.take().unwrap());
        assert_eq!(pool.insert(None, &[1, 2, 3, 4, 5], p0), Insert::Added(None));
        assert_eq!(pool.insert(Some(p0), &[5, 6, 7, 8, 9], p1), Insert::Added(None));
        // another token after the second page: only the first matches
        let held = pool.lookup(&[1, 2, 3, 4, 5, 6, 7, 8, 0]);
        assert_eq!(held, vec![p0]);
        // a page of the same tokens followed by another is a node of its own (no Shared)
        let p2 = pool.take().unwrap();
        assert_eq!(pool.insert(Some(p0), &[5, 6, 7, 8, 0], p2), Insert::Added(None));
        let both = pool.lookup(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(both, vec![p0, p1]);
        // no token after the page yet: no match
        assert!(pool.lookup(&[1, 2, 3, 4]).is_empty());
        assert_eq!(pool.peek(None, &[1, 2, 3, 4, 5, 6, 7, 8, 0]), 2);
        pool.check();
        pool.release(&both);
        pool.release(&[p0, p2]);
        pool.release(&[p0, p1]);
        assert_eq!((pool.cached(), pool.available()), (3, 8));
        pool.check();
    }

    #[test]
    fn evicts_unreferenced_leaves_oldest_first() {
        let mut pool = PagePool::new(6, 2, true);
        let a = fill(&mut pool, &[1, 2, 3, 4, 5, 6]);
        let b = fill(&mut pool, &[1, 2, 9, 9]);
        assert_eq!(b.pages[0], a.pages[0]);
        let (a_pages, b_pages) = (a.pages.clone(), b.pages.clone());
        drop_seq(&mut pool, a);
        drop_seq(&mut pool, b);
        assert_eq!((pool.cached(), pool.available()), (4, 6));
        // two free pages, then a's leaf (released first), then its parent, then b's leaf; the
        // shared root page goes last
        let order: Vec<u32> = (0..6).map(|_| pool.take().unwrap()).collect();
        assert_eq!(order[2..], [a_pages[2], a_pages[1], b_pages[1], a_pages[0]]);
        assert!(pool.take().is_none());
        assert_eq!((pool.cached(), pool.idle, pool.leaves.len()), (0, 0, 0));
    }

    #[test]
    fn referenced_pages_are_never_evicted() {
        let mut pool = PagePool::new(4, 2, true);
        let a = fill(&mut pool, &[1, 2, 3, 4]);
        drop_seq(&mut pool, a);
        let held = pool.lookup(&[1, 2, 3, 4]);
        assert_eq!(pool.available(), 2);
        assert!(pool.take().is_some() && pool.take().is_some() && pool.take().is_none());
        pool.release(&held);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn a_page_filled_twice_is_shared() {
        let mut pool = PagePool::new(8, 2, true);
        // both start before either has filled a page: the second switches to the first's copy
        let p1 = pool.take().unwrap();
        let p2 = pool.take().unwrap();
        assert_eq!(pool.insert(None, &[7, 7], p1), Insert::Added(None));
        assert_eq!(pool.insert(None, &[7, 7], p2), Insert::Shared(p1));
        pool.give_back(p2);
        assert_eq!((pool.cached(), pool.node(p1).refs), (1, 2));
        // and its next page enters the tree under the shared one
        let p3 = pool.take().unwrap();
        assert_eq!(pool.insert(Some(p1), &[8, 8], p3), Insert::Added(None));
        pool.release(&[p1]);
        pool.release(&[p1, p3]);
        assert_eq!((pool.available(), pool.cached()), (8, 2));
        assert_eq!(pool.flush(), 2);
        assert_eq!((pool.available(), pool.cached(), pool.free.len()), (8, 0, 8));
    }

    /// The device drops its copies to make room and the nodes stay, host-only: a lookup stops
    /// before them, `peek` sees them, `load` brings them back into fresh device pages and names
    /// the host page each comes from.
    #[test]
    fn the_host_keeps_what_the_device_drops() {
        let mut pool = PagePool::with_host(4, 16, 2);
        let a = fill(&mut pool, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!((a.shared, pool.cached(), pool.host_cached()), (4, 4, 4));
        let a_nodes: Vec<u32> = a.pages.iter().map(|&pg| pool.owner_of(pg)).collect();
        drop_seq(&mut pool, a);
        let b = fill(&mut pool, &[9, 9, 9, 9, 9, 9, 9, 9]);
        assert_eq!((pool.cached(), pool.host_cached()), (4, 8), "a's device copies dropped, its nodes kept");
        drop_seq(&mut pool, b);
        pool.check();
        let prompt = [1, 2, 3, 4, 5, 6, 7, 8];
        assert!(pool.lookup(&prompt).is_empty(), "nothing of a is resident");
        assert_eq!(pool.peek(None, &prompt), 4);
        assert_eq!(pool.peek(None, &[1, 2, 3, 4, 0, 0]), 2);
        let got = pool.load(None, &prompt, 3);
        assert_eq!(
            got.iter().map(|g| g.1).collect::<Vec<_>>(),
            a_nodes[..3].iter().map(|&n| Some(n)).collect::<Vec<_>>()
        );
        pool.check();
        // the rest after what is held now: one more page
        let held: Vec<u32> = got.iter().map(|g| g.0).collect();
        assert_eq!(pool.peek(held.last().copied(), &prompt[6..]), 1);
        assert_eq!(pool.available(), 1);
        pool.release(&held);
        // resident again: a plain lookup finds the three
        assert_eq!(pool.lookup(&prompt), held);
        pool.release(&held);
        pool.check();
    }

    /// A page computed again while only the host holds its prefix becomes the node's device
    /// copy (nothing to write back); evicting from the host removes a leaf with its device copy.
    #[test]
    fn host_only_nodes_adopt_and_host_eviction_frees_the_device() {
        let mut pool = PagePool::with_host(2, 3, 1);
        let a = fill(&mut pool, &[5]);
        drop_seq(&mut pool, a);
        let p = pool.take().unwrap();
        let q = pool.take().unwrap(); // drops a's device copy
        pool.give_back(p);
        assert_eq!((pool.cached(), pool.host_cached()), (0, 1));
        assert_eq!(pool.insert(None, &[5], q), Insert::Added(None));
        assert_eq!((pool.cached(), pool.host_cached()), (1, 1));
        pool.release(&[q]);
        // two new roots fill the host tier; the third evicts the oldest leaf: [5], resident
        let b = fill(&mut pool, &[6]);
        drop_seq(&mut pool, b);
        let avail = pool.available();
        let c = fill(&mut pool, &[7]);
        assert_eq!(pool.host_cached(), 3);
        drop_seq(&mut pool, c);
        let d = fill(&mut pool, &[8]);
        assert_eq!(pool.host_cached(), 3);
        assert_eq!(pool.peek(None, &[5]), 0, "the oldest leaf left the tree");
        drop_seq(&mut pool, d);
        assert_eq!(pool.available(), avail);
        pool.check();
        assert_eq!(pool.flush(), 3);
        assert_eq!((pool.cached(), pool.host_cached(), pool.available()), (0, 0, 2));
        pool.check();
    }

    /// xorshift64*: deterministic randomness without a dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// What the pool's caller keeps in the pages: a fingerprint of the prefix whose cache each
    /// device and host page holds (0: nothing of use), moved the way `Paged` moves the bytes.
    struct Mem {
        dev: Vec<u64>,
        host: Vec<u64>,
    }

    const FP_START: u64 = 0xcbf2_9ce4_8422_2325;

    fn fp_extend(h: u64, toks: &[u32]) -> u64 {
        toks.iter().fold(h, |h, &t| (h ^ u64::from(t + 1)).wrapping_mul(0x0100_0000_01b3))
    }

    fn fingerprint(toks: &[u32]) -> u64 {
        fp_extend(FP_START, toks)
    }

    /// A sequence driven the way `Paged` drives the pool: lookup at start (one token short), the
    /// host tier's part loaded, pages taken as it grows, full pages entered (switching to the
    /// tree's copy, writing new ones through), released at the end. With next-token keys a page's
    /// cache also depends on the token after it (as an EAGLE-3 head's does), and it enters once
    /// the sequence holds that token.
    struct Sim {
        toks: Vec<u32>,
        pages: Vec<u32>,
        shared: usize,
    }

    fn sim_start(pool: &mut PagePool, mem: &mut Mem, prompt: &[u32]) -> (Sim, usize) {
        let extra = pool.key_len() - pool.page;
        let n = prompt.len().saturating_sub(1) / pool.page * pool.page;
        let n_key = (n + extra).min(prompt.len());
        let held = pool.lookup(&prompt[..n_key]);
        let mut s = Sim { toks: prompt[..held.len() * pool.page].to_vec(), shared: held.len(), pages: held };
        let rest = &prompt[s.toks.len()..n_key];
        let more = pool.peek(s.pages.last().copied(), rest);
        let mut loaded = 0;
        for (pg, from) in pool.load(s.pages.last().copied(), rest, more) {
            if let Some(h) = from {
                mem.dev[pg as usize] = mem.host[h as usize];
                loaded += 1;
            }
            s.pages.push(pg);
            s.shared += 1;
        }
        s.toks = prompt[..s.shared * pool.page].to_vec();
        (s, loaded)
    }

    /// Append `more`; false (and nothing changed) when the pool cannot supply the pages.
    fn sim_grow(pool: &mut PagePool, mem: &mut Mem, s: &mut Sim, more: &[u32]) -> bool {
        let (page, extra) = (pool.page, pool.key_len() - pool.page);
        let want = (s.toks.len() + more.len()).div_ceil(page);
        if want - s.pages.len() > pool.available() {
            return false;
        }
        while s.pages.len() < want {
            let pg = pool.take().expect("available said so");
            mem.dev[pg as usize] = 0;
            s.pages.push(pg);
        }
        // the step writes the new positions' cache (with next-token keys, the last full page's too:
        // its last position now reads the token after it)
        let from = s.toks.len().saturating_sub(extra) / page;
        s.toks.extend_from_slice(more);
        for i in from..s.pages.len() {
            mem.dev[s.pages[i] as usize] = fingerprint(&s.toks[..((i + 1) * page + extra).min(s.toks.len())]);
        }
        while (s.shared + 1) * page + extra <= s.toks.len() {
            let i = s.shared;
            let parent = i.checked_sub(1).map(|j| s.pages[j]);
            match pool.insert(parent, &s.toks[i * page..(i + 1) * page + extra], s.pages[i]) {
                Insert::Added(save) => {
                    if let Some(h) = save {
                        mem.host[h as usize] = mem.dev[s.pages[i] as usize];
                    }
                }
                Insert::Shared(have) => {
                    pool.give_back(s.pages[i]);
                    s.pages[i] = have;
                }
                Insert::Full => break,
            }
            s.shared += 1;
        }
        true
    }

    fn sim_end(pool: &mut PagePool, s: Sim) {
        pool.release(&s.pages[..s.shared]);
        for &pg in &s.pages[s.shared..] {
            pool.give_back(pg);
        }
    }

    /// The path from the root to a sequence's last shared page spells its tokens.
    fn path_matches(pool: &PagePool, s: &Sim) -> bool {
        let Some(last) = s.shared.checked_sub(1) else { return true };
        let mut id = pool.owner_of(s.pages[last]);
        for i in (0..s.shared).rev() {
            let n = pool.node(id);
            if n.dev != Some(s.pages[i])
                || s.toks.get(i * pool.page..i * pool.page + pool.key_len()) != Some(&n.key[..])
            {
                return false;
            }
            match n.parent {
                Some(p) => id = p,
                None => return i == 0,
            }
        }
        false
    }

    /// Every page holds what the tree (or the sequence owning it) says: each node's device and
    /// host copies the cache of its path, each sequence's full pages that of its tokens.
    fn contents_match(pool: &PagePool, mem: &Mem, live: &[Sim]) -> bool {
        // depth first from the roots: a node holds the cache of its path's tokens and its key's (the
        // token after its page too, with next-token keys); its children extend its page's tokens
        let page = pool.page;
        let mut stack: Vec<(u32, u64)> = pool.roots.values().map(|&r| (r, FP_START)).collect();
        while let Some((id, above)) = stack.pop() {
            let n = pool.node(id);
            let want = fp_extend(above, &n.key);
            if n.dev.is_some_and(|pg| mem.dev[pg as usize] != want) || (pool.host && mem.host[id as usize] != want) {
                return false;
            }
            let path = fp_extend(above, &n.key[..page]);
            stack.extend(n.children.values().map(|&c| (c, path)));
        }
        let extra = pool.key_len() - page;
        live.iter().all(|s| {
            (0..s.toks.len() / page).all(|i| {
                mem.dev[s.pages[i] as usize] == fingerprint(&s.toks[..((i + 1) * page + extra).min(s.toks.len())])
            })
        })
    }

    /// Thousands of random starts, appends, ends and flushes over a 3-token alphabet (so prefixes
    /// collide and pages are filled twice), in pools small enough to evict constantly, with and
    /// without a host tier: every invariant holds after every operation, no page of either tier
    /// is lost or duplicated, every sequence's shared pages spell its tokens, and every page
    /// holds what the tree says it holds (so a hit, from either tier, reads the right cache); with
    /// page keys and with next-token keys.
    #[test]
    fn random_operations_keep_every_invariant() {
        let shapes = [
            (1u64, 24, 0, 2),
            (2, 64, 0, 2),
            (3, 40, 0, 4),
            (4, 200, 0, 3),
            (5, 16, 0, 1),
            (6, 24, 40, 2),
            (7, 16, 64, 2),
            (8, 40, 41, 4),
            (9, 20, 200, 3),
            (10, 8, 30, 1),
        ];
        for ((seed, n_pages, n_host, page), next_token) in shapes.into_iter().flat_map(|x| [(x, false), (x, true)]) {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut pool = if n_host > 0 {
                PagePool::with_host(n_pages, n_host, page)
            } else {
                PagePool::new(n_pages, page, true)
            };
            if next_token {
                pool.key_next_token();
            }
            let mut mem = Mem { dev: vec![0; n_pages], host: vec![0; n_host] };
            let mut live: Vec<Sim> = Vec::new();
            let (mut hits, mut loads, mut evicted_on_take) = (0usize, 0usize, 0usize);
            for op in 0..6000 {
                match rng.below(10) {
                    0..=2 if live.len() < 8 => {
                        let len = 1 + rng.below(6 * page + 3);
                        let prompt: Vec<u32> = (0..len).map(|_| rng.below(3) as u32).collect();
                        let (mut s, loaded) = sim_start(&mut pool, &mut mem, &prompt);
                        hits += s.shared;
                        loads += loaded;
                        let rest = prompt[s.toks.len()..].to_vec();
                        if sim_grow(&mut pool, &mut mem, &mut s, &rest) {
                            live.push(s);
                        } else {
                            sim_end(&mut pool, s);
                        }
                    }
                    3..=6 if !live.is_empty() => {
                        let i = rng.below(live.len());
                        let more: Vec<u32> = (0..1 + rng.below(2 * page)).map(|_| rng.below(3) as u32).collect();
                        let before = pool.cached();
                        if !sim_grow(&mut pool, &mut mem, &mut live[i], &more) {
                            let s = live.swap_remove(i);
                            sim_end(&mut pool, s);
                        } else if pool.cached() < before {
                            evicted_on_take += 1;
                        }
                    }
                    7..=8 if !live.is_empty() => {
                        let s = live.swap_remove(rng.below(live.len()));
                        sim_end(&mut pool, s);
                    }
                    9 if rng.below(20) == 0 => {
                        pool.flush();
                    }
                    _ => {}
                }
                pool.check();
                let private: usize = live.iter().map(|s| s.pages.len() - s.shared).sum();
                assert_eq!(pool.free.len() + pool.cached() + private, n_pages, "seed {seed} op {op}: pages lost");
                assert!(live.iter().all(|s| path_matches(&pool, s)), "seed {seed} op {op}: a path is wrong");
                assert!(contents_match(&pool, &mem, &live), "seed {seed} op {op}: a page holds the wrong cache");
            }
            for s in live.drain(..) {
                sim_end(&mut pool, s);
            }
            pool.check();
            assert_eq!(pool.available(), n_pages);
            pool.flush();
            assert_eq!((pool.free.len(), pool.cached(), pool.host_cached()), (n_pages, 0, 0));
            assert!(hits > 0 && evicted_on_take > 0, "seed {seed}: hits {hits}, evictions {evicted_on_take}");
            assert!(n_host == 0 || loads > 0, "seed {seed}: nothing came back from the host");
        }
    }

    /// A 20,000-token prompt (10,000 pages deep): lookups, releases and leaf-first eviction of
    /// the whole chain stay iterative, and the chain is reclaimed deepest page first.
    #[test]
    fn a_very_long_chain() {
        let (n, page) = (10_000usize, 2usize);
        let mut pool = PagePool::new(n + 1, page, true);
        let mut mem = Mem { dev: vec![0; n + 1], host: Vec::new() };
        let toks: Vec<u32> = (0..n * page).map(|i| (i % 7) as u32).collect();
        let (mut s, _) = sim_start(&mut pool, &mut mem, &toks);
        assert!(sim_grow(&mut pool, &mut mem, &mut s, &toks));
        assert_eq!(s.shared, n);
        let chain = s.pages.clone();
        sim_end(&mut pool, s);
        let mut again = toks.clone();
        again.push(0);
        let held = pool.lookup(&again[..n * page]);
        assert_eq!(held, chain);
        pool.release(&held);
        pool.check();
        let _spare = pool.take().unwrap();
        let order: Vec<u32> = (0..n).map(|_| pool.take().unwrap()).collect();
        assert!(order.iter().eq(chain.iter().rev()), "evicted deepest first");
        assert_eq!(pool.cached(), 0);
    }

    /// The same chain with a host tier: the device gives every page up, the host keeps the whole
    /// chain, and one load brings it back with the right bytes.
    #[test]
    fn a_very_long_chain_through_the_host() {
        let (n, page) = (10_000usize, 2usize);
        let mut pool = PagePool::with_host(n + 1, 2 * n, page);
        let mut mem = Mem { dev: vec![0; n + 1], host: vec![0; 2 * n] };
        let toks: Vec<u32> = (0..n * page).map(|i| (i % 7) as u32).collect();
        let (mut s, _) = sim_start(&mut pool, &mut mem, &toks);
        assert!(sim_grow(&mut pool, &mut mem, &mut s, &toks));
        sim_end(&mut pool, s);
        let all: Vec<u32> = (0..=n).map(|_| pool.take().unwrap()).collect();
        for pg in all {
            mem.dev[pg as usize] = 0;
            pool.give_back(pg);
        }
        assert_eq!((pool.cached(), pool.host_cached()), (0, n));
        let mut again = toks.clone();
        again.push(0);
        let (s, loaded) = sim_start(&mut pool, &mut mem, &again);
        assert_eq!((s.shared, loaded), (n, n));
        assert!(contents_match(&pool, &mem, std::slice::from_ref(&s)));
        sim_end(&mut pool, s);
        pool.check();
    }

    /// 300 prompts sharing one first page and diverging in the second: one shared node with 300
    /// children; eviction takes the children in release order and the shared page last.
    #[test]
    fn a_wide_fan_out() {
        let mut pool = PagePool::new(302, 2, true);
        let mut mem = Mem { dev: vec![0; 302], host: Vec::new() };
        let mut seconds = Vec::new();
        for i in 0..300u32 {
            let prompt = [9, 9, i, i];
            let (mut s, _) = sim_start(&mut pool, &mut mem, &prompt);
            let rest = prompt[s.toks.len()..].to_vec();
            assert!(sim_grow(&mut pool, &mut mem, &mut s, &rest));
            seconds.push(s.pages[1]);
            sim_end(&mut pool, s);
        }
        pool.check();
        assert_eq!(pool.cached(), 301);
        let root = pool.lookup(&[9, 9]);
        assert_eq!(pool.node(root[0]).children.len(), 300);
        pool.release(&root);
        let order: Vec<u32> = (0..302).map(|_| pool.take().unwrap()).collect();
        assert!(order[1..301].iter().eq(seconds.iter()), "children in release order");
        assert_eq!(order[301], root[0], "the shared page last");
    }

    /// Flushing drops only what no sequence holds; the same tokens at another depth are another
    /// node; a prompt shorter than a page matches nothing.
    #[test]
    fn flush_depth_and_short_prompts() {
        let mut pool = PagePool::new(16, 2, true);
        let mut mem = Mem { dev: vec![0; 16], host: Vec::new() };
        let (mut a, _) = sim_start(&mut pool, &mut mem, &[1, 1, 1, 1, 2]);
        assert!(sim_grow(&mut pool, &mut mem, &mut a, &[1, 1, 1, 1, 2]));
        // [1, 1] at depth 0 and again at depth 1: two nodes
        assert_eq!(pool.cached(), 2);
        assert_ne!(a.pages[0], a.pages[1]);
        let (mut b, _) = sim_start(&mut pool, &mut mem, &[5, 5, 5]);
        assert!(sim_grow(&mut pool, &mut mem, &mut b, &[5, 5, 5]));
        sim_end(&mut pool, b);
        assert_eq!(pool.flush(), 1, "only the released page");
        pool.check();
        assert_eq!(pool.lookup(&[1, 1, 1, 1]), a.pages[..2]);
        pool.release(&a.pages[..2]);
        assert!(pool.lookup(&[1]).is_empty() && pool.lookup(&[]).is_empty());
        sim_end(&mut pool, a);
        assert_eq!(pool.flush(), 2);
        assert_eq!(pool.available(), 16);
    }

    #[test]
    fn disabled_is_a_free_list() {
        let mut pool = PagePool::new(3, 2, false);
        let pg = pool.take().unwrap();
        assert!(pool.lookup(&[1, 2]).is_empty());
        assert_eq!(pool.peek(None, &[1, 2]), 0);
        pool.give_back(pg);
        assert_eq!(pool.available(), 3);
    }
}
