# Concurrent Kama

`kama_conc.h` provides typed concurrent maps. Ordinary `kama.h` is unchanged.
Include the header and use `kama_conc(NAME, VALUE)` for string keys or
`kama_conc_u32`, `kama_conc_u64`, `kama_conc_ptr`, `kama_conc_fixed(NAME, N, VALUE)`
for specialized keys. `kama_conc_soa`, `kama_conc_aos` and `kama_conc_packed`
select the corresponding string layouts. Values follow ordinary Kama's C11
and trivial C++17 type requirements.

## Parallel reads and writes

The default has 128 independently locked tables. Each operation computes a full
hash before locking, using RapidHash or the specialized integer mixer. Bits 32
upward select the shard, using different bits from the low home-slot bits and high seven
control bits. The same hash token feeds the shard's probe; the target key is not
hashed again under the lock. Long strings with equal ends still distribute according
to their complete contents. Built-in string hashing is always full RapidHash in
this variant, including when the supplied ordinary options select adaptive
hashing.

Each shard grows and compacts independently. Threads operating on different
shards can read, write, allocate, run callbacks and rebuild at the same time.
Each shard's lock/table control block starts on a 128-byte boundary, so adjacent
blocks do not share cache lines. Empty shard tables allocate lazily; the
lock/table control array is allocated at initialization.

The default uses `os_unfair_lock` on supported Apple targets (macOS 10.12+,
iOS 10+), POSIX mutexes on other POSIX systems, or Windows SRW locks in exclusive
mode. Define `KAMA_CONC_NATIVE_LOCK=0` to retain POSIX mutexes on Apple targets.
The native lock uses the OS blocking primitive directly, with no try-lock retry
loop or busy-spin policy around callbacks or allocation. Define
`KAMA_CONC_RWLOCK=1` to use POSIX reader/writer locks or shared Windows SRW reads.
Multiple readers can then share a shard; writes exclude its readers and other
writers. This can help with long read callbacks, but the extra synchronization
can slow short operations. OS lock scheduling
determines fairness. The [Windows lock contract](https://learn.microsoft.com/en-us/windows/win32/sync/slim-reader-writer--srw--locks)
does not promise FIFO or fairness.

The map uses your threads and does not create a worker pool. A hot key, poor
hash distribution, memory limits or too little independent work can limit scaling. Different keys can also share a shard. Increasing the number
of shards trades more control memory for less contention. At the default cache
line setting, this host's string/uint64 map with mutexes and statistics disabled
uses 32 KiB of shard control storage, plus its handle and any allocated tables/owned keys. Smaller shard
counts are available for small maps.

## A complete multithreaded example

This POSIX C11 example starts four writers, each inserting independent integer
keys, then reads the result after joining them. Windows applications can use the
same map API from their native threads; the library's Windows backend uses SRW
locks and does not require pthreads.

```c
#include "kama_conc.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

kama_conc_u64(shared_map, uint64_t)

typedef struct { shared_map_t *map; uint64_t first; } writer_arg;

static void *write_keys(void *context) {
    writer_arg *arg = (writer_arg *)context;
    for (uint64_t i = 0; i < 100000; ++i) {
        uint64_t key = arg->first + i;
        if (!shared_map_put(arg->map, key, key * 3)) abort();
    }
    return NULL;
}

int main(void) {
    shared_map_t map;
    if (!shared_map_init(&map, 0)) return 1;
    pthread_t threads[4]; writer_arg args[4];
    for (size_t i = 0; i < 4; ++i) {
        args[i].map = &map; args[i].first = i * UINT64_C(100000);
        if (pthread_create(&threads[i], NULL, write_keys, &args[i])) abort();
    }
    for (size_t i = 0; i < 4; ++i)
        if (pthread_join(threads[i], NULL)) abort();
    uint64_t value;
    if (!shared_map_get(&map, 123456, &value)) abort();
    printf("%zu entries; value = %llu\n", shared_map_count(&map),
           (unsigned long long)value);
    shared_map_free(&map);
    return 0;
}
```

Compile with `cc -std=c11 -O3 -pthread -I. example.c -o example` on POSIX.
C++17 callers can use the same header and APIs. For the optional POSIX
reader/writer backend, define `_POSIX_C_SOURCE=200809L` before system headers,
usually as a compiler definition alongside `KAMA_CONC_RWLOCK=1`.

## Operations

The string signatures below omit the leading `NAME_` prefix. Integer, pointer
and fixed-key maps omit `len`, just like ordinary Kama. All single-key operations
are linearizable: each call observes or changes its entry while holding the
corresponding shard lock.

| Function | Behavior |
| --- | --- |
| `init(map, capacity)` | Initialize before sharing; capacity is a total slot hint, distributed across shards |
| `init_ex(map, capacity, options)` | Select shard count, ownership, seed, allocator and hash/equality hooks |
| `free(map)` | Release everything after all users have stopped; repeatable on a zero/freed object |
| `put(map, key, len, value)` | Insert or overwrite; boolean success |
| `get(map, key, len, out)` | Copy a value under the lock; false on miss, leaving output untouched; NULL output is allowed |
| `contains(map, key, len)` | Membership test |
| `delete(map, key, len)` | Remove if present; return whether removed |
| `update(map, key, len, fn, context)` | Find or insert a zero value, then call `void fn(VALUE *, void *)` while exclusively locked |
| `visit(map, key, len, fn, context)` | Call `void fn(VALUE const *, void *)` on a hit while locked, avoiding a value copy |
| `hash_key(map, key, len)` | Compute a reusable full token without taking a shard lock |
| `put_hashed(map, key, len, value, hash)` | Insert/overwrite using a token |
| `get_hashed(map, key, len, hash, out)` | Copy a value using a token |
| `contains_hashed(map, key, len, hash)` | Membership using a token |
| `delete_hashed(map, key, len, hash)` | Remove using a token |
| `update_hashed(map, key, len, hash, fn, context)` | Atomic update using a token |
| `visit_hashed(map, key, len, hash, fn, context)` | Locked visit using a token |
| `count(map)`, `empty(map)` | Exact count or emptiness at one instant; acquire all shards |
| `capacity(map)` | Current total slot capacity; acquire all shards |
| `allocated_bytes(map)` | Current requested shard-control, table and owned-key bytes; acquire all shards |
| `clear(map)` | Atomically clear all entries, retaining table capacity and configuration |
| `reserve_entries(map, entries)` | Distribute an entry budget across shards; acquire all shards |
| `compact(map)`, `shrink_to_fit(map)` | Compact/shrink every shard; retain shard count and control storage |
| `get_stats(map, out)` | Snapshot per-shard statistics when `KAMA_ENABLE_STATS` is defined |
| `for_each(map, fn, context)` | Visit entries one shard at a time; return visited count; stop if the callback returns false |

`update` makes read/modify/write operations such as increments atomic. Separate
`get` and `put` calls remain separate operations. Allocation failure prevents the
update callback from running. Mutation results use the same `true`/`false`
convention as ordinary Kama.

Tokens require the same key bytes, seed and hash/equality configuration. Prehashed
operations reuse the target key's token. Growth or maintenance can still invoke
hash hooks for stored keys: the default string layout caches the routing bits,
but some other layouts/key types recompute hashes during migration. Shard count
does not change the token, and built-in string tokens match ordinary Kama's full
RapidHash tokens.

Operations on the whole map acquire locks in ascending order and release them
in reverse. They can run alongside other operations, but temporarily block the
whole map. A hot shard can still grow after distributed reservation. A failed reserve/compact/shrink can leave earlier shards resized, while
preserving entries and values. Writes can also rebuild before a final owned-key
allocation fails; no pointers into the affected table escape through `get`.

The `for_each` callback has signature
`bool fn(const char *key, size_t len, VALUE const *value, void *context)` for
strings, or `bool fn(KEY key, VALUE const *value, void *context)` for other key
types (`const void *` for fixed keys). Each shard is stable during its callback
sequence, but the traversal as a whole is not a global snapshot. A concurrent
insert/delete in a different shard may or may not appear. An exact `count` can
therefore differ from a concurrent traversal's visited count.

## Ownership and lifetime

Initialize successfully before publishing the handle to other threads, using
thread creation or the application's synchronization. No other thread may use the map during initialization, reinitialization or
destruction. Join all users, or otherwise stop their access, before freeing. Do not copy an initialized handle
by value. Zero/freed handles have no locks or tables; entry lookups and writes
fail until explicitly initialized. Aggregate queries return zero/empty, and
`clear`/`free` are harmless. There is no concurrent lazy initialization of the
handle. Treat shard fields and generated inner-table helpers as private.

String ownership matches ordinary Kama: up to 8 bytes are inline, packed maps
also inline exact 16-byte keys, and longer keys are borrowed unless
`options.map.copy_keys=1`. Borrowed bytes must stay alive and immutable until
removal, including while other threads might be reading them. Input bytes must
remain valid for the duration of a call. Copied keys provide independent storage.
Values are copied by value, with no automatic destruction or pointee management.

Use `get` for copied values and `visit`/`update` for locked access. Pointers passed
to callbacks are valid only during that callback. Callbacks must not reenter the
same map or retain pointers into it. C++ callback exceptions release their locks;
entry creation or changes already made by an update are not rolled back.
Do not leave a C callback through `longjmp` or cancel a thread while it holds a
map lock. Allocator/hash/equality hooks follow their C contracts, must not throw,
and must be safe to invoke concurrently from different threads. Equal keys must
have identical complete hashes; changing hooks, contexts or seed while the map
is in use is unsupported.

## Options and build integration

Start `kama_conc_options_t options = {0}` and choose fields before `init_ex`:

| Field | Default / contract |
| --- | --- |
| `shards` | 0 selects `KAMA_CONC_SHARDS` (128); otherwise a power of two, up to 2^24 |
| `map` | Ordinary `kama_options_t`, including `copy_keys`, seed and thread-safe hooks; built-in string hash policy is forced to full |
| `map.stats` | Must be NULL; use `NAME_get_stats` for snapshots |

Define `KAMA_CONC_SHARDS`, `KAMA_CONC_CACHE_LINE` (128; power of two at least 64),
`KAMA_CONC_RWLOCK` (0), or `KAMA_CONC_NATIVE_LOCK` (1) before inclusion to change
them. `KAMA_CONC_LOCK_BACKEND` names the selected primitive for diagnostics.
All translation units sharing map objects must use the same Kama and
concurrent-header configuration.

Statistics are private to each shard, so their updates do not contend on one
global counter. Statistics builds take exclusive locks for reads too, protecting
ordinary Kama's non-atomic probe counters. `get_stats` returns false without
changing output when statistics are disabled. Allocation totals include shard
control storage. `peak_allocated_bytes` is the sum of the per-shard peaks plus
control storage, an upper bound rather than a simultaneous global high-water
measurement. `count` uses no shared global atomic counter on the write path.

The `dev` source checkout includes CMake integration. Its normal target keeps
its existing dependencies. Enable the concurrent target and tests there with:

```sh
cmake -S . -B build -DKAMA_BUILD_CONCURRENT=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

In a source build, link `kama::conc`. Installed consumers use
`find_package(kamaConc CONFIG REQUIRED)` and link `kama::conc`; this package adds
the platform thread dependency. Ordinary `find_package(kama)` / `kama::kama`
remain independent. Header-only users need just `kama_conc.h`, `kama.h` and their
platform thread library.

## Benchmarks and validation

The [README](README.md#concurrent-maps) compares concurrent Kama with Parallel Hashmap. Raw measurements, source snapshots, sanitizer logs and reproduction tools stay on `dev`.

The native suite passed 27 ASan/UBSan CTests and nine ThreadSanitizer configurations. Both peer adapters passed sanitizer checks, and the existing x86 Docker checks passed. Tests cover same-key increments, independent shards, shared readers, collisions, growth and maintenance, owned keys, allocation failures, seeded/prehashed routing, over-aligned values and C++ exception unwinding. Overlapping histories are checked against a sequential model.

Ordinary `kama.h` and its source templates are unchanged by the concurrent implementation. Representative optimized C11/C++17 put/get assembly is byte-identical with either header included.
