# kama

A fast, single-header C11 hash map. Built around a Swiss-table layout with SIMD-accelerated probing, embedded [RapidHash](https://github.com/Nicoshev/rapidhash), and inline short keys. Supports typed maps and sets with string, integer, pointer, or fixed-size binary keys. C++17 is supported for trivial C-compatible value types.

---

## Features

- **Single header** - drop `kama.h` into your project, done; no library to link
- **Type-safe via macro codegen** - `kama(mymap, int)` generates a fully typed map
- **SIMD probe acceleration** - AVX-512BW, AVX2, SSE2, ARM NEON, with a word-at-a-time scalar fallback
- **Short key optimization** - string keys ≤ 8 bytes are copied inline, no pointer indirection
- **Optional packed layout** - `kama_packed` also copies exact 16-byte keys inline
- **Specialized keys and sets** - uint32, uint64, pointer identity, fixed-size binary keys, and sets without dummy values
- **RapidHash V3** - pinned and embedded with its license; cached short-key seed preparation preserves hash output
- **Load factor ~85%** - triangular probing, tombstone-aware cleanup, and direct migration
- **Allocation control** - lazy initialization, checked allocation failures, reserve, compact, clear, and shrink
- **Fewer repeated lookups** - value pointers, entry/update, prehashed operations, and batch APIs
- **Configurable ownership** - borrowed or copied long string keys, custom allocators, hash/equality hooks, and optional statistics
- **Optional keyed hashing** - SipHash-2-4 hook for caller-supplied secret keys

---

## Usage

```c
#include "kama.h"
#include <stdio.h>

// Generate a map type: binary string keys -> int values
kama(imap, int)

int main(void) {
    imap_t map;
    if (!imap_init(&map, 0)) return 1; // allocate on first write

    if (imap_put(&map, "hello", 5, 42) == KAMA_ERROR ||
        imap_put(&map, "world", 5, 99) == KAMA_ERROR) {
        imap_free(&map);
        return 1;
    }

    int val;
    if (imap_get(&map, "hello", 5, &val))
        printf("%d\n", val); // 42

    imap_entry_t entry = imap_entry(&map, "counter", 7);
    if (!entry.value) {
        imap_free(&map);
        return 1;
    }
    ++*entry.value; // one search; a new value starts at zero

    imap_delete(&map, "hello", 5);
    imap_free(&map);
    return 0;
}
```

Compile a saved example with `cc -std=c11 -O3 example.c -o example`. Use `c++ -std=c++17 -O3 example.cpp -o example` for C++.

The `kama(NAME, VAL_TYPE)` macro generates `NAME_t`, `NAME_entry_t`, `NAME_keyarg_t`, and the functions below. All map pointers must refer to initialized, zero-initialized, or freed objects as appropriate. Index accessors require an occupied slot.

### Initialization and capacity

| Function | Description |
| -------- | ----------- |
| `NAME_init(map, cap)` | Initialize with a slot-capacity hint; zero defers allocation |
| `NAME_init_ex(map, cap, options)` | Initialize with a `const kama_options_t *`; NULL selects defaults |
| `NAME_free(map)` | Free owned keys and table storage; reset the object |
| `NAME_reserve_entries(map, n)` | Reserve room for at least n live entries, accounting for tombstones |
| `NAME_rehash(map, cap)` | Rebuild at a rounded slot capacity large enough for existing entries |
| `NAME_resize(map, cap)` | Alias for checked `rehash` |
| `NAME_compact(map)` | Remove tombstones without changing capacity |
| `NAME_shrink_to_fit(map)` | Shrink to fit live entries; release table storage when empty |
| `NAME_clear(map)` | Remove entries and owned keys while retaining capacity and options |
| `NAME_count(map)` | Return the number of live entries |
| `NAME_empty(map)` | Return 1 when empty |
| `NAME_allocated_bytes(map)` | Return currently requested table and owned-key bytes, excluding allocator overhead |

`init`, `init_ex`, `reserve_entries`, `rehash`, `resize`, `compact`, and `shrink_to_fit` return 1 on success or 0 on failure. `free` and `clear` return void. Capacities round up to a power of two and at least one probe group; capacity hints count slots, while `reserve_entries` counts entries.

`NAME_t map = {0}` is also valid. Initialize only an uninitialized or already freed object: initializing a live map leaks its allocation. A failed initialization is safely freeable. `free` can be repeated; a freed map is reusable with default options and seed zero.

### Reads and writes

| Function | Description |
| -------- | ----------- |
| `NAME_put(map, key, len, val)` | Insert or overwrite; return write status |
| `NAME_put_unique(map, key, len, val)` | Insert without checking for duplicates; key **must be absent** |
| `NAME_get(map, key, len, out)` | Return 1 on hit, 0 on miss; missing output stays untouched; out may be NULL |
| `NAME_contains(map, key, len)` | Return membership without copying a value |
| `NAME_get_ptr(map, key, len)` | Return a pointer to the const value, or NULL |
| `NAME_get_mut(map, key, len)` | Return a pointer to the mutable value, or NULL |
| `NAME_entry(map, key, len)` | Return `{ value, result }`; insert a zero-initialized value if absent |
| `NAME_update(map, key, len, fn, context)` | Find or insert, call `fn(value_pointer, context)`, and return write status |
| `NAME_delete(map, key, len)` | Return 1 if removed, 0 if absent |
| `NAME_find_idx(map, key, len)` | Return an occupied index, or `map->capacity` on miss |
| `NAME_erase_idx(map, index)` | Remove an occupied index; return 1 if removed, otherwise 0 |
| `NAME_key(map, index)` | Return the stored key; string keys are binary spans |
| `NAME_key_len(map, index)` | Return key length in bytes |
| `NAME_val(map, index)` | Return a copy of the stored value |

**Check write statuses:** `KAMA_INSERTED` is 1, `KAMA_UPDATED` is 0, and `KAMA_ERROR` is -1. Test `== KAMA_ERROR`, not boolean truth, for failure. `entry` returns a NULL value pointer on error. The update callback has signature `void fn(VAL_TYPE *value, void *context)`; passing NULL returns an error.

Overwriting preserves the original stored key and does not allocate or grow. Allocation failure preserves existing entries and values. An owned-key insertion may already have rebuilt the table before its key allocation fails, so even a failed write can invalidate pointers. A failed reserve/rehash allocation leaves the old table intact.

### Hash reuse and batches

| Function | Description |
| -------- | ----------- |
| `NAME_hash_key(map, key, len)` | Return a full `uint64_t` hash token |
| `NAME_find_hashed(map, key, len, hash)` | Find an index using a token; return capacity on miss |
| `NAME_get_hashed(map, key, len, hash, out)` | Lookup with the same output rules as `get` |
| `NAME_put_hashed(map, key, len, val, hash)` | Insert or overwrite with a token |
| `NAME_entry_hashed(map, key, len, hash)` | Entry lookup/insertion with a token |
| `NAME_get_batch(map, keys, count, values, found)` | Return hit count; prepare hashes/prefetches in chunks |
| `NAME_put_batch(map, keys, values, count, results)` | Return the number of completed writes; stop on failure |

Hash tokens are valid only for the same key bytes and hash configuration. Prehashed operations still check collisions and manage capacity.

Batch keys are `NAME_keyarg_t`: `kama_key_t { const char *data; size_t len; }` for strings, the scalar key type for integer/pointer maps, or `const void *` for fixed keys. `get_batch` accepts optional value output and `uint8_t found[]` arrays; missing values remain untouched. `put_batch` requires input values; its `kama_result_t results[]` output is optional. On failure, earlier writes remain, `results[completed]` is `KAMA_ERROR` when supplied, and later results are untouched. Batch writes are not atomic.

Input/output arrays must not overlap storage that a map mutation can invalidate. Input key bytes must stay valid throughout the batch.

---

## Key variants

| Macro | Key argument | Storage |
| ----- | ------------ | ------- |
| `kama(NAME, VALUE)` | `const char *key, size_t len` | Separate metadata; keys ≤ 8 bytes inline; balanced default |
| `kama_soa(NAME, VALUE)` | `const char *key, size_t len` | Explicit name for the default layout |
| `kama_aos(NAME, VALUE)` | `const char *key, size_t len` | Metadata beside each key/value pair |
| `kama_packed(NAME, VALUE)` | `const char *key, size_t len` | Colocated metadata; keys ≤ 8 bytes and exactly 16 bytes inline |
| `kama_u32(NAME, VALUE)` | `uint32_t key` | Inline integer key, no metadata array |
| `kama_u64(NAME, VALUE)` | `uint64_t key` | Inline integer key, no metadata array |
| `kama_ptr(NAME, VALUE)` | `const void *key` | Pointer identity; NULL is a valid key |
| `kama_fixed(NAME, N, VALUE)` | `const void *key` | Exactly N readable bytes copied inline; N must be positive |

Integer, pointer and fixed-key maps have the same functions as string maps, with the `len` argument omitted. Integer/pointer keys are returned by value from `NAME_key`; fixed keys return a `const void *` byte span. Integer and pointer maps use specialized mixers by default. Fixed keys use RapidHash, with cached seed preparation for sizes up to 16 bytes.

### Integer key variant

```c
#include "kama.h"
#include <stdio.h>

kama_u32(ids, float)

int main(void) {
    ids_t map = {0};
    if (ids_put(&map, 42, 3.14f) == KAMA_ERROR) {
        ids_free(&map);
        return 1;
    }

    float val;
    if (ids_get(&map, 42, &val))
        printf("%.2f\n", val); // 3.14

    ids_delete(&map, 42);
    ids_free(&map);
    return 0;
}
```

### Sets

Every layout has a set generator: `kama_set(NAME)`, `kama_soa_set(NAME)`, `kama_aos_set(NAME)`, `kama_packed_set(NAME)`, `kama_u32_set(NAME)`, `kama_u64_set(NAME)`, `kama_ptr_set(NAME)`, and `kama_fixed_set(NAME, N)`. Sets have no value field.

| Set function | Description |
| ------------ | ----------- |
| `NAME_insert(map, key, len)` | Insert; return `KAMA_INSERTED`, `KAMA_UPDATED` if already present, or `KAMA_ERROR` |
| `NAME_insert_hashed(map, key, len, hash)` | Insert with a hash token |
| `NAME_contains(map, key, len)` | Return membership |
| `NAME_contains_batch(map, keys, count, found)` | Return hit count; optional `uint8_t found[]` |
| `NAME_insert_batch(map, keys, count, results)` | Return completed count; same partial-failure rules as `put_batch` |

Sets also expose all initialization/capacity functions, `delete`, `erase_idx`, `hash_key`, `find_idx`, `find_hashed`, `key`, `key_len`, and iteration. Omit `len` for integer/pointer/fixed sets. Value accessors, `put`, `entry`, and `update` are map-only.

---

## Key ownership and options

**Short keys are copied.** String keys ≤ 8 bytes are always stored inline. Longer keys are borrowed by default and must remain alive and immutable until removed. Overwriting an entry with an equal key preserves the original key buffer.

**Packed keys.** `kama_packed` additionally copies exact 16-byte keys into the pair's existing key/metadata words. A one-bit-per-slot bitmap preserves all key patterns and all seven control-tag bits. Lengths 9–15 and 17+ follow ordinary borrowed/owned behavior. Inline 16-byte keys are rehashed during migration. With uint64 values, packed storage uses 25.125 bytes per capacity slot versus 25 for the default, excluding the common control clone tail and alignment padding: about 0.5% more.

**Owned strings.** Pass `copy_keys = 1` in `kama_options_t` to copy long string keys too. These copies are also NUL-terminated, but their explicit binary length determines identity. Fixed keys are always copied. Pointer maps store the pointer value and never own its pointee. Values are assigned by value and are never automatically destroyed or freed.

`kama_options_t` is copied by `NAME_init_ex`; start with `{0}` and set the fields needed:

| Field | Description |
| ----- | ----------- |
| `allocate`, `deallocate` | Custom allocator pair; provide both or neither |
| `allocator_context` | Borrowed context for allocation callbacks |
| `seed` | Deterministic 64-bit hash seed; zero by default |
| `copy_keys` | Nonzero copies long string keys |
| `hash` | Optional `uint64_t fn(const void *key, size_t len, uint64_t seed, void *context)` |
| `equal` | Optional `int fn(const void *a, const void *b, size_t len, void *context)`; nonzero means equal |
| `key_context` | Borrowed context for hash/equality callbacks |
| `stats` | Optional pointer to zero-initialized `kama_stats_t`; requires `KAMA_ENABLE_STATS` |

Allocation callbacks have signatures `void *allocate(void *context, size_t bytes, size_t alignment)` and `void deallocate(void *context, void *ptr, size_t bytes, size_t alignment)`. Allocation must honor the requested power-of-two alignment and return NULL on failure.

Equal keys must have identical complete hashes. String equality is called only for equal lengths. Integer/pointer hash and equality hooks receive the key's object bytes; fixed-key hooks receive its N bytes. Keep callback contexts and statistics alive for the map's lifetime. Do not change its hash/equality behavior, seed, allocator or options while entries exist.

Statistics include `probe_groups`, `key_comparisons`, `rehashes`, `migrated_entries`, `allocated_bytes`, and `peak_allocated_bytes`. Reads update probe/comparison counters. Allocation counters track requested bytes, including temporary old/new tables during migration; they exclude allocator overhead and caller-owned data.

For SipHash-2-4, set `options.hash = kama_siphash24` and `options.key_context` to a live `kama_siphash_key_t { k0, k1 }` containing a caller-supplied secret 128-bit key. The options seed is XORed into the first key word. A public seed or fixed public key does not protect against attacker-selected collisions.

---

## Iteration

`kama_foreach` scans occupied indices, skipping empty and deleted slots. String/fixed keys are byte spans, not necessarily NUL-terminated strings. For an `imap` declared in the usage example:

```c
kama_foreach(&map, i) {
    fwrite(imap_key(&map, i), 1, imap_key_len(&map, i), stdout);
    printf(": %d\n", imap_val(&map, i));
}
```

`NAME_erase_idx(map, i)` can remove the current entry during iteration: deletion does not move other entries. Do not insert, reserve, rehash, compact, shrink or clear while continuing an existing index iteration. Use `NAME_key` and `NAME_key_len` rather than interpreting packed entries' raw fields.

Pointers into slots can be invalidated by growth/rebuild, clear, free, or deletion of the pointed-to entry. An update callback must not invalidate its supplied value pointer while using it. Stored keys must never be modified in place.

---

## Benchmarks

The latest lookup measurements compare the reviewed original header (`fed7c6d`, with two collision early-return defects repaired), the first correctness/performance rewrite, and the current default and packed layouts. These are **Kama version comparisons**, not the old Abseil speed claims.

Measured on macOS 26.6.2 ARM64 with Apple Clang 21, C++17, `-O3 -march=native`, NEON groups of 16, uint64 values, and statistics disabled. Each number is median **ns/op** across 12 timed samples: three processes, each with one warmup and four timed iterations. Lower is better.

At **1,048,576 slots and 80% occupancy**:

| Keys | Operation | Original | First rewrite | Current default | Packed |
| ---- | --------- | -------: | ------------: | --------------: | -----: |
| 16 bytes | Shuffled hit | 50.64 | 57.42 | 56.65 | 37.06 |
| 16 bytes | Miss | 12.30 | 13.12 | 11.42 | 12.47 |
| Mixed lengths | Shuffled hit | 88.43 | 99.77 | 94.15 | 95.46 |
| Mixed lengths | Miss | 14.58 | 14.99 | 13.44 | 14.75 |

Mixed lengths cycle through **8, 9, 15, 16, 17, 32 and 64 bytes in the same map**. Queries use independently allocated equal buffers; missing keys are disjoint. The trace seed is 12345678 and the map hash seed is zero. Correctness checks run outside timing. Key lengths below eight are excluded from the original-header comparison because that header had a short-key overread.

The current default reduces mixed-length hit time by **5.6%** and miss time by **10.3%** versus the first rewrite in this configuration. Packed 16-byte hits take **26.8% less time than the original** and **34.6% less than the current default**.

The default's shuffled-hit regression is **not fully removed**: it still takes 11.9% more time than the original for 16-byte keys and 6.5% more for mixed keys here. Packed storage does not consistently improve mixed workloads, so separate metadata remains the default.

These are workload-specific results. Processes were interleaved serially and requested user-initiated macOS QoS; CPU affinity, frequency and background activity were not controlled. For the mixed-hit row, observed sample ranges were 86.45–94.18, 97.26–103.25, 91.53–98.20 and 91.61–102.61 ns/op respectively. Small differences should not be treated as universal gains; no cycles/op estimates or native x86 speed claims are made.

### Validation

The development validation passed 14 native ASan/UBSan test targets, including the 91 historical tests, all key generators, scalar widths 8/16/32/64, packed layouts, forced collisions, allocation failures, over-aligned values, ownership and upstream hash equivalence. Default and packed deterministic fuzz drivers passed 10,000 inputs each. The benchmark harness passed 741 correctness smoke cases across Kama, packed Kama and Abseil with four value types and boundary key lengths; those small cases are not performance comparisons.

GCC 15.2 C11 and G++ C++17 checks also passed in an existing amd64 Docker image under OrbStack emulation: SSE2, scalar widths 8/16/32/64, packed/AoS layouts, hash vectors and UBSan. AVX2 and AVX-512 compiled, but could not execute because the emulator did not expose those CPU flags. Physical x86 performance and AVX runtime validation remain outstanding.

---

## How it works

**Control bytes.** Each slot has a 1-byte tag: `0xFF` (empty), `0xFE` (deleted), or the top seven hash bits. Probing compares a full group of tags at once. A cloned tail permits groups to wrap across the end of the table.

**Collision resolution.** Triangular probing visits groups of `KAMA_GROUP_WIDTH` bytes. Matching tags select candidates for key comparison; unequal candidates never terminate the search. Default string maps also check cached length and hash bits. An empty byte ends a search only after all candidate matches in that group are checked.

**Key storage.** The default separates metadata from values so misses can reject candidates without fetching large values. Short strings, fixed keys and integers avoid external key-pointer loads. Packed strings reuse the key/metadata area for exact 16-byte keys.

**Deletion and migration.** Deletion marks a slot empty only when doing so preserves every overlapping probe window; otherwise it leaves a tombstone. Capacity management distinguishes live occupancy from tombstone pressure and can compact without growing. Migration uses a dedicated vacancy-only path. Overwriting an existing entry never grows the table.

---

## SIMD support

| Architecture | Native group width | Intrinsics |
| ------------ | ------------------ | ---------- |
| AVX-512BW | 64 bytes | `_mm512_*` |
| AVX2 | 32 bytes | `_mm256_*` |
| SSE2 | 16 bytes | `_mm_*` |
| ARM NEON | 16 bytes; optional 8 | `vceqq_u8`, narrowing and sparse masks |
| Scalar | 8 when forced; otherwise the selected width | Word-at-a-time C; supports 8/16/32/64 |

Selection is at compile time, with no runtime dispatch. The header never changes compiler ISA flags; use target flags appropriate for deployment CPUs. Unsupported requested widths fall back to scalar code.

Define tuning options **before** including `kama.h`:

| Option | Default / effect |
| ------ | ---------------- |
| `KAMA_GROUP_WIDTH` | 64 with AVX-512BW, 32 with AVX2, otherwise 16; scalar/NEON8 overrides below; accepts 8/16/32/64 |
| `KAMA_FORCE_SCALAR` | Force scalar probing; defaults to width 8 unless explicitly overridden |
| `KAMA_NEON8` | Select width 8 when width is not explicitly set and AVX is not selected; intended for NEON |
| `KAMA_LINEAR_PROBING=1` | Use linear probing instead of triangular |
| `KAMA_LOOKUP_PREFETCH` | Default 0: off; 1: controls; 2: controls and metadata/slots |
| `KAMA_LOAD_PERCENT` | Default 85; accepts 25 through 95 |
| `KAMA_BATCH_SIZE` | Default 16 independent queries per chunk; must be positive |
| `KAMA_SEED` | Default zero; seed used by explicit initialization with default options |
| `KAMA_ENABLE_STATS` | Compile in statistics updates when an options stats pointer is supplied |

A zero-initialized or freed object uses seed zero regardless of `KAMA_SEED`. All translation units sharing map objects must use the same header and configuration. Wider groups, prefetch and alternative layouts should be measured against the application's workload.

---

## Caveats

- **Borrowed keys need stable storage.** Long strings are borrowed unless copying is enabled; packed exact 16-byte keys are an inline exception.
- **Binary lengths are explicit.** String lengths 0 through UINT32_MAX are supported. NULL is valid only for a zero-length string, or as a pointer-identity key.
- **C++ values must be trivial.** No constructors or destructors run. Use a typedef when a value type contains commas.
- **Mutation needs external synchronization.** Immutable maps can be read concurrently if callbacks are safe and statistics are disabled or NULL. The library provides no locks.
- **Pointers are not stable across rebuilds.** Use the documented invalidation rules for value pointers, key spans and indices.
- **Capacity is bounded.** At most 2^32 slots on 64-bit targets; 32-bit targets and allocation sizes impose lower limits. Overflow checks return failure.
- **Recompile existing consumers.** Hash outputs and object layouts changed from older versions. Writes now return status; the old private `put_hashed` signature with a truncated hash and separate tag has changed to the full-token API above.

---

## License

MIT. The complete license and the embedded RapidHash copyright/license notices are included in `kama.h`.
