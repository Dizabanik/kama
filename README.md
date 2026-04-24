# kama

A fast, single-header string hash map for C. Built around a Swiss-table layout with SIMD-accelerated probing, [RapidHash](https://github.com/Nicoshev/rapidhash), and a short-key inline storage optimization.

---

## Features

- **Single header** - drop `kama.h` into your project, done
- **Type-safe via macro codegen** - `kama(mymap, int)` generates a fully typed map
- **SIMD probe acceleration** - AVX-512, AVX2, SSE2, ARM NEON, with a scalar fallback
- **Short key optimization** - keys ≤ 8 bytes are stored inline, no pointer indirection
- **RapidHash** - a fast, high-quality 64-bit hash function embedded directly
- **Load factor ~85%** with tombstone-aware resizing

---

## Usage

```c
#include "kama.h"

// Generate a map type: string keys -> int values
kama(imap, int)

int main(void) {
    imap_t map;
    imap_init(&map, 64);

    imap_put(&map, "hello", 5, 42);
    imap_put(&map, "world", 5, 99);

    int val;
    if (imap_get(&map, "hello", 5, &val))
        printf("%d\n", val); // 42

    imap_delete(&map, "hello", 5);

    imap_free(&map);
}
```

The `kama(NAME, VAL_TYPE)` macro generates:

| Function                       | Description                     |
| ------------------------------ | ------------------------------- |
| `NAME_init(map, cap)`          | Initialize with a capacity hint |
| `NAME_put(map, key, len, val)` | Insert or overwrite             |
| `NAME_get(map, key, len, out)` | Lookup, returns 1 on hit        |
| `NAME_delete(map, key, len)`   | Mark slot as deleted            |
| `NAME_free(map)`               | Release all memory              |

---

## Benchmarks

Measured against [Abseil's `flat_hash_map`](https://abseil.io/) with `absl::string_view` for fairness. Lower `ns/op`, `c/op` and higher `Mop/s` is better.

```
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┓
┃                        ┃      kama ┃    kama ┃    kama ┃    abseil ┃  abseil ┃  abseil ┃      Δ vs ┃
┃ Benchmark              ┃     Mop/s ┃   ns/op ┃    c/op ┃     Mop/s ┃   ns/op ┃    c/op ┃    abseil ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━┩
│ insert_grow            │     36.30 │    27.5 │   124.2 │     26.28 │    38.0 │   171.6 │    +38.1% │
│ insert_prealloc        │    114.55 │     8.7 │    39.4 │     28.26 │    35.4 │   159.6 │   +305.3% │
│ insert_overwrite       │     50.99 │    19.6 │    88.5 │     47.14 │    21.2 │    95.7 │     +8.2% │
│ lookup_hit             │     51.10 │    19.6 │    88.2 │     50.39 │    19.9 │    89.5 │     +1.4% │
│ lookup_miss            │    199.33 │     5.0 │    22.6 │    118.67 │     8.4 │    38.0 │    +68.0% │
│ lookup_mixed           │     59.48 │    16.8 │    75.8 │     56.82 │    17.6 │    79.4 │     +4.7% │
│ lookup_tombstone       │     49.89 │    20.0 │    90.4 │     44.52 │    22.5 │   101.3 │    +12.1% │
│ delete_all             │     51.81 │    19.3 │    87.0 │     36.67 │    27.3 │   123.0 │    +41.3% │
│ delete_random_half     │     19.21 │    52.1 │   234.8 │     15.16 │    66.0 │   297.6 │    +26.7% │
│ churn                  │     55.15 │    18.1 │    81.8 │     40.51 │    24.7 │   111.3 │    +36.1% │
│ read_heavy_90_10       │     47.14 │    21.2 │    95.7 │     46.93 │    21.3 │    96.1 │     +0.5% │
│ write_heavy_10_90      │    116.53 │     8.6 │    38.7 │     30.63 │    32.6 │   147.3 │   +280.5% │
│ mixed_crud             │     19.19 │    52.1 │   235.0 │     16.06 │    62.2 │   280.8 │    +19.5% │
│ zipfian_hotpath        │      3.82 │   262.1 │  1182.0 │      3.45 │   289.6 │  1306.1 │    +10.5% │
│ insert_fixed_16b       │    138.17 │     7.2 │    32.6 │     27.77 │    36.0 │   162.4 │   +397.5% │
└────────────────────────┴───────────┴─────────┴─────────┴───────────┴─────────┴─────────┴───────────┘
```

The biggest wins are in write-heavy workloads, as pre-allocated insertion is 4x faster, write-heavy mixed traffic is 3.8x faster, and miss lookups are 68% faster. In read-heavy or overwrite-dominated workloads the gap narrows, but kama stays ahead.

---

## How it works

**Control bytes.** Each slot has a 1-byte tag: `0xFF` (empty), `0xFE` (deleted), or the top 7 bits of the hash. Probing loads a full SIMD register of control bytes at once and checks all of them in parallel.

**Collision resolution.** Linear probing in groups of `KAMA_GROUP_WIDTH` bytes (64 for AVX-512, 32 for AVX2, 16 for SSE/NEON). On a match in the control byte, the full hash and key are checked to confirm.

**Short keys.** Keys of 8 bytes or fewer are stored directly in the slot as a `uint64_t`. This eliminates a pointer dereference on every lookup and comparison.

**Tombstone-aware resizing.** When `size + tombstones` exceeds the load threshold, the map rehashes. If there are few live entries but many tombstones, it rehashes at the same capacity instead of doubling.

---

## SIMD support

| Architecture | Width    | Intrinsics       |
| ------------ | -------- | ---------------- |
| AVX-512      | 64 bytes | `_mm512_*`       |
| AVX2         | 32 bytes | `_mm256_*`       |
| SSE2         | 16 bytes | `_mm_*`          |
| ARM NEON     | 16 bytes | `vceqq_u8`, etc. |
| Scalar       | 8 bytes  | plain C          |

The correct path is selected at compile time via preprocessor. No runtime dispatch.

---

## Caveats

- **Keys are not copied.** For keys longer than 8 bytes, kama stores the pointer as-is. The caller is responsible for keeping the key alive.
- **String-keyed only.** The API takes `const char *` + `size_t len`. If you need integer keys, wrap them.
- **Not thread-safe.** No locking of any kind.
- **GCC/Clang only.** The header uses GCC/Clang extensions (`__attribute__`, `__builtin_*`). MSVC is partially supported.

---

## License

MIT
