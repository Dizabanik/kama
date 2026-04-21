#ifndef KAMA_H
#define KAMA_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
// Forces the compiler to allow AVX2 instructions
#pragma GCC target("avx2,bmi,bmi2,popcnt")
#endif
#endif
#if defined(_WIN32) || defined(_WIN64)
#include <malloc.h> /* _aligned_malloc, _aligned_free */
#endif

/* DEFINES */
#define KAMA_INLINE __attribute__((always_inline)) inline
#if defined(__GNUC__) || defined(__clang__)
#define KAMA_PREFETCH(ptr) __builtin_prefetch((ptr), 0, 3)
#else
#define KAMA_PREFETCH(ptr) (void)(ptr)
#endif
#define KAMA_UNLIKELY(x) __builtin_expect(!!(x), 0)

#if defined(_MSC_VER)
#define KAMA_RESTRICT __restrict
#else
#define KAMA_RESTRICT __restrict__
#endif

#define KAMA_MATCH(a, b) kama_int_cmpeq_mask(a, b)

// We try to identificate the SIMD system
#if defined(__AVX512F__)
#include <immintrin.h>
#define KAMA_SIMD_AVX512
#elif defined(__AVX2__)
#include <immintrin.h>
#define KAMA_SIMD_AVX
#elif defined(__SSE__)
#include <xmmintrin.h>
#define KAMA_SIMD_SSE
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define KAMA_SIMD_NEON
#else
#warning "[Kama] No SIMD extension detected, using scalar fallback."
#define KAMA_ARCH_GENERIC
#endif

#ifdef KAMA_SIMD_AVX512
typedef __m512i kama_simd_int;
typedef __mmask64 kama_simd_mask;
#define KAMA_GROUP_WIDTH 64
#elif defined(KAMA_SIMD_AVX)
typedef __m256i kama_simd_int;
typedef uint32_t kama_simd_mask;
#define KAMA_GROUP_WIDTH 32
#elif defined(KAMA_SIMD_SSE)
typedef __m128i kama_simd_int;
typedef uint32_t kama_simd_mask;
#define KAMA_GROUP_WIDTH 16
#elif defined(KAMA_SIMD_NEON)
typedef uint8x16_t kama_simd_int;
typedef uint32_t kama_simd_mask;
#define KAMA_GROUP_WIDTH 16
#else
typedef int8_t kama_simd_int;
typedef uint32_t kama_simd_mask;
#define KAMA_GROUP_WIDTH 8
#endif

static inline kama_simd_int kama_cmpeq_vec(kama_simd_int a, kama_simd_int b) {
#ifdef KAMA_SIMD_AVX512
	return _mm512_movm_epi8(_mm512_cmpeq_epi8_mask(a, b));
#elif defined(KAMA_SIMD_AVX)
	return _mm256_cmpeq_epi8(a, b);
#elif defined(KAMA_SIMD_SSE)
	return _mm_cmpeq_epi8(a, b);
#elif defined(KAMA_SIMD_NEON)
	return vceqq_u8(a, b);
#else
	return (a == b) ? 0xFF : 0x00;
#endif
}

static inline kama_simd_mask kama_movemask(kama_simd_int vec) {
#ifdef KAMA_SIMD_AVX512
	return _mm512_movepi8_mask(vec);
#elif defined(KAMA_SIMD_AVX)
	return (kama_simd_mask)_mm256_movemask_epi8(vec);
#elif defined(KAMA_SIMD_SSE)
	return (kama_simd_mask)_mm_movemask_epi8(vec);
#elif defined(KAMA_SIMD_NEON)
	// Isolate the sign bit (MSB)
	uint16x8_t b0 = vreinterpretq_u16_u8(vshrq_n_u8(vec, 7));

	// Compress 16 bytes into 8 bytes
	uint8x8_t p0 = vmovn_u16(vsraq_n_u16(b0, b0, 7));

	// Compress 8 bytes into 4 bytes
	uint16x8_t p1 = vreinterpretq_u16_u8(vcombine_u8(p0, p0));
	uint8x8_t p2 = vmovn_u16(vsraq_n_u16(p1, p1, 6));

	// Compress 4 bytes into 2 bytes
	uint16x8_t p3 = vreinterpretq_u16_u8(vcombine_u8(p2, p2));
	uint8x8_t p4 = vmovn_u16(vsraq_n_u16(p3, p3, 4));

	// The first 16 bits of our final vector contain the mask.
	return (kama_simd_mask)vget_lane_u16(vreinterpret_u16_u8(p4), 0);
#else
	return (vec) ? 1 : 0;
#endif
}

static inline kama_simd_mask kama_int_cmpeq_mask(kama_simd_int a,
												 kama_simd_int b) {
#ifdef KAMA_SIMD_AVX512
	return _mm512_cmpeq_epi8_mask(a, b);
#elif defined(KAMA_SIMD_AVX)
	return (kama_simd_mask)_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b));
#elif defined(KAMA_SIMD_SSE)
	return (kama_simd_mask)_mm_movemask_epi8(_mm_cmpeq_epi8(a, b));
#elif defined(KAMA_SIMD_NEON)
	uint8x16_t cmp = vceqq_u8(a, b);

	static const uint8_t __attribute__((aligned(16))) mask_lut[16] = {
		1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};

	// Mismatches stay 0x00. Matches become their respective power of 2.
	uint8x16_t masked = vandq_u8(cmp, vld1q_u8(mask_lut));

#if defined(__aarch64__)
	// AArch64: Vector Add Across Vector
	// The max possible sum of a half-vector is 255, so an 8-bit add is
	// perfectly safe.
	uint32_t low = vaddv_u8(vget_low_u8(masked));
	uint32_t high = vaddv_u8(vget_high_u8(masked));

	return low | (high << 8);
#else
	// Pairwise addition
	uint8x8_t sum = vpadd_u8(vget_low_u8(masked), vget_high_u8(masked));
	sum = vpadd_u8(sum, sum);
	sum = vpadd_u8(sum, sum);

	return vget_lane_u16(vreinterpret_u16_u8(sum), 0);
#endif

#else
	return (a == b) ? 1 : 0;
#endif
}

static inline kama_simd_int kama_int_set1(int8_t val) {
#ifdef KAMA_SIMD_AVX512
	return _mm512_set1_epi8(val);
#elif defined(KAMA_SIMD_AVX)
	return _mm256_set1_epi8(val);
#elif defined(KAMA_SIMD_SSE)
	return _mm_set1_epi8(val);
#elif defined(KAMA_SIMD_NEON)
	return vdupq_n_u8((uint8_t)val);
#else
	return val;
#endif
}

static inline kama_simd_int kama_int_load(const void *KAMA_RESTRICT ptr) {
#ifdef KAMA_SIMD_AVX512
	return _mm512_loadu_si512(ptr);
#elif defined(KAMA_SIMD_AVX)
	return _mm256_loadu_si256((const __m256i *)ptr);
#elif defined(KAMA_SIMD_SSE)
	return _mm_loadu_si128((const __m128i *)ptr);
#elif defined(KAMA_SIMD_NEON)
	return vld1q_u8((const uint8_t *)ptr);
#else
	return *(const int8_t *)ptr;
#endif
}

static inline kama_simd_int kama_int_or(kama_simd_int a, kama_simd_int b) {
#ifdef KAMA_SIMD_AVX512
	return _mm512_or_si512(a, b);
#elif defined(KAMA_SIMD_AVX)
	return _mm256_or_si256(a, b);
#elif defined(KAMA_SIMD_SSE)
	return _mm_or_si128(a, b);
#elif defined(KAMA_SIMD_NEON)
	return vorrq_u8(a, b);
#else
	return a | b;
#endif
}

#if defined(_WIN32) || defined(_WIN64)
#define KAMA_ALIGNED_MALLOC(bytes, align) _aligned_malloc((bytes), (align))
#define KAMA_ALIGNED_ALLOC(ptr, size, align)                                   \
	*(ptr) = _aligned_malloc((size), (align))
#define KAMA_ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
static inline void *kama_aligned_malloc_posix(size_t bytes, size_t align) {
	void *p = NULL;
	if (posix_memalign(&p, align, bytes) != 0)
		return NULL;
	return p;
}
#define KAMA_ALIGNED_MALLOC(bytes, align)                                      \
	kama_aligned_malloc_posix((bytes), (align))
#define KAMA_ALIGNED_FREE(ptr) free(ptr)
static inline void kama_aligned_alloc_wrap(void **ptr, size_t size,
										   size_t align) {
	if (posix_memalign(ptr, align, size) != 0)
		*ptr = NULL;
}
#define KAMA_ALIGNED_ALLOC(ptr, size, align)                                   \
	kama_aligned_alloc_wrap((void **)(ptr), (size), (align))
#endif

#if defined(_MSC_VER)

static inline int kama_ctz(kama_simd_mask mask) {
	unsigned long idx;

#ifdef _WIN64
	if (sizeof(mask) == 8) {
		_BitScanForward64(&idx, mask);
		return idx;
	}
#endif
	_BitScanForward(&idx, (unsigned long)mask);
	return idx;
}

#else

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L

#define kama_ctz(mask)                                                         \
	_Generic((mask),                                                           \
		unsigned long long: __builtin_ctzll(mask),                             \
		unsigned long: __builtin_ctzl(mask),                                   \
		default: __builtin_ctz(mask))

#else

#ifdef KAMA_SIMD_AVX512
// AVX512 uses 64-bit __mmask64
#define kama_ctz(mask) __builtin_ctzll(mask)
#else
// AVX, SSE, NEON, and Generic use 32-bit uint32_t
#define kama_ctz(mask) __builtin_ctz(mask)
#endif

#endif

#endif

/* RAPIDHASH */

static const uint64_t RAPID_SECRET[3] = {
	0x2d358dccaa6c78a5ULL, 0x8bb84b93962eacc9ULL, 0x4b33a62ed433d4a3ULL};

KAMA_INLINE uint64_t kama_rapid_read64(const uint8_t *p) {
	uint64_t v;
	memcpy(&v, p, 8);
	return v;
}
KAMA_INLINE uint64_t kama_rapid_read32(const uint8_t *p) {
	uint32_t v;
	memcpy(&v, p, 4);
	return v;
}
KAMA_INLINE uint64_t kama_rapid_mix(uint64_t A, uint64_t B) {
#if defined(__SIZEOF_INT128__)
	unsigned __int128 r = (unsigned __int128)A * B;
	return (uint64_t)(r) ^ (uint64_t)(r >> 64);
#else
	uint64_t ha = A >> 32, hb = B >> 32, la = (uint32_t)A, lb = (uint32_t)B;
	uint64_t rh = ha * hb, rm0 = ha * lb, rm1 = hb * la, rl = la * lb,
			 t = rl + (rm0 << 32), c = t < rl;
	uint64_t lo = t + (rm1 << 32);
	c += lo < t;
	uint64_t hi = rh + (rm0 >> 32) + (rm1 >> 32) + c;
	return hi ^ lo;
#endif
}
static KAMA_INLINE uint64_t kama_rapidhash(const char *key, size_t len,
										   uint64_t seed) {
	const uint8_t *p = (const uint8_t *)key;
	uint64_t a = RAPID_SECRET[0], b = RAPID_SECRET[1], c = RAPID_SECRET[2];
	seed ^= a ^ b ^ c ^ len;
	if (len <= 16) {
		if (len >= 4) {
			const uint8_t *end = p + len - 4;
			a ^= kama_rapid_read32(p) + (kama_rapid_read32(end) << 32);
			return kama_rapid_mix(seed ^ kama_rapid_read64(p + len / 2 - 4), a);
		} else if (len > 0) {
			uint64_t v =
				(uint64_t)((uint8_t)p[0] + ((uint8_t)p[len >> 1] << 8) +
						   ((uint8_t)p[len - 1] << 16));
			return kama_rapid_mix(seed ^ v, a);
		}
		return seed;
	}
	uint64_t see1 = seed, see2 = seed;
	while (len >= 48) {
		seed = kama_rapid_mix(kama_rapid_read64(p) ^ RAPID_SECRET[0],
							  kama_rapid_read64(p + 8) ^ seed);
		see1 = kama_rapid_mix(kama_rapid_read64(p + 16) ^ RAPID_SECRET[1],
							  kama_rapid_read64(p + 24) ^ see1);
		see2 = kama_rapid_mix(kama_rapid_read64(p + 32) ^ RAPID_SECRET[2],
							  kama_rapid_read64(p + 40) ^ see2);
		p += 48;
		len -= 48;
	}
	seed ^= see1 ^ see2;
	if (len > 16) {
		seed =
			kama_rapid_mix(kama_rapid_read64(p) ^ RAPID_SECRET[2],
						   kama_rapid_read64(p + 8) ^ seed ^ RAPID_SECRET[1]);
		if (len > 32)
			seed = kama_rapid_mix(kama_rapid_read64(p + 16) ^ RAPID_SECRET[2],
								  kama_rapid_read64(p + 24) ^ seed);
	}
	return kama_rapid_mix(seed ^ kama_rapid_read64(p + len - 16),
						  RAPID_SECRET[0] ^ kama_rapid_read64(p + len - 8));
}

#define KAMA_LOAD_FACTOR 85 / (100)
#define KAMA_CTRL_EMPTY 0xFF
#define KAMA_CTRL_DELETED 0xFE
#define KAMA_H2(hash) ((hash) >> 57)

// Common Metadata Struct
typedef struct {
	uint32_t hash;
	uint32_t len;
} kama_meta_t;

// Common Comparison
static KAMA_INLINE int kama_key_eq(const char *a, const char *b, size_t len) {
	if (len == 16)
		return (kama_rapid_read64((const uint8_t *)a) ==
				kama_rapid_read64((const uint8_t *)b)) &&
			   (kama_rapid_read64((const uint8_t *)a + 8) ==
				kama_rapid_read64((const uint8_t *)b + 8));
	return memcmp(a, b, len) == 0;
}

// Common Empty/Del Matcher
#if defined(KAMA_ARCH_GENERIC)
static KAMA_INLINE kama_simd_mask kama_match_empty_or_del(const int8_t *ctrl,
														  size_t idx) {
	uint32_t mask = 0;
	const int8_t *p = ctrl + idx;
	for (int i = 0; i < 8; i++)
		if (p[i] == (int8_t)KAMA_CTRL_EMPTY ||
			p[i] == (int8_t)KAMA_CTRL_DELETED)
			mask |= (1U << i);
	return mask;
}
#else
static KAMA_INLINE
	kama_simd_mask kama_match_empty_or_del(kama_simd_int ctrl_vec) {
	kama_simd_int empty =
		kama_cmpeq_vec(ctrl_vec, kama_int_set1((int8_t)KAMA_CTRL_EMPTY));
	kama_simd_int del =
		kama_cmpeq_vec(ctrl_vec, kama_int_set1((int8_t)KAMA_CTRL_DELETED));
	return kama_movemask(kama_int_or(empty, del));
}
#endif

// Helper Macros for Code Generation
#if defined(KAMA_ARCH_GENERIC)
#define KAMA_SIMD_SETUP(h2)
#define KAMA_SIMD_LOOP(ctrl, idx, target)                                      \
	uint32_t match = 0;                                                        \
	for (int i = 0; i < 8; i++)                                                \
		if (ctrl[idx + i] == h2)                                               \
			match |= (1U << i);
#define KAMA_CHECK_EMPTY(ctrl, idx)                                            \
	for (int i = 0; i < 8; i++)                                                \
		if (ctrl[idx + i] == (int8_t)KAMA_CTRL_EMPTY)                          \
			return map->capacity;
#define KAMA_GET_INSERT_MASK(res, map, idx, vec)                               \
	res = kama_match_empty_or_del(map->ctrl, idx)
#else
#define KAMA_SIMD_SETUP(h2)                                                    \
	kama_simd_int target = kama_int_set1(h2);                                  \
	kama_simd_int empty_v = kama_int_set1((int8_t)KAMA_CTRL_EMPTY);
#define KAMA_SIMD_LOOP(ctrl, idx, target)                                      \
	kama_simd_int vec = kama_int_load(ctrl + idx);                             \
	kama_simd_mask match = KAMA_MATCH(vec, target);
#define KAMA_CHECK_EMPTY(ctrl, idx)                                            \
	if (KAMA_MATCH(vec, empty_v))                                              \
		return map->capacity;
#define KAMA_GET_INSERT_MASK(res, map, idx, vec)                               \
	res = kama_match_empty_or_del(vec)
#endif

/* Kama Generator
 * VAL_TYPE stands for type of values
 */
#define kama(NAME, VAL_TYPE)                                                   \
                                                                               \
	typedef struct {                                                           \
		union {                                                                \
			const char *ptr;                                                   \
			uint64_t inline_key;                                               \
		} key;                                                                 \
		VAL_TYPE val;                                                          \
	} NAME##_pair_t;                                                           \
                                                                               \
	typedef struct {                                                           \
		int8_t *ctrl;                                                          \
		kama_meta_t *meta;                                                     \
		NAME##_pair_t *slots;                                                  \
		void *heap;                                                            \
		size_t size;                                                           \
		size_t capacity;                                                       \
		size_t resize_threshold;                                               \
		size_t tombstones;                                                     \
	} NAME##_t;                                                                \
                                                                               \
	static void NAME##_init(NAME##_t *map, size_t cap) {                       \
		if (cap < 16)                                                          \
			cap = 16;                                                          \
		size_t n = 1;                                                          \
		while (n < cap)                                                        \
			n <<= 1;                                                           \
		cap = n;                                                               \
		map->capacity = cap;                                                   \
		map->size = 0;                                                         \
		map->tombstones = 0;                                                   \
		map->resize_threshold = (size_t)(cap * KAMA_LOAD_FACTOR);              \
		size_t sz_m = cap * sizeof(kama_meta_t);                               \
		size_t sz_d = cap * sizeof(NAME##_pair_t);                             \
		size_t sz_c = cap + KAMA_GROUP_WIDTH;                                  \
		KAMA_ALIGNED_ALLOC(&map->heap, sz_m + sz_d + sz_c, 32);                \
		uint8_t *b = (uint8_t *)map->heap;                                     \
		map->meta = (kama_meta_t *)b;                                          \
		map->slots = (NAME##_pair_t *)(b + sz_m);                              \
		map->ctrl = (int8_t *)(b + sz_m + sz_d);                               \
		memset(map->ctrl, KAMA_CTRL_EMPTY, sz_c);                              \
	}                                                                          \
                                                                               \
	static void NAME##_free(NAME##_t *map) {                                   \
		if (map->heap)                                                         \
			KAMA_ALIGNED_FREE(map->heap);                                      \
		map->ctrl = NULL;                                                      \
		map->heap = NULL;                                                      \
	}                                                                          \
                                                                               \
	static KAMA_INLINE size_t NAME##_find_idx(NAME##_t *map, const char *key,  \
											  size_t len) {                    \
		uint64_t hash = kama_rapidhash(key, len, 0);                           \
		size_t mask = map->capacity - 1;                                       \
		size_t idx = hash & mask;                                              \
		KAMA_PREFETCH((const void *)(map->ctrl + idx));                        \
		KAMA_PREFETCH((const void *)(map->meta + idx));                        \
		uint64_t query_inline_key = 0;                                         \
		if (len <= 8)                                                          \
			memcpy(&query_inline_key, key, len);                               \
		int8_t h2 = (int8_t)(KAMA_H2(hash) & 0x7F);                            \
		uint32_t hash32 = (uint32_t)hash;                                      \
		KAMA_SIMD_SETUP(h2)                                                    \
		while (1) {                                                            \
			KAMA_SIMD_LOOP(map->ctrl, idx, target)                             \
			while (match) {                                                    \
				int bit = kama_ctz(match);                                     \
				size_t probe = (idx + (size_t)bit) & mask;                     \
				if (map->meta[probe].hash == hash32 &&                         \
					map->meta[probe].len == (uint32_t)len) {                   \
					if (len <= 8) {                                            \
						if (map->slots[probe].key.inline_key ==                \
							query_inline_key)                                  \
							return probe;                                      \
					} else if (kama_key_eq(map->slots[probe].key.ptr, key,     \
										   len)) {                             \
						return probe;                                          \
					}                                                          \
				}                                                              \
				match &= ~(((kama_simd_mask)1) << bit);                        \
			}                                                                  \
			KAMA_CHECK_EMPTY(map->ctrl, idx)                                   \
			idx = (idx + KAMA_GROUP_WIDTH) & mask;                             \
		}                                                                      \
	}                                                                          \
                                                                               \
	static int NAME##_get(NAME##_t *map, const char *key, size_t len,          \
						  VAL_TYPE *out) {                                     \
		size_t idx = NAME##_find_idx(map, key, len);                           \
		if (idx != map->capacity) {                                            \
			*out = map->slots[idx].val;                                        \
			return 1;                                                          \
		}                                                                      \
		return 0;                                                              \
	}                                                                          \
                                                                               \
	static int NAME##_delete(NAME##_t *map, const char *key, size_t len) {     \
		size_t idx = NAME##_find_idx(map, key, len);                           \
		if (idx == map->capacity)                                              \
			return 0;                                                          \
		map->ctrl[idx] = (int8_t)KAMA_CTRL_DELETED;                            \
		if (KAMA_UNLIKELY(idx < KAMA_GROUP_WIDTH))                             \
			map->ctrl[idx + map->capacity] = (int8_t)KAMA_CTRL_DELETED;        \
		map->size--;                                                           \
		map->tombstones++;                                                     \
		return 1;                                                              \
	}                                                                          \
                                                                               \
	static KAMA_INLINE void NAME##_put_hashed(NAME##_t *map, const char *key,  \
											  size_t len, VAL_TYPE val,        \
											  uint32_t hash32, int8_t h2) {    \
		size_t mask = map->capacity - 1;                                       \
		size_t idx = hash32 & mask;                                            \
		size_t first_del = map->capacity;                                      \
		KAMA_SIMD_SETUP(h2)                                                    \
		uint64_t inline_k = 0;                                                 \
		if (len <= 8)                                                          \
			memcpy(&inline_k, key, len);                                       \
		while (1) {                                                            \
			KAMA_SIMD_LOOP(map->ctrl, idx, target)                             \
			while (match) {                                                    \
				int bit = kama_ctz(match);                                     \
				size_t probe = (idx + (size_t)bit) & mask;                     \
				if (map->meta[probe].hash == hash32 &&                         \
					map->meta[probe].len == (uint32_t)len) {                   \
					if (len <= 8) {                                            \
						if (map->slots[probe].key.inline_key == inline_k) {    \
							map->slots[probe].val = val;                       \
							return;                                            \
						}                                                      \
					} else if (kama_key_eq(map->slots[probe].key.ptr, key,     \
										   len)) {                             \
						map->slots[probe].val = val;                           \
						return;                                                \
					}                                                          \
					return;                                                    \
				}                                                              \
				match &= ~(((kama_simd_mask)1) << bit);                        \
			}                                                                  \
                                                                               \
			uint32_t empties_dels;                                             \
			KAMA_GET_INSERT_MASK(empties_dels, map, idx, vec);                 \
			if (empties_dels) {                                                \
				while (empties_dels) {                                         \
					int bit = kama_ctz(empties_dels);                          \
					size_t probe = (idx + (size_t)bit) & mask;                 \
					if (map->ctrl[probe] == (int8_t)KAMA_CTRL_EMPTY) {         \
						size_t ins =                                           \
							(first_del != map->capacity) ? first_del : probe;  \
						if (first_del != map->capacity)                        \
							map->tombstones--;                                 \
                                                                               \
						map->ctrl[ins] = h2;                                   \
						map->ctrl[((ins - KAMA_GROUP_WIDTH) & map->capacity) + \
								  ins] = h2;                                   \
                                                                               \
						map->meta[ins].hash = hash32;                          \
						map->meta[ins].len = (uint32_t)len;                    \
						if (len <= 8) {                                        \
							map->slots[ins].key.inline_key = inline_k;         \
						} else {                                               \
							map->slots[ins].key.ptr = key;                     \
						}                                                      \
						map->slots[ins].val = val;                             \
						map->size++;                                           \
						return;                                                \
					}                                                          \
					if (first_del == map->capacity)                            \
						first_del = probe;                                     \
					empties_dels &= ~(((kama_simd_mask)1) << bit);             \
				}                                                              \
			}                                                                  \
			idx = (idx + KAMA_GROUP_WIDTH) & mask;                             \
		}                                                                      \
	}                                                                          \
                                                                               \
	static void NAME##_resize(NAME##_t *map, size_t new_cap) {                 \
		NAME##_t new_map;                                                      \
		NAME##_init(&new_map, new_cap);                                        \
		for (size_t i = 0; i < map->capacity; ++i) {                           \
			if ((map->ctrl[i] & 0x80) == 0) {                                  \
				const char *actual_key =                                       \
					(map->meta[i].len <= 8)                                    \
						? (const char *)&map->slots[i].key.inline_key          \
						: map->slots[i].key.ptr;                               \
				NAME##_put_hashed(&new_map, actual_key, map->meta[i].len,      \
								  map->slots[i].val, map->meta[i].hash,        \
								  map->ctrl[i]);                               \
			}                                                                  \
		}                                                                      \
		NAME##_free(map);                                                      \
		*map = new_map;                                                        \
	}                                                                          \
                                                                               \
	static void NAME##_put(NAME##_t *map, const char *key, size_t len,         \
						   VAL_TYPE val) {                                     \
		if (KAMA_UNLIKELY((map->size + map->tombstones) >=                     \
						  map->resize_threshold)) {                            \
			if (map->size < map->resize_threshold / 2)                         \
				NAME##_resize(map, map->capacity);                             \
			else                                                               \
				NAME##_resize(map, map->capacity * 2);                         \
		}                                                                      \
		uint64_t hash = kama_rapidhash(key, len, 0);                           \
		NAME##_put_hashed(map, key, len, val, (uint32_t)hash,                  \
						  (int8_t)(KAMA_H2(hash) & 0x7F));                     \
	}

#endif
