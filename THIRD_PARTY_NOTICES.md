RapidHash V3 is Copyright (C) 2025 Nicolas De Carli and is based on wyhash by Wang Yi. Its MIT notice is included in `kama.h`. The embedded copy prefixes identifiers and removes the standalone `#pragma once`; the algorithm is unchanged.

Upstream: https://github.com/Nicoshev/rapidhash

The adaptive string fingerprint is an altered C adaptation of FoldHash v0.2.0 by Orson Peters, Copyright (C) 2024, under the Zlib license. Kama extends the short-byte construction to sample long keys, checks complete keys, and falls back to full RapidHash after a sampled collision. It is not upstream FoldHash for long inputs. The complete notice is included in `kama.h`.

Upstream: https://github.com/orlp/foldhash/tree/v0.2.0

The optional SipHash-2-4 hook implements the algorithm by Jean-Philippe Aumasson and Daniel J. Bernstein. Its test vectors come from the reference project: https://github.com/veorq/SipHash/blob/master/vectors.h . Algorithm description: https://www.aumasson.jp/siphash/ .
