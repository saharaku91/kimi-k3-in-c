# Kimi K3 in C — Laporan Analisa & Implementasi

> Analisa mendalam oleh Hermes (Saharakubot AI) — 29 Agu 2026

## Ringkasan

**Kimi K3 (Moonshot AI) — model 2.78T parameter — dijalankan di CPU murni (C99), tanpa GPU/BLAS/framework, dalam 8.24GB RAM.** Engine streaming dari disk. Repo asli: `FareedKhan-dev/kimi-k3-in-c` (Apache 2.0, 64 commits). Fork user: `github.com/saharaku91/kimi-k3-in-c`.

## Teknik Reduksi (kenapa muat di CPU kecil)

| Teknik | Efek |
|---|---|
| MXFP4 | Expert weight 4-bit (0.5 byte/w) — 896 expert jalan hemat |
| KDA recurrence | 69/93 layer: recurrent state tidak membesar dengan konteks |
| MLA latent | 1 latent (53x lebih kecil) ganti 96 expanded heads |
| Trunk streaming | Layer trunk di-stream tiap token (O_DIRECT + ring buffer) |
| Expert LRU | 16/896 expert per layer — budget memori "diputar" |

**Fitur kunci**: output **byte-identical** di seluruh memory budget 8GB→224GB. Memory beli kecepatan, bukan akurasi (28x memori = 1.70x speed, terukur).

## Verifikasi Live (29 Agu 2026, server Olif)

- `git clone` [OK] 75MB, 302 files non-.git
- `make -j` [OK] build 0 error, bin/k3 242K ELF x86-64
- `make test` [OK] **ALL PASS**: "ENGINE MATCHES THE REFERENCE EXACTLY", tokenizer parity, oracle 20/20, greedy = tf. Fixture 24MB 13-layer.
- `k3-doctor.sh` [OK] "this machine can run Kimi K3"

## Spesifikasi Mesin Olif vs Kebutuhan

| Metric | Mesin | Butuh | Status |
|---|---|---|---|
| RAM | 54GB | ≥8GB | [OK] |
| Core | 4 | 4+ | [OK] |
| AVX2/AVX-512 | Ada | AVX2 | [OK] |
| **Disk free** | **6.3GB** | **1.56TB + 109GB trunk** | **[FAIL] blocker utama** |
| Storage read | 6.9GB/s | NVMe | [OK] |

## Preset & Kecepatan (referensi EPYC 7763, 124 core, NVMe 3.2GB/s)

| Preset | RAM | s/token | Cocok server Olif? |
|---|---|---|---|
| ultra | ~3GB | proof-of-life | [OK] |
| laptop | ~10GB | ~32s | tipis |
| desktop | ~32GB | ~31s | [OK] |
| workstation | ~96GB | ~24s | RAM kurang |
| server | ~128GB | ~17s | RAM kurang |
| max | ~224GB | ~19s | RAM kurang |

Mesin Olif 4 core ≈ 1-2x lebih lambat dari referensi → realistik 60-120s/token.

## Cara Implementasi

### Opsi A — Minimal (tanpa checkpoint besar)
Engine + test sudah terverifikasi di server. Tiny fixture proof-of-life sudah jalan.

### Opsi B — Model asli (butuh disk)
```bash
pipx install huggingface_hub        # hf CLI
export HF_TOKEN=***                  # token akses Kimi-K3 (gated model Moonshot)
./scripts/download-model.sh ~/k3model   # 1.56TB, 96 shards, ~15-25 min @6.9GB/s
./scripts/pack-trunk.sh ~/k3model ~/k3trunk  # 109GB
./bin/k3 ~/k3model --trunk ~/k3trunk --preset laptop \
    --tok ~/k3model --prompt "Hello" --gen 16 --incremental
```
Blocker: butuh disk NVMe baru ~2TB (Azure managed disk P30/P40 terpisah OS disk).

### Opsi C — Semi
Tidak ada jalan pintas; checkpoint memang 1.56TB.

## Analisa Kritis

**Positif:**
- Kualitas kode tinggi: 7,738 LOC C, kontrak float eksplisit (bit-identical), komentar jelas
- Test suite komprehensif (ops, cache, ST reader, config, tokenizer BPE 163,584 ranks, oracle end-to-end)
- "Memory is a dial" nyata, bukan hype — output identical di semua budget

**Waspada:**
1. **Bukan chat** — tidak ada chat template (base model continuation). ROADMAP #6.
2. ~30s/token di mesin referensi → ~60-120s di server Olif (4 core)
3. Butuh download 1.56TB + disk besar (biaya Azure)
4. Storage I/O bottleneck: 40-60% wall clock nunggu disk, ~135GB/token traffik
5. Tokenizer tiktoken.model download terpisah (kecil)

## Rekomendasi

- Sekarang: engine terverifikasi (build + test pass) — bisa demo tiny, gratis
- Model asli: butuh disk ~2TB baru + HF token akses Kimi-K3
- Fitur chat template (XTML), sampling, vision = ROADMAP priority — belum ada

## File Kunci Repo

- `Makefile` — build (make / make test / make portable)
- `src/core/k3_ops.c` — kernel (AVX2 matmul, MXFP4, KDA, MLA)
- `src/cli/k3_run.c` — CLI entrypoint (20+ flags)
- `scripts/k3-doctor.sh` — cek kesiapan mesin
- `scripts/download-model.sh` — download 1.56TB checkpoint
- `scripts/pack-trunk.sh` — pack trunk 109GB
- `docs/QUICKSTART.md`, `docs/TUNING.md`, `docs/PERFORMANCE.md`, `docs/ROADMAP.md`
- `tools/dqd_trunk.py` — hybrid speculative decode (draft + verify)
