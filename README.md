# Lensing vs precession: network fitting factors (stage 1)

How well can an unlensed **precessing** binary imitate a **lensed non-precessing** one, as
seen by H, HL, HLV, HLVI and ET?

The idea being tested: lensing multiplies h+ and h× by the same F(f), so it never changes
the polarization state. Precession does change it. A single detector, or the nearly
co-aligned HL pair, sees only one polarization mix, which is where the degeneracy lives.
Detectors with other orientations should break it.

## What it computes

Each row covers one combination of source, time delay Δt, flux ratio I, cos ι, sky position,
network and sky treatment.

**Injection.** IMRPhenomXAS, 22 mode only, zero spins. It is lensed by a geometric-optics
two-image factor F(f) = 1 + i√I·e^{−2πifΔt}: the second image is type II, in LAL's sign
convention. F(f) is applied to h+ and h× before projecting onto the detectors.

**FF_aligned.** The best aligned-spin XAS template.
- Distance, inclination, polarization and phase are maximized exactly by the network
  F-statistic.
- Arrival time is maximized by FFT, then refined.
- Masses and spins (plus sky, if free) are found by differential evolution, then Nelder–Mead.

**FF_precessing.** The best IMRPhenomXP template.
- Parameters: chirp mass, q, primary spin magnitude and tilt, secondary aligned spin, θJN, φJL.
- ψ is maximized exactly (a 2×2 generalized eigenvalue problem).
- Phase and time are handled as above.
- The search starts from the aligned optimum (with a θJN scan), so FF_prec ≥ FF_aligned up to
  optimizer noise.

**rho_crit** = √(4.6 / (1 − FF_prec)). This is the network SNR above which lensing beats the
best precessing explanation by ln B ≈ 4.6. Occam factors are ignored.

**d_crit_mpc.** The distance at which the source reaches rho_crit (detector-frame masses, no
cosmology).

**n_fringes** = Δt × (f95 − f_min), where f95 encloses 95% of the unlensed SNR².

## Caveats

- **Noise curves:**
  - A+ design for H1, L1 and I1.
  - Virgo uses AdV design (P1200087), which is below the O5 target, so **Virgo's leverage is
    underestimated**.
  - ET uses P1600143 for E1–E3, from 10 Hz.
  - Others start at 20 Hz; f_max is 1024 Hz for all.
- **Phase freedom.** The common-phase maximization gives precessing templates a little more
  freedom than φ_ref really has. This errs toward a stronger degeneracy.
- **Sky treatment.** `fixed`: templates use the true sky, an upper bound on
  distinguishability. `free`: templates pick their own sky, which is what a real analysis
  allows. `free` is the headline.
- **Scope.** 22-mode only, zero-spin injections, one GPS time. Higher modes come next; they
  also change the polarization content.

## Files

| File | Contents |
|---|---|
| `signals.py` | Waveforms and the lens factor |
| `network.py` | Detectors, PSDs, projection, time-maximized overlaps, network F-statistic, ψ-maximized precessing overlap |
| `fit.py` | `Problem`, the optimizers, `run_problem` |
| `run_grid.py` | Grid runner: multiprocessing, appends rows to a CSV, resumes |
| `checks.py` | Sanity checks |

## Checks (about 2 min; fine on the Mac)

```bash
python checks.py
```

Expect `ALL PASS`. On 2026-10-17 the unlensed truth overlaps were 1.0000000 in every network
for both template families, and the HL seed spread was 6e-8.

## Run on walle

```bash
scp -r ~/Downloads/lensing_precession walle.weizmann.ac.il:/mnt/storage/saif/
```

Pilot (360 tasks: 100+80 M☉, Δt 5 and 20 ms, I 0.3 and 0.9, cos ι 0.9/0.5/0, 6 skies,
H/HL/HLV/HLVI/ET, sky free). Expect about 10 CPU-hours, roughly 5–10 min on 150 workers:

```bash
cd /mnt/storage/saif/lensing_precession && OMP_NUM_THREADS=1 nohup ~/PruningCW/.venv_new/bin/python run_grid.py --preset pilot --workers 150 --out results/pilot.csv > pilot.log 2>&1 &
```

Full grid (23,040 tasks). Expect about 600–700 CPU-hours, roughly 4–5 h on 150 workers:

```bash
cd /mnt/storage/saif/lensing_precession && OMP_NUM_THREADS=1 nohup ~/PruningCW/.venv_new/bin/python run_grid.py --preset full --workers 150 --out results/full.csv > full.log 2>&1 &
```

Notes:
- Rerunning the same command resumes; finished rows are skipped.
- `--limit N` runs only the first N pending tasks.
- Runtime estimates are extrapolated from Mac timings: about 4 ms per evaluation for HL and
  8 ms for ET, and roughly 15–20k evaluations per task at the default settings.

## Reading the result

For each geometry, compare rho_crit across networks. GO means HLV, HLVI or ET cut rho_crit
substantially (say 2× or more) where HL is degenerate, ideally to below realistic O5/ET SNRs.
The `ff_precessing − ff_aligned` gap shows how much of the lensing pattern precession absorbs;
it should shrink as the network adds polarization information.
