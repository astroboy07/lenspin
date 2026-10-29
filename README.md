# lenspin

Can spin precession mimic gravitational-wave lensing, and how much does the detector network
break that degeneracy? Network fitting factors of precessing and aligned-spin templates against
lensed non-precessing signals, for H, HL, HLV, HLVI and ET.

Requires Python 3.11+; dependencies are listed in `pyproject.toml`. Install into an environment that
already has them, and run from the repository root (outputs go to `results/`).

```bash
pip install --no-deps -e .
python -m lenspin.fitting.checks
python -m lenspin.fitting.run_grid --preset pilot --workers 48 --out results/pilot.csv
python -m lenspin.fitting.analyze results/pilot.csv
```

Presets: `pilot`, `trimmed`, `full`.
