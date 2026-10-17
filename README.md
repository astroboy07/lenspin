# lenspin

Can spin precession mimic gravitational-wave lensing, and how much does the detector network
break that degeneracy? Network fitting factors of precessing and aligned-spin templates against
lensed non-precessing signals, for H, HL, HLV, HLVI and ET.

Requires lalsuite, numpy, scipy, pandas and matplotlib.

```bash
python checks.py
python run_grid.py --preset pilot --workers 48 --out results/pilot.csv
python analyze.py results/pilot.csv
```

Presets: `pilot`, `trimmed`, `full`.
