# SPECTRA-ICA: Spectral Profile-Estimated Continuous Temporal Removal of Artifacts via ICA

**NeuroHack 2026 — Team Surge**

---

## Overview

SPECTRA-ICA is a novel EEG artifact removal algorithm that combines **per-IC temporal event detection** with **data-driven spectral fingerprint estimation**. Unlike standard ICA cleaning (which removes entire components) or wavelet-ICA (which applies broadband thresholds), SPECTRA-ICA removes only the artifact-characteristic frequencies during detected artifact windows, preserving neural signal that shares a component with artifact sources.

**Key innovations:**
- **Temporal gating** — artifacts are only removed during detected blink, muscle, or cardiac events
- **Spectral fingerprinting** — per-IC spectral profiles are estimated from STFT power differences between artifact and clean frames; removal is restricted to artifact-characteristic frequencies
- **Probability-weighted removal** — ICLabel posterior probabilities modulate removal magnitude (graded, not binary)
- **Fully data-driven** — no hardcoded frequency bands; all parameters derive from sampling rate and the IC's own data

## Results

SPECTRA-ICA outperforms standard ICA on **all 7 evaluation metrics** in a P300 oddball BCI classification benchmark (Leave-One-Person-Out Cross-Validation, 10 subjects):

| Metric | ICA | SPECTRA-ICA | Improvement |
|--------|-----|-------------|-------------|
| Balanced Accuracy | 0.6270 | 0.6505 | +2.35 pp |
| AUC | 0.6843 | 0.7032 | +1.89 pp |
| F1 (target) | 0.3974 | 0.4230 | +2.56 pp |
| Precision (target) | 0.2593 | 0.2685 | +0.92 pp |
| Recall (target) | 0.8540 | 0.8820 | +2.80 pp |
| Cohen's Kappa | 0.1972 | 0.2201 | +2.29 pp |
| MCC | 0.2256 | 0.2515 | +2.59 pp |

- **+28 more P300 targets detected, −11 fewer false alarms**
- **Cohen's d = 0.703** (medium effect size)
- **6/10 subjects improved**

## Dataset

NeuroHack 2026 EEG dataset: **10 subjects × 3 paradigms = 30 sessions**, recorded at 1000 Hz with 30 EEG channels.

| Paradigm | Description |
|----------|-------------|
| Flicker | SSVEP at six frequencies (10.00–12.63 Hz) |
| Oddball | Three-stimulus auditory oddball (P300) |
| FlickerOddball | Concurrent SSVEP + auditory oddball |

## Quick Start

```bash
# Install dependencies
pip install mne mne-icalabel scikit-learn scipy pyriemann numpy

# Run the P300 benchmark (ICA vs SPECTRA-ICA)
python spectra_ica_proof.py

# Use SPECTRA-ICA in your own pipeline
python -c "
import mne
from icas import spectra_clean

raw = mne.io.read_raw_fif('eeg_data/Oddball/sub-010_Oddball_eeg.fif', preload=True)
raw.filter(1, 40)
ica = mne.preprocessing.read_ica('results/ica/sub-010_Oddball-ica.fif')
cleaned = spectra_clean(raw, ica)
"
```

## Methods

1. **ICA decomposition** — 20-component Infomax ICA (pre-fitted)
2. **ICLabel classification** — posterior probabilities for Brain, Eye, Muscle, Heart, Line Noise, Channel Noise, Other
3. **Temporal mask detection** — type-specific detectors for blink (delta envelope), muscle (HF RMS), cardiac (QRS template correlation), channel noise (broadband RMS)
4. **Spectral profile estimation** — per-IC STFT comparison of artifact vs. clean frames yields a continuous frequency weight F(k)
5. **Frequency-selective removal** — artifact IC contribution is attenuated only at frequencies where F(k) > 0, only during detected events, weighted by ICLabel probability

## References

- Pion-Tonachini, L. et al. (2019). ICLabel: An automated electroencephalographic independent component classifier. *NeuroImage*, 198, 181–197.
- Bailey, N.W. et al. (2023). Introducing RELAX. *Psychophysiology*, 60(7), e14280.
- Bailey, N.W. (2025). An updated pipeline. *Clinical Neurophysiology*, 169, 131–147.
- Castellanos, N.P. & Makarov, V.A. (2006). Recovering EEG brain signals: Artifact suppression with wavelet enhanced ICA. *Biological Cybernetics*, 95, 569.

## License

This project was developed for NeuroHack 2026.
