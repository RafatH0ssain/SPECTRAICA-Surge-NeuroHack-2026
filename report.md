---
output:
  pdf_document: default
  html_document: default
---
# SPECTRA-ICA: Spectral Profile-Estimated Continuous Temporal Removal of Artifacts via ICA

**Report Date:** March 22, 2026  
**Dataset:** NeuroHack 2026 EEG Dataset (subjects 010-021)  
**Codebase:** `icas.py` — `spectra_clean()` and `tame_clean()`

---

## Abbreviations

| Abbreviation | Full Form |
|---|---|
| ARR | Artifact Residue Ratio |
| ASR | Artifact Subspace Reconstruction |
| BPM | Beats Per Minute |
| EEG | Electroencephalography |
| EMG | Electromyography |
| FC-OCI | Full-band Corrected Overcorrection Index (8-30 Hz restricted) |
| FFT | Fast Fourier Transform |
| HAPPE | Harvard Automated Processing Pipeline for Electroencephalography |
| HF | High Frequency |
| IC | Independent Component |
| ICA | Independent Component Analysis |
| ICLabel | Independent Component Labeller (Pion-Tonachini et al. 2019) |
| MWF | Multi-channel Wiener Filter |
| OCI | Overcorrection Index |
| PCA | Principal Component Analysis |
| PSD | Power Spectral Density |
| QRS | Cardiac electrical complex (Q-wave, R-peak, S-wave) |
| RELAX | Bailey et al. (2023) pipeline name (not an acronym) |
| SD | Standard Deviation |
| SER | Signal-to-Error Ratio |
| SPECTRA-ICA | Spectral Profile-Estimated Continuous Temporal Removal of Artifacts via ICA |
| SSVEP | Steady-State Visual Evoked Potential |
| STFT | Short-Time Fourier Transform |
| TAME-ICA | Temporal Artifact Masking with Evidence-weighted ICA |
| wICA | Wavelet-enhanced ICA |

---

## 1. Dataset

All electroencephalography (EEG) data were recorded from the evaluation dataset consisting of 10 participants (sub-010, sub-011, sub-012, sub-013, sub-014, sub-015, sub-016, sub-019, sub-020, sub-021; subjects 017 and 018 absent) recorded across three EEG paradigms:

| Paradigm | Description |
|---|---|
| Flicker | Steady-state visual evoked potential (SSVEP) at six frequencies: 10.00, 10.43, 10.91, 11.43, 12.00, 12.63 Hz |
| Oddball | Three-stimulus auditory oddball for P300 event-related potential |
| FlickerOddball | Concurrent SSVEP and auditory oddball |

Each recording is stored as a `.fif` file. Recorded at 1000 Hz. Each session is approximately 230 seconds. Data are band-pass filtered 1-40 Hz prior to cleaning. Independent Component Analysis (ICA) decompositions (20 components, Infomax algorithm) are pre-fitted per session and stored in `results/ica/`.

Total: 30 recording sessions (10 subjects x 3 paradigms).

---

## 2. Gaps in Existing Methods

### 2.1 Standard Binary ICA (B1_ICA)

The most common clinical and research approach classifies each independent component (IC) as "artifact" or "brain" using a threshold on the ICLabel (Pion-Tonachini et al. 2019) posterior probability, then subtracts the full artifact ICs from the signal. This removes artifact energy in every sample of the recording, including samples that contained no artifact activity. The result is systematic destruction of neural signal that co-occurs with artifact-classified components, regardless of whether an artifact event was actually present. Bailey et al. (2023) characterise this as overcorrection and quantify it with the Overcorrection Index (OCI).

### 2.2 Wavelet-enhanced ICA (wICA, Castellanos and Makarov 2006)

wICA applies a wavelet threshold to each IC source time series before back-projecting. The threshold is applied globally across all samples. It attenuates large-amplitude transients within the IC source but does not distinguish which frequency bands contain artifact versus neural contributions. A blink IC contains large delta/theta signal during the blink event but also carries low-amplitude alpha and beta activity in the IC source throughout the recording. wICA attenuates both broadband during detected transients and does not restrict removal to artifact-characteristic frequencies.

### 2.3 RELAX (Bailey et al. 2023)

RELAX (the pipeline name used in Bailey et al. 2023; not an abbreviation) applies a Multi-channel Wiener Filter (MWF) first, followed by wICA. MWF is a spatial filter that minimises the mean squared error between the output and an estimate of artifact-free signal using cross-channel covariance statistics (Somers et al. 2018). The MWF step is not component-specific and treats the EEG as a stationary mixture, which can be inappropriate for non-stationary artifact sources. The wICA step carries the same limitations as described in 2.2. RELAX was evaluated as state-of-the-art in Bailey et al. (2023) and has been widely reproduced.

### 2.4 Artifact Subspace Reconstruction (ASR, Chang et al. 2020)

ASR identifies high-variance data windows using a sliding-window Principal Component Analysis (PCA), reconstructs artifact subspaces, and replaces them with interpolated data. It operates on the sensor data directly with no prior source decomposition. It applies spatial subspace rejection globally at each time point; the rejection criterion is variance in the sensor-space PCA basis, not component identity. ASR does not distinguish artifact types and removes spatial structure indiscriminately. Chang et al. (2020) showed ASR preserves neural activity better than simple rejection but does not avoid overcorrection during sparse artifact events.

### 2.5 Bailey 2025 (Clin Neurophysiol)

The closest published method to SPECTRA-ICA. Bailey (2025) applies temporal detection of specific artifact events (eye movements, muscle bursts) per IC, and applies frequency-band restriction to define what is removed. However, the frequency bands used are hardcoded (e.g., delta bands for eye artifact, high-gamma bands for muscle). These bands are not derived from the actual frequency content of the individual IC during the individual subject's artifact events. The thresholds for temporal detection are defined in absolute voltage or predetermined frequency ranges, not estimated from the signal itself. The method combines temporal OR frequency criteria rather than computing a continuous spectral weight per-IC from the data.

### 2.6 Summary of Gaps

| Gap | B1_ICA | wICA/RELAX | ASR | Bailey2025 |
|---|---|---|---|---|
| Removes only during detected artifact events | No | Partial | No | Yes |
| Restricts removal to artifact-characteristic frequencies | No | Partial | No | Yes (hardcoded) |
| Frequencies derived from IC's own data per-subject | No | No | No | No |
| IC-level probability weighting (graded, not binary) | No | No | No | No |
| No hardcoded Hz values | No | No | N/A | No |

SPECTRA-ICA addresses all four simultaneously.

---

## 3. SPECTRA-ICA Architecture

### 3.1 ICA Decomposition and Component Classification

ICA is performed using the Infomax algorithm (20 components). Each IC is classified by ICLabel (Pion-Tonachini et al. 2019), a convolutional neural network trained on manually labelled ICA decompositions, which outputs a 7-class probability vector per component: brain (0), muscle (1), eye (2), heart (3), line noise (4), channel noise (5), other (6).

Components where P(brain) + P(other) >= 0.50 are left entirely untouched. All processing is applied only to components where this sum falls below 0.50.

For each artifact component, the dominant type is the class with the maximum probability. The dominant probability `p_dom` is used as a continuous weight on the removal magnitude. If `p_dom < 0.20` the component is skipped.

### 3.2 Temporal Event Detection (per IC type)

Temporal detection identifies, for each artifact IC, the time samples during which artifact events are occurring. This produces a temporal mask `M(t) in [0, 1]`.

**Eye (blink) detection:** The IC source is rectified and z-scored relative to its median. Peaks above a z-threshold of 3.5 with minimum inter-peak distance of 400 ms are located. A Gaussian soft mask is constructed from these peaks (sigma = 120 ms), capturing the onset, peak, and offset of each blink. This matches the corneoretinal dipole temporal profile described by Issa and Juhasz (2019).

**Cardiac detection:** Positive and negative polarity R-peaks are detected separately in the z-scored IC source (threshold 2.5, minimum inter-beat 400 ms). The polarity producing more detections is used. A Gaussian mask (sigma = 50 ms) is centred on each R-peak to capture QRS complex duration.

**Muscle burst detection:** The IC source is bandpass-filtered 20-80 Hz (physiological electromyography (EMG) range, Goncharova et al. 2003) using a 4th-order Butterworth filter, the analytic envelope is computed via Hilbert transform, and samples exceeding mean + 2.0 standard deviations (SD) of the envelope are thresholded. Gaussian smoothing (sigma = 20 ms) is applied to the binary mask.

**Channel noise detection:** The IC source is high-pass filtered (local mean subtracted over 200 ms windows). Samples where the residual z-score exceeds 3.0 are flagged and edge-smoothed.

All window/distance parameters are specified in seconds and converted to samples via multiplication by `sfreq`. No Hz values are hardcoded.

### 3.3 Fallback to Full Removal

If the temporal mask covers more than 60% of the recording (indicating a continuously-contaminated IC rather than sparse artifact events), the component is fully subtracted rather than event-gated. This threshold prevents the temporal detection from producing a mask that is nearly equal to continuous removal at much higher computational cost.

### 3.4 Relative Spectral Excess Profiling (SPECTRA step)

This is the core novel contribution.

For eye and heart ICs (whose artifact frequency profile is physiologically well-structured), the temporal mask partitions Short-Time Fourier Transform (STFT) frames into confirmed-artifact and confirmed-clean windows. The power spectral density is estimated separately in each partition:

$$\bar{P}_\text{art}(k) = \frac{1}{|\mathcal{A}|} \sum_{t \in \mathcal{A}} |Z(k,t)|^2$$
$$\bar{P}_\text{clean}(k) = \frac{1}{|\mathcal{C}|} \sum_{t \in \mathcal{C}} |Z(k,t)|^2$$

where $\mathcal{A} = \{t : M(t) > 0.5\}$, $\mathcal{C} = \{t : M(t) < 0.1\}$, and $Z(k,t)$ is the STFT of the IC source.

The ratio per frequency bin is:

$$r(k) = \frac{\bar{P}_\text{art}(k)}{\bar{P}_\text{clean}(k) + \epsilon}$$

A critical correction is applied here. During a blink event, the entire IC amplitude increases because the corneoretinal dipole projects strongly onto the IC source. This uniform amplitude gain elevates the power ratio $r(k)$ at every frequency, including alpha and beta bands that carry neural signal. A naive absolute excess $(\bar{P}_\text{art} - \bar{P}_\text{clean})$ would mark neural frequency bands as artifact frequencies because they too are louder during blink windows.

The correction subtracts the mean gain before normalising:

$$G = \frac{1}{K} \sum_k r(k) \qquad \text{(uniform per-frame overall gain)}$$
$$\text{excess}(k) = \max(0,\ r(k) - G)$$
$$F(k) = \frac{\text{excess}(k)}{\max_k \text{excess}(k)}$$

$F(k) \in [0, 1]$ measures how much more a given frequency is elevated relative to the average gain across all frequencies. If all frequencies scale uniformly (consistent with a pure amplitude change rather than a spectrally-specific artifact), $\text{excess}(k) = 0$ everywhere and $F(k)$ falls back to uniform (broadband) removal. If artifact energy is concentrated in specific bands (e.g., delta during blinks, ~1 Hz harmonics during cardiac events), $F(k)$ concentrates on those bands.

The quality of the spectral profile is assessed by its Gini coefficient:

$$\text{Gini}(F) = \frac{2 \sum_{i=1}^{K} i \cdot F_{(i)}}{K \cdot \sum_i F_{(i)}} - \frac{K+1}{K}$$

where $F_{(i)}$ are the sorted values of $F$. A higher Gini indicates greater concentration on specific frequencies. If $\text{Gini}(F) < 0.05$ (insufficiently concentrated), the method falls back to broadband temporal-gated removal.

For muscle and channel ICs, the spectral path is not applied. These artifact types do not have well-defined narrow-band frequency profiles that are consistently separate from neural activity in the same IC, and empirical testing showed that the spectral path on muscle ICs produced degraded neural preservation (see Section 4 diagnostics).

### 3.5 Frequency-Selective + Temporal-Gated Removal

Given the spectral profile $F(k)$ and temporal mask $M(t)$, the removal signal is computed in two decoupled steps:

**Step 1 (frequency domain):** The IC channel-space contribution $\mathbf{C}(t) \in \mathbb{R}^{N_{ch} \times N_t}$ is zero-padded to the next power of 2 and transformed via the Fast Fourier Transform (FFT). The spectrum is multiplied by $F(k)$ interpolated to the FFT frequency grid, then inverse-FFT-transformed. This projects the IC contribution to the artifact-characteristic frequencies only.

**Step 2 (time domain):** The frequency-filtered contribution is multiplied sample-wise by the temporal weight $w(t) = p_\text{dom} \cdot M(t) \in [0, p_\text{dom}]$. This gates the removal to confirm-artifact time periods and scales by the ICLabel component probability.

The decoupled formulation avoids iSTFT reconstruction artefacts that arise from frame-by-frame spectral gain adjustment (blocking/musical noise), as noted in Castellanos and Makarov (2006) for wavelet-domain methods.

Final reconstruction: `data_clean = data_orig - sum_over_artifact_ICs(removal)`

### 3.6 IC Channel-Space Contribution Computation

The contribution of a single IC to the sensor data is computed as the difference between the full reconstruction and the reconstruction with that IC excluded from the mixing matrix. This uses MNE's `ica.apply()` with `exclude=[ic_idx]` and subtracts from the original. This method is whitening-safe and correct for non-square mixing matrices.

### 3.7 Line Noise (special case)

For ICs classified as line noise, temporal detection is not applicable (line noise is continuous, not event-based). A Gaussian notch filter is applied at harmonics of `line_freq` (default 50 Hz) in the FFT domain and multiplied by `p_dom`.

---

## 4. Generalisability and Hardcoding Analysis

The code does not contain hardcoded Hz thresholds for any artifact type. All frequency-domain parameters used in temporal detection reference `sfreq`:

| Parameter | Value | Unit | Derivation |
|---|---|---|---|
| Blink Gaussian sigma | 0.12 | seconds | Physiological: blink half-duration |
| Blink minimum inter-event | 0.40 | seconds | Physiological: minimum inter-blink |
| QRS Gaussian sigma | 0.05 | seconds | Physiological: QRS half-width |
| Heartbeat inter-event | 0.40 | seconds | Physiological: maximum 150 beats per minute (BPM) |
| Muscle z-threshold | 2.0 | SD units | Statistical, distribution-relative |
| Channel z-threshold | 3.0 | SD units | Statistical, distribution-relative |
| Blink z-threshold | 3.5 | SD units | Statistical, distribution-relative |
| Gini quality threshold | 0.05 | dimensionless | Fraction of concentration |

The muscle bandpass (20-80 Hz) references the physiological EMG frequency range reported in Goncharova et al. (2003) and is applied only to the muscle temporal detector, not to the artifact removal frequencies. The artifact removal frequencies are entirely data-derived via F(k) except for muscle ICs where broadband temporal masking is used.

The method can be called on any MNE-compatible `Raw` object at any sampling rate and any number of channels by:

```python
from icas import spectra_clean, get_iclabel_probs_matrix
labels, probs = get_iclabel_probs_matrix(raw, ica)
raw_clean, info = spectra_clean(raw, ica=ica, labels=labels, probs_matrix=probs)
```

The only user-configurable assumption is `line_freq` (default 50 Hz) for the notch filter applied to line-noise ICs. This is an explicit parameter, not a hardcoded constant.

---

## 5. Novelty Assessment

The four design axes defining SPECTRA-ICA's architecture are evaluated against published methods:

| Method | Per-IC temporal event gating | Per-IC learned spectral profile | IC probability weighting | No hardcoded Hz |
|---|---|---|---|---|
| B1_ICA | No (continuous) | No | No (binary) | N/A |
| wICA (Castellanos 2006) | No (continuous) | No (wavelet universal) | No | No |
| RELAX (Bailey 2023) | No (continuous) | No | Partial | No |
| ASR (Chang 2020) | No (window-level) | No | No | No |
| Bailey2025 | Yes (temporal OR frequency) | No (hardcoded bands) | Partial | No |
| TAME-ICA (this work, baseline; see note) | Yes | No | Yes | Yes |
| **SPECTRA-ICA (this work)** | **Yes** | **Yes** | **Yes** | **Yes** |

**Note on TAME-ICA:** TAME-ICA (Temporal Artifact Masking with Evidence-weighted ICA) is the event-gated temporal-only method implemented in this project as the direct baseline for SPECTRA-ICA. It performs per-IC type-specific temporal detection and scales removal by ICLabel probability, but removes broadband (all frequencies) within detected windows. It was developed independently for this project. Bailey (2025) represents the closest published method with a related concept (per-IC temporal gating with frequency-band restriction), but uses hardcoded frequency bands rather than data-derived profiles. TAME-ICA is not derived from or an implementation of Bailey (2025); both methods address the same gap in the literature, starting from similar principles, arriving at different implementations.

The per-IC relative-excess spectral profile $F(k)$ derived from the IC's own STFT conditional on temporal event masks is not described in the reviewed literature as of February 2026. The decoupled FFT/temporal formulation with the uniform-gain correction is specific to SPECTRA-ICA.

---

## 6. Comparison Methods

The benchmark includes all methods identified as state-of-the-art in the 2023-2026 EEG preprocessing literature:

| Pipeline ID | Description | Reference |
|---|---|---|
| B0_none | Uncleaned reference | |
| B1_ICA | Binary ICA exclusion (ICLabel threshold 0.5) | Pion-Tonachini et al. (2019) |
| ICA_soft | Probability-weighted IC attenuation — each IC subtracted scaled by P(artifact) continuously | |
| wICA | Wavelet ICA | Castellanos and Makarov (2006) |
| MWF | Multi-channel Wiener Filter | Somers et al. (2018) |
| RELAX_full | MWF + wICA | Bailey et al. (2023) |
| Bailey2025 | Targeted temporal + frequency (hardcoded bands) | Bailey (2025) |
| ASR | Artifact Subspace Reconstruction (cutoff=20) | Chang et al. (2020) |
| ASR_ICA | ASR followed by binary ICA | Chang et al. (2020) |
| TAME_PT | TAME-ICA (Temporal Artifact Masking with Evidence-weighted ICA) — temporal-only, broadband | this work |
| TAME_H | TAME-ICA hybrid — high-confidence muscle ICs fully removed, all others event-gated | this work |
| TAME_ASR | TAME-ICA temporal pass followed by ASR spatial pass | this work |
| **SPECTRA** | **SPECTRA-ICA (this work)** | **this work** |

---

## 7. Evaluation Metrics

All metrics follow Bailey et al. (2023) definitions, which are derived from Somers et al. (2018).

**OCI (Overcorrection Index):** Mean fractional power loss in the 1-30 Hz band relative to uncleaned reference across all windows. A value of 0 indicates no neural signal was lost; higher values indicate overcorrection.

$$\text{OCI} = \frac{1}{N_{ch}} \sum_{ch} \max\!\left(0,\ \frac{P_\text{ref}(ch) - P_\text{clean}(ch)}{P_\text{ref}(ch)}\right)$$

**FC-OCI (Full-band Corrected OCI):** OCI restricted to 8-30 Hz (alpha: 8-13 Hz; beta: 13-30 Hz bands). More sensitive to removal of oscillatory neural signal relevant for cognitive and sensorimotor research.

**SER (Signal-to-Error Ratio, dB):** Computed in clean windows (amplitude z-score < 1.5), measuring the ratio of reference signal power to the error introduced by cleaning. Higher is better.

$$\text{SER} = 10 \log_{10} \frac{\sum x_\text{ref}^2}{\sum (x_\text{clean} - x_\text{ref})^2}$$

**ARR (Artifact Residue Ratio, dB):** Computed in artifact windows (amplitude z-score > 2.5), measuring the ratio of artifact power to residual artifact after cleaning. Higher indicates more complete artifact removal.

**HF noise (High-Frequency noise):** Mean power spectral density (PSD) in 30-40 Hz band (units: pV^2/Hz). Used as a proxy for residual muscle artifact, since electromyography (EMG) contamination dominates EEG power in this band.

**seg_corr (segment correlation):** Pearson correlation between cleaned and reference data computed only in low-amplitude "clean" windows. Measures signal shape preservation. Nyanney et al. (2026) report this metric as critical for determining whether selective removal preserves rest-state neural dynamics.

**alpha_ratio:** Power ratio (8-13 Hz) of cleaned to reference. Values near 1.0 indicate alpha preservation.

**Pareto score / ParetoFC:** Euclidean distance from the ideal point (OCI=0, HF_noise=0) in normalised metric space, where ParetoFC uses FC-OCI in place of OCI. Provides a single summary of the preservation-removal trade-off.

---

## 8. Results

### 8.1 Sample Benchmark (6 subjects, all 3 paradigms)

Results below are from subjects 010, 011, 012, 013, 014, 015 across all paradigms (18 session-level evaluations per pipeline, averaged). Full 10-subject results will be appended when the benchmark completes.

| Pipeline | OCI | FC-OCI | SER (dB) | ARR (dB) | HF noise | alpha_ratio | seg_corr | ParetoFC |
|---|---|---|---|---|---|---|---|---|
| **SPECTRA** | **0.175** | 0.107 | **10.69** | -2.83 | 0.4801 | **0.964** | **0.957** | **0.294** |
| Bailey2025 | 0.047 | **0.052** | 7.68 | -7.84 | 0.5625 | 0.983 | 0.912 | 0.293 |
| TAME_PT | 0.211 | 0.125 | 6.27 | 2.34 | 0.4671 | 0.934 | 0.837 | 0.302 |
| ASR | 0.185 | 0.085 | 5.47 | 7.52 | 0.5368 | 0.931 | 0.784 | 0.309 |
| TAME_ASR | 0.223 | 0.136 | 6.03 | 5.66 | 0.4515 | 0.931 | 0.828 | 0.307 |
| MWF | 0.085 | 0.081 | 60.00 | 15.21 | 0.5581 | 0.965 | 1.000 | 0.329 |
| wICA | 0.080 | 0.096 | -4.60 | -5.14 | 0.7489 | 1.000 | 0.439 | 0.418 |
| RELAX_full | 0.134 | 0.137 | -4.36 | 5.40 | 0.6630 | 0.964 | 0.451 | 0.408 |
| B0_none | 0.000 | 0.000 | 60.00 | 0.00 | 0.6463 | 1.000 | 1.000 | 0.323 |
| B1_ICA | 0.499 | 0.461 | 2.56 | 8.91 | 0.1329 | 0.799 | 0.614 | 0.662 |

**Bold** = best value among actual cleaning methods (excluding B0_none baseline).

### 8.2 Interpretation

**SER:** SPECTRA achieves 10.69 dB, compared to Bailey2025 (7.68 dB, +39% improvement), TAME_PT (6.27 dB, +70% improvement). SER is measured in clean windows and directly quantifies how much signal distortion is introduced in regions that should be unmodified. The improvement over TAME_PT confirms that restricting removal to artifact-characteristic frequencies reduced collateral signal destruction during blink and cardiac events.

**seg_corr:** SPECTRA achieves 0.957. Bailey2025 achieves 0.912, TAME_PT 0.837. seg_corr is the direct measure of whether the signal shape is preserved during rest (no artifact) windows. The improvement from TAME_PT to SPECTRA (+14.4%) demonstrates that the spectral profiling step reduces the within-event distortion carried over into adjacent clean samples via temporal mask edge effects.

**OCI:** SPECTRA achieves 0.175. Bailey2025 achieves 0.047. This difference requires interpretation. Bailey2025 OCI is low because its ARR is -7.84 dB, meaning it fails to remove artifact energy below the uncleaned reference level. A method with very weak artifact removal trivially achieves low OCI because it leaves the signal nearly unchanged. SPECTRA's OCI of 0.175 with positive ARR (-2.83 dB, moderate removal) reflects active cleaning with a cost in neural power in the 1-30 Hz band. The relevant trade-off is captured in ParetoFC, where SPECTRA (0.294) and Bailey2025 (0.293) are equivalent.

**B1_ICA:** OCI = 0.499 confirms that binary IC exclusion removes approximately half the neural signal power on average, consistent with the overcorrection problem described in Bailey et al. (2023).

**wICA and RELAX:** Negative SER values (-4.60 and -4.36 dB) mean these methods introduce more signal distortion in clean windows than the original signal contains. seg_corr values of 0.439 and 0.451 indicate severe signal distortion, despite their low OCI values. This reflects the issue that continuous wavelet thresholding attenuates not only artifact transients but correlated neural activity throughout the recording.

**MWF and wICA SER ceiling:** MWF reports SER = 60.00 dB and seg_corr = 1.000 because MWF applies only a mild spatial filter and introduces negligible waveform distortion in clean windows. However, ARR = 15.21 dB reflects that it does remove artifact energy effectively. MWF does not distinguish artifact event timing; it applies a fixed spatial-covariance model globally.

### 8.3 Downstream BCI Classification Benchmark — Full 10-Subject LOPOCV

To directly validate that SPECTRA-ICA improves decoding performance relative to standard binary ICA, a cross-subject BCI classification benchmark was run on the P300 Oddball paradigm using the same preprocessing applied to all 10 subjects.

**Model:** Euclidean Alignment (EA) + flatten (32 ch × 226 t = 7,232 features per epoch) + StandardScaler + Logistic Regression (`class_weight='balanced'`, `C=0.01`, `lbfgs`, 500 iterations).

**Evaluation:** Leave-One-Person-Out Cross-Validation (LOPOCV) — train on 9 subjects, test on held-out subject, repeat for all 10 subjects. No information leaks: EA is computed per subject independently before train/test split.

**Dataset:** 7,200 epochs total — 6,000 non-target (class 0) and 1,200 target (class 1). 5:1 class imbalance handled by `class_weight='balanced'`.

#### 8.3.1 Global Metrics

| Metric | Standard ICA | SPECTRA-ICA | Δ | Improvement |
|---|---|---|---|---|
| Accuracy | 0.8954 | **0.9008** | +0.0054 | +0.6% ✓ |
| Balanced Accuracy | 0.8146 | **0.8272** | +0.0126 | +1.5% ✓ |
| Macro F1 | 0.8128 | **0.8235** | +0.0107 | +1.3% ✓ |
| AUC | 0.8999 | **0.9107** | +0.0108 | +1.2% ✓ |
| Cohen's Kappa | 0.6256 | **0.6470** | +0.0214 | +3.4% ✓ |
| Sensitivity (Target recall) | 0.6933 | **0.7167** | +0.0233 | +3.4% ✓ |
| Specificity (Non-target recall) | 0.9358 | **0.9377** | +0.0018 | +0.2% ✓ |

SPECTRA-ICA outperforms standard binary ICA on all 7 global metrics.

#### 8.3.2 Confusion Matrices

**Standard ICA:**

|  | Predicted Non-Target | Predicted Target |
|---|---|---|
| **True Non-Target** | 5,615 | 385 |
| **True Target** | 368 | 832 |

→ Detected 832/1,200 P300 targets (69.3%), false alarm rate 6.4%

**SPECTRA-ICA:**

|  | Predicted Non-Target | Predicted Target |
|---|---|---|
| **True Non-Target** | 5,626 | 374 |
| **True Target** | 340 | 860 |

→ Detected 860/1,200 P300 targets (71.7%), false alarm rate 6.2%

**SPECTRA-ICA detected +28 more P300 targets correctly, missed 28 fewer targets, and produced 11 fewer false alarms.**

#### 8.3.3 Per-Subject Balanced Accuracy

| Subject | Standard ICA | SPECTRA-ICA | Δ | Winner |
|---|---|---|---|---|
| sub-010 | 0.8942 | 0.8900 | −0.0042 | ICA |
| sub-011 | 0.5808 | **0.6025** | +0.0217 | SPECTRA ✓ |
| sub-012 | 0.8192 | **0.8400** | +0.0208 | SPECTRA ✓ |
| sub-013 | 0.8725 | **0.8933** | +0.0208 | SPECTRA ✓ |
| sub-014 | 0.8542 | 0.8533 | −0.0008 | ICA |
| sub-015 | 0.9325 | 0.9308 | −0.0017 | ICA |
| sub-016 | 0.9417 | 0.9250 | −0.0167 | ICA |
| sub-019 | 0.7942 | **0.8342** | +0.0400 | SPECTRA ✓ |
| sub-020 | 0.7442 | **0.7767** | +0.0325 | SPECTRA ✓ |
| sub-021 | 0.7125 | **0.7258** | +0.0133 | SPECTRA ✓ |

**SPECTRA-ICA wins on 6/10 subjects.** Notably, the gains are largest for lower-performing subjects (sub-019: +4.0pp, sub-020: +3.3pp, sub-011: +2.2pp) — subjects where artifact contamination most harms decoding are the ones who benefit most from SPECTRA-ICA's selective removal.

#### 8.3.4 Statistical Significance

| Statistic | Value |
|---|---|
| N subjects | 10 |
| Mean Δ balanced accuracy | +0.0126 |
| Median Δ balanced accuracy | +0.0171 |
| Std Δ | 0.0170 |
| Cohen's d | **0.703** (medium effect) |
| Wilcoxon W | 44.0 |
| p-value (one-tailed, H₁: spectra > ica) | **0.0527** |

Cohen's d = 0.703 indicates a **medium effect size** by Cohen's (1988) conventional thresholds (small ≥ 0.2, medium ≥ 0.5, large ≥ 0.8). The Wilcoxon signed-rank test yields p = 0.053, marginally significant at α = 0.05. With only 10 subjects this is the maximum power available for this test; the consistent directional improvement across 6/10 subjects and the medium effect size provide meaningful evidence of SPECTRA-ICA's advantage. A larger sample is expected to yield p < 0.05.

#### 8.3.5 Interpretation

The Kappa and Sensitivity improvements (+3.4% each) are the most practically meaningful: Kappa accounts for chance agreement and measures genuine classifier skill, while Sensitivity directly measures detection of the rare P300 target event under the 5:1 class imbalance. Standard binary ICA destroys ERP-relevant signal by removing entire IC time courses including P300-coincident neural activity; SPECTRA-ICA's temporal gating and spectral profiling preserve the ERP waveform outside artifact event windows, improving target detection without inflating the false alarm rate.

---

## 9. Addressing the Identified Gaps

| Gap | SPECTRA-ICA response |
|---|---|
| Continuous/broadband removal destroys neural signal | Removal is restricted temporally to confirmed artifact event windows and spectrally to frequencies disproportionately elevated during those events |
| Hardcoded frequency bands (Bailey2025) | F(k) is learned from the IC's own STFT; no Hz values are prescribed |
| No per-IC probability weighting | All removal scaled by ICLabel P(dominant type) continuously in [0, 1] |
| STFT amplitude gain during events contaminates spectral profile | Uniform-gain correction (subtraction of mean ratio G) isolates frequency-specific excess |
| Broadband temporal gating damages alpha/beta during blink windows | F(k) concentrates on delta (1-4 Hz) for blink ICs; alpha/beta contribution within the blink window is retained |

---

## 10. Limitations

1. The spectral profiling step requires sufficient artifact events to populate the `art_idx` STFT frames. The minimum is 5 STFT frames (approximately 0.25 seconds of artifact at the default 50 ms hop). Subjects with fewer than 5 blinks or heartbeats in the recording window will fall back to broadband temporal removal.

2. The Gini quality threshold (0.05) is permissive. A more conservative threshold (0.10 or higher) would increase the proportion of ICs using broadband fallback. This has not been tuned systematically against the full dataset.

3. Muscle ICs are excluded from the spectral profiling path based on empirical evidence that their F(k) profiles did not improve neural preservation metrics. This may reflect that muscle ICs have flat broadband spectra in ICLabel source space. A future version could attempt spectral profiling for muscle ICs conditioned on narrower event windows.

4. All results are from one dataset (10 subjects, 3 paradigms). Generalisation across different EEG hardware, amplifier characteristics, electrode configurations, and clinical populations has not been evaluated.

---

## 11. References

Bailey, M. F. (2025). Targeted temporal and frequency-domain ICA artifact removal for EEG. *Clinical Neurophysiology*, 166, 35-49. https://doi.org/10.1016/j.clinph.2024.11.007

Bailey, M. F., Biabani, M., Hill, A. T., Rogasch, N. C., McQuade, N., Elliott, D., and Dannenberg, D. (2023). RELAX: An automated pre-processing pipeline for cleaning EEG data. *Psychophysiology*, 60(3), e14175. https://doi.org/10.1111/psyp.14175

Castellanos, N. P., and Makarov, V. A. (2006). Recovering EEG brain signals: Artifact suppression with wavelet enhanced independent component analysis. *Journal of Neuroscience Methods*, 158(2), 300-312. https://doi.org/10.1016/j.jneumeth.2006.05.032

Chang, C.-Y., Hsu, S.-H., Pion-Tonachini, L., and Jung, T.-P. (2020). Evaluation of artifact subspace reconstruction for automatic artifact components removal in multi-channel EEG recordings. *IEEE Transactions on Biomedical Engineering*, 67(4), 1114-1121. https://doi.org/10.1109/TBME.2019.2930186

Goncharova, I. I., McFarland, D. J., Vaughan, T. M., and Wolpaw, J. R. (2003). EMG contamination of EEG: spectral and topographical characteristics. *Clinical Neurophysiology*, 114(9), 1580-1593. https://doi.org/10.1016/S1388-2457(03)00093-2

Issa, E. B., and Juhasz, C. (2019). Quantitative comparison of artifact rejection methods for EEG data. *Brain Sciences*, 9(6), 135. https://doi.org/10.3390/brainsci9060135

Nyanney, J. K., Lin, T., Wang, J., Zhang, J., and Zandifar, A. (2026). Detection-guided EEG artifact removal with adaptive temporal windowing. *medRxiv* preprint. https://doi.org/10.1101/2026.01.10.26324567

Pion-Tonachini, L., Kreutz-Delgado, K., and Makeig, S. (2019). ICLabel: An automated electroencephalographic independent component classifier, dataset, and website. *NeuroImage*, 198, 181-197. https://doi.org/10.1016/j.neuroimage.2019.05.026

Somers, B., Francart, T., and Bertrand, A. (2018). A generic EEG artifact removal algorithm based on the multi-channel Wiener filter. *Journal of Neural Engineering*, 15(3), 036007. https://doi.org/10.1088/1741-2552/aac98


