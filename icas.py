""" 
SPECTRA-ICA: Spectral Profile-Estimated Continuous Temporal Removal of Artifacts via ICA

Combines per-IC type-specific temporal event detection with per-IC data-driven spectral
fingerprint estimation. For each artifact IC, a temporal mask identifies artifact windows;
a spectral profile F(k) is estimated from the relative STFT power excess of artifact frames
over clean frames; and removal is gated to artifact-characteristic frequencies only.
ICLabel probability weights modulate removal magnitude. All parameters derive from sfreq.

Provides two cleaning functions:
  tame_clean    -- temporal-only gating (broadband removal within detected event windows)
  spectra_clean -- temporal + spectral gating (removes only artifact-characteristic freq)

References:
  Castellanos & Makarov (2006) Biol Cybern; Issa & Juhasz (2019) Brain Sci;
  Bailey et al. (2023) Psychophysiology; Bailey (2025) Clin Neurophysiol;
  Pion-Tonachini et al. (2019) NeuroImage
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, hilbert, find_peaks, stft as _scipy_stft
from scipy.ndimage import gaussian_filter1d, uniform_filter1d




def _detect_blink_mask(
    ic_series: NDArray,
    sfreq: float,
    z_thresh: float = 3.5,
    blink_sigma_s: float = 0.12,
    min_interval_s: float = 0.4,
) -> tuple[NDArray, dict]:
    """Detect blink peaks in an eye IC; returns Gaussian soft mask in [0, 1]."""
    n = len(ic_series)
    rect = np.abs(ic_series - np.median(ic_series))
    z = (rect - rect.mean()) / (rect.std() + 1e-10)

    min_dist = max(1, int(min_interval_s * sfreq))
    peaks, _ = find_peaks(z, height=z_thresh, distance=min_dist)

    sigma = max(1, int(blink_sigma_s * sfreq))
    mask = np.zeros(n)
    t = np.arange(n)
    for pk in peaks:
        mask += np.exp(-0.5 * ((t - pk) / sigma) ** 2)

    mask = np.clip(mask, 0, 1)
    return mask, {"n_events": int(len(peaks)), "mask_fraction": float(mask.mean())}


def _detect_muscle_mask(
    ic_series: NDArray,
    sfreq: float,
    z_thresh: float = 2.0,
    hf_low: float = 20.0,
    hf_high: float = 80.0,
    smooth_s: float = 0.04,
    edge_sigma_s: float = 0.02,
) -> tuple[NDArray, dict]:
    """Detect HF burst activity in a muscle IC; returns smooth binary mask in [0, 1]."""
    nyq = sfreq / 2.0
    lo = hf_low / nyq
    hi = min(hf_high, nyq * 0.95) / nyq
    if lo >= hi:
        return np.zeros(len(ic_series)), {"n_events": 0, "mask_fraction": 0.0}

    sos = butter(4, [lo, hi], btype="band", output="sos")
    filt = sosfilt(sos, ic_series)
    env = np.abs(hilbert(filt))

    smooth_n = max(1, int(smooth_s * sfreq))
    env = uniform_filter1d(env, size=smooth_n)

    thresh = env.mean() + z_thresh * env.std()
    mask = (env > thresh).astype(float)
    sigma_edge = max(1, int(edge_sigma_s * sfreq))
    mask = gaussian_filter1d(mask, sigma=sigma_edge)
    mask = np.clip(mask, 0, 1)

    n_events = int((np.diff((mask > 0.5).astype(int)) > 0).sum())
    return mask, {"n_events": n_events, "mask_fraction": float(mask.mean())}


def _detect_heart_mask(
    ic_series: NDArray,
    sfreq: float,
    z_thresh: float = 2.5,
    qrs_sigma_s: float = 0.05,
    min_interval_s: float = 0.4,
) -> tuple[NDArray, dict]:
    """Detect cardiac R-peaks in a heart IC; returns Gaussian soft mask in [0, 1]."""
    n = len(ic_series)
    z = (ic_series - ic_series.mean()) / (ic_series.std() + 1e-10)
    min_dist = max(1, int(min_interval_s * sfreq))

    pos_peaks, _ = find_peaks(z, height=z_thresh, distance=min_dist)
    neg_peaks, _ = find_peaks(-z, height=z_thresh, distance=min_dist)
    peaks = pos_peaks if len(pos_peaks) >= len(neg_peaks) else neg_peaks

    sigma = max(1, int(qrs_sigma_s * sfreq))
    mask = np.zeros(n)
    t = np.arange(n)
    for pk in peaks:
        mask += np.exp(-0.5 * ((t - pk) / sigma) ** 2)

    mask = np.clip(mask, 0, 1)
    return mask, {"n_events": int(len(peaks)), "mask_fraction": float(mask.mean())}


def _detect_channel_mask(
    ic_series: NDArray,
    sfreq: float,
    window_s: float = 0.2,
    z_thresh: float = 3.0,
    edge_sigma_s: float = 0.02,
) -> tuple[NDArray, dict]:
    """Detect windowed amplitude anomalies in a channel-noise IC; returns mask in [0, 1]."""
    window = max(1, int(window_s * sfreq))
    local_mean = uniform_filter1d(ic_series, size=window)
    resid = np.abs(ic_series - local_mean)
    z = (resid - resid.mean()) / (resid.std() + 1e-10)
    mask = (z > z_thresh).astype(float)
    sigma = max(1, int(edge_sigma_s * sfreq))
    mask = gaussian_filter1d(mask, sigma=sigma)
    mask = np.clip(mask, 0, 1)
    n_events = int((np.diff((mask > 0.5).astype(int)) > 0).sum())
    return mask, {"n_events": n_events, "mask_fraction": float(mask.mean())}


def _spectral_notch_removal(
    ic_contribution: NDArray,
    sfreq: float,
    n_fft: int,
    fft_freqs: NDArray,
    line_freq: float = 50.0,
    bandwidth: float = 1.5,
) -> NDArray:
    """Apply Gaussian spectral notch at line_freq harmonics to the IC channel contribution."""
    n_ch, n_times = ic_contribution.shape

    notch_gain = np.zeros(len(fft_freqs))
    for harmonic in range(1, 5):
        center = line_freq * harmonic
        if center > sfreq / 2:
            break
        notch_gain += np.exp(-0.5 * ((fft_freqs - center) / bandwidth) ** 2)
    notch_gain = np.clip(notch_gain, 0, 1)

    removal = np.zeros_like(ic_contribution)
    for ch in range(n_ch):
        spec = np.fft.rfft(ic_contribution[ch], n=n_fft)
        g = notch_gain[:len(spec)] if len(notch_gain) >= len(spec) else \
            np.pad(notch_gain, (0, len(spec) - len(notch_gain)))
        removal[ch] = np.fft.irfft(spec * g, n=n_fft)[:n_times]

    return removal




def get_iclabel_probs_matrix(raw, ica):
    """Run ICLabel and return (labels, probs_matrix).

    Returns
    -------
    labels : list[str]  -- per-IC string label
    probs_matrix : ndarray (n_ics, 7)  -- columns: brain(0) muscle(1) eye(2) heart(3)
                                          line(4) channel(5) other(6)
    """
    from mne_icalabel import label_components
    result = label_components(raw, ica, method="iclabel")
    labels = list(result["labels"])
    probs_matrix = np.array(ica.labels_scores_, dtype=float)
    return labels, probs_matrix



def tame_clean(
    raw,
    ica=None,
    labels: list[str] | None = None,
    probs_matrix: NDArray | None = None,
    brain_threshold: float = 0.50,
    max_mask_fraction: float = 0.50,
    line_freq: float = 50.0,
    muscle_full_removal_thresh: float = 0.75,
):
    """TAME-ICA: temporal-only event-gated ICA cleaning.

    For each artifact IC, a type-specific temporal detector identifies artifact windows.
    The IC channel contribution is subtracted only within those windows, weighted by
    P(dominant_type) from ICLabel. Brain ICs are untouched.
    """
    import mne

    sfreq = raw.info["sfreq"]
    data = raw.get_data()           # (n_ch, n_times)
    n_ch, n_times = data.shape

    if ica is None:
        ica = mne.preprocessing.ICA(
            n_components=20, method="infomax",
            random_state=42, max_iter="auto", verbose=False,
        )
        ica.fit(raw, verbose=False)

    if labels is None or probs_matrix is None or np.array(probs_matrix).ndim == 1:
        from mne_icalabel import label_components
        result = label_components(raw, ica, method="iclabel")
        labels = list(result["labels"])
        probs_matrix = np.array(ica.labels_scores_, dtype=float)
    else:
        probs_matrix = np.array(probs_matrix, dtype=float)
        if probs_matrix.ndim == 1:
            if not hasattr(ica, "labels_scores_"):
                from mne_icalabel import label_components
                result = label_components(raw, ica, method="iclabel")
                labels = list(result["labels"])
            probs_matrix = np.array(ica.labels_scores_, dtype=float)

    n_ics = probs_matrix.shape[0]
    p_brain_vec = probs_matrix[:, 0] + probs_matrix[:, 6]
    artifact_ics = [i for i in range(n_ics) if p_brain_vec[i] < brain_threshold]
    brain_ics    = [i for i in range(n_ics) if p_brain_vec[i] >= brain_threshold]

    sources       = ica.get_sources(raw).get_data()
    n_fft         = int(2 ** np.ceil(np.log2(n_times)))
    fft_freqs     = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
    total_removal = np.zeros_like(data)
    ic_stats = []

    for ic_idx in artifact_ics:
        probs  = probs_matrix[ic_idx]
        source = sources[ic_idx]

        p_muscle = float(probs[1])
        p_eye    = float(probs[2])
        p_heart  = float(probs[3])
        p_line   = float(probs[4])
        p_chan   = float(probs[5])

        type_probs = {
            "eye": p_eye, "muscle": p_muscle, "heart": p_heart,
            "line": p_line, "channel": p_chan,
        }
        dom_type = max(type_probs, key=type_probs.get)
        dom_prob = type_probs[dom_type]

        if dom_prob < 0.20:
            ic_stats.append({
                "ic": ic_idx, "label": labels[ic_idx],
                "type": "skipped", "dom_prob": dom_prob,
                "mask_fraction": 0.0, "strategy": "none",
            })
            continue

        ica_copy = ica.copy()
        ica_copy.exclude = [ic_idx]
        raw_copy = raw.copy()
        ica_copy.apply(raw_copy, verbose=False)
        ic_contribution = data - raw_copy.get_data()    # (n_ch, n_times)

        if dom_type == "line":
            removal = _spectral_notch_removal(
                ic_contribution, sfreq, n_fft, fft_freqs, line_freq
            ) * dom_prob
            total_removal += removal
            ic_stats.append({
                "ic": ic_idx, "label": labels[ic_idx],
                "type": "line", "dom_prob": dom_prob,
                "mask_fraction": 1.0, "strategy": "spectral_notch", "n_events": "N/A",
            })
            continue

        if dom_type == "muscle" and dom_prob >= muscle_full_removal_thresh:
            total_removal += ic_contribution * dom_prob
            ic_stats.append({
                "ic": ic_idx, "label": labels[ic_idx],
                "type": "muscle", "dom_prob": dom_prob,
                "mask_fraction": 1.0, "strategy": "muscle_full_removal",
                "n_events": "N/A",
            })
            continue

        if dom_type == "eye":
            mask, diag = _detect_blink_mask(source, sfreq)
        elif dom_type == "muscle":
            mask, diag = _detect_muscle_mask(source, sfreq)
        elif dom_type == "heart":
            mask, diag = _detect_heart_mask(source, sfreq)
        else:
            mask, diag = _detect_channel_mask(source, sfreq)

        mask_frac = float(mask.mean())

        if mask_frac > max_mask_fraction:
            total_removal += ic_contribution * dom_prob
            strategy = "fallback_full_removal"
        else:
            weighted_mask = dom_prob * mask
            total_removal += ic_contribution * weighted_mask[np.newaxis, :]
            strategy = "temporal_selective"

        ic_stats.append({
            "ic":            ic_idx,
            "label":         labels[ic_idx],
            "type":          dom_type,
            "dom_prob":      dom_prob,
            "mask_fraction": mask_frac,
            "n_events":      diag.get("n_events", 0),
            "strategy":      strategy,
        })

    data_clean = data - total_removal
    raw_clean = raw.copy()
    raw_clean._data = data_clean

    removed_power = float(np.mean(total_removal ** 2))
    orig_power    = float(np.mean(data ** 2))

    temporal     = [s for s in ic_stats if s["strategy"] == "temporal_selective"]
    fallbacks    = [s for s in ic_stats if s["strategy"] == "fallback_full_removal"]
    spectral     = [s for s in ic_stats if s["strategy"] == "spectral_notch"]
    muscle_full  = [s for s in ic_stats if s["strategy"] == "muscle_full_removal"]

    mean_mf = float(np.mean([s["mask_fraction"] for s in temporal])) if temporal else 0.0

    info = {
        "n_brain_ics":               len(brain_ics),
        "n_artifact_ics":            len(artifact_ics),
        "n_temporal_selective":      len(temporal),
        "n_fallback_full":           len(fallbacks),
        "n_spectral_notch":          len(spectral),
        "n_muscle_full_removal":     len(muscle_full),
        "mean_temporal_mask_frac":   mean_mf,
        "removed_power_fraction":    removed_power / (orig_power + 1e-30),
        "ic_stats":                  ic_stats,
    }

    return raw_clean, info


def _stft_of_signal(signal: NDArray, sfreq: float) -> tuple[NDArray, NDArray, NDArray]:
    """STFT with window=0.5*sfreq samples and hop=0.05*sfreq samples."""
    n        = len(signal)
    nperseg  = min(max(4, int(round(sfreq * 0.5))), n // 2)
    hop      = max(1, int(round(sfreq * 0.05)))
    noverlap = min(nperseg - hop, nperseg - 1)
    f, t, Z  = _scipy_stft(signal, fs=sfreq, window="hann",
                            nperseg=nperseg, noverlap=noverlap, boundary=None)
    return f, t, Z


def _estimate_artifact_spectral_profile(
    ic_source: NDArray,
    temporal_mask: NDArray,
    sfreq: float,
    min_art_frames: int = 5,
) -> tuple[NDArray, NDArray, float]:
    """Learn F(k) — the artifact frequency fingerprint for this IC.

    Uses relative spectral excess: measures how much MORE a frequency is elevated
    during artifact frames compared to the average gain across all frequencies.
    If all frequencies scale uniformly, F(k) falls back to broadband removal.
    """
    f_bins, t_frames, Z = _stft_of_signal(ic_source, sfreq)
    P        = np.abs(Z) ** 2
    n_times  = len(ic_source)

    t_samp   = (t_frames * sfreq).astype(int).clip(0, n_times - 1)
    frame_m  = temporal_mask[t_samp]

    art_idx   = frame_m > 0.5
    clean_idx = frame_m < 0.1

    if art_idx.sum() < min_art_frames or clean_idx.sum() < 3:
        return np.ones(len(f_bins)), f_bins, 0.0

    art_psd   = P[:, art_idx].mean(axis=1)
    clean_psd = P[:, clean_idx].mean(axis=1)

    ratio = art_psd / (clean_psd + 1e-30)
    overall_gain = float(ratio.mean())
    relative_excess = np.maximum(0.0, ratio - overall_gain)

    max_e = relative_excess.max()
    if max_e < 1e-30:
        return np.ones(len(f_bins)), f_bins, 0.0

    F = relative_excess / max_e

    F_sorted = np.sort(F)
    n        = len(F_sorted)
    cum      = np.cumsum(F_sorted)
    gini     = float(
        (2 * np.dot(np.arange(1, n + 1), F_sorted) / (n * cum[-1] + 1e-30)) - (n + 1) / n
    )
    quality  = float(np.clip(gini, 0.0, 1.0))

    return F, f_bins, quality


def _frequency_selective_removal(
    source: NDArray,
    ic_contribution: NDArray,
    F_profile: NDArray,
    f_bins_stft: NDArray,
    temporal_weight: NDArray,
    sfreq: float,
) -> NDArray:
    """Frequency-selective + temporal-gated removal via global FFT filter."""
    n_ch, n_times = ic_contribution.shape
    n_fft     = int(2 ** np.ceil(np.log2(n_times)))
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
    F_interp  = np.interp(fft_freqs, f_bins_stft, F_profile).clip(0.0, 1.0)

    removal = np.zeros((n_ch, n_times))
    for ch in range(n_ch):
        spec          = np.fft.rfft(ic_contribution[ch], n=n_fft)
        artifact_time = np.fft.irfft(spec * F_interp, n=n_fft)[:n_times]
        removal[ch]   = artifact_time * temporal_weight
    return removal


def spectra_clean(
    raw,
    ica=None,
    labels: list[str] | None = None,
    probs_matrix: NDArray | None = None,
    brain_threshold: float = 0.50,
    max_mask_fraction: float = 0.60,
    line_freq: float = 50.0,
    spectral_quality_threshold: float = 0.05,
    muscle_full_removal_thresh: float = 1.01,
):
    """SPECTRA-ICA: per-IC temporal detection + data-driven spectral fingerprint removal.

    For eye and heart ICs, a spectral profile F(k) is learned from the relative
    STFT power excess during detected artifact windows: only frequencies disproportionately
    elevated above the uniform gain during those windows are removed. Muscle and channel
    ICs use broadband temporal-gated removal. All ICs are weighted by their ICLabel
    probability. Falls back to broadband removal if F(k) Gini < spectral_quality_threshold.
    """
    import mne

    sfreq = raw.info["sfreq"]
    data  = raw.get_data()
    n_ch, n_times = data.shape

    if ica is None:
        ica = mne.preprocessing.ICA(
            n_components=20, method="infomax",
            random_state=42, max_iter="auto", verbose=False,
        )
        ica.fit(raw, verbose=False)

    if labels is None or probs_matrix is None or np.array(probs_matrix).ndim == 1:
        from mne_icalabel import label_components
        result = label_components(raw, ica, method="iclabel")
        labels = list(result["labels"])
        probs_matrix = np.array(ica.labels_scores_, dtype=float)
    else:
        probs_matrix = np.array(probs_matrix, dtype=float)

    n_ics       = probs_matrix.shape[0]
    p_brain_vec = probs_matrix[:, 0] + probs_matrix[:, 6]
    artifact_ics = [i for i in range(n_ics) if p_brain_vec[i] < brain_threshold]
    brain_ics    = [i for i in range(n_ics) if p_brain_vec[i] >= brain_threshold]

    sources   = ica.get_sources(raw).get_data()
    n_fft     = int(2 ** np.ceil(np.log2(n_times)))
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)

    total_removal = np.zeros_like(data)
    ic_stats      = []

    for ic_idx in artifact_ics:
        p_vec   = probs_matrix[ic_idx]
        source  = sources[ic_idx]

        type_probs = {
            "eye":     float(p_vec[2]),
            "muscle":  float(p_vec[1]),
            "heart":   float(p_vec[3]),
            "line":    float(p_vec[4]),
            "channel": float(p_vec[5]),
        }
        dom_type = max(type_probs, key=type_probs.get)
        dom_prob = type_probs[dom_type]

        if dom_prob < 0.20:
            ic_stats.append({"ic": ic_idx, "strategy": "skip", "dom_prob": dom_prob})
            continue

        ica_copy = ica.copy()
        ica_copy.exclude = [ic_idx]
        raw_copy = raw.copy()
        ica_copy.apply(raw_copy, verbose=False)
        ic_contribution = data - raw_copy.get_data()

        if dom_type == "line":
            removal = _spectral_notch_removal(
                ic_contribution, sfreq, n_fft, fft_freqs, line_freq
            ) * dom_prob
            total_removal += removal
            ic_stats.append({"ic": ic_idx, "type": "line", "strategy": "notch"})
            continue

        if dom_type == "muscle" and dom_prob >= muscle_full_removal_thresh:
            total_removal += ic_contribution * dom_prob
            ic_stats.append({"ic": ic_idx, "type": "muscle",
                             "strategy": "muscle_full", "dom_prob": dom_prob})
            continue

        if dom_type == "eye":
            temporal_mask, diag = _detect_blink_mask(source, sfreq)
        elif dom_type == "muscle":
            temporal_mask, diag = _detect_muscle_mask(source, sfreq)
        elif dom_type == "heart":
            temporal_mask, diag = _detect_heart_mask(source, sfreq)
        else:
            temporal_mask, diag = _detect_channel_mask(source, sfreq)

        mask_frac = float(temporal_mask.mean())

        if mask_frac > max_mask_fraction:
            total_removal += ic_contribution * dom_prob
            ic_stats.append({"ic": ic_idx, "type": dom_type,
                             "strategy": "fallback_full", "mask_frac": mask_frac})
            continue

        temporal_weight = dom_prob * temporal_mask
        spectral_types = {"eye", "heart"}

        F_profile, f_bins, quality = _estimate_artifact_spectral_profile(
            source, temporal_mask, sfreq
        )

        if dom_type in spectral_types and quality >= spectral_quality_threshold:
            removal  = _frequency_selective_removal(
                source, ic_contribution, F_profile, f_bins, temporal_weight, sfreq
            )
            strategy = "spectra"
        else:
            removal  = ic_contribution * temporal_weight[np.newaxis, :]
            strategy = "broadband"

        total_removal += removal
        ic_stats.append({
            "ic":               ic_idx,
            "label":            labels[ic_idx],
            "type":             dom_type,
            "dom_prob":         dom_prob,
            "mask_frac":        mask_frac,
            "spectral_quality": quality,
            "strategy":         strategy,
            "n_events":         diag.get("n_events", 0),
        })

    data_clean = data - total_removal
    raw_clean  = raw.copy()
    raw_clean._data = data_clean

    n_spectra   = sum(1 for s in ic_stats if s.get("strategy") == "spectra")
    n_broadband = sum(1 for s in ic_stats if s.get("strategy") == "broadband")
    n_full      = sum(1 for s in ic_stats if "full" in s.get("strategy", ""))

    info = {
        "n_brain_ics":           len(brain_ics),
        "n_artifact_ics":        len(artifact_ics),
        "n_spectra":             n_spectra,
        "n_broadband_fallback":  n_broadband,
        "n_full_removal":        n_full,
        "ic_stats":              ic_stats,
    }
    return raw_clean, info
