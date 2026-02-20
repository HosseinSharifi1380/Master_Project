import numpy as np
import pandas as pd
import zipfile
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import butter, filtfilt, iirnotch, welch

def bandpass(ecg, fs, low=0.5, high=10, order=4):
    """
    Simple bandpass filter for ECG or PPG signals.
    
    Parameters
    ----------
    ecg : array-like
        Input signal.
    fs : float
        Sampling frequency (Hz).
    low : float
        Low cutoff frequency (Hz).
    high : float
        High cutoff frequency (Hz).
    order : int
        Filter order.

    Returns
    -------
    filtered_signal : np.ndarray
        The bandpass filtered signal.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, ecg)

## bad segment detection using PSD

def bandpower_from_psd(f, Pxx, band):
    """
    Compute bandpower by integrating PSD in [band[0], band[1]].

    Parameters
    ----------
    f : array-like
        Frequencies corresponding to Pxx (Hz)
    Pxx : array-like
        Power spectral density
    band : tuple
        (low_freq, high_freq) band to integrate

    Returns
    -------
    float
        Integrated power in the specified band
    """
    lo, hi = band
    m = (f >= lo) & (f <= hi)
    if not np.any(m):
        return 0.0
    return float(np.trapezoid(Pxx[m], f[m]))

def ecg_qc_filter(x, fs, hp=0.5, notch_hz=50.0, notch_q=30.0):
    """
    Step-1 QC filter for ECG (for PSD-based noise/motion detection):
    - High-pass at ~0.5 Hz to reduce baseline drift
    - Notch at 50/60 Hz to suppress powerline interference

    Parameters
    ----------
    x : array-like
        Raw ECG signal
    fs : float
        Sampling frequency (Hz)
    hp : float
        High-pass cutoff (Hz)
    notch_hz : float
        Notch frequency (Hz) to remove powerline
    notch_q : float
        Quality factor for notch filter

    Returns
    -------
    y : np.ndarray
        Filtered ECG signal
    """
    x = np.asarray(x, dtype=float)

    # 1) High-pass Butterworth
    nyq = 0.5 * fs
    hp_n = hp / nyq
    b_hp, a_hp = butter(2, hp_n, btype="highpass")
    y = filtfilt(b_hp, a_hp, x)

    # 2) Notch filter (powerline)
    if notch_hz is not None and notch_hz > 0:
        w0 = notch_hz / nyq
        if 0 < w0 < 1:
            b_notch, a_notch = iirnotch(w0, notch_q)
            y = filtfilt(b_notch, a_notch, y)

    return y

def compute_ecg_bandpowers(
    ecg_signal,
    fs=250,
    in_band=(15, 40),
    out_band=(45, 100),
    hp=0.5,
    notch_hz=50.0,
    notch_q=30.0,
    nperseg_sec=2.0,
    noverlap_ratio=0.8,
    apply_qc_filter=True,
):
    """
    Apply (optional) QC filter to ECG and compute PSD and bandpowers.

    Parameters
    ----------
    ecg_signal : array-like
        Raw ECG samples
    fs : float
        Sampling frequency
    in_band : tuple
        Low-frequency band for P_in
    out_band : tuple
        High-frequency band for P_out
    hp : float
        High-pass frequency for QC filter (used if apply_qc_filter=True)
    notch_hz : float
        Notch frequency for powerline (used if apply_qc_filter=True)
    notch_q : float
        Notch Q factor (used if apply_qc_filter=True)
    nperseg_sec : float
        Length of segment (sec) for Welch
    noverlap_ratio : float
        Fraction of segment overlap
    apply_qc_filter : bool
        If True, apply ecg_qc_filter before PSD. If False, use raw signal.

    Returns
    -------
    dict
        'P_in', 'P_out', 'ratio', 'f', 'Pxx', 'ecg_used'
    """
    x = np.asarray(ecg_signal, dtype=float)

    # 1) Optional QC filter
    if apply_qc_filter:
        ecg_used = ecg_qc_filter(x, fs=fs, hp=hp, notch_hz=notch_hz, notch_q=notch_q)
    else:
        ecg_used = x

    # 2) Welch PSD
    nperseg = int(round(nperseg_sec * fs))
    nperseg = min(nperseg, len(ecg_used))
    noverlap = int(noverlap_ratio * nperseg)

    f, Pxx = welch(ecg_used, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")

    # 3) Bandpower
    P_in = bandpower_from_psd(f, Pxx, in_band)
    P_out = bandpower_from_psd(f, Pxx, out_band)
    ratio = P_out / (P_in + 1e-12)

    return {
        "P_in": P_in,
        "P_out": P_out,
        "ratio": ratio,
        "f": f,
        "Pxx": Pxx,
        "ecg_used": ecg_used,
        "ecg_raw": ecg_signal,
    }


def _runs_of_true(mask: np.ndarray):
    """Return list of (start, end) index pairs for contiguous True regions. end is exclusive."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    return list(zip(starts.tolist(), ends.tolist()))

def detect_bad_segments_hf_energy(
    ecg_raw: np.ndarray,
    fs: float,
    out_band: tuple[float, float] = (45.0, 100.0),

    # PSD windowing
    win_sec: float = 6.0,
    step_sec: float = 1.0,
    nperseg_sec: float = 2.0,
    noverlap_ratio: float = 0.5,
    pad_sec: float = 0.2,

    # Thresholding (PSD)
    thr_method: str = "robust",   # "robust" or "fixed"
    thr_fixed: float | None = None,
    thr_alpha: float = 6.0,

    # Optional QC filtering
    apply_qc_filter: bool = True,
    hp: float = 0.5,
    notch_hz: float | None = 50.0,
    notch_q: float = 30.0,

    # Glitch (sample-level outliers, local median/MAD)
    k_amp: float = 15.0,
    glitch_win_sec: float = 10.0,
    glitch_step_sec: float = 1.0,
    pad_glitch_sec: float = 0.05,
    min_glitch_len_sec: float = 0.02,

    # Optional: keep PSD/Glitch separate in info
    keep_sources: bool = True,
):
    """
    Detect bad ECG segments using:
      1) High-frequency bandpower (Welch PSD) on sliding windows (PSD branch)
      2) Glitch detection via local median/MAD outliers (Glitch branch)

    Returns
    -------
    bad_mask : np.ndarray (bool)
        True where samples are considered bad.
    bad_segments : list[tuple[int,int]]
        Contiguous True runs as (start, end) with end exclusive.
    info : dict
        Diagnostics (threshold, Pout, etc.). If keep_sources=True also includes
        PSD-only and Glitch-only masks/segments.
    """
    x = np.asarray(ecg_raw, dtype=float)
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=bool), [], {}

    # 0) Optional QC filter for stability
    x_qc = (ecg_qc_filter(x, fs=fs, hp=hp, notch_hz=notch_hz, notch_q=notch_q) if apply_qc_filter else x)

    # ---------------- PSD branch ----------------
    nyq = 0.5 * fs
    out_lo, out_hi = out_band
    out_hi = min(out_hi, nyq * 0.999)
    if not (0 < out_lo < out_hi < nyq):
        raise ValueError(f"out_band invalid for fs={fs}: out_band={(out_lo, out_hi)}, nyq={nyq}")

    win = int(round(win_sec * fs))
    step = int(round(step_sec * fs))
    pad = int(round(pad_sec * fs))
    if win < 8 or step < 1:
        raise ValueError("win_sec/step_sec too small for given fs.")

    starts = np.arange(0, n - win + 1, step, dtype=int)
    centers = starts + win // 2

    Pouts = np.zeros_like(starts, dtype=float)

    # IMPORTANT: seg is already from x_qc => disable QC inside compute_ecg_bandpowers
    for i, s in enumerate(starts):
        seg = x_qc[s:s + win]

        res = compute_ecg_bandpowers(
            seg,
            fs=fs,
            in_band=(15, 40),           # not used for P_out (kept for compatibility)
            out_band=(out_lo, out_hi),
            nperseg_sec=nperseg_sec,
            noverlap_ratio=noverlap_ratio,
            apply_qc_filter=False,
        )
        Pouts[i] = res["P_out"]

    # Threshold on PSD bandpower
    if thr_method == "fixed":
        if thr_fixed is None:
            raise ValueError("thr_fixed must be provided when thr_method='fixed'")
        thr = float(thr_fixed)
    elif thr_method == "robust":
        medP = np.median(Pouts)
        madP = np.median(np.abs(Pouts - medP))
        scaleP = 1.4826 * madP
        if scaleP < 1e-12:
            scaleP = np.std(Pouts) + 1e-12
        thr = float(medP + thr_alpha * scaleP)
    else:
        raise ValueError("thr_method must be 'robust' or 'fixed'")

    bad_win = Pouts > thr

    bad_mask_psd = np.zeros(n, dtype=bool)
    for s, is_bad in zip(starts, bad_win):
        if is_bad:
            s2 = max(0, s - pad)
            e2 = min(n, s + win + pad)
            bad_mask_psd[s2:e2] = True

    # ---------------- Glitch branch ----------------
    bad_mask_glitch = np.zeros(n, dtype=bool)

    gwin = int(round(glitch_win_sec * fs))
    gstep = int(round(glitch_step_sec * fs))
    if gwin < 8 or gstep < 1:
        raise ValueError("glitch_win_sec/glitch_step_sec too small for given fs.")

    gstarts = np.arange(0, n - gwin + 1, gstep, dtype=int)

    for s in gstarts:
        seg_amp = x_qc[s:s + gwin]
        med = np.median(seg_amp)
        mad = np.median(np.abs(seg_amp - med))
        scale = 1.4826 * mad
        if scale < 1e-12:
            scale = np.std(seg_amp) + 1e-12

        thr_amp = k_amp * scale
        bad_mask_glitch[s:s + gwin] |= (np.abs(seg_amp - med) > thr_amp)

    # Remove too-short glitch runs
    min_glitch_len = max(1, int(round(min_glitch_len_sec * fs)))
    if min_glitch_len > 1 and bad_mask_glitch.any():
        filtered = np.zeros(n, dtype=bool)
        for rs, re in _runs_of_true(bad_mask_glitch):
            if (re - rs) >= min_glitch_len:
                filtered[rs:re] = True
        bad_mask_glitch = filtered

    # Padding around glitch runs
    pad_glitch = int(round(pad_glitch_sec * fs))
    if pad_glitch > 0 and bad_mask_glitch.any():
        padded = bad_mask_glitch.copy()
        for rs, re in _runs_of_true(bad_mask_glitch):
            s2 = max(0, rs - pad_glitch)
            e2 = min(n, re + pad_glitch)
            padded[s2:e2] = True
        bad_mask_glitch = padded

    # ---------------- Final combine ----------------
    bad_mask = bad_mask_psd | bad_mask_glitch
    bad_segments = _runs_of_true(bad_mask)

    info = {
        "Pout": Pouts,
        "threshold": thr,
    }

    if keep_sources:
        info["bad_mask_psd"] = bad_mask_psd
        info["bad_mask_glitch"] = bad_mask_glitch
        info["bad_segments_psd"] = _runs_of_true(bad_mask_psd)
        info["bad_segments_glitch"] = _runs_of_true(bad_mask_glitch)

        # optional: intersection / disagreements between detectors
        info["bad_mask_psd_only"] = bad_mask_psd & ~bad_mask_glitch
        info["bad_mask_glitch_only"] = bad_mask_glitch & ~bad_mask_psd
        info["bad_mask_both"] = bad_mask_psd & bad_mask_glitch
        if apply_qc_filter:
            info["ecg_used"] = x_qc

    return bad_mask, bad_segments, info


def load_pkl_from_zip(z, file_name):
    # Read a single .pkl member from an already-open ZipFile object
    with z.open(file_name) as f:
        return pickle.load(f)
    
def plot_ecg_bad_segments_from_zip(
    zip_path: str,
    n_files: int = 7,
    start_idx: int = 0,
    fs: float = 250,
    detector_kwargs: dict | None = None,
    keep_sources: bool = False,
    show: bool = True,
):
    """
    Plot ECG signals from a zip of pickle files and overlay detected bad regions.

    detector_kwargs: dict of keyword args passed to detect_bad_segments_hf_energy
    keep_sources: if True, keep PSD/Glitch masks in info (heavier)
    show: if True, calls plt.show(). Otherwise returns fig/axes for further editing.
    """

    detector_kwargs = detector_kwargs or {}

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        selected = names[start_idx : start_idx + n_files]

        fig, axes = plt.subplots(len(selected), 1, figsize=(14, 2.2 * len(selected)), sharex=False)
        if len(selected) == 1:
            axes = [axes]

        for ax, name in zip(axes, selected):
            data = load_pkl_from_zip(z, name)

            ecg_raw = np.asarray(data["zephyr"]["ECG"]["EcgWaveform"], dtype=float)
            ecg_time = pd.to_datetime(data["zephyr"]["ECG"]["Time"])

            bad_mask, _, info = detect_bad_segments_hf_energy(
                ecg_raw,
                fs=fs,
                keep_sources=keep_sources,
                **detector_kwargs,
            )
            bad_mask_psd = info['bad_mask_psd']
            bad_mask_glitch = info['bad_mask_glitch']
            ecg_filtered = info["ecg_used"]

            # ecg_bad = np.full_like(ecg_raw, np.nan, dtype=float)
            # ecg_bad_PSD = np.full_like(ecg_raw, np.nan, dtype=float)
            # ecg_bad_glitch = np.full_like(ecg_raw, np.nan, dtype=float)
            ecg_bad = np.full_like(ecg_filtered, np.nan, dtype=float)
            ecg_bad_PSD = np.full_like(ecg_filtered, np.nan, dtype=float)
            ecg_bad_glitch = np.full_like(ecg_filtered, np.nan, dtype=float)

            # ecg_bad[bad_mask] = ecg_raw[bad_mask]
            # ecg_bad_PSD[bad_mask_psd] = ecg_raw[bad_mask_psd]
            # ecg_bad_glitch[bad_mask_glitch] = ecg_raw[bad_mask_glitch]
            ecg_bad[bad_mask] = ecg_filtered[bad_mask]
            ecg_bad_PSD[bad_mask_psd] = ecg_filtered[bad_mask_psd]
            ecg_bad_glitch[bad_mask_glitch] = ecg_filtered[bad_mask_glitch]

            # ax.plot(ecg_time, ecg_raw, lw=0.8, label="Raw ECG")
            ax.plot(ecg_time, ecg_filtered, lw=0.8, label="Raw ECG")
            ax.plot(ecg_time, ecg_bad_PSD, 'r.', lw=0.8, label="Bad regions(PSD)")
            ax.plot(ecg_time, ecg_bad_glitch, 'g.',lw=0.8, label="Bad regions(glitch)")

            
            # ax.plot(ecg_time, ecg_bad, lw=1.2, label="Bad regions")
            ax.legend(loc="upper right")

            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.set_title(name, fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes
