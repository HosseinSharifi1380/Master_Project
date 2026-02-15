import numpy as np
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
