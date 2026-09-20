

import pywt
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.signal import welch

"""
=========================================================
Wavelet Preprocessing
---------------------------------------------------------
Paper:
Noninvasive Blood Glucose Monitoring Using
Spatiotemporal ECG and PPG Feature Fusion
=========================================================
"""
class WaveletPreprocessor:
    """
    Perform DWT decomposition and reconstruction.

    Parameters
    ----------
    wavelet : str
        Mother wavelet.
        Default = 'db4'

    level : int
        Number of decomposition levels.
        Default = 7
    """
    def __init__(self, wavelet: str = "db4", level: int = 7):
        self.wavelet = wavelet
        self.level = level

    # ----------------------------------------------------
    # DWT decomposition
    # ----------------------------------------------------
    def decompose(self, signal: np.ndarray):
        coeffs = pywt.wavedec(signal, wavelet=self.wavelet, level=self.level)
        return coeffs

    # ----------------------------------------------------
    # Reconstruct one detail level
    # ----------------------------------------------------
    def reconstruct_detail(self, coeffs, signal_length: int, detail_level: int):
        # coeffs = self.decompose(signal)
        new_coeffs = []
        new_coeffs.append(np.zeros_like(coeffs[0]))

        for i in range(1, len(coeffs)):

            current_level = len(coeffs) - i

            if current_level == detail_level:
                new_coeffs.append(coeffs[i])

            else:
                new_coeffs.append(np.zeros_like(coeffs[i]))

        reconstructed = pywt.waverec(new_coeffs, self.wavelet)

        return reconstructed[:signal_length] 
    

    # ----------------------------------------------------
    # Build signal database
    # ----------------------------------------------------
    def build_signal_database(self, signal: np.ndarray):

        signal_database = {}
        signal_database["original"] = signal
        coeffs = self.decompose(signal)

        for level in range(1, self.level + 1):
            signal_database[f"D{level}"] = self.reconstruct_detail(coeffs, len(signal), detail_level=level)

        return signal_database
    
    # ----------------------------------------------------
    # Extract statistical features from one signal
    # ----------------------------------------------------
    def extract_features(self, signal: np.ndarray) -> dict:

        features = {}

        features["Mobility"]   = signal_mobility(signal)
        features["Complexity"] = signal_complexity(signal)
        features["Kurtosis"]   = signal_kurtosis(signal)
        features["Skewness"]   = signal_skewness(signal)
        features["SE"]         = shannon_entropy(signal)
        features["PSE"]        = power_spectral_entropy(signal)
        features["C0"]         = c0_complexity(signal)
        features["FD"]         = higuchi_fd(signal)

        return features
    
    # ----------------------------------------------------
    # Extract features from all wavelet signals
    # ----------------------------------------------------
    def extract_wavelet_features(self, signal: np.ndarray) -> dict:

        signal_database = self.build_signal_database(signal)

        feature_database = {}

        for signal_name, wavelet_signal in signal_database.items():

            feature_database[signal_name] = self.extract_features(wavelet_signal)

        return feature_database
    
    
"""
=========================================================
Statistical Features
=========================================================
"""

def signal_mobility(signal: np.ndarray, step: int = 15) -> float:
    """
    Hjorth Mobility

    Parameters
    ----------
    signal : ndarray
    step : int

    Returns
    -------
    mobility : float
    """

    signal = np.asarray(signal)
    d = signal[step:] - signal[:-step]
    S0 = np.sqrt(np.mean(signal ** 2))
    S1 = np.sqrt(np.mean(d ** 2))

    return S1 / S0

def signal_complexity(signal: np.ndarray, step: int = 15) -> float:
    """
    Hjorth Complexity
    """

    signal = np.asarray(signal)

    d = signal[step:] - signal[:-step]
    g = d[step:] - d[:-step]

    S0 = np.sqrt(np.mean(signal ** 2))
    S1 = np.sqrt(np.mean(d ** 2))
    S2 = np.sqrt(np.mean(g ** 2))

    return np.sqrt((S2 ** 2) / (S1 ** 2) - (S1 ** 2) / (S0 ** 2))

def signal_skewness(signal: np.ndarray) -> float:
    return skew(signal,bias=False)

def signal_kurtosis(signal: np.ndarray) -> float:
    return kurtosis(signal, fisher=False, bias=False)

def shannon_entropy(signal: np.ndarray) -> float:
    """
    Shannon Entropy
    (Implementation exactly follows the paper)
    """

    signal = np.asarray(signal)

    se_total = 0.0

    for nbins in range(2, 201):


        hist, _ = np.histogram(signal, bins=nbins)
        hist = hist.astype(float)

        hist /= hist.sum()

        entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
        se_total += entropy

    return se_total / 199

def power_spectral_entropy(signal: np.ndarray, fs: int = 250) -> float:
    """
    Standard Power Spectral Entropy
    """

    signal = np.asarray(signal)

    psd_freq, psd = welch(signal, fs=fs)

    psd = psd / np.sum(psd)

    pse = -np.sum(psd * np.log2(psd + np.finfo(float).eps))

    return pse

def c0_complexity(signal: np.ndarray) -> float:
    """
    C0 Complexity

    Parameters
    ----------
    signal : ndarray

    Returns
    -------
    float
        C0 complexity
    """

    signal = np.asarray(signal)

    # ---------------------------------------------
    # FFT
    # ---------------------------------------------
    spectrum = np.fft.fft(signal)

    # ---------------------------------------------
    # Power Spectrum
    # ---------------------------------------------
    power = np.abs(spectrum) ** 2

    # ---------------------------------------------
    # Mean spectral power
    # ---------------------------------------------
    threshold = np.mean(power)

    # ---------------------------------------------
    # Keep only dominant frequencies
    # ---------------------------------------------
    filtered_spectrum = spectrum.copy()

    filtered_spectrum[power < threshold] = 0

    # ---------------------------------------------
    # Signal reconstruction
    # ---------------------------------------------
    reconstructed = np.fft.ifft(filtered_spectrum).real

    # ---------------------------------------------
    # Error Energy
    # ---------------------------------------------
    error_energy = np.sum((signal - reconstructed) ** 2)

    # ---------------------------------------------
    # Signal Energy
    # ---------------------------------------------
    signal_energy = np.sum(signal ** 2)

    # ---------------------------------------------
    # C0 Complexity
    # ---------------------------------------------
    c0 = error_energy / signal_energy

    return c0

# ==========================================================
# Higuchi Fractal Dimension
# ==========================================================
def higuchi_fd(signal: np.ndarray, kmax: int = 20) -> float:
    """
    Higuchi Fractal Dimension (HFD)

    Parameters
    ----------
    signal : ndarray
        1D signal

    kmax : int
        Maximum scale

    Returns
    -------
    float
        Higuchi Fractal Dimension
    """

    signal = np.asarray(signal, dtype=np.float64)

    N = len(signal)

    Lk = np.zeros(kmax)

    # -----------------------------------------------------
    # Loop over scales
    # -----------------------------------------------------
    for k in range(1, kmax + 1):

        Lm = np.zeros(k)

        # ---------------------------------------------
        # Different starting points
        # ---------------------------------------------
        for m in range(k):

            # Build subsequence
            idx = np.arange(m, N, k)

            subseq = signal[idx]

            if len(subseq) < 2:
                continue

            # -----------------------------------------
            # Curve length
            # -----------------------------------------
            length = np.sum(np.abs(np.diff(subseq)))

            # -----------------------------------------
            # Normalization
            # -----------------------------------------
            norm = (N - 1) / ((len(subseq) - 1) * k)
            Lm[m] = (length * norm) / k

        # Mean curve length for this scale
        Lk[k - 1] = np.mean(Lm)

    # -----------------------------------------------------
    # Linear fitting in log-log space
    # -----------------------------------------------------
    scales = np.arange(1, kmax + 1)
    coeff = np.polyfit(np.log(1 / scales), np.log(Lk), 1)
    fd = coeff[0]

    return fd

"""
=========================================================
Build Feature Vector
=========================================================
"""
FEATURE_ORDER = ["Mobility", "Complexity", "Kurtosis", "Skewness", "SE", "PSE", "C0", "FD",]
SIGNAL_ORDER = ["original", "D1", "D2", "D3", "D4", "D5", "D6", "D7",]

def build_feature_vector(feature_database: dict) -> np.ndarray:
    """
    Convert feature dictionary into one ordered feature vector.

    Parameters
    ----------
    feature_database : dict
        Output of
            WaveletPreprocessor.extract_wavelet_features()

    Returns
    -------
    ndarray
        Shape:
            (64,)
    """

    feature_vector = []

    for feature_name in FEATURE_ORDER:
        for signal_name in SIGNAL_ORDER:
            feature_vector.append(feature_database[signal_name][feature_name])

    return np.asarray(feature_vector, dtype=np.float32)

