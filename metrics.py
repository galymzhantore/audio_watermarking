import numpy as np
from scipy.signal import resample  # Add this import

def calculate_snr(original, watermarked):
    original = np.asarray(original, dtype=np.float64)
    watermarked = np.asarray(watermarked, dtype=np.float64)
    
    if len(original) != len(watermarked):
        watermarked = resample(watermarked, len(original))
    
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - watermarked) ** 2)
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power) - 10 * np.log10(noise_power)


def calculate_psnr(original, watermarked):
    original = np.asarray(original, dtype=np.float64)
    watermarked = np.asarray(watermarked, dtype=np.float64)
    
    if len(original) != len(watermarked):
        watermarked = resample(watermarked, len(original))
    
    max_val = np.max(np.abs(original))
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return np.inf
    return 20 * np.log10(max_val) - 10 * np.log10(mse)

def calculate_nc(original_wm, extracted_wm):
    original_wm = np.asarray(original_wm, dtype=np.float64).flatten()
    extracted_wm = np.asarray(extracted_wm, dtype=np.float64).flatten()
    numerator = np.sum(original_wm * extracted_wm)
    denominator = np.sqrt(np.sum(original_wm ** 2) * np.sum(extracted_wm ** 2))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_ber(original_wm, extracted_wm):
    original_wm = np.asarray(original_wm, dtype=np.uint8).flatten()
    extracted_wm = np.asarray(extracted_wm, dtype=np.uint8).flatten()
    total_bits = len(original_wm)
    error_bits = np.sum(original_wm != extracted_wm)
    return (error_bits / total_bits) * 100
