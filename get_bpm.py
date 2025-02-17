#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
# Updated BPM Detector Python Script
# Corrects issues related to doubled or halved BPM.
# ------------------------------------------------------------------------------
import argparse
import array
import math
import wave

import matplotlib.pyplot as plt
import numpy
import pywt
from scipy import signal


def read_wav(filename):
    """
    Reads a 16-bit mono WAV file and returns:
      - samps: list of audio samples
      - fs: sampling frequency
    """
    try:
        wf = wave.open(filename, "rb")
    except IOError as e:
        print("Error opening WAV file:", e)
        return None, None

    # Print some debug info about the file
    num_channels = wf.getnchannels()
    sample_width = wf.getsampwidth()
    fs = wf.getframerate()
    nsamps = wf.getnframes()

    print("File:", filename)
    print("Channels:", num_channels)
    print("Sample width:", sample_width, "bytes")
    print("Frame rate (fs):", fs, "Hz")
    print("Number of frames (nsamps):", nsamps)

    if num_channels != 1:
        print("Warning: This script expects a mono WAV file.")
        print("If your file is stereo or multi-channel, please convert to mono first.")

    if sample_width == 2:
        # 16-bit PCM data => 'h' type (signed short)
        samp_type = "h"
    else:
        # Fall back to 'i' if 32-bit or something else
        samp_type = "i"
        print("Warning: Detected sample width != 16-bit; using 'i' (32-bit).")

    # Read the raw frames into an array
    raw_frames = wf.readframes(nsamps)
    wf.close()

    if not raw_frames:
        print("No frames read from WAV file.")
        return None, None

    # Convert to array
    samps_array = array.array(samp_type, raw_frames)

    # If stereo, you'd need to downmix. For now, we assume mono.
    samps = list(samps_array)

    # Basic checks
    if nsamps != len(samps):
        print(f"Warning: {nsamps} frames reported, but {len(samps)} samples read.")
    if fs <= 0:
        print("Invalid sampling frequency:", fs)
        return None, None

    return samps, fs


def no_audio_data():
    """
    Print a message indicating no valid audio data was found
    and return (None, None) for BPM and correlation array.
    """
    print("No audio data for sample, skipping...")
    return None, None


def peak_detect(data):
    """
    Simple peak detection that returns the index of the maximum absolute value.
    """
    max_val = numpy.amax(abs(data))
    peak_ndx = numpy.where(data == max_val)
    if len(peak_ndx[0]) == 0:  # if nothing found, look for -max_val
        peak_ndx = numpy.where(data == -max_val)
    return peak_ndx


def bpm_detector(data, fs):
    """
    Main BPM detection logic using Discrete Wavelet Transform (pywt)
    and Auto-Correlation Function (ACF).
    
    Returns:
      - bpm: estimated beats per minute (float) or None
      - correl: the auto-correlation array (numpy array) or None
    """
    # If there's no data, abort
    if not data:
        return no_audio_data()

    # Decompose levels
    levels = 4
    max_decimation = 2 ** (levels - 1)

    # BPM range: 60-180 (adjustable)
    min_ndx = math.floor(60.0 / 180 * (fs / max_decimation))  # Adjusted max BPM = 180
    max_ndx = math.floor(60.0 / 60 * (fs / max_decimation))   # Adjusted min BPM = 60

    # Initialize wavelet decomposition
    cA = []
    cD_sum = numpy.zeros(1)  # just a placeholder; will be replaced by actual length

    # Perform multi-level wavelet decomposition
    for loop in range(levels):
        if loop == 0:
            [cA, cD] = pywt.dwt(data, "db4")
            cD_minlen = len(cD) / max_decimation + 1
            cD_sum = numpy.zeros(math.floor(cD_minlen))
        else:
            [cA, cD] = pywt.dwt(cA, "db4")
        cD = signal.lfilter([0.01], [1 - 0.99], cD)
        cD = abs(cD[:: (2 ** (levels - loop - 1))])
        cD = cD - numpy.mean(cD)
        cD_sum = cD[0 : math.floor(cD_minlen)] + cD_sum

    # If approximation coefficients are all zero, skip
    if all(v == 0.0 for v in cA):
        return no_audio_data()

    # incorporate approximate coefficients
    cA = signal.lfilter([0.01], [1 - 0.99], cA)
    cA = abs(cA)
    cA = cA - numpy.mean(cA)
    cD_sum = cA[0 : math.floor(cD_minlen)] + cD_sum

    # auto-correlation
    correl = numpy.correlate(cD_sum, cD_sum, "full")
    midpoint = math.floor(len(correl) / 2)
    correl_midpoint_tmp = correl[midpoint:]

    # look for a peak in the range of min_ndx to max_ndx
    if min_ndx < 0 or max_ndx > len(correl_midpoint_tmp):
        # Edge case: if window is too small or something else
        return no_audio_data()

    peak_range = correl_midpoint_tmp[min_ndx:max_ndx]
    peak_ndx = peak_detect(peak_range)
    if len(peak_ndx) == 0 or len(peak_ndx[0]) == 0:
        return no_audio_data()

    # if there's more than one peak, just use the first
    peak_ndx_adjusted = peak_ndx[0][0] + min_ndx
    bpm = 60.0 / peak_ndx_adjusted * (fs / max_decimation)

    # Correct doubling/halving issues
    if bpm > 150:
        bpm /= 2
    elif bpm < 50:
        bpm *= 2

    print(f"Detected BPM: {bpm}")
    return bpm, correl


def main():
    parser = argparse.ArgumentParser(
        description="Process a 16-bit mono .wav file to determine the Beats Per Minute."
    )
    parser.add_argument("--filename", required=True, help=".wav file for processing")
    parser.add_argument(
        "--window",
        type=float,
        default=3,
        help="Size of the window (in seconds) for scanning to determine the bpm. Default = 3.",
    )

    args = parser.parse_args()
    samps, fs = read_wav(args.filename)
    if samps is None or fs is None:
        print("Failed to read WAV file. Exiting.")
        return

    # Set up window size (in samples)
    window_samps = int(args.window * fs)
    nsamps = len(samps)
    if window_samps <= 0 or window_samps > nsamps:
        print("Invalid window size. Exiting.")
        return

    max_window_ndx = math.floor(nsamps / window_samps)
    bpms = numpy.zeros(max_window_ndx)
    correl = None

    # Process each window
    samps_ndx = 0
    for window_ndx in range(max_window_ndx):
        # Extract a slice of samples
        data = samps[samps_ndx : samps_ndx + window_samps]
        if len(data) != window_samps:
            print("Skipping last chunk (not enough samples).")
            continue

        bpm, correl_temp = bpm_detector(data, fs)
        if bpm is not None:
            bpms[window_ndx] = bpm
            correl = correl_temp  # keep the last valid correlation for plotting

        samps_ndx += window_samps

    # Median BPM across all windows
    final_bpm = numpy.median(bpms) if len(bpms) > 0 else 0.0
    print("Completed!  Estimated Beats Per Minute:", final_bpm)

    # Plot the correlation if valid
    if correl is not None and isinstance(correl, numpy.ndarray):
        n = range(len(correl))
        plt.plot(n, abs(correl), label="Auto-correlation")
        plt.title("Auto-correlation of last processed window")
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        plt.legend()
        plt.show()
    else:
        print("No valid correlation data to plot.")


if __name__ == "__main__":
    main()
