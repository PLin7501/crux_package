import warnings
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


# gets a single channel (e.g. "EXG Channel 0") from a .txt file and returns as np array
def get_channel_from_txt(file, channel):
    df = pd.read_csv(file, sep=', ', comment="%", engine='python')
    values = np.array(df.loc[:, channel])
    return values


# takes raw data (2d np array) and prints whether there are missing data
def check_missing_samples(data):
    count = 1
    for val1, val2 in zip(data[0], data[0, 1:]):
        if val2 == val1 + 1 or (val1, val2) == (255, 0):
            count += 1
    print(f"{count} samples found with {data.shape[1]} samples expected: {data.shape[1] - count} missing samples")


# checks if any samples were not recieved from the hardware side and fills missing samples
# by linearly interpolating between the two neighboring samples
def interpolate_missing_samples(sample_indices, voltages, cycle_size=256):
    sample_indices = sample_indices.astype(int)

    if sample_indices.shape != voltages.shape:
        raise ValueError("sample_indices and voltages must have same shape")

    sample_indices = np.asarray(sample_indices)
    voltages = np.asarray(voltages)

    output_voltages = []

    last_idx = None
    last_v = None

    for idx, v in zip(sample_indices, voltages):

        if last_idx is None:
            output_voltages.append(v)
            last_idx = idx
            last_v = v
            continue

        # Detect wraparound (new cycle)
        if idx <= last_idx:
            # Fill remaining indices in previous cycle with last value
            for _ in range(last_idx + 1, cycle_size):
                output_voltages.append(last_v)

            # Fill missing at start of new cycle with current value
            for _ in range(0, idx):
                output_voltages.append(v)

            output_voltages.append(v)

        else:
            gap = idx - last_idx

            if gap > 1:
                # Linear interpolation for missing internal samples
                steps = np.linspace(last_v, v, gap + 1)[1:-1]
                output_voltages.extend(steps)

            output_voltages.append(v)

        last_idx = idx
        last_v = v

    return np.array(output_voltages)


# DEPRECATED. Use mne.filter.filter_data instead
# applies a bandpass filter to data
# fs is the sampling rate
# lowcut and highcut are the minimum and maximum frequencies
def bandpass(data, fs, lowcut, highcut, order=4):
    warnings.warn("processing.bandpass is deprecated. Use mne.filter.filter_data instead.")
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data


# gets average signal of data across all channels
def get_avg(data, channels):
    return data[np.array(list(channels.keys()))].mean(axis=0)


# rereferences data to average and returns a new array
def reference_to_avg(data, channels):
    return data - get_avg(data, channels)


# indexes signal at the indices in ref_idx and returns a matrix of subarrays
# each subarray has "length" values and starts at ref_idx
def get_subarrays(signal, ref_idx, length):
    out = []
    for idx in ref_idx:
        out.append(signal[idx:idx + length])
    return np.stack(out)


# gets subarrays, but also adjusts each subarray by a baseline period
def get_subarrays_with_baseline(signal, ref_idx, length, base_length):
    sub = get_subarrays(signal, ref_idx, length)
    baseline = get_subarrays(
        signal, ref_idx - base_length
        , base_length
        ).mean(axis=1).reshape(-1, 1)
    return sub - baseline


# returns of array of ranges for each signal in subarrays
def get_ranges(subarrays):
    out = []
    for arr in subarrays:
        out.append(arr.max() - arr.min())
    return np.array(out)


# finds outliers using a metric
def find_outliers(subarrays, metric):
    ranges = metric(subarrays)
    Q1 = np.percentile(ranges, 25)
    Q3 = np.percentile(ranges, 75)
    IQR = Q3 - Q1
    return ((ranges < Q1 - 1.5 * IQR) | (ranges > Q3 + 1.5 * IQR))