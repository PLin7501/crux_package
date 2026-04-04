# columns labels of raw EEG data given by index
COLUMNS = [
    'Sample Index', 'EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2',
    'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6',
    'EXG Channel 7', 'Accel Channel 0', 'Accel Channel 1',
    'Accel Channel 2', 'Not Used', 'Digital Channel 0 (D11)',
    'Digital Channel 1 (D12)', 'Digital Channel 2 (D13)',
    'Digital Channel 3 (D17)', 'Not Used.1', 'Digital Channel 4 (D18)',
    'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2', 'Timestamp',
    'Marker Channel'
]

# padding in seconds before and after playing audio
DELAY = 20

# sampling rate of OpenBCI hardware
SAMPLING_RATE = 250

# sampling rate of generated audio
AUDIO_SAMPLING_RATE = 10000

# required parameters for data dictionary
REQUIRED_PARAMS = (
    "data",
    "click_idx",
    "soa_seconds",
    "soa_indices",
    "click_number",
    "delay",
    "channels",
    "end_idx",
    "audio_dict"
)


def print_required_params():
    print(REQUIRED_PARAMS)
