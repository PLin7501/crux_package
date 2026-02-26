import time
import pickle
import numpy as np
from tqdm.auto import tqdm
from playsound import playsound
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from .processing import interpolate_missing_samples
from .utils import next_valid


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
DELAY = 5
SAMPLING_RATE = 250


def _set_delay(delay):
    global DELAY
    DELAY = delay


# runs OpenBCI recording and plays audio file during recording
# saves data in a .pkl file
def record(filepath, serial_port):
    BoardShim.enable_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = serial_port # can change
    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    board.prepare_session()
    
    board.start_stream()
    print("start_openbci")
    times = {'start_openbci': time.time()}
    
    for _ in tqdm(range(DELAY)):
        time.sleep(1)    
    print("start_audio")
    board.insert_marker(1)

    times['start_audio'] = time.time()
    playsound(filepath)

    board.insert_marker(2)
    print("end_audio")
    times['end_audio'] = time.time()
    for _ in tqdm(range(DELAY)):
        time.sleep(1)

    data = board.get_board_data()
    board.stop_stream()
    board.release_session()
    print("end_openbci")
    times['end_openbci'] = time.time()

    data = data[:, 1:]

    filename = next_valid("data", ".pkl")
    with open(filename, "wb") as file:
        pickle.dump({
            "data": data,
            "times": times
        }, file)
    print(f"Saved data as: {filename}")
    
    return data, times


# opens a .pkl file and returns data and times
def open_data(filepath):
    with open(filepath, "rb") as file:
        loaded_data = pickle.load(file)
    return loaded_data['data'], loaded_data['times']


def extract(data, channels, transform=None): 

    if transform is None:
        transform = lambda data: data
    
    # returns processed values in a channel
    def get_channel_data(channel):
        vals = data[COLUMNS.index(channel)]
        vals = interpolate_missing_samples(
            data[COLUMNS.index('Sample Index')],
            vals
        )
        return transform(vals)[audio_start:audio_end]

    audio_start = np.where(data[COLUMNS.index('Marker Channel')] == 1)[0][0]
    audio_end = np.where(data[COLUMNS.index('Marker Channel')] == 2)[0][0]

    if len(channels) == 1:
        channels = channels[0]
    if isinstance(channels, str):
        return get_channel_data(channels)
    
    out = {}
    for channel in channels:
        out[channel] = get_channel_data(channel)
    return out


def print_time_latency(times):
    latency = times['start_audio'] - times['start_openbci']
    print(f"Intended latency: {DELAY} seconds")
    print(f"Measured latency (start_openbci --> start_audio): {latency} seconds")
    print(f"Sample latency: {(latency - DELAY) * SAMPLING_RATE} samples")


def record_extract(filepath, serial_port, channels, transform=None):
    data, times = record(filepath, serial_port)
    print_time_latency(times)
    return extract(data, channels, transform=transform), times
