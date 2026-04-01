import time
import pickle
import numpy as np
from tqdm.auto import tqdm
from playsound import playsound
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from .globals import DELAY, REQUIRED_PARAMS
from .utils import next_valid


# context manager that starts and ends recording
# returns self, board and data can be retrieved using self.board and self.data
class Recording:
    def __init__(self, serial_port):
        self.serial_port = serial_port
        self.data = None

    def __enter__(self):
        print("Starting recording ...")
        BoardShim.enable_board_logger()
        params = BrainFlowInputParams()
        params.serial_port = self.serial_port
        board = BoardShim(BoardIds.CYTON_BOARD.value, params)

        self.board = board
        self.board.prepare_session()
        self.board.start_stream()

        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.data = self.board.get_board_data()
        self.board.stop_stream()
        self.board.release_session()
        print("Finished recording")


# runs OpenBCI recording and plays audio file during recording
# pads audio file with globals.DELAY seconds
# returns raw data (2d numpy array)
def record(filepath, serial_port):
    recording = Recording(serial_port)
    with recording as rec:
        board = rec.board

        for _ in tqdm(range(DELAY), desc="Padding ..."):
            time.sleep(1)    
            board.insert_marker(1)
        
        print("started audio")
        playsound(filepath)
        print("finished audio")

        board.insert_marker(2)
        for _ in tqdm(range(DELAY), desc="Padding ..."):
            time.sleep(1)

    return recording.data


def save_dict(data_dict, save_path, check_params=True):
    save = True

    if check_params:
        for key, value in data_dict.items():
            if key not in REQUIRED_PARAMS:
                print(f"warning: key not in required params: \"{key}\"")
        for key in REQUIRED_PARAMS:
            if key not in data_dict.keys():
                print(f"following key must be in data_dict: \"{key}\"")
                save = False
            
    if save:
        with open(save_path, 'wb') as f:
            pickle.dump(data_dict)
        print(f"saved to {save_path}")
    else:
        print("save aborted")


'''
DELETE AFTER RECORD FUNCTION ABOVE HAS BEEN TESTED

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
    
    return data
'''

