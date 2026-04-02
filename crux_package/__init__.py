from .sound_generation import (
    generate_gaussian, 
    generate_pulse, 
    generate_envelope, 
    generate_train, 
    generate_oddball_train,
    array_to_wav
)
from .processing import (
    get_channel_from_txt, 
    check_missing_samples,
    interpolate_missing_samples,
    bandpass,
    get_subarrays,
    get_subarrays_with_baseline,
    get_avg,
    reference_to_avg,
    get_ranges,
    find_outliers
)
from .recording import (
    Recording,
    record, 
    save_dict
)
from .globals import (
    _set_delay,
    print_required_params
)

__all__ = [
    # sound_generation.py
    "generate_gaussian",
    "generate_pulse",
    "generate_envelope",
    "generate_train",
    "generate_oddball_train",
    "array_to_wav",

    # processing.py
    "get_channel_from_txt",
    "check_missing_samples",
    "interpolate_missing_samples",
    "bandpass",
    "get_subarrays",
    "get_subarrays_with_baseline",
    "get_avg",
    "reference_to_avg",
    "get_ranges",
    "find_outliers",

    # recording.py
    "Recording",
    "record",
    "save_dict",
    
    # globals.py
    "_set_delay",
    "print_required_params"
]