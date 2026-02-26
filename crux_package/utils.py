import os


# returns a file name that is not currently being used
def next_valid(prefix, extension=".wav"):
    counter = 0
    filename = f"{prefix}{counter}{extension}"

    while os.path.exists(filename):
        counter += 1
        filename = f"{prefix}{counter}{extension}"
    return filename