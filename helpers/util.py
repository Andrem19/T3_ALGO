import os


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' was created.")
    else:
        print(f"Folder is already exist '{path}'")
        

def find_candle_index(timestamp, candles):

    left, right = 0, len(candles) - 1

    while left <= right:
        mid = (left + right) // 2
        mid_timestamp = candles[mid][0]

        if mid_timestamp == timestamp:
            return mid
        elif mid_timestamp < timestamp:
            left = mid + 1
        else:
            right = mid - 1

    return -1