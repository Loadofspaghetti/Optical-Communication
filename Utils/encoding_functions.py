# utils/encoding_functions.py

import numpy as np

from utils.global_definitions import (
    rows, columns
)

class Encode:

    def __init__(self, rows=rows, columns=columns):
        self.rows = rows
        self.columns = columns


    def message_to_bit_arrays(self, message, bits_per_cell):

        # Converts message to a string encoded in binary
        binary_string = "".join([format(ord(character), "08b") for character in message])

        chunks = [binary_string[i:i+bits_per_cell] for i in range(0, len(binary_string), bits_per_cell)]

        cell_values = [int(chunk, 2) for chunk in chunks]

        frame_capacity = self.rows * self.columns

        frame_bit_arrays = []

        for start_idx in range(0, len(cell_values), frame_capacity):
            frame_chunk = cell_values[start_idx:start_idx + frame_capacity]

            if len(frame_chunk) < frame_capacity:
                frame_chunk += [0] * (frame_capacity - len(frame_chunk))
                
            frame_array = np.array(frame_chunk, dtype=np.uint8).reshape((self.rows, self.columns))
            frame_bit_arrays.append(frame_array)

        return frame_bit_arrays
    
encode = Encode()