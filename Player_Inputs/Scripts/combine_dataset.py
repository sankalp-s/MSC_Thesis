import numpy as np

def open_npy_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        return data
    except FileNotFoundError:
        print("File not found.")
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None
    
def binary_array_to_integer(binary_array):
    """
    Convert a binary array to an integer.
    """
    binary_string = ''.join(map(str, binary_array))
    decimal_integer = int(binary_string, 2)
    return decimal_integer

def main():
    # Example usage
    file_path = '/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/Level1/Sankalp_Session1.npy'
    loaded_data = open_npy_file(file_path)

    Integer_array = np.array([binary_array_to_integer(binary_array) for binary_array in loaded_data[1]])

    # Append state space and action space arrays
    appended_arrays = []
    for i in range(len(Integer_array)):
        appended_array = np.append(loaded_data[0][i], Integer_array[i])
        appended_arrays.append(appended_array)

    appended_arrays = np.array(appended_arrays)
    print("Resulting array shape:", appended_arrays.shape)

if __name__ == "__main__":
    main()
