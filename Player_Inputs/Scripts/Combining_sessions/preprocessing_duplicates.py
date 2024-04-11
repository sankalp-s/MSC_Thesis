import numpy as np
import matplotlib.pyplot as plt

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


# Example usage
file_path = '/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/Combined_dataset/Integer/Master_integer.npy'
loaded_data = open_npy_file(file_path)

if loaded_data is not None:
    print("----------------------------------------")
    print(loaded_data[:1])  # Print first row
    print("1.) Combined Dataset size", loaded_data.shape)
    print("----------------------------------------")

    # Count the occurrences of each unique row
    unique_rows, counts = np.unique(loaded_data, axis=0, return_counts=True)

    # Find indices of duplicate rows
    duplicate_indices = np.where(counts > 1)[0]

    # Print duplicate rows and their counts
    total_duplicates = 0
    if len(duplicate_indices) > 0:
        print("Duplicate rows and their counts:")
        for idx in duplicate_indices:
            print("Row:", unique_rows[idx], "Count:", counts[idx])
            total_duplicates += counts[idx]
    else:
        print("No duplicate rows found.")
    print("Total duplicate rows found:", total_duplicates)
