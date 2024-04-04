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

def integer_to_binary_array(integer_value, array_length=12):
    """
    Convert an integer to a binary array of specified length.
    Parameters:
        integer_value (int): The integer value to be converted.
        array_length (int): The desired length of the binary array.
    Returns:
        list: The binary array representing the integer value.
    """
    binary_string = format(integer_value, 'b')  # Convert integer to binary string
    binary_array = [int(bit) for bit in binary_string.zfill(array_length)]  # Pad with leading zeros if needed
    return np.array(binary_array)

# Example usage
file_path = '/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/Combined_dataset/Integer/Master_integer.npy'
loaded_data = open_npy_file(file_path)

if loaded_data is not None:
    print("----------------------------------------")
    print(loaded_data[:1])  # Print first 5 elements
    print("1.) Combined Dataset size", loaded_data.shape)
    print("----------------------------------------")

X = loaded_data[:, :-1]
y = loaded_data[:, -1] #Last column is the target variable

# Extract unique integer labels
original_labels = np.unique(y)

# Create a mapping between integer labels and class indices which will be later used in classification as classes.
label_to_index = {label: index for index, label in enumerate(original_labels)}

# Create a mapping from class indices to original labels which will be later used to track the labels from the class index.
index_to_label = {index: label for label, index in label_to_index.items()}

# Replace labels with corresponding class indices
# Modifying the target variables for classification
y_categorical = np.array([label_to_index[label] for label in y])

# Count occurrences of each label
label_counts = np.unique(y, return_counts=True)

print("Label Counts:")
for label, count in zip(label_counts[0], label_counts[1]):
    print(f"Label {label}: {count} times")

print('Label Values:')
for label in original_labels:
    print('label', label, 'binary array', integer_to_binary_array(label))

#print("label counts:")
#print(label_counts)

"""
import numpy as np
import matplotlib.pyplot as plt

# Labels and counts
labels = ['No action', 'Spin Jump', 'Right', 'Right+Spin Jump', 'Left', 'Left+Spin Jump', 'Left+Right', 'Left+Right+Spin Jump', 'Down', 'Down+Left', 'Up+Right', 'Jump', 'Jump+Right', 'Jump+Spin+Right', 'Jump+Left', 'Jump+Left+Right']
counts = [10067, 141, 78899, 6888, 7713, 758, 57, 3, 2536, 125, 19, 77, 11877, 9, 1045, 10]

# Plotting
plt.figure(figsize=(19, 6))
bars = plt.barh(labels, counts, color='skyblue')  # Horizontal bar plot
plt.xlabel('Count')
plt.ylabel('Label Value')
plt.title('Label Counts')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Annotate bars with counts
for bar, count in zip(bars, counts):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, count, 
             va='center', ha='left', fontsize=10, color='black')

plt.show()
"""



