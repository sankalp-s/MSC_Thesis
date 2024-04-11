# Dataset Loading and Preparation

[Preprocessing_remove_instance.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Preprocessing/Preprocessing_remove_instance.py)

The provided script loads a dataset from a .npy file and prepares it for further processing. Here's a breakdown of the code:

## Functions
- `open_npy_file(file_path)`: Loads data from a .npy file, handling potential errors such as file not found or other exceptions.
- `integer_to_binary_array(integer_value, array_length=12)`: Converts an integer value to a binary array of specified length.

## Example Usage
The script loads a dataset stored in a .npy file located at a specific file path and prints the size of the loaded dataset.

## Data Processing
- The loaded dataset is separated into feature matrix `X` and target vector `y`.
- Unique integer labels present in the target vector `y` are extracted.
- A mapping is created between the original integer labels and class indices for classification.

## Dataset Filtering
- Instances in the dataset where the target value is either 0, 16, 32, 40, 48, 56, 80, 2080, or 2096 are filtered out.
- The filtered dataset is saved to a new .npy file.
- The script prints the size of the filtered dataset.

This script serves as a template for loading, processing, and filtering datasets stored in .npy files, facilitating data preprocessing tasks in machine learning projects.

