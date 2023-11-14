"""
CUR Matrix Decomposition and Reconstruction

Author: Muhammad Ahmed
Date: [Insert Date]
Version: 1.0

Description:
This script demonstrates the CUR matrix decomposition technique applied to a structured matrix with missing values. The primary goal is to reconstruct the missing values in a matrix that has a clear sequential pattern. For instance, in a sequence like 1, 2, 3, NaN, 5, a successful CUR reconstruction would approximate the NaN value with a number close to 4.

The matrix is initially generated with structured values and then selectively introduced with missing values (NaNs). The script then performs CUR decomposition by selecting columns and rows that do not contain NaN values. The U matrix is constructed and inverted as part of the decomposition process. Finally, the matrix is reconstructed, and the norm of the difference between the original and reconstructed matrices is computed, focusing only on the non-NaN elements.

This script serves as an educational tool to demonstrate CUR decomposition in action and its potential applications in dealing with incomplete datasets.

Requirements:
- Python 3
- NumPy library

"""
import numpy as np

# Set the size of the matrix and the seed for reproducibility
rows, cols = 10, 8
np.random.seed(0)

# Generate a non-square matrix with structured values
matrix = np.arange(1, rows * cols + 1).reshape((rows, cols))
matrix = matrix.astype(float)

# Print the rank of the original matrix
rank = np.linalg.matrix_rank(matrix)
print(f'The rank of the matrix is {rank}')

# Introduce missing values (5% of the data)
num_missing = int(0.05 * rows * cols)
missing_indices = np.random.choice(rows * cols, num_missing, replace=False)
for index in missing_indices:
    matrix[np.unravel_index(index, (rows,cols))] = np.nan

# Display the original matrix with missing values
print('This is the Original Matrix with missing values: \n')
print(matrix)

# Select columns and rows for CUR decomposition avoiding NaNs
# Initialize lists for storing indices and actual data of selected columns
col_index = []
selected_columns = np.zeros((10, rank))

for i in range(cols):
    if not np.isnan(matrix[:, i]).any():
        col_index.append(i)
        selected_columns[:, len(col_index) - 1] = matrix[:, i]
        if len(col_index) == rank:
            break

# Initialize lists for storing indices and actual data of selected rows
row_index = []
selected_rows = np.zeros((rank, 8))

for i in range(rows):
    if not np.isnan(matrix[i, :]).any():
        row_index.append(i)
        selected_rows[len(row_index) - 1, :] = matrix[i, :]
        if len(row_index) == rank:
            break

# Construct U matrix and its inverse for CUR decomposition
U = np.array([
    [matrix[row_index[0], col_index[0]], matrix[row_index[0], col_index[1]]],
    [matrix[row_index[1], col_index[0]], matrix[row_index[1], col_index[1]]]
])
U_inverse = np.linalg.inv(U)

# Reconstruct the matrix using CUR decomposition
matrix_reconstructed = selected_columns @ (U_inverse @ selected_rows)

# Display the reconstructed matrix
print('This is the reconstructed Matrix: \n')
print(matrix_reconstructed)

# Compute and print the norm of the difference between the original and reconstructed matrices
# for non-NaN elements
mask = ~np.isnan(matrix)
norm = np.linalg.norm((matrix - matrix_reconstructed)[mask])
print(f"L2 norm between reconstructed and original matrix is: {norm}")
