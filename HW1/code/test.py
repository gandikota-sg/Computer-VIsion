import numpy as np

file1 = np.load('dictionary.npy')
file2 = np.load('1.npy')

diff_indices = np.where(file1 != file2)
print(file2)
#print(diff_indices)