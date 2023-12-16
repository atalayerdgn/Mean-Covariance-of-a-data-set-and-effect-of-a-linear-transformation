import	numpy as np
from numpy.testing import assert_allclose

def	affine_mean(mean, A, b):
    affine_m = np.dot(A, mean) + b
    
    return affine_m

def	affine_cov(S,A,b):
    affine_cov = np.dot(A, np.dot(S, A.T))
    return affine_cov
A = np.array([[0, 1], [2, 3]])
b = np.ones(2)
m = np.full((2,), 2)
S = np.eye(2)*2

expected_affine_mean = np.array([ 3., 11.])
expected_affine_cov = np.array(
    	[[ 2.,  6.],
    	[ 6., 26.]])

assert_allclose(affine_mean(m, A, b), expected_affine_mean, rtol=1e-4)
print("Test Passed")
