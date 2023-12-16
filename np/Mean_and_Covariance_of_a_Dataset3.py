import	numpy as np
from numpy.testing import assert_allclose

def	cov_naive(X):
    
	N, D = X.shape
	covariance = np.zeros((D,D))
	mean_vector = np.mean(X, axis = 0)
	for	n in range(N):
		x_diff = X[n] - mean_vector
		covariance += np.outer(x_diff, x_diff)
	covariance /= N
	return covariance

# Test case 1
X = np.array([[0., 1.], 
              [1., 2.],
     [0., 1.], 
     [1., 2.]])
expected_cov = np.array(
    [[0.25, 0.25],
    [0.25, 0.25]])
assert_allclose(cov_naive(X), expected_cov, rtol=1e-5)
# Test case 2
X = np.array([[0., 1.], 
              [2., 3.]])
expected_cov = np.array(
    [[1., 1.],
    [1., 1.]])
assert_allclose(cov_naive(X), expected_cov, rtol=1e-5)
# Test covariance is zero
X = np.array([[0., 1.], 
              [0., 1.],
              [0., 1.]])
expected_cov = np.zeros((2, 2))
assert_allclose(cov_naive(X), expected_cov, rtol=1e-5)
print("All Tests Passed")
