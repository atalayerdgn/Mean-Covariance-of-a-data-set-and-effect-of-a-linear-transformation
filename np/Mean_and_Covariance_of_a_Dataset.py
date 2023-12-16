import	numpy as np
from numpy.testing import assert_allclose
def	mean_naive(X):
    
	mean = np.zeros(X.shape[1])
	for	n in range(X.shape[0]):
		mean += X[n]
	mean /= X.shape[0]
	return mean

# Test case 1
X = np.array([[0., 1., 1.], 
              [1., 2., 1.]])
expected_mean = np.array([0.5, 1.5, 1.])
assert_allclose(mean_naive(X), expected_mean, rtol=1e-5)
# Test case 2
X = np.array([[0., 1., 0.], 
              [2., 3., 1.]])
expected_mean = np.array([1., 2., 0.5])
assert_allclose(mean_naive(X), expected_mean, rtol=1e-5)
# Test covariance is zero
X = np.array([[0., 1.], 
              [0., 1.]])
expected_mean = np.array([0., 1.])
assert_allclose(mean_naive(X), expected_mean, rtol=1e-5)
print("All Tests Passed")
