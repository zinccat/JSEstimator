import numpy as np
import matplotlib.pyplot as plt

def james_stein_estimator(X):
    """
    Compute the James-Stein estimator for a dataset X.
    Assumptions:
      - Observations in rows of X: shape (n, d)
      - Identity covariance
      - Target shrinkage point: the zero vector

    The James-Stein estimator for the mean is:
      theta_JS = (1 - (d-2)/sum(xbar^2)) * xbar, if sum(xbar^2) > 0
      If sum(xbar^2) = 0, it's just zero.
    """
    n, d = X.shape
    xbar = np.mean(X, axis=0)
    norm2 = np.sum(xbar**2)
    if norm2 > 0:
        shrinkage = max(0, 1 - (d - 2) / (n * norm2))
        return shrinkage * xbar
    else:
        return xbar  # which is zero anyway in this case

def mle_estimator(X):
    """
    The MLE for the mean under a normal model is just the sample mean.
    """
    return np.mean(X, axis=0)

def simulate_comparison(n=20, dimensions=[1,2,5,10,20,50], num_trials=10000, random_seed=42):
    """
    Simulate and compare MLE and James-Stein estimators for different dimensions.

    Parameters:
      - n: sample size
      - dimensions: list of dimensions to test
      - num_trials: number of simulated datasets per dimension
      - random_seed: for reproducibility
    """
    np.random.seed(random_seed)
    
    mse_mle = []
    mse_js = []
    
    for d in dimensions:
        # True parameters
        true_mean = np.zeros(d)  # we assume a zero mean to highlight shrinkage
        cov = np.eye(d)
        
        # Track errors for this dimension
        errors_mle = []
        errors_js = []

        for _ in range(num_trials):
            X = np.random.multivariate_normal(mean=true_mean, cov=cov, size=n)

            # Compute estimators
            theta_mle = mle_estimator(X)
            theta_js = james_stein_estimator(X)

            # Compute squared error
            error_mle = np.sum((theta_mle - true_mean)**2)
            error_js = np.sum((theta_js - true_mean)**2)

            errors_mle.append(error_mle)
            errors_js.append(error_js)

        mse_mle.append(np.mean(errors_mle))
        mse_js.append(np.mean(errors_js))
    
    return dimensions, mse_mle, mse_js


if __name__ == "__main__":
    # Example parameters
    n = 20
    dimensions = [2, 5, 10, 20]
    num_trials = 1000

    dims, mle_risks, js_risks = simulate_comparison(n=n, dimensions=dimensions, num_trials=num_trials)

    # Print results
    print("Dimension | MLE Risk | James-Stein Risk")
    for d, m_mle, m_js in zip(dims, mle_risks, js_risks):
        print(f"{d:9d} | {m_mle:9.6f} | {m_js:9.6f}")

    # Plot results
    plt.figure(figsize=(8,6))
    plt.plot(dims, mle_risks, marker='o', label='MLE')
    plt.plot(dims, js_risks, marker='s', label='James-Stein')
    plt.xlabel('Dimension (d)')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Comparison of MLE and James-Stein (n={n}, trials={num_trials})')
    plt.legend()
    plt.grid(True)
    plt.savefig('js_vs_mle.png')
    plt.show()

