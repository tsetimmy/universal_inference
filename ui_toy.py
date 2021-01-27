import numpy as np
import argparse
import sys
from scipy.stats import norm
import scipy.linalg as spla
import matplotlib.pyplot as plt
from tqdm import tqdm

def universal_inference(y, X, noise_var, beta, beta0, alpha):
    assert y.shape == X.shape

    half = len(y) // 2
    y0 = y[:half]
    X0 = X[:half]
    y1 = y[half:]
    X1 = X[half:]

    beta_hat_1 = np.dot(X1, y1) / np.dot(X1, X1)

    T = np.exp(np.square(y0 - X0 * beta_hat_1).sum() / (-2. * noise_var)) / np.exp(np.square(y0 - X0 * beta0).sum() / (-2. * noise_var))

    return 1. if T > 1. / alpha else 0.

def plot_theoretical_power(X, XX_inv, z, noise_var, beta0, betas):
    deltas = (betas - beta0) / np.sqrt(noise_var * XX_inv)
    t2_error = norm.cdf(z - deltas) - norm.cdf(-z - deltas) 
    power = 1. - np.array(t2_error)
    plt.plot(betas, power, label='theoretical power')
    plt.scatter(beta0, .05, color='red', marker='x')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=400)
    parser.add_argument('--noise_var', type=float, default=1.)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--beta0', type=float, default=1.)
    parser.add_argument('--n_iterations', type=int, default=250)
    parser.add_argument('--n_plot_points', type=int, default=100001)
    args = parser.parse_args()
    print(sys.argv)
    print(args)

    n_samples = args.n_samples
    noise_var = args.noise_var
    alpha = args.alpha
    beta0 = args.beta0

    X = np.random.normal(size=n_samples)

    n_iterations = args.n_iterations
    n_plot_points = args.n_plot_points
    betas = np.linspace(-2., 2., n_plot_points) + beta0

    XX_inv = 1. / np.dot(X, X)
    z = norm.ppf(1. - alpha / 2.)
    power = []
    power_ui = []
    for beta in tqdm(betas):
    #for beta in betas:
        rejections = 0.
        rejections_ui = 0.
        for _ in range(n_iterations):
            y = X * beta + np.random.normal(scale=np.sqrt(noise_var), size=n_samples)
            beta_hat = XX_inv * np.dot(X, y)
            stat = (beta_hat - beta0) / np.sqrt(noise_var * XX_inv)
            if np.abs(stat) >= z:
                rejections += 1.
            rejections_ui += universal_inference(y, X, noise_var, beta, beta0, alpha)
        power.append(rejections / float(n_iterations))
        power_ui.append(rejections_ui / float(n_iterations))
    power = np.array(power)
    power_ui = np.array(power_ui)

    #---------------------
    plt.plot(betas, power, label='empirical power')
    plt.plot(betas, power_ui, label='empirical power (UI)')
    #plt.grid()
    #plt.show()

    plot_theoretical_power(X, XX_inv, z, noise_var, beta0, betas)

    plt.grid()
    plt.legend()
    plt.xlabel('beta')
    plt.ylabel('power')
    plt.title('Testing: H0: beta = 1 vs H1: beta != 1')
    plt.show()
    #plt.savefig('universal_inference.pdf')

if __name__ == '__main__':
    main()
