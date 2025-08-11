
import numpy as np
from scipy.special import logsumexp
from numpy.linalg import slogdet, inv

def log_multivariate_gaussian_pdf(x, mean, cov):
    """Log PDF for multivariate Gaussian."""
    d = x.shape[0]
    sign, logdet = slogdet(cov)
    if sign <= 0:
        cov = cov + np.eye(d) * 1e-6
        sign, logdet = slogdet(cov)
    diff = x - mean
    return -0.5 * (d * np.log(2 * np.pi) + logdet + diff.T @ inv(cov) @ diff)

def forward_log(log_pi, log_A, log_B):
    T, K = log_B.shape
    alpha = np.zeros((T, K))
    alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        alpha[t] = log_B[t] + logsumexp(alpha[t-1][:, None] + log_A, axis=0)
    loglik = logsumexp(alpha[-1])
    return alpha, loglik

def backward_log(log_A, log_B):
    T, K = log_B.shape
    beta = np.zeros((T, K))
    for t in range(T-2, -1, -1):
        beta[t] = logsumexp(log_A + log_B[t+1] + beta[t+1], axis=1)
    return beta

def baum_welch_multivariate(obs, K=3, max_iter=200, tol=1e-6, verbose=True):
    """
    Baum-Welch for multivariate Gaussian HMM.
    obs: shape (T, D)
    """
    T, D = obs.shape
    np.random.seed(42)

    # Initialization
    pi = np.full(K, 1.0 / K)
    A = np.full((K, K), 1.0 / K)
    means = np.array([obs[np.random.choice(T, size=10)].mean(axis=0) for _ in range(K)])
    covs = np.array([np.cov(obs.T) + np.eye(D) * 1e-6 for _ in range(K)])

    loglik_hist = []
    for it in range(max_iter):
        # Emission log-probs
        log_B = np.zeros((T, K))
        for k in range(K):
            for t in range(T):
                log_B[t, k] = log_multivariate_gaussian_pdf(obs[t], means[k], covs[k])

        log_pi = np.log(pi + 1e-16)
        log_A = np.log(A + 1e-16)

        alpha, loglik = forward_log(log_pi, log_A, log_B)
        beta = backward_log(log_A, log_B)

        # Gamma
        log_gamma = alpha + beta
        log_gamma -= logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)

        # Xi
        xi_sum = np.zeros((K, K))
        for t in range(T-1):
            log_xi_t = alpha[t][:, None] + log_A + log_B[t+1][None, :] + beta[t+1][None, :]
            log_xi_t -= logsumexp(log_xi_t)
            xi_sum += np.exp(log_xi_t)

        # Updates
        pi_new = gamma[0]
        A_new = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        A_new = np.where(np.isnan(A_new), 1.0 / K, A_new)

        means_new = np.zeros((K, D))
        covs_new = np.zeros((K, D, D))
        for k in range(K):
            weight = gamma[:, k][:, None]
            means_new[k] = (weight * obs).sum(axis=0) / (gamma[:, k].sum() + 1e-16)
            diff = obs - means_new[k]
            covs_new[k] = (weight * diff).T @ diff / (gamma[:, k].sum() + 1e-16)
            covs_new[k] += np.eye(D) * 1e-6

        change = (np.max(np.abs(means_new - means)) +
                  np.max(np.abs(covs_new - covs)) +
                  np.max(np.abs(A_new - A)) +
                  np.max(np.abs(pi_new - pi)))

        pi, A, means, covs = pi_new, A_new, means_new, covs_new
        loglik_hist.append(loglik)

        if verbose:
            print(f"Iter {it+1:03d} loglik={loglik:.6f} change={change:.6e}")
        if change < tol:
            break

    # Viterbi
    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = np.log(pi + 1e-16) + log_B[0]
    for t in range(1, T):
        for k in range(K):
            seq_probs = delta[t-1] + np.log(A[:, k] + 1e-16)
            psi[t, k] = np.argmax(seq_probs)
            delta[t, k] = np.max(seq_probs) + log_B[t, k]
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    return {
        "pi": pi,
        "A": A,
        "means": means,
        "covs": covs,
        "gamma": gamma,
        "viterbi": path
    }
