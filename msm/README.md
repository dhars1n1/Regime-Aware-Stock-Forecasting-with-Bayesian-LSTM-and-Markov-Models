# Multivariate Gaussian Hidden Markov Model (HMM)

This code file contains the implementation of the **Baum-Welch algorithm** (Expectation-Maximization for HMMs) with **multivariate Gaussian emissions**, written in pure NumPy and SciPy.  

The code supports:
- Training HMM parameters (`π`, `A`, means, covariances) from observed sequences  
- Forward-Backward algorithm in **log-space** for numerical stability  
- Expectation-Maximization (Baum-Welch) updates  
- Viterbi decoding for most likely state sequence  

---

## 🚀 Features
- **Numerically stable log-space** forward-backward algorithms  
- **Full covariance matrices** for Gaussian emissions  
- **Random initialization** with reproducibility (`np.random.seed(42)`)  
- **Convergence check** with tolerance (`tol`)  
- **Verbose training log** showing log-likelihood and parameter change  
- **Viterbi decoding** to infer most probable hidden state path  

---

## 📈 Output Example
During training, you’ll see logs like:
```
Iter 001 loglik=-354.812340 change=1.234e-01
Iter 002 loglik=-342.678912 change=9.210e-02
Iter 003 loglik=-335.456789 change=5.670e-02
...
```

---

## ⚙️ Functions Overview
- `log_multivariate_gaussian_pdf(x, mean, cov)`  
  Computes log-probability density for multivariate Gaussian.  

- `forward_log(log_pi, log_A, log_B)`  
  Forward algorithm in log-space.  

- `backward_log(log_A, log_B)`  
  Backward algorithm in log-space.  

- `baum_welch_multivariate(obs, K, max_iter, tol, verbose)`  
  Full training loop for multivariate Gaussian HMM. Returns trained parameters and Viterbi path.  

---

## 📚 Requirements
- Python 3.8+  
- NumPy  
- SciPy  

Install dependencies with:
```bash
pip install numpy scipy
```

---

## 📝 Notes
- Works best with **continuous multivariate data**.  
- Covariance matrices are regularized with `1e-6` on diagonal for numerical stability.  
- Convergence is based on parameter change (`tol`).  

