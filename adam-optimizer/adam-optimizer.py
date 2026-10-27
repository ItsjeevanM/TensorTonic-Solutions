import numpy as np


def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (param_new, m_new, v_new) as NumPy arrays."""
    # Convert inputs to numpy arrays
    param = np.array(param, dtype=np.float64)
    grad = np.array(grad, dtype=np.float64)
    m = np.array(m, dtype=np.float64)
    v = np.array(v, dtype=np.float64)

    # 1. Update biased first moment estimate
    m_new = beta1 * m + (1 - beta1) * grad

    # 2. Update biased second raw moment estimate
    v_new = beta2 * v + (1 - beta2) * (grad**2)

    # 3. Compute bias-corrected estimates
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)

    # 4. Update parameters
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param_new, m_new, v_new