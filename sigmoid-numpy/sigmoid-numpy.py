import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    x_arr = np.array(x)
    result = 1 / (1 + np.exp(-x_arr))
    
    # Return a scalar float if input was a scalar, else return np.ndarray
    if isinstance(x, (float, int)):
        return float(result)
    
    return result