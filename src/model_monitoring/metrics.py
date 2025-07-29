import numpy as np
import pandas as pd


def gini(t, p, w):
    """Calculates the Gini coefficient from true values, predicted scores, and weights.

    Args:
        t (np.array or list): An array of the true binary outcomes (0 or 1).
        p (np.array or list): An array of the predicted scores or probabilities.
        w (np.array or list): An array of weights for each observation.

    Returns:
        float: The calculated Gini coefficient, a value between -1 and 1.
    """
    data = pd.DataFrame({"t": t, "p": p, "w": w})

    # Handle edge case: empty data
    if len(data) == 0:
        return 0.0

    data = data.sort_values("p", ascending=False)
    data["w"] = data["w"] / data["w"].sum()

    nu_ = np.cumsum(data["t"] * data["w"])

    # Handle edge case: all targets are 0 (no positive events)
    if len(nu_) == 0 or nu_.iloc[-1] == 0:
        return 0.0

    nu_ = nu_ / nu_.iloc[-1]
    nu_ = [0, *list(nu_)]
    dx = np.cumsum(data["w"])
    dx = [0, *list(dx)]

    auc = sum(np.add(nu_[:-1], nu_[1:]) * (np.array(dx[1:]) - np.array(dx[:-1])) / 2)
    gini_coefficient = 2 * auc - 1

    return gini_coefficient
