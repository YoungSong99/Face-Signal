import numpy as np


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else np.nan


def safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size else np.nan


def safe_var(x: np.ndarray) -> float:
    return float(np.var(x)) if x.size else np.nan


def safe_skewness(x: np.ndarray, eps: float = 1e-8) -> float:
    if x.size == 0:
        return np.nan

    x = x.astype(np.float64)
    mu = np.mean(x)
    sigma = np.std(x)

    if sigma < eps:
        return 0.0

    return float(np.mean(((x - mu) / sigma) ** 3))


def safe_kurtosis(x: np.ndarray, eps: float = 1e-8) -> float:
    if x.size == 0:
        return np.nan

    x = x.astype(np.float64)
    mu = np.mean(x)
    sigma = np.std(x)

    if sigma < eps:
        return 0.0

    return float(np.mean(((x - mu) / sigma) ** 4))


def safe_uniformity(hist: np.ndarray, eps: float = 1e-12) -> float:
    if hist.size == 0:
        return np.nan

    total = np.sum(hist)
    if total < eps:
        return 0.0

    p = hist / total
    return float(np.sum(p ** 2))


def safe_entropy(x: np.ndarray, eps: float = 1e-12) -> float:
    if x.size == 0:
        return np.nan

    x = x.astype(np.float64)

    total = np.sum(x)
    if total < eps:
        return 0.0

    p = x / total
    p = p[p > 0]

    return float(-np.sum(p * np.log2(p)))


def safe_corrcoef(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    if x.size == 0 or y.size == 0:
        return np.nan

    x = x.reshape(-1).astype(np.float64)
    y = y.reshape(-1).astype(np.float64)

    x_std = np.std(x)
    y_std = np.std(y)

    if x_std < eps or y_std < eps:
        return 0.0

    return float(np.corrcoef(x, y)[0, 1])