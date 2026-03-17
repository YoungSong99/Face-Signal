import numpy as np


def safe_mean(x):
    x = np.asarray(x)
    return float(np.mean(x)) if x.size else np.nan


def safe_std(x):
    x = np.asarray(x)
    return float(np.std(x)) if x.size else np.nan


def safe_var(x):
    x = np.asarray(x)
    return float(np.var(x)) if x.size else np.nan


def safe_skewness(x, eps=1e-8):
    x = np.asarray(x)

    if x.size == 0:
        return np.nan

    x = x.astype(np.float64, copy=False)
    mu = np.mean(x)
    sigma = np.std(x)

    if sigma < eps:
        return 0.0

    return float(np.mean(((x - mu) / sigma) ** 3))


def safe_kurtosis(x, eps=1e-8):
    x = np.asarray(x)

    if x.size == 0:
        return np.nan

    x = x.astype(np.float64, copy=False)
    mu = np.mean(x)
    sigma = np.std(x)

    if sigma < eps:
        return 0.0

    return float(np.mean(((x - mu) / sigma) ** 4))


def safe_uniformity(hist, eps=1e-12):
    hist = np.asarray(hist)

    if hist.size == 0:
        return np.nan

    total = np.sum(hist)
    if total < eps:
        return 0.0

    p = hist / total
    return float(np.sum(p ** 2))


def safe_entropy(x, eps=1e-12):
    x = np.asarray(x)

    if x.size == 0:
        return np.nan

    x = x.astype(np.float64, copy=False)
    total = np.sum(x)

    if total < eps:
        return 0.0

    p = x / total
    p = p[p > 0]

    return float(-np.sum(p * np.log2(p)))


def safe_corrcoef(x, y, eps=1e-8):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.size == 0 or y.size == 0:
        return np.nan

    x = x.reshape(-1).astype(np.float64, copy=False)
    y = y.reshape(-1).astype(np.float64, copy=False)

    if np.std(x) < eps or np.std(y) < eps:
        return 0.0

    return float(np.corrcoef(x, y)[0, 1])


def safe_stats(arr):
    flat = np.asarray(arr).ravel()

    return {
        "mean": safe_mean(flat),
        "std": safe_std(flat),
        "kurt": safe_kurtosis(flat),
        "p95": float(np.percentile(np.abs(flat), 95)) if flat.size else np.nan,
    }


def hist_entropy(arr, bins=32, value_range=None):
    arr = np.asarray(arr)

    if arr.size == 0:
        return np.nan

    hist, _ = np.histogram(arr, bins=bins, range=value_range, density=False)
    return safe_entropy(hist)