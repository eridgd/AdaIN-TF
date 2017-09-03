import numpy as np

def matSqrt(x):
    U,D,V = np.linalg.svd(x)
    result = np.dot(U, np.dot(np.diag(np.sqrt(D)), V.T))
    return result


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def coral(source, target):
    n_channels = source.shape[-1]

    source_flatten = source.reshape(-1, source.shape[0]*source.shape[1])
    target_flatten = target.reshape(-1, target.shape[0]*target.shape[1])

    source_flatten_mean = source_flatten.mean(axis=1, keepdims=True)
    source_flatten_std = source_flatten.std(axis=1, keepdims=True)
    source_flatten_norm = np.divide((source_flatten - source_flatten_mean), source_flatten_std)

    target_flatten_mean = target_flatten.mean(axis=1, keepdims=True)
    target_flatten_std = target_flatten.std(axis=1, keepdims=True)
    target_flatten_norm = np.divide((target_flatten - target_flatten_mean), target_flatten_std)

    source_flatten_cov_eye = np.dot(source_flatten_norm, source_flatten_norm.T) + np.eye(n_channels)
    target_flatten_cov_eye = np.dot(target_flatten_norm, target_flatten_norm.T) + np.eye(n_channels)

    source_flatten_norm_transfer = np.dot(np.dot(matSqrt(target_flatten_cov_eye), np.linalg.inv(matSqrt(source_flatten_cov_eye))), source_flatten_norm)
    source_flatten_transfer = np.multiply(source_flatten_norm_transfer, target_flatten_std) + target_flatten_mean
    
    coraled = source_flatten_transfer.reshape(source.shape)

    return coraled
