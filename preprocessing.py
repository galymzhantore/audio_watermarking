import numpy as np

def bernoulli_map(length, y0=0.3, r=2.0):
    """Bernoulli shift map: y(n+1) = (r * y(n)) mod 1"""
    y = np.zeros(length)
    y[0] = y0
    for i in range(1, length):
        y[i] = (r * y[i - 1]) % 1.0
    return y, (y >= 0.5).astype(np.uint8)


def gaussian_map(length, x0=0.5, alpha=6.2, beta=-0.5):
    """Gaussian map: x(n+1) = exp(-alpha * x(n)^2) + beta"""
    x = np.zeros(length)
    x[0] = x0
    for i in range(1, length):
        x[i] = np.exp(-alpha * x[i - 1] ** 2) + beta
    return x, (x >= 0).astype(np.uint8)


def tent_map(length, x0=0.3, mu=2.0):
    """Tent map: x(n+1) = mu * min(x(n), 1 - x(n))"""
    x = np.zeros(length)
    x[0] = x0
    for i in range(1, length):
        x[i] = mu * min(x[i - 1], 1 - x[i - 1])
    return x, (x >= 0.5).astype(np.uint8)


def get_chaotic_mask(length, method='bernoulli', encryption_params = {}):
    if method == 'bernoulli':
        _, mask = bernoulli_map(length, **encryption_params)
    elif method == 'gaussian':
        _, mask = gaussian_map(length, **encryption_params)
    elif method == 'tent':
        _, mask = tent_map(length, **encryption_params)
    else:
        raise ValueError(f"Unknown method: {method}")
    return mask

def generate_permutation_key(length, seed=42):
    rng = np.random.default_rng(seed)
    return rng.permutation(length)


def encrypt_watermark(watermark, method='bernoulli',
                      encryption_params = {}, seed = 42):
    q = watermark.flatten().astype(np.uint8)
    length = len(q)
    C = generate_permutation_key(length, seed = seed)
    e = q[C]
    z = get_chaotic_mask(length, method, encryption_params)
    u = np.bitwise_xor(e, z)
    return u


def decrypt_watermark(encrypted_sequence, original_shape, method='bernoulli',
                      encryption_params = {}, seed = 42):
    u_star = encrypted_sequence.astype(np.uint8)
    length = len(u_star)
    z = get_chaotic_mask(length, method, encryption_params)
    e_star = np.bitwise_xor(u_star, z)
    C = generate_permutation_key(length, seed = seed)
    C_inv = np.argsort(C)
    q_star = e_star[C_inv]
    return q_star.reshape(original_shape)