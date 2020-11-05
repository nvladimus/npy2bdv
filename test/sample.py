import numpy as np

def generate_test_image(dim_yx, iz, nz):
    """Gaussian blob spanning the range of uint16 type."""
    x = np.linspace(-3, 3, dim_yx[1])
    y = np.linspace(-3, 3, dim_yx[0])
    sigma = 1.0 - abs(iz - nz/2) / nz
    x, y = np.meshgrid(x, y)
    return (65535 * np.exp(- ((x ** 2) + (y ** 2)) / (2 * sigma**2) )).astype("uint16")