import numpy as np
from scipy.stats import norm
import numpy as np
from scipy.stats import multivariate_normal

def kde_gaussian(x, data, h):
    """
    Computes the kernel density estimate at point x using a Gaussian kernel
    with bandwidth h and data array data.
    """
    return np.sum(norm.pdf((x - data) / h) / (len(data) * h))

def kde_pdf(data, h):
    """
    Computes the probability density function of an array of data using
    kernel density estimation with a Gaussian kernel with bandwidth h.
    """
    x_min = np.min(data)
    x_max = np.max(data)
    x_range = x_max - x_min
    n_points = len(data)
    n_bins = int(np.sqrt(n_points))
    bin_width = x_range / n_bins
    x_vals = np.linspace(x_min, x_max, n_bins)
    pdf_vals = np.zeros_like(x_vals)
    for i, x in enumerate(x_vals):
        pdf_vals[i] = kde_gaussian(x, data, h)
    return x_vals, pdf_vals

def kde_bivariate_gaussian(x, y, data, h):
    """
    Computes the kernel density estimate at point (x, y) using a bivariate
    Gaussian kernel with bandwidth h and data array data.
    """
    xy = np.column_stack([x, y])
    return np.sum(multivariate_normal.pdf(xy, mean=data, cov=h**2) / (len(data) * h**2))

def kde_joint_pdf(data_x, data_y, h):
    """
    Computes the joint probability density function of two arrays of data
    using kernel density estimation with a bivariate Gaussian kernel with
    bandwidth h.
    """
    x_min = np.min(data_x)
    x_max = np.max(data_x)
    x_range = x_max - x_min
    y_min = np.min(data_y)
    y_max = np.max(data_y)
    y_range = y_max - y_min
    n_points = len(data_x)
    n_bins = int(np.sqrt(n_points))
    x_bin_width = x_range / n_bins
    y_bin_width = y_range / n_bins
    x_vals = np.linspace(x_min, x_max, n_bins)
    y_vals = np.linspace(y_min, y_max, n_bins)
    pdf_vals = np.zeros((n_bins, n_bins))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            pdf_vals[i, j] = kde_bivariate_gaussian(x, y, np.column_stack([data_x, data_y]), h)
    return x_vals, y_vals, pdf_vals


h = 0.5

data_x = np.random.normal(0, 1, 1000)
data_y = np.random.normal(0, 1, 1000)


x_vals, pdf_vals_x = kde_pdf(data_x, h)
x_vals, pdf_vals_y = kde_pdf(data_y, h)
x_vals, y_vals, pdf_vals_xy = kde_joint_pdf(data_x, data_y, h)

