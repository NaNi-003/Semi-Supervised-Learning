import numpy as np
import matplotlib.pyplot as plt

mean = 5
std_dev = 9

num_points = 500
points = np.random.normal(mean, std_dev, num_points)

# Create x-axis values for the scatter plot
x_values = np.arange(num_points)

# Plot scatter plot of the generated points
plt.figure(figsize=(8, 6))
plt.scatter(x_values, points, alpha=0.5, color='r')
plt.title('Scatter Plot of Points Generated from Gaussian Distribution')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Set mean vector and covariance matrix
mean = [0, 0]   # Mean vector of the multivariate Gaussian distribution
cov_matrix = [[1, 0.5], [0.5, 1]]   # Covariance matrix of the multivariate Gaussian distribution

# Generate 1000 points using a Gaussian multivariate distribution
num_points = 1000
points = np.random.multivariate_normal(mean, cov_matrix, num_points)

# Extract x and y coordinates of the points
x_values = points[:, 0]
y_values = points[:, 1]

# Plot scatter plot of the generated points
plt.figure(figsize=(7, 7))
plt.scatter(x_values, y_values, alpha=0.6, color='g')
plt.title('Scatter Plot of Points Generated from Gaussian Multivariate Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Set parameters for the Gaussian mixture model
num_components = 3   # Number of Gaussian components
num_points_per_component = 100   # Number of points per component
covariance_type = 'full'   # Type of covariance matrix for each component ('full', 'tied', 'diag', 'spherical')

# Set mean vectors and covariance matrices for each component
mean_vectors = np.array([[0, 0], [3, 3], [6, 0]])
cov_matrices = np.array([[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], [[1, 0], [0, 1]]])

# Generate points using the Gaussian mixture model
points = []
for i in range(num_components):
    component_points = np.random.multivariate_normal(mean_vectors[i], cov_matrices[i], num_points_per_component)
    points.append(component_points)
points = np.vstack(points)

# Fit a Gaussian mixture model to the generated points
gmm = GaussianMixture(n_components=num_components, covariance_type=covariance_type)
gmm.fit(points)

# Generate random samples from the learned Gaussian mixture model
sample_points, _ = gmm.sample(1000)

# Plot scatter plot of the generated points and sample points
plt.figure(figsize=(7, 7))
plt.scatter(points[:, 0], points[:, 1], alpha=0.6, color='g', label='Original Points')
plt.scatter(sample_points[:, 0], sample_points[:, 1], alpha=0.6, color='r', label='Sampled Points')
plt.title('Scatter Plot of Points Generated from Gaussian Mixture Model')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

gmm.means_

gmm.covariances_

gmm.weights_

