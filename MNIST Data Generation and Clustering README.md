# MNIST Data Generation and Clustering
This README file on the ipynb file demonstrates the generation, visualization, and clustering of MNIST data, including both real and synthetic (fake) data. The steps include loading the data, preprocessing, performing PCA for dimensionality reduction, and using K-means for clustering.

## Table of Contents:-
1. Overview
2. Installation
3. Usage
4. Code Explanation
5. Results
6. Contributing
7. License

## Overview:-
### The notebook performs the following tasks:-

#### Loads real MNIST data.
#### Generates synthetic MNIST data using a GAN (Generative Adversarial Network).
#### Preprocesses the data.
#### Combines real and fake data.
#### Applies PCA for reducing data to 3 dimensions.
#### Clusters the data using K-means.
#### Visualizes the clusters in a 3D scatter plot.

## Installation:-
### To run this notebook, you need the following libraries:-

#### numpy
#### matplotlib
#### sklearn
#### tensorflow
#### keras

### You can install them using pip:-

` pip install numpy matplotlib sklearn tensorflow keras `

## Usage:-
1. Clone the repository or download the notebook file.
2. Ensure you have the required libraries installed.
3. Open the notebook in Jupyter Notebook or Jupyter Lab.
4. Run the cells sequentially.
   
## Code Explanation:-
### 1. Loading Real MNIST Data
#### The code loads the real MNIST data using tensorflow.keras.datasets.mnist.

### 2. Generating Fake MNIST Data
#### A pre-trained GAN model generates fake MNIST images. The GAN model structure includes a generator and a discriminator.

### 3. Preprocessing Data
#### The data (both real and fake) is normalized and reshaped for further processing.

### 4. PCA for Dimensionality Reduction
#### PCA reduces the combined dataset to 3 dimensions for easier visualization and clustering.

### 5. K-means Clustering
#### K-means algorithm is applied to cluster the reduced data into 10 clusters.

### 6. 3D Scatter Plot Visualization
#### The clusters are visualized in a 3D scatter plot with distinct colors for each cluster.

## Results:-
### The notebook concludes with a 3D scatter plot showing the clustering of real and fake MNIST data.

## Contributing:-
### Contributions are welcome! Please create an issue or submit a pull request.

## License:-
### This project is licensed under the MIT License. See the LICENSE file for details.
