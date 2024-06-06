import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
import warnings
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import mode
from sklearn.impute import SimpleImputer
warnings.simplefilter(action='ignore')
sns.set()
plt.style.use("ggplot") 
# %matplotlib inline

# read the dataset from dir
original_data = pd.read_csv("diabetes.csv")
# Separate features and target variable
X = original_data.drop('Outcome', axis=1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit KMeans to the scaled data
kmeans.fit(X_scaled)

# Get cluster labels for each sample
cluster_labels = kmeans.labels_

# Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
plt.title('K-means Clustering')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Calculate silhouette score
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")
# original_data.info()

from google.colab import drive
drive.mount('/content/drive')

# Randomize the data values
randomized_data = original_data.apply(np.random.permutation, axis=0)

# Remove the labels from the dataset
unlabeled_randomized_data = randomized_data.drop('Outcome', axis=1)

# Verify the shape of the unlabeled randomized dataset
print("Shape of unlabeled randomized dataset:", unlabeled_randomized_data.shape)

# Now, unlabeled_randomized_data contains the dataset with randomized data values and without labels.

# Randomly select 77 rows from the PIMA Indian dataset
random_pima_data = original_data.sample(n=77, random_state=42)

# Randomly select 681 rows from the fake unlabeled randomized dataset
random_fake_data = unlabeled_randomized_data.sample(n=681, random_state=42)

# Verify the shape of the randomly selected subsets
print("Shape of randomly selected PIMA Indian dataset:", random_pima_data.shape)
print("Shape of randomly selected fake dataset:", random_fake_data.shape)

# Now, random_pima_data and random_fake_data contain the randomly selected rows from their respective datasets.

# Merge both random_pima_data and random_fake_data into a single dataframe
merged_data = pd.concat([random_pima_data, random_fake_data], ignore_index=True)

# Verify the shape of the merged dataframe
print("Shape of merged dataframe:", merged_data.shape)

# Now, merged_data contains both random_pima_data and random_fake_data merged into a single dataframe.

print(merged_data)

# Filter rows based on values in the "Outcome" column
outcome_1_or_0 = merged_data[(merged_data['Outcome'] == 1) | (merged_data['Outcome'] == 0)]

# Print the filtered rows
print("Rows containing '1' or '0' in the Outcome column:")
print(outcome_1_or_0)

# Select features for clustering (excluding the Outcome column)
features_for_clustering = merged_data.drop('Outcome', axis=1)

# Initialize KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit KMeans to the data
kmeans.fit(features_for_clustering)

# Get cluster labels for each sample
cluster_labels = kmeans.labels_

# Add cluster labels to the merged dataset
merged_data['Cluster'] = cluster_labels

# Display the counts of samples in each cluster
print("Counts of samples in each cluster:")
print(merged_data['Cluster'].value_counts())

# Display the first few rows of the merged dataset with cluster labels
print("\nMerged dataset with cluster labels:")
print(merged_data.head())

# Perform PCA to reduce dimensions to 2
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_for_clustering)

# Create a scatter plot of the data points with cluster labels
plt.figure(figsize=(10, 6))

# Loop through each cluster label and plot the points in that cluster
for cluster_label in merged_data['Cluster'].unique():
    cluster_data = principal_components[cluster_labels == cluster_label] # Indentation added here
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_label}')

plt.title('K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

from scipy.stats import mode

# Determine the majority vote label for each cluster
cluster_labels_predicted = np.zeros_like(cluster_labels)
for cluster in range(3):
    mask = (cluster_labels == cluster)
    mode_result = mode(merged_data['Outcome'][mask])

    if isinstance(mode_result.mode, np.ndarray):
        # If there are multiple modes, choose the smallest one
        majority_label = mode_result.mode[0]
    else:
        majority_label = mode_result.mode
    cluster_labels_predicted[mask] = majority_label

from scipy.stats import mode

# Determine the majority vote label for each cluster
cluster_labels_predicted = np.zeros_like(cluster_labels)
cluster_labels_names = {}  # Dictionary to store cluster labels

for cluster in range(3):
    mask = (cluster_labels == cluster)
    mode_result = mode(merged_data['Outcome'][mask])

    if isinstance(mode_result.mode, np.ndarray):
        # If there are multiple modes, choose the smallest one
        majority_label = mode_result.mode[0]
    else:
        majority_label = mode_result.mode

    cluster_labels_predicted[mask] = majority_label

    # Assign label based on majority outcome in the cluster
    if majority_label == 0:
        cluster_labels_names[cluster] = "Low Risk"
    elif majority_label == 1:
        cluster_labels_names[cluster] = "Moderate Risk"
    else:
        cluster_labels_names[cluster] = "High Risk"

# Now plot the clusters with labels
plt.figure(figsize=(10, 6))

for cluster_label in merged_data['Cluster'].unique():
    cluster_data = principal_components[cluster_labels_predicted == cluster_label]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_label}: {cluster_labels_names[cluster_label]}')

plt.title('K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import silhouette_score

# Compute the silhouette score
silhouette_avg = silhouette_score(features_for_clustering, cluster_labels)
print("Silhouette Score:", silhouette_avg)

from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

# Compute silhouette scores for each sample
silhouette_values = silhouette_samples(features_for_clustering, cluster_labels)

# Create a plot with a distinct color for each cluster
plt.figure(figsize=(10, 6))
y_lower = 10
for i in range(3):  # Assuming 3 clusters
    cluster_silhouette_values = silhouette_values[cluster_labels == i]
    cluster_silhouette_values.sort()
    cluster_size = cluster_silhouette_values.shape[0]
    y_upper = y_lower + cluster_size
    color = cm.nipy_spectral(float(i) / 3)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * cluster_size, str(i))
    y_lower = y_upper + 10

plt.axvline(x=silhouette_avg, color="red", linestyle="--")  # Add silhouette average line
plt.xlabel("Silhouette Coefficient Values")
plt.ylabel("Cluster Label")
plt.title("Silhouette Plot for K-means Clustering")
plt.show()

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
import warnings
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import mode
warnings.simplefilter(action='ignore')
sns.set()
plt.style.use("ggplot")
# %matplotlib inline

# read the dataset from dir
original_data = pd.read_csv("diabetes.csv")
# Separate features and target variable
X = original_data.drop('Outcome', axis=1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit KMeans to the scaled data
kmeans.fit(X_scaled)

# Get cluster labels for each sample
cluster_labels = kmeans.labels_

# Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
plt.title('K-means Clustering')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Calculate silhouette score
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")
# original_data.info()

# Randomize the data values
randomized_data = original_data.apply(np.random.permutation, axis=0)

# Remove the labels from the dataset
unlabeled_randomized_data = randomized_data.drop('Outcome', axis=1)

# Verify the shape of the unlabeled randomized dataset
print("Shape of unlabeled randomized dataset:", unlabeled_randomized_data.shape)

# Now, unlabeled_randomized_data contains the dataset with randomized data values and without labels.

# Randomly select 77 rows from the PIMA Indian dataset
random_pima_data = original_data.sample(n=77, random_state=42)

# Randomly select 681 rows from the fake unlabeled randomized dataset
random_fake_data = unlabeled_randomized_data.sample(n=681, random_state=42)

# Verify the shape of the randomly selected subsets
print("Shape of randomly selected PIMA Indian dataset:", random_pima_data.shape)
print("Shape of randomly selected fake dataset:", random_fake_data.shape)

# Now, random_pima_data and random_fake_data contain the randomly selected rows from their respective datasets.

# Merge both random_pima_data and random_fake_data into a single dataframe
merged_data = pd.concat([random_pima_data, random_fake_data], ignore_index=True)

# Verify the shape of the merged dataframe
print("Shape of merged dataframe:", merged_data.shape)

# Now, merged_data contains both random_pima_data and random_fake_data merged into a single dataframe.

print(merged_data)

# Filter rows based on values in the "Outcome" column
outcome_1_or_0 = merged_data[(merged_data['Outcome'] == 1) | (merged_data['Outcome'] == 0)]

# Print the filtered rows
print("Rows containing '1' or '0' in the Outcome column:")
print(outcome_1_or_0)

# Select features for clustering (excluding the Outcome column)
features_for_clustering = merged_data.drop('Outcome', axis=1)

# Initialize KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit KMeans to the data
kmeans.fit(features_for_clustering)

# Get cluster labels for each sample
cluster_labels = kmeans.labels_

# Add cluster labels to the merged dataset
merged_data['Cluster'] = cluster_labels

# Display the counts of samples in each cluster
print("Counts of samples in each cluster:")
print(merged_data['Cluster'].value_counts())

# Display the first few rows of the merged dataset with cluster labels
print("\nMerged dataset with cluster labels:")
print(merged_data.head())

# Perform PCA to reduce dimensions to 2
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_for_clustering)

# Create a scatter plot of the data points with cluster labels
plt.figure(figsize=(10, 6))

# Loop through each cluster label and plot the points in that cluster
for cluster_label in merged_data['Cluster'].unique():
    cluster_data = principal_components[cluster_labels == cluster_label] # Indentation added here
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_label}')

plt.title('K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

from scipy.stats import mode

# Determine the majority vote label for each cluster
cluster_labels_predicted = np.zeros_like(cluster_labels)
for cluster in range(3):
    mask = (cluster_labels == cluster)
    mode_result = mode(merged_data['Outcome'][mask])

    if isinstance(mode_result.mode, np.ndarray):
        # If there are multiple modes, choose the smallest one
        majority_label = mode_result.mode[0]
    else:
        majority_label = mode_result.mode
    cluster_labels_predicted[mask] = majority_label

from scipy.stats import mode

# Determine the majority vote label for each cluster
cluster_labels_predicted = np.zeros_like(cluster_labels)
cluster_labels_names = {}  # Dictionary to store cluster labels

for cluster in range(3):
    mask = (cluster_labels == cluster)
    mode_result = mode(merged_data['Outcome'][mask])

    if isinstance(mode_result.mode, np.ndarray):
        # If there are multiple modes, choose the smallest one
        majority_label = mode_result.mode[0]
    else:
        majority_label = mode_result.mode

    cluster_labels_predicted[mask] = majority_label

    # Assign label based on majority outcome in the cluster
    if majority_label == 0:
        cluster_labels_names[cluster] = "Low Risk"
    elif majority_label == 1:
        cluster_labels_names[cluster] = "Moderate Risk"
    else:
        cluster_labels_names[cluster] = "High Risk"

# Now plot the clusters with labels
plt.figure(figsize=(10, 6))

for cluster_label in merged_data['Cluster'].unique():
    cluster_data = principal_components[cluster_labels_predicted == cluster_label]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_label}: {cluster_labels_names[cluster_label]}')

plt.title('K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import silhouette_score

# Compute the silhouette score
silhouette_avg = silhouette_score(features_for_clustering, cluster_labels)
print("Silhouette Score:", silhouette_avg)

from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

# Compute silhouette scores for each sample
silhouette_values = silhouette_samples(features_for_clustering, cluster_labels)

# Create a plot with a distinct color for each cluster
plt.figure(figsize=(10, 6))
y_lower = 10
for i in range(3):  # Assuming 3 clusters
    cluster_silhouette_values = silhouette_values[cluster_labels == i]
    cluster_silhouette_values.sort()
    cluster_size = cluster_silhouette_values.shape[0]
    y_upper = y_lower + cluster_size
    color = cm.nipy_spectral(float(i) / 3)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * cluster_size, str(i))
    y_lower = y_upper + 10

plt.axvline(x=silhouette_avg, color="red", linestyle="--")  # Add silhouette average line
plt.xlabel("Silhouette Coefficient Values")
plt.ylabel("Cluster Label")
plt.title("Silhouette Plot for K-means Clustering")
plt.show()

