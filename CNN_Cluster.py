import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import cKDTree  # Fast nearest neighbor search

def load_sensor_data(sensor_file):
    """ Load sensor positions from a CSV file. """
    df = pd.read_csv(sensor_file)
    sensors = df[['cx', 'cy', 'cz']].values  # Extract sensor positions
    return sensors

def load_hit_data(hit_file):
    """ Load hit positions from a CSV file. """
    df = pd.read_csv(hit_file)
    hits = df[['x_global','y_global','z_global']].values  # Extract hit positions
    return hits

def find_activated_sensors(sensors, hits):
    """ Find the nearest sensor for each hit and mark as activated. """
    tree = cKDTree(sensors)  # Create a KD-tree for fast lookup
    _, sensor_indices = tree.query(hits)  # Find nearest sensors for each hit
    activated = np.unique(sensor_indices)  # Get unique activated sensor indices
    return activated

def convert_to_2d_grid(sensors, activated_sensors, z_bins=100, theta_bins=200):
    """ Convert sensor activation into a 2D grid representation. """
    module_x,module_y,module_z = sensors[:, 0], sensors[:, 1], sensors[:, 2]

    # Compute θ (circumferential angle)
    theta = np.arctan2(module_y,module_x)
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)  # Normalize to [0, 2π]

    # Define bin edges
    z_min, z_max = module_z.min(), module_z.max()
    z_idx = ((module_z-z_min) / (z_max - z_min) * (z_bins - 1)).astype(int)
    theta_idx = ((theta - 0) / (2 * np.pi) * (theta_bins - 1)).astype(int)

    # Initialize 2D grid
    grid = np.zeros((z_bins, theta_bins))

    for sensor_id in range(len(sensors)):
        zi, ti = z_idx[sensor_id], theta_idx[sensor_id]
        grid[zi, ti] = 0.3  # Light color for all sensors

    # Mark activated sensors in the grid
    for sensor_id in activated_sensors:
        zi, ti = z_idx[sensor_id], theta_idx[sensor_id]
        grid[zi, ti] = 1  # Mark as activated (binary activation) (this is 1 but decrease for light colour)

    return grid, z_idx, theta_idx

def visualize_grid(grid):
    """ Display the 2D grid as an activation map. """
    plt.figure(figsize=(10, 6))
    plt.imshow(grid, cmap="hot", aspect="auto", extent=[0, 360, 0, 100])
    plt.colorbar(label="Activation")
    plt.xlabel("Circumference (θ bins) [degrees]")
    plt.ylabel("Length (z bins)")
    plt.title("Activated Sensor Map")
    plt.show()

# === Define CNN Model ===
class CNNClusterDetector(nn.Module):
    def __init__(self):
        super(CNNClusterDetector, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))  # Output probability of activation
        return x

# === Detect Clusters ===
def extract_clusters(grid, min_hits=2):
    clusters = []
    visited = np.zeros_like(grid)

    def dfs(i, j, cluster):
        if i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1]:
            return
        if grid[i, j] == 0 or visited[i, j] == 1:
            return
        visited[i, j] = 1
        cluster.append((i, j))
        dfs(i + 1, j, cluster)
        dfs(i - 1, j, cluster)
        dfs(i, j + 1, cluster)
        dfs(i, j - 1, cluster)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1 and visited[i, j] == 0:
                cluster = []
                dfs(i, j, cluster)
                if len(cluster) >= min_hits:
                    clusters.append(cluster)

    return clusters

# === Visualize Clusters ===
def visualize_clusters(grid, clusters):
    plt.figure(figsize=(12, 6))

    # Background sensors in light gray
    plt.imshow(grid, cmap="gray", aspect="auto", extent=[0, 360, 0, 100])

    # Plot clusters in different colors
    cluster_colors = sns.color_palette("husl", len(clusters))  # Generate distinct colors

    for i, cluster in enumerate(clusters):
        cluster_x = [point[1] for point in cluster]  # Theta bins
        cluster_y = [point[0] for point in cluster]  # Z bins
        plt.scatter(cluster_x, cluster_y, color=cluster_colors[i], label=f"Cluster {i+1}")

    plt.colorbar(label="Activation Intensity")
    plt.xlabel("Circumference (θ bins) [degrees]")
    plt.ylabel("Length (z bins)")
    plt.title("Clustered Activated Sensors")
    plt.legend()
    plt.show()
    
# === RUN PIPELINE ===
file_path = r"D:\traccc-data-v9\tml_detector\transformedAndMerged_Strip.csv"

# Load Data
sensors = load_sensor_data(file_path)
hits = load_hit_data(file_path)

# Find Activated Sensors
activated_sensors = find_activated_sensors(sensors, hits)

# Convert to 2D Grid
grid = convert_to_2d_grid(sensors, activated_sensors)

# Convert Grid to Tensor for CNN
grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W) 
# Apply CNN Model
model = CNNClusterDetector()
output_grid = model(grid_tensor).detach().numpy().squeeze()

# Extract Clusters
clusters = extract_clusters(grid)

# Print Clusters
print(f"Detected {len(clusters)} clusters with at least 2 hits.")
for idx, cluster in enumerate(clusters):
    print(f"Cluster {idx + 1}: {cluster}")

# Visualize Grid
visualize_grid(grid)

# def plot_sensors_vs_activated(sensors, activated_sensors):
#     """Plot original sensors vs. activated bins."""
#     x, y, z = sensors[:, 0], sensors[:, 1], sensors[:, 2]
#     theta = np.arctan2(y, x)
#     theta = np.where(theta < 0, theta + 2 * np.pi, theta)  # Normalize to [0, 2π]

#     z_activated = z[activated_sensors]
#     theta_activated = theta[activated_sensors]

#     plt.figure(figsize=(10, 6))
#     plt.scatter(np.degrees(theta), z, color="darkgray", s=5, label="All Sensors")  # Original Sensors
#     plt.scatter(np.degrees(theta_activated), z_activated, color="red", s=10, label="Activated Sensors")  # Activated Sensors

#     plt.xlabel("Circumference (θ bins) [degrees]")
#     plt.ylabel("Length (z bins)")
#     plt.title("Sensor Activation Map")
#     plt.legend()
#     plt.grid(True, linestyle="--", linewidth=0.5)
#     plt.savefig('Activated vs Sensors.png')
#     plt.show()

# # === Run Visualization ===
# plot_sensors_vs_activated(sensors, activated_sensors)

