import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns 

detector = pd.read_csv(r"D:\traccc-data-v9\tml_detector\trackml-detector.csv")
tml_cell_ttbar_mu60 = pd.read_csv(r"d:\traccc-data-v9\tml_full\ttbar_mu60\event000000000-cells.csv") 

print(detector['volume_id'].unique())

#Define mapping of volume_id to module parameters
volume_params = {
    (5, 10): {'module_t':0.0, 'pitch_u':0.0, 'pitch_v':0.0},
    (7, 8, 9): {'module_t':0.15, 'pitch_u':0.05, 'pitch_v':0.05635},
    (12, 13, 14): {'module_t':0.25, 'pitch_u':0.08, 'pitch_v':1.2},
    (16, 18): {'module_t':0.35, 'pitch_u':0.12, 'pitch_v':10.4},
    (17, ): {'module_t':0.35, 'pitch_u':0.12, 'pitch_v':10.8},
}

detector['module_t'] = 0.0
detector['pitch_u'] = 0.0
detector['pitch_v'] = 0.0

#iterate through mapping and update params
for volume_ids, params in volume_params.items():
    detector.loc[detector['volume_id'].isin(volume_ids), 'module_t'] = params ['module_t']
    detector.loc[detector['volume_id'].isin(volume_ids), 'pitch_u'] = params ['pitch_u']
    detector.loc[detector['volume_id'].isin(volume_ids), 'pitch_v'] = params ['pitch_v']

#Merge detector and cell_file DataFrames
merged_df = pd.merge(tml_cell_ttbar_mu60,detector, on= 'geometry_id', how='inner')

def compute_global_coordinates(row):
    #Compute local 2D point in homogenous 3D space
    x_local_3D = np.array ([row['channel0'] * row['pitch_u'],row['channel1']* row['pitch_v'], 0])
    #ROTATION MATRIX
    R = np.array ([
        [row['rot_xu'], row['rot_xv'], row['rot_xw']],
        [row['rot_yu'], row['rot_yv'], row['rot_yw']],
        [row['rot_zu'], row['rot_zv'], row['rot_zw']]
    ])

    # Define translation vector
    t = np.array([row['cx'], row['cy'], row['cz']])

    # Compute global coordinates
    x_global = np.dot(R, x_local_3D) + t
    return pd.Series({'x_global': x_global[0], 'y_global': x_global[1], 'z_global': x_global[2]})
global_coord_df = merged_df.copy()

# Apply the transformation to each row of the DataFrame
global_coord_df[['x_global', 'y_global', 'z_global']] = merged_df.apply(compute_global_coordinates, axis=1)

# Print resulting DataFrame
print(global_coord_df[['x_global', 'y_global', 'z_global']])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot x, y, z
ax.scatter(global_coord_df['x_global'], global_coord_df['y_global'],global_coord_df['z_global'], c='black', label='x, y, z', marker='o', s=3)

# Plot cx, cy, cz
ax.scatter(global_coord_df['cx'], global_coord_df['cy'], global_coord_df['cz'], c='red', label='cx, cy, cz', marker='x', s=12)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot of x, y, z and cx, cy, cz')
ax.legend()
# Save the plot as an image file
plt.savefig('3d_scatter_plot.png')
# plt.show()

print(merged_df['layer_id'].unique())

#parameterize STRIP layers
# Calculate radius
global_coord_df['radius'] = np.sqrt(global_coord_df['cx']**2 + global_coord_df['cy']**2)

# Print resulting DataFrame with radius
print(global_coord_df[['x_global', 'y_global', 'z_global', 'radius', 'cz']])

#make a new_df where global_coord_df['cz'] >= -1160) & (global_coord_df['cz'] <= -1150)) & (global_coord_df['radius'] >= 250 ) & (global_coord_df['radius'] <= 1100)]

strip_df = global_coord_df[(global_coord_df['cz'] >= -1160) & (global_coord_df['cz'] <= 1150) & (global_coord_df['radius'] >= 250) & (global_coord_df['radius'] <= 1100)]

print(strip_df['layer_id'].unique())

strip_df.to_csv('transformedAndMerged_Strip.csv')