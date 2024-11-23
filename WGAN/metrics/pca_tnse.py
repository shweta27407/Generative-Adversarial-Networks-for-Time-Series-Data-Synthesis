import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load Real Data
real_data_df = pd.read_csv("data/cnc.csv")  # Replace with the path to your real data CSV
selected_columns = ['f_x_sim', 'f_y_sim', 'f_z_sim', 'f_sp_sim', 'm_sp_sim', 
                    'materialremoved_sim', 'a_x', 'a_y', 'a_z', 'a_sp', 
                    'v_x', 'v_y', 'v_z', 'v_sp', 'pos_x', 'pos_y', 
                    'pos_z', 'pos_sp']
real_data_df = real_data_df[selected_columns]
real_data = real_data_df.values

# Load Synthetic Data
synthetic_data_df = pd.read_csv("synthetic_data.csv")  # Replace with the path to your WGAN synthetic data CSV
synthetic_data_df = synthetic_data_df[selected_columns]
synthetic_data = synthetic_data_df.values

# Combine Real and Synthetic Data
combined_data = np.vstack((real_data, synthetic_data))
labels = np.array([0] * real_data.shape[0] + [1] * synthetic_data.shape[0])

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(combined_data)

# Apply t-SNE
tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(combined_data)

# Plot PCA Results
plt.figure(figsize=(12, 6))
plt.scatter(pca_result[labels == 0, 0], pca_result[labels == 0, 1], alpha=0.5, label='Real Data')
plt.scatter(pca_result[labels == 1, 0], pca_result[labels == 1, 1], alpha=0.5, label='Synthetic Data')
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA of Real and Synthetic Data")
plt.legend()
plt.savefig('pca_plot_wgan.png')  # Save PCA plot
plt.show()

# Plot t-SNE Results
plt.figure(figsize=(12, 6))
plt.scatter(tsne_result[labels == 0, 0], tsne_result[labels == 0, 1], alpha=0.5, label='Real Data')
plt.scatter(tsne_result[labels == 1, 0], tsne_result[labels == 1, 1], alpha=0.5, label='Synthetic Data')
plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")
plt.title("t-SNE of Real and Synthetic Data")
plt.legend()
plt.savefig('tsne_plot_wgan.png')  # Save t-SNE plot
plt.show()
