import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- Configuration ---
# Change this to the path of your embeddings folder.
embedding_folder = "data/metaworld_generation_data_new/test/embeddings"
# File extension of your embedding files (assuming .npy here)
file_extension = ".npy"

# --- Step 1: Read embeddings ---
embedding_list = []

# Loop through files in the folder
for idx, file_name in enumerate(sorted(os.listdir(embedding_folder))):
    # if idx>=1800 and idx<2400:
        # break
        if file_name.endswith(file_extension):
            file_path = os.path.join(embedding_folder, file_name)
            embedding = np.load(file_path)  # assumes each file stores a single vector
            embedding_list.append(embedding)

# Convert list to a numpy array.
# This assumes each file is a single vector, so embeddings will have shape (N, D)
embeddings = np.array(embedding_list)
print(f"Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}.")

# --- Step 2: Examine Representation Statistics ---

# 2.1 Compute per-dimension variance
variances = np.var(embeddings, axis=0)
print("\nPer-dimension variance:")
# print(variances)
print("Mean variance across dimensions:", np.mean(variances))

print(embeddings.shape)
# 2.2 Compute covariance matrix and its eigenvalues
cov_matrix = np.cov(embeddings, rowvar=False)
eigenvalues = np.linalg.eigvalsh(cov_matrix)
print(eigenvalues.shape)
print("\nEigenvalues of the covariance matrix:")
# print(eigenvalues)
print("Number of eigenvalues near zero (e.g., < 1e-6):", np.sum(eigenvalues < 1e-6))

# 2.3 Compute average cosine similarity (excluding self-similarity)
# Normalize the embeddings
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norm_embeddings = embeddings / (norms + 1e-10)  # add small constant to avoid division by zero

# Compute the cosine similarity matrix
cosine_sim_matrix = np.dot(norm_embeddings, norm_embeddings.T)
print(cosine_sim_matrix.shape)

# Exclude self-similarity (diagonal)
N = embeddings.shape[0]
mask = np.ones((N, N), dtype=bool)
np.fill_diagonal(mask, 0)
average_cosine_similarity = cosine_sim_matrix[mask].mean()
print("\nAverage cosine similarity (excluding self similarity):", average_cosine_similarity)

# --- Step 3: Visualize the Embeddings ---
N1, N2 = 0, len(embeddings)

# 3.1 PCA visualization
# Assuming embeddings is already defined
num_points = len(embeddings)
colors = np.linspace(0, 1, num_points)  # Continuous color scale
color_classes = np.digitize(colors, np.linspace(0, 1, 5))  # Divide into 4 classes
color_classes[-1] = 4

color_classes = color_classes[N1:N2]

# purple: window open
# blue: button press
# green: door unlock
# yellow: pick place

cmap = plt.get_cmap("viridis", 4)
# Generate color values
colors_name = [cmap(i) for i in range(4)]

# Print RGBA values
for i, color_name in enumerate(colors_name):
    print(f"Color {i}: {color_name}")

fig, ax = plt.subplots(figsize=(6, 2))
for i in range(4):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=cmap(i)))

ax.set_xticks(np.arange(4) + 0.5)
ax.set_xticklabels([f"Color {i}" for i in range(4)])
ax.set_yticks([])
ax.set_xlim(0, 4)
ax.set_ylim(0, 1)
plt.savefig('photos_new/color_map.png')

pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(embeddings_pca[N1:N2, 0], embeddings_pca[N1:N2, 1], c=color_classes, cmap=cmap, alpha=0.6, edgecolor='k')
plt.title("PCA Visualization of Embeddings")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
# plt.xlim(-2,11)
# plt.ylim(-2,5)
plt.savefig("photos_new/pca_visualization_2losses.png")

# 3.2 t-SNE visualization
# Note: t-SNE can be slow for large numbers of points.
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(embeddings_tsne[N1:N2, 0], embeddings_tsne[N1:N2, 1], c=color_classes, cmap=cmap, alpha=0.6, edgecolor='k')
plt.title("t-SNE Visualization of Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
# plt.xlim(-70,70)
# plt.ylim(-70,70)
plt.savefig("photos_new/tsne_visualization_2losses.png")
