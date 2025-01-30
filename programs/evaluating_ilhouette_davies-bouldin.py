import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
import pandas as pd
import umap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import warnings  # Importar módulo de warnings

# Filtrar advertencias específicas de UMAP
warnings.filterwarnings("ignore", category=UserWarning, module="umap")


# Function to calculate color indices
def calculate_earnings(df, index_pairs):
    for index_pair in index_pairs:
        color_index_name = f"{index_pair[0]} - {index_pair[1]}"
        df.loc[:, color_index_name] = df[index_pair[0]] - df[index_pair[1]]
    return df

# Read the data
combined_df = pd.read_csv("Ha-emitters/Halpha_Mine_PerField_total-unique_wise.csv")

# Filter out rows with errors in measurements
m_err_splus = (combined_df["e_r_PStotal"] <= 0.2) & (combined_df["e_g_PStotal"] <= 0.2) & \
        (combined_df["e_i_PStotal"] <= 0.2) & (combined_df["e_u_PStotal"] <= 0.2) & \
        (combined_df["e_J0378_PStotal"] <= 0.2) & (combined_df["e_J0395_PStotal"] <= 0.2) & \
        (combined_df["e_J0410_PStotal"] <= 0.2) & (combined_df["e_J0430_PStotal"] <= 0.2) & \
        (combined_df["e_J0515_PStotal"] <= 0.2) & (combined_df["e_J0660_PStotal"] <= 0.2) & \
        (combined_df["e_J0861_PStotal"] <= 0.2) & (combined_df["e_z_PStotal"] <= 0.2)

# Choose a threshold for the maximum allowed magnitude error
max_allowed_e_Wmag = 0.5  # Example threshold value

# Apply the threshold to filter the dataset

m_err_wise = (combined_df["e_W1mag"] <= max_allowed_e_Wmag) & \
              (combined_df["e_W2mag"] <= max_allowed_e_Wmag) 
        

mask_total = (m_err_splus & m_err_wise)

df_cleanErr = combined_df[mask_total]
print("Number of final objects:", len(df_cleanErr))

# Selecting columns with magnitude values
columns = ["r_PStotal",
           "g_PStotal",
           "i_PStotal",
           "u_PStotal",
           "z_PStotal",
           "J0378_PStotal",
           "J0395_PStotal",
           "J0410_PStotal",
           "J0430_PStotal",
           "J0515_PStotal",
           "J0660_PStotal",
           "J0861_PStotal"]

df_mag = df_cleanErr[columns]
print("Look like the new table:", df_mag.head())

# Generate all combinations of magnitude columns
color_index_pairs = list(combinations(df_mag, 2))
print("Numbers of colors:", len(color_index_pairs))

df_colors_mag = calculate_earnings(df_mag, color_index_pairs)
print("Look likes the new table,", df_colors_mag.head())


# Drop magnitude columns, keeping only color indices
df_colors = df_colors_mag.drop(columns=columns)
print("Look likes the colors:", df_colors.head())

# Calculate differences between W1 and each magnitude
for col in ["r_PStotal", "g_PStotal", "i_PStotal", "u_PStotal", "z_PStotal"]:
    df_colors[f'diff_W1_{col}'] = df_cleanErr["W1mag"] - df_cleanErr[col]

# Calculate differences between W2 and each magnitude
for col in ["r_PStotal", "g_PStotal", "i_PStotal", "u_PStotal", "z_PStotal"]:
    df_colors[f'diff_W2_{col}'] = df_cleanErr["W2mag"] - df_cleanErr[col]

# Calculate difference between W1 and W2
df_colors['diff_W1_W2'] = df_cleanErr['W1mag'] - df_cleanErr['W2mag']

print(df_colors)

# Standardizing the data
X_stand = StandardScaler().fit_transform(df_colors)

# Split data into training and validation sets
X_train, X_val = train_test_split(X_stand, test_size=0.2, random_state=42)

# Define a range of number of components to try
n_components_range = [2, 3, 4, 5, 10, 20, 50]
n_neighbors_range = [5, 10, 15, 20, 30, 50, 70, 100]

# Initialize best scores and parameters
best_silhouette_score = -1
best_davies_bouldin_score = float('inf')
best_num_components = None
best_n_neighbors = None
best_labels = None

# Set a fixed random state for KMeans reproducibility
random_state = 42

# Initialize a list to store results for plotting
results = []

# Loop over different numbers of components and neighbors
for num_components in n_components_range:
    for n_neighbors in n_neighbors_range:
        # Fit UMAP model without random_state for parallelism
        reducer_ = umap.UMAP(n_neighbors=n_neighbors, n_components=num_components)
        X_train_trans = reducer_.fit_transform(X_train)

        # Cluster the transformed data using KMeans
        kmeans = KMeans(n_clusters=num_components, random_state=random_state)
        labels = kmeans.fit_predict(X_train_trans)

        # Evaluate performance using Silhouette Score and Davies-Bouldin Index
        silhouette = silhouette_score(X_train_trans, labels)
        davies_bouldin = davies_bouldin_score(X_train_trans, labels)
        print(f"Components: {num_components}, Neighbors: {n_neighbors}, Silhouette Score: {silhouette}, DB Index: {davies_bouldin}")

        # Store the results
        results.append({
            'n_components': num_components,
            'n_neighbors': n_neighbors,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin
        })

        # Update best parameters based on combined metrics
        if silhouette > best_silhouette_score and davies_bouldin < best_davies_bouldin_score:
            best_silhouette_score = silhouette
            best_davies_bouldin_score = davies_bouldin
            best_num_components = num_components
            best_n_neighbors = n_neighbors
            best_labels = labels

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Print the best parameters
print(f"Best Silhouette Score: {best_silhouette_score}")
print(f"Best Davies-Bouldin Index: {best_davies_bouldin_score}")
print(f"Best Number of Components: {best_num_components}")
print(f"Best Number of Neighbors: {best_n_neighbors}")

# Plot the results using seaborn
sns.set(style="ticks")  # Use a more appropriate style for A&A
plt.rc('axes', grid=False)  # Ensure that grid is turned off

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Panel 1: Silhouette Score vs n_neighbors for different n_components
for n_components in n_components_range:
    subset = results_df[results_df['n_components'] == n_components]
    sns.lineplot(x='n_neighbors', y='silhouette_score', data=subset, ax=axes[0], marker='o', label=f'n_components={n_components}', legend=False, markersize=9)
axes[0].set_xlabel('Number of neighbors (n_neighbors)', fontsize=17)
axes[0].set_ylabel('Silhouette Score', fontsize=17)
axes[0].tick_params(axis='x', labelsize=17)
axes[0].tick_params(axis='y', labelsize=17)
axes[0].set_ylim(0.25, 0.85)

# Panel 2: Davies-Bouldin Index vs n_neighbors for different n_components
for n_components in n_components_range:
    subset = results_df[results_df['n_components'] == n_components]
    sns.lineplot(x='n_neighbors', y='davies_bouldin_score', data=subset, ax=axes[1], marker='o', label=f'n_components={n_components}', legend=False, markersize=9)
axes[1].set_xlabel('Number of neighbors (n_neighbors)', fontsize=17)
axes[1].set_ylabel('Davies-Bouldin Index', fontsize=17)
axes[1].tick_params(axis='x', labelsize=17)
axes[1].tick_params(axis='y', labelsize=17)
axes[1].set_ylim(0.1, max(results_df['davies_bouldin_score']) + 0.1)

# Create a single legend outside the subplots
handles, labels = axes[0].get_legend_handles_labels()

# Adjust the legend position and ensure it appears
kw = dict(ncol=4, loc="upper center", frameon=False, fontsize=17)
leg1 = fig.legend(handles[:4], labels[:4], bbox_to_anchor=(0.5, 1.00), **kw)
leg2 = fig.legend(handles[4:], labels[4:], bbox_to_anchor=(0.5, 0.95), **kw)
fig.add_artist(leg1)

plt.tight_layout(rect=[0, 0, 1, 0.90])  # Adjust layout to make space for the legend
#plt.savefig("Figs/Silhouette_Score_Davies-Bouldin_wise.pdf")
