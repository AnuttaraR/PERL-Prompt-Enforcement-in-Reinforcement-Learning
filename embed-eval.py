import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load your Excel file
file_path = "D:/output.xlsx"  # Replace with your actual file path
xls = pd.ExcelFile(file_path)

# Load the relevant sheets
gte_data = pd.read_excel(xls, sheet_name='GTE-Base Results')
bge_data = pd.read_excel(xls, sheet_name='BGE-Base Results')
e5_data = pd.read_excel(xls, sheet_name='E5-Base Results')

# Clean data and extract chunk metadata
gte_chunks = gte_data['Chunk Metadata'].dropna().astype(str).tolist()
bge_chunks = bge_data['Chunk Metadata'].dropna().astype(str).tolist()
e5_chunks = e5_data['Chunk Metadata'].dropna().astype(str).tolist()

# Vectorizing the chunks using TF-IDF (term frequency-inverse document frequency)
vectorizer = TfidfVectorizer()

# Transform the text chunks into vectors
gte_vectors = vectorizer.fit_transform(gte_chunks)
bge_vectors = vectorizer.fit_transform(bge_chunks)
e5_vectors = vectorizer.fit_transform(e5_chunks)


# Function to perform K-means clustering
def perform_clustering(vectors, chunks, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    clusters = kmeans.labels_

    # Group chunks by their assigned cluster
    categorized_chunks = {i: [] for i in range(num_clusters)}
    for idx, cluster_id in enumerate(clusters):
        categorized_chunks[cluster_id].append(chunks[idx])

    return categorized_chunks


# Performing clustering on each model's chunks
gte_clusters = perform_clustering(gte_vectors, gte_chunks)
bge_clusters = perform_clustering(bge_vectors, bge_chunks)
e5_clusters = perform_clustering(e5_vectors, e5_chunks)

# Output the clustering results for each model
print("GTE-Base Clusters:")
for cluster_id, chunks in gte_clusters.items():
    print(f"Cluster {cluster_id}: {chunks[:3]}...")  # Displaying first 3 chunks per cluster for brevity

print("\nBGE-Base Clusters:")
for cluster_id, chunks in bge_clusters.items():
    print(f"Cluster {cluster_id}: {chunks[:3]}...")

print("\nE5-Base Clusters:")
for cluster_id, chunks in e5_clusters.items():
    print(f"Cluster {cluster_id}: {chunks[:3]}...")


# Optionally, visualize the number of chunks per cluster
def plot_clusters(clusters, model_name):
    cluster_sizes = [len(chunks) for chunks in clusters.values()]
    plt.bar(range(len(clusters)), cluster_sizes)
    plt.title(f'Cluster Sizes for {model_name}')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Chunks')
    plt.show()


plot_clusters(gte_clusters, 'GTE-Base')
plot_clusters(bge_clusters, 'BGE-Base')
plot_clusters(e5_clusters, 'E5-Base')
