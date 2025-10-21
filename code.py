# Premier League Clustering Analysis
# K-means vs AGNES Comparison

# Step 1: Install required packages and import libraries
!pip install kaggle seaborn plotly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Step 2: Upload and load the dataset
print("DATASET UPLOAD - Choose one of the methods below:")
print("="*60)

# File upload widget for Google Colab
from google.colab import files
import io

def load_csv_with_encoding(file_data, filename):
    """Try different encodings to load CSV file"""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']

    for encoding in encodings:
        try:
            print(f"Trying encoding: {encoding}")
            df = pd.read_csv(io.BytesIO(file_data), encoding=encoding)
            print(f"✅ Successfully loaded with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            print(f"❌ Failed with {encoding}")
            continue
        except Exception as e:
            print(f"❌ Error with {encoding}: {str(e)[:50]}...")
            continue

    # If all encodings fail, try with error handling
    try:
        print("Trying with error handling (replacing invalid characters)...")
        df = pd.read_csv(io.BytesIO(file_data), encoding='utf-8', errors='replace')
        print("✅ Loaded with character replacement")
        return df
    except Exception as e:
        raise Exception(f"Could not load file with any encoding method: {str(e)}")

# METHOD 1: Direct Upload
print("\n🔄 METHOD 1: Direct File Upload")
print("Click 'Choose Files' below to upload your EPL CSV file:")

uploaded = files.upload()

if uploaded:
    # Get the uploaded file name
    filename = list(uploaded.keys())[0]
    print(f"Uploaded file: {filename}")

    try:
        print("Loading dataset...")
        df = load_csv_with_encoding(uploaded[filename], filename)
        upload_success = True
    except Exception as e:
        print(f"❌ Direct upload failed: {str(e)}")
        upload_success = False
        df = None
else:
    print("No file uploaded via direct method.")
    upload_success = False
    df = None

# METHOD 2: Manual Upload (if direct upload failed)
if not upload_success or df is None:
    print("\n📁 METHOD 2: Manual File Browser Upload")
    print("Follow these steps:")
    print("1. Click the 'Files' icon 📁 on the left sidebar")
    print("2. Click 'Upload to session storage' button")
    print("3. Select your CSV file")
    print("4. Enter the filename below:")

    # Manual filename input
    manual_filename = input("Enter your uploaded filename (e.g., 'EPL_20_21.csv'): ").strip()

    if manual_filename:
        # Try different encodings for manual upload
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None

        for encoding in encodings:
            try:
                print(f"Trying to load '{manual_filename}' with {encoding} encoding...")
                df = pd.read_csv(manual_filename, encoding=encoding)
                print(f"✅ Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                print(f"❌ File '{manual_filename}' not found. Please check the filename.")
                break
            except Exception as e:
                print(f"❌ Error loading file: {str(e)}")
                continue

        # Last resort: try with error replacement
        if df is None and manual_filename:
            try:
                df = pd.read_csv(manual_filename, encoding='utf-8', errors='replace')
                print("✅ Loaded with character replacement")
            except Exception as e:
                print(f"❌ Could not load file: {str(e)}")
                df = None

# Check if dataset was loaded successfully
if df is not None and not df.empty:

    print("✅ Dataset loaded successfully!")
    print("Dataset Shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())
else:
    print("❌ Failed to load dataset. Please try the following:")
    print("1. Make sure your file is a valid CSV")
    print("2. Try downloading a different EPL dataset")
    print("3. Check that the file isn't corrupted")
    print("4. Try converting your file to UTF-8 encoding first")

    # Provide sample data structure for reference
    print("\n📋 Expected CSV structure:")
    print("Columns should include: HomeTeam, AwayTeam, FTHG, FTAG, FTR")
    print("Example row: Arsenal,Chelsea,2,1,H")

    # Stop execution if no data
    raise SystemExit("Please fix the data loading issue and run again.")

# Step 3: Data Exploration and Preprocessing
print("\nDataset Description:")
print(df.describe())

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Step 4: Feature Engineering for Team Performance Analysis
# We'll create team-level statistics for clustering
def create_team_stats(df):
    """Create comprehensive team statistics"""

    # Home team stats
    home_stats = df.groupby('HomeTeam').agg({
        'FTHG': ['mean', 'sum'],  # Goals scored at home
        'FTAG': ['mean', 'sum'],  # Goals conceded at home
        'FTR': lambda x: (x == 'H').sum()  # Home wins
    }).reset_index()

    home_stats.columns = ['Team', 'Home_Goals_Per_Game', 'Total_Home_Goals',
                         'Home_Goals_Conceded_Per_Game', 'Total_Home_Goals_Conceded', 'Home_Wins']

    # Away team stats
    away_stats = df.groupby('AwayTeam').agg({
        'FTAG': ['mean', 'sum'],  # Goals scored away
        'FTHG': ['mean', 'sum'],  # Goals conceded away
        'FTR': lambda x: (x == 'A').sum()  # Away wins
    }).reset_index()

    away_stats.columns = ['Team', 'Away_Goals_Per_Game', 'Total_Away_Goals',
                         'Away_Goals_Conceded_Per_Game', 'Total_Away_Goals_Conceded', 'Away_Wins']

    # Merge home and away stats
    team_stats = pd.merge(home_stats, away_stats, on='Team', how='outer')

    # Calculate overall statistics
    team_stats['Total_Goals_Scored'] = team_stats['Total_Home_Goals'] + team_stats['Total_Away_Goals']
    team_stats['Total_Goals_Conceded'] = team_stats['Total_Home_Goals_Conceded'] + team_stats['Total_Away_Goals_Conceded']
    team_stats['Goal_Difference'] = team_stats['Total_Goals_Scored'] - team_stats['Total_Goals_Conceded']
    team_stats['Total_Wins'] = team_stats['Home_Wins'] + team_stats['Away_Wins']
    team_stats['Goals_Per_Game'] = (team_stats['Total_Goals_Scored'] / 38)  # 38 games per season
    team_stats['Goals_Conceded_Per_Game'] = (team_stats['Total_Goals_Conceded'] / 38)
    team_stats['Win_Rate'] = team_stats['Total_Wins'] / 38

    # Calculate draws and losses
    home_draws = df.groupby('HomeTeam')['FTR'].apply(lambda x: (x == 'D').sum()).reset_index()
    home_draws.columns = ['Team', 'Home_Draws']
    away_draws = df.groupby('AwayTeam')['FTR'].apply(lambda x: (x == 'D').sum()).reset_index()
    away_draws.columns = ['Team', 'Away_Draws']

    team_stats = pd.merge(team_stats, home_draws, on='Team', how='left')
    team_stats = pd.merge(team_stats, away_draws, on='Team', how='left')
    team_stats['Total_Draws'] = team_stats['Home_Draws'] + team_stats['Away_Draws']
    team_stats['Total_Losses'] = 38 - team_stats['Total_Wins'] - team_stats['Total_Draws']

    return team_stats

# Create team statistics
team_stats = create_team_stats(df)
print("\nTeam Statistics:")
print(team_stats.head())

# Step 5: Select features for clustering
clustering_features = [
    'Goals_Per_Game',
    'Goals_Conceded_Per_Game',
    'Goal_Difference',
    'Win_Rate',
    'Total_Wins',
    'Total_Draws',
    'Total_Losses'
]

# Prepare data for clustering
X = team_stats[clustering_features].fillna(0)
team_names = team_stats['Team'].values

print(f"\nClustering features shape: {X.shape}")
print(f"Features used: {clustering_features}")

# Step 6: Data Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Scaled data shape: {X_scaled.shape}")

# Step 7: K-means Clustering Analysis

# Find optimal number of clusters using silhouette analysis only
def find_optimal_clusters(X, max_k=10):
    """Find optimal number of clusters using silhouette score"""

    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    return k_range, silhouette_scores

k_range, sil_scores = find_optimal_clusters(X_scaled)

# Plot silhouette score only
plt.figure(figsize=(10, 6))
plt.plot(k_range, sil_scores, 'ro-', linewidth=2, markersize=8)
plt.title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(k_range)
plt.tight_layout()
plt.show()

# Find optimal k
optimal_k_sil = k_range[np.argmax(sil_scores)]

print(f"\nOptimal k based on Silhouette Score: {optimal_k_sil}")

# Use the optimal k or default to 4 for interpretability
optimal_k = 4  # You can adjust based on the metrics above

# Step 8: Apply K-means Clustering
print(f"\nApplying K-means with k={optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Calculate clustering metrics for K-means
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

print(f"K-means Clustering Metrics:")
print(f"Silhouette Score: {kmeans_silhouette:.4f}")

# Step 9: AGNES (Agglomerative Hierarchical Clustering)
print(f"\nApplying AGNES with n_clusters={optimal_k}")

# Create linkage matrix for dendrogram
linkage_matrix = linkage(X_scaled, method='ward')

# Apply AGNES
agnes = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
agnes_labels = agnes.fit_predict(X_scaled)

# Calculate clustering metrics for AGNES
agnes_silhouette = silhouette_score(X_scaled, agnes_labels)

print(f"AGNES Clustering Metrics:")
print(f"Silhouette Score: {agnes_silhouette:.4f}")

# Step 10: Create Dendrogram
plt.figure(figsize=(15, 8))
dendrogram(linkage_matrix,
           labels=team_names,
           leaf_rotation=90,
           leaf_font_size=10)
plt.title('AGNES Dendrogram - Premier League Teams Clustering', fontsize=16)
plt.xlabel('Teams', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.tight_layout()
plt.show()

# Step 11: Visualize Clustering Results
# Add cluster labels to original data
team_stats['KMeans_Cluster'] = kmeans_labels
team_stats['AGNES_Cluster'] = agnes_labels

# Create comparison dataframe
cluster_comparison = pd.DataFrame({
    'Team': team_names,
    'KMeans_Cluster': kmeans_labels,
    'AGNES_Cluster': agnes_labels
})

print("\nCluster Assignments Comparison:")
print(cluster_comparison.sort_values('Team'))

# Step 12: Analyze Cluster Characteristics
def analyze_clusters(data, cluster_column, features):
    """Analyze characteristics of each cluster"""

    cluster_stats = data.groupby(cluster_column)[features].agg(['mean', 'std', 'count']).round(3)
    return cluster_stats

print("\n" + "="*80)
print("K-MEANS CLUSTER ANALYSIS")
print("="*80)
kmeans_cluster_stats = analyze_clusters(team_stats, 'KMeans_Cluster', clustering_features)
print(kmeans_cluster_stats)

print("\n" + "="*80)
print("AGNES CLUSTER ANALYSIS")
print("="*80)
agnes_cluster_stats = analyze_clusters(team_stats, 'AGNES_Cluster', clustering_features)
print(agnes_cluster_stats)

# Step 13: Visualize clusters in 2D using PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create subplots for comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# K-means visualization
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                         c=kmeans_labels, cmap='viridis', alpha=0.7, s=100)
axes[0].set_title(f'K-means Clustering (k={optimal_k})\nSilhouette Score: {kmeans_silhouette:.3f}')
axes[0].set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
axes[0].set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')

# Add team labels
for i, team in enumerate(team_names):
    axes[0].annotate(team, (X_pca[i, 0], X_pca[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)

# AGNES visualization
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                         c=agnes_labels, cmap='viridis', alpha=0.7, s=100)
axes[1].set_title(f'AGNES Clustering (k={optimal_k})\nSilhouette Score: {agnes_silhouette:.3f}')
axes[1].set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
axes[1].set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')

# Add team labels
for i, team in enumerate(team_names):
    axes[1].annotate(team, (X_pca[i, 0], X_pca[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)

plt.tight_layout()
plt.show()

# Step 14: Create detailed cluster profiles
def create_cluster_profiles(data, cluster_column):
    """Create detailed profiles for each cluster"""

    profiles = {}
    for cluster in sorted(data[cluster_column].unique()):
        cluster_data = data[data[cluster_column] == cluster]
        teams = cluster_data['Team'].tolist()

        profile = {
            'Teams': teams,
            'Count': len(teams),
            'Avg_Goals_Per_Game': cluster_data['Goals_Per_Game'].mean(),
            'Avg_Goals_Conceded': cluster_data['Goals_Conceded_Per_Game'].mean(),
            'Avg_Goal_Difference': cluster_data['Goal_Difference'].mean(),
            'Avg_Win_Rate': cluster_data['Win_Rate'].mean(),
            'Avg_Wins': cluster_data['Total_Wins'].mean()
        }
        profiles[f'Cluster_{cluster}'] = profile

    return profiles

print("\n" + "="*80)
print("DETAILED K-MEANS CLUSTER PROFILES")
print("="*80)
kmeans_profiles = create_cluster_profiles(team_stats, 'KMeans_Cluster')
for cluster, profile in kmeans_profiles.items():
    print(f"\n{cluster}:")
    print(f"  Teams ({profile['Count']}): {', '.join(profile['Teams'])}")
    print(f"  Avg Goals/Game: {profile['Avg_Goals_Per_Game']:.2f}")
    print(f"  Avg Goals Conceded/Game: {profile['Avg_Goals_Conceded']:.2f}")
    print(f"  Avg Goal Difference: {profile['Avg_Goal_Difference']:.2f}")
    print(f"  Avg Win Rate: {profile['Avg_Win_Rate']:.2%}")
    print(f"  Avg Wins: {profile['Avg_Wins']:.1f}")

print("\n" + "="*80)
print("DETAILED AGNES CLUSTER PROFILES")
print("="*80)
agnes_profiles = create_cluster_profiles(team_stats, 'AGNES_Cluster')
for cluster, profile in agnes_profiles.items():
    print(f"\n{cluster}:")
    print(f"  Teams ({profile['Count']}): {', '.join(profile['Teams'])}")
    print(f"  Avg Goals/Game: {profile['Avg_Goals_Per_Game']:.2f}")
    print(f"  Avg Goals Conceded/Game: {profile['Avg_Goals_Conceded']:.2f}")
    print(f"  Avg Goal Difference: {profile['Avg_Goal_Difference']:.2f}")
    print(f"  Avg Win Rate: {profile['Avg_Win_Rate']:.2%}")
    print(f"  Avg Wins: {profile['Avg_Wins']:.1f}")

# Step 15: Model Comparison Summary
print("\n" + "="*80)
print("CLUSTERING METHODS COMPARISON SUMMARY")
print("="*80)

comparison_metrics = pd.DataFrame({
    'Metric': ['Silhouette Score'],
    'K-means': [kmeans_silhouette],
    'AGNES': [agnes_silhouette],
    'Better Method': ['K-means' if kmeans_silhouette > agnes_silhouette else 'AGNES']
})

print(comparison_metrics.to_string(index=False))

# Calculate cluster agreement
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari_score = adjusted_rand_score(kmeans_labels, agnes_labels)
nmi_score = normalized_mutual_info_score(kmeans_labels, agnes_labels)

print(f"\nCluster Agreement Metrics:")
print(f"Adjusted Rand Index: {ari_score:.4f}")
print(f"Normalized Mutual Information: {nmi_score:.4f}")

# Step 16: Save results
results_df = team_stats[['Team'] + clustering_features + ['KMeans_Cluster', 'AGNES_Cluster']]
results_df.to_csv('clustering_results.csv', index=False)
print(f"\nResults saved to 'clustering_results.csv'")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
