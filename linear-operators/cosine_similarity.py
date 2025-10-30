"""
COSINE SIMILARITY - MACHINE LEARNING APPLICATION

Theory:
Cosine similarity measures the cosine of the angle between two vectors:
    cos(θ) = (A·B) / (|A||B|)
    
Range: [-1, 1]
- cos(θ) = +1: Vectors point in same direction (θ = 0°)
- cos(θ) = 0: Vectors are orthogonal (θ = 90°)
- cos(θ) = -1: Vectors point in opposite directions (θ = 180°)

Key Property:
Unlike Euclidean distance, cosine similarity is invariant to vector magnitude,
focusing only on orientation. This makes it ideal for comparing patterns
regardless of scale.

Applications in Machine Learning:
- Document Similarity (TF-IDF, word embeddings)
- Recommendation Systems (user-item similarity)
- Image Recognition (feature vector comparison)
- Natural Language Processing (semantic similarity)
- Clustering (K-means with cosine distance)
- Information Retrieval (query-document matching)
- Neural Networks (attention mechanisms, similarity layers)
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Wedge
import matplotlib.patches as mpatches

# Set random seed for reproducibility
np.random.seed(42)

# Define document vectors (simplified TF-IDF-like representation)
# Each dimension represents a word/feature
# Example: [science, tech, sports, politics, health, arts, business]

doc_vectors = {
    'Doc1: AI Research': np.array([5, 4, 0, 0, 1, 0, 2]),
    'Doc2: Machine Learning': np.array([4, 5, 0, 0, 0, 0, 1]),
    'Doc3: Sports News': np.array([0, 0, 5, 1, 2, 0, 0]),
    'Doc4: Tech Startup': np.array([2, 5, 0, 0, 0, 0, 4]),
    'Doc5: Medical AI': np.array([3, 2, 0, 0, 5, 0, 1]),
    'Doc6: Art Review': np.array([0, 0, 0, 0, 0, 5, 1]),
    'Doc7: Political Tech': np.array([1, 3, 0, 5, 0, 0, 2])
}

# User query vector
query = np.array([4, 4, 0, 0, 0, 0, 1])  # Looking for tech/AI content

def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity and related metrics"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return {
            'cosine_sim': 0,
            'angle': 90,
            'dot_product': 0,
            'euclidean_dist': np.linalg.norm(vec1 - vec2)
        }
    
    cosine_sim = dot_product / (norm1 * norm2)
    cosine_sim = np.clip(cosine_sim, -1, 1)  # Numerical stability
    angle_deg = np.degrees(np.arccos(cosine_sim))
    euclidean_dist = np.linalg.norm(vec1 - vec2)
    
    # Similarity interpretation
    if cosine_sim > 0.9:
        similarity_level = "Very High"
        color = "darkgreen"
    elif cosine_sim > 0.7:
        similarity_level = "High"
        color = "green"
    elif cosine_sim > 0.5:
        similarity_level = "Moderate"
        color = "orange"
    elif cosine_sim > 0.2:
        similarity_level = "Low"
        color = "coral"
    else:
        similarity_level = "Very Low"
        color = "red"
    
    return {
        'cosine_sim': cosine_sim,
        'angle': angle_deg,
        'dot_product': dot_product,
        'euclidean_dist': euclidean_dist,
        'similarity_level': similarity_level,
        'color': color,
        'norm1': norm1,
        'norm2': norm2
    }

# Calculate similarities between query and all documents
results = {}
for doc_name, doc_vec in doc_vectors.items():
    results[doc_name] = calculate_cosine_similarity(query, doc_vec)

# Sort by cosine similarity (for ranking)
sorted_results = sorted(results.items(), key=lambda x: x[1]['cosine_sim'], reverse=True)

print("\n" + "="*110)
print("COSINE SIMILARITY - DOT PRODUCT APPLICATION IN MACHINE LEARNING")
print("="*110)
print("\nFormula: cos(θ) = (A·B) / (|A||B|)  where cos(θ) ∈ [-1, 1]")
print("Query Vector (User Interest): [4, 4, 0, 0, 0, 0, 1] → Tech/AI focused")
print("Feature Dimensions: [science, tech, sports, politics, health, arts, business]")
print("-"*110)

# Create comprehensive table
data = {
    "Rank": list(range(1, len(sorted_results) + 1)),
    "Document": [name for name, _ in sorted_results],
    "Vector": [str(doc_vectors[name]) for name, _ in sorted_results],
    "Query·Doc": [f"{result['dot_product']:.2f}" for _, result in sorted_results],
    "Cosine Similarity": [f"{result['cosine_sim']:.4f}" for _, result in sorted_results],
    "Angle θ": [f"{result['angle']:.2f}°" for _, result in sorted_results],
    "Euclidean Dist": [f"{result['euclidean_dist']:.3f}" for _, result in sorted_results],
    "Relevance": [result['similarity_level'] for _, result in sorted_results]
}

df = pd.DataFrame(data)
print(df.to_string(index=False))
print("-"*110)
print("\nKey Insight: Cosine similarity ranks documents by orientation, not magnitude.")
print("             Documents with similar topic distributions score higher.")
print("="*110)

# Visualization
fig = plt.figure(figsize=(16, 11))

# Plot 1: Cosine Similarity Ranking
ax1 = plt.subplot(2, 3, 1)
doc_names_short = [name.split(':')[0] for name, _ in sorted_results]
cosine_vals = [result['cosine_sim'] for _, result in sorted_results]
colors = [result['color'] for _, result in sorted_results]

bars = ax1.barh(doc_names_short, cosine_vals, color=colors, alpha=0.7,
                edgecolor='black', linewidth=2)

# Add value labels
for bar, val in zip(bars, cosine_vals):
    width = bar.get_width()
    ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}',
             ha='left', va='center', fontsize=10, fontweight='bold')

ax1.axvline(x=0, color='black', linewidth=2)
ax1.axvline(x=1, color='green', linewidth=1, linestyle='--', alpha=0.5)
ax1.grid(True, axis='x', linestyle=':', alpha=0.4)
ax1.set_xlabel('Cosine Similarity', fontsize=11, fontweight='bold')
ax1.set_title('Document Ranking by Relevance\n(Higher = More Similar to Query)', 
              fontsize=11, fontweight='bold', pad=10)
ax1.set_xlim(0, 1.1)

# Add legend for colors
legend_elements = [
    mpatches.Patch(color='darkgreen', label='Very High (>0.9)'),
    mpatches.Patch(color='green', label='High (>0.7)'),
    mpatches.Patch(color='orange', label='Moderate (>0.5)'),
    mpatches.Patch(color='coral', label='Low (>0.2)'),
    mpatches.Patch(color='red', label='Very Low (≤0.2)')
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.95)

# Plot 2: 2D Projection of Document Vectors (PCA-like)
ax2 = plt.subplot(2, 3, 2)

# Use first two dimensions (science, tech) for visualization
all_docs = list(doc_vectors.keys())
x_coords = [doc_vectors[doc][0] for doc in all_docs]
y_coords = [doc_vectors[doc][1] for doc in all_docs]

# Plot documents
for i, doc_name in enumerate(all_docs):
    result = results[doc_name]
    ax2.scatter(x_coords[i], y_coords[i], s=200, c=result['color'],
               edgecolors='black', linewidths=2, alpha=0.7, zorder=3)
    ax2.annotate(doc_name.split(':')[0], (x_coords[i], y_coords[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8,
                fontweight='bold')

# Plot query
ax2.scatter(query[0], query[1], s=300, marker='*', c='gold',
           edgecolors='black', linewidths=2.5, label='Query', zorder=5)
ax2.annotate('Query', (query[0], query[1]),
            xytext=(5, -15), textcoords='offset points', fontsize=10,
            fontweight='bold', color='darkred')

# Draw vectors from origin
for i, doc_name in enumerate(all_docs):
    ax2.arrow(0, 0, x_coords[i]*0.9, y_coords[i]*0.9, 
             head_width=0.15, head_length=0.2, fc='gray', ec='gray',
             alpha=0.3, linewidth=1)

ax2.arrow(0, 0, query[0]*0.9, query[1]*0.9,
         head_width=0.2, head_length=0.25, fc='gold', ec='black',
         linewidth=2.5, alpha=0.8, zorder=4)

ax2.set_xlim(-0.5, 6)
ax2.set_ylim(-0.5, 6)
ax2.set_aspect('equal', adjustable='box')
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax2.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
ax2.set_xlabel('Science Content (Feature 1)', fontsize=10)
ax2.set_ylabel('Tech Content (Feature 2)', fontsize=10)
ax2.set_title('Document Space Projection\n(Science vs Tech Dimensions)', 
              fontsize=11, fontweight='bold', pad=10)
ax2.legend(loc='upper left', fontsize=9)

# Plot 3: Angle Comparison
ax3 = plt.subplot(2, 3, 3)
angles = [result['angle'] for _, result in sorted_results]
colors3 = [result['color'] for _, result in sorted_results]

bars3 = ax3.barh(doc_names_short, angles, color=colors3, alpha=0.7,
                 edgecolor='black', linewidth=2)

# Add value labels
for bar, val in zip(bars3, angles):
    width = bar.get_width()
    ax3.text(width + 1, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}°',
             ha='left', va='center', fontsize=10, fontweight='bold')

ax3.axvline(x=0, color='green', linewidth=1.5, linestyle='--', 
            alpha=0.6, label='0° (Identical)')
ax3.axvline(x=90, color='gray', linewidth=1.5, linestyle='--',
            alpha=0.6, label='90° (Orthogonal)')
ax3.grid(True, axis='x', linestyle=':', alpha=0.4)
ax3.set_xlabel('Angle θ (degrees)', fontsize=11, fontweight='bold')
ax3.set_title('Angular Distance from Query\n(Smaller = More Similar)', 
              fontsize=11, fontweight='bold', pad=10)
ax3.set_xlim(0, 100)
ax3.legend(loc='lower right', fontsize=8)

# Plot 4: Dot Product vs Cosine Similarity
ax4 = plt.subplot(2, 3, 4)
dot_products = [result['dot_product'] for _, result in sorted_results]
cosine_sims = [result['cosine_sim'] for _, result in sorted_results]

ax4.scatter(dot_products, cosine_sims, s=200, c=colors, 
           edgecolors='black', linewidths=2, alpha=0.7)

for i, name in enumerate(doc_names_short):
    ax4.annotate(name, (dot_products[i], cosine_sims[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax4.grid(True, linestyle='--', alpha=0.4)
ax4.set_xlabel('Dot Product (Query·Doc)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Cosine Similarity', fontsize=10, fontweight='bold')
ax4.set_title('Dot Product vs Cosine Similarity\n(Magnitude vs Orientation)', 
              fontsize=11, fontweight='bold', pad=10)

# Add explanation text
ax4.text(0.5, 0.05, 'Note: Cosine normalizes for vector magnitude',
         transform=ax4.transAxes, fontsize=9, ha='center', style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Plot 5: Feature Heatmap
ax5 = plt.subplot(2, 3, 5)
feature_names = ['Science', 'Tech', 'Sports', 'Politics', 'Health', 'Arts', 'Business']

# Create matrix for heatmap
matrix = []
row_labels = ['Query'] + [name.split(':')[0] for name in all_docs]
matrix.append(query)
for doc in all_docs:
    matrix.append(doc_vectors[doc])

matrix = np.array(matrix)

im = ax5.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')

# Set ticks
ax5.set_xticks(np.arange(len(feature_names)))
ax5.set_yticks(np.arange(len(row_labels)))
ax5.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
ax5.set_yticklabels(row_labels, fontsize=9)

# Add text annotations
for i in range(len(row_labels)):
    for j in range(len(feature_names)):
        text = ax5.text(j, i, str(int(matrix[i, j])),
                       ha="center", va="center", color="black",
                       fontsize=8, fontweight='bold')

ax5.set_title('Feature Vector Heatmap\n(Document Content Distribution)', 
              fontsize=11, fontweight='bold', pad=10)
cbar = plt.colorbar(im, ax=ax5)
cbar.set_label('Feature Weight', fontsize=9)

# Plot 6: Comparison - Cosine vs Euclidean Distance
ax6 = plt.subplot(2, 3, 6)

# Normalize Euclidean distances to [0, 1] for comparison
euclidean_dists = [result['euclidean_dist'] for _, result in sorted_results]
max_dist = max(euclidean_dists)
euclidean_normalized = [1 - (d / max_dist) for d in euclidean_dists]

x = np.arange(len(doc_names_short))
width = 0.35

bars1 = ax6.bar(x - width/2, cosine_sims, width, label='Cosine Similarity',
                alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
bars2 = ax6.bar(x + width/2, euclidean_normalized, width, 
                label='Euclidean Similarity (normalized)',
                alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)

ax6.set_ylabel('Similarity Score', fontsize=10, fontweight='bold')
ax6.set_xlabel('Documents (ranked by cosine similarity)', fontsize=10, fontweight='bold')
ax6.set_title('Cosine vs Euclidean Distance Comparison\n(Both normalized to [0,1])', 
              fontsize=11, fontweight='bold', pad=10)
ax6.set_xticks(x)
ax6.set_xticklabels(doc_names_short, rotation=45, ha='right', fontsize=8)
ax6.legend(loc='upper right', fontsize=9)
ax6.grid(True, axis='y', linestyle=':', alpha=0.4)
ax6.set_ylim(0, 1.1)

# Add explanation
textstr = 'Cosine focuses on direction\nEuclidean on distance'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax6.text(0.02, 0.98, textstr, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', bbox=props, style='italic')

plt.suptitle('Cosine Similarity via Dot Product (Machine Learning & Information Retrieval)', 
             fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout(rect=(0, 0, 1, 0.98))
plt.show()

print("\n✓ Visualization complete!")
print("\nRecommendation: Based on cosine similarity, the top 3 most relevant documents are:")
for i, (name, result) in enumerate(sorted_results[:3], 1):
    print(f"  {i}. {name} (similarity: {result['cosine_sim']:.4f})")
