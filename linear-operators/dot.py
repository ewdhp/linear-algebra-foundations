"""
VECTOR OPERATIONS AND PROPERTIES - THEORETICAL BACKGROUND

This script visualizes fundamental vector operations and their geometric interpretations
through four comprehensive subplots:

1. DOT PRODUCT AND PROJECTION
   Theory: The dot product (inner product) A·B = |A||B|cos(θ) measures the similarity
   between two vectors. Geometrically, it represents the projection of one vector onto
   another scaled by the magnitude of the second vector. The angle θ between vectors
   can be computed as θ = arccos(A·B / (|A||B|)). A positive dot product indicates
   vectors pointing in similar directions (θ < 90°), zero indicates orthogonality (θ = 90°),
   and negative indicates opposite directions (θ > 90°).
   
   Key Concepts:
   - Projection formula: proj_B(A) = (A·B̂)B̂ where B̂ = B/|B| is the unit vector
   - Dot product properties: commutative (A·B = B·A), distributive, and linear
   - Applications: Work in physics (W = F·d), cosine similarity in ML

2. VECTOR ADDITION AND SUBTRACTION
   Theory: Vector addition follows the parallelogram law (or triangle rule). The sum
   A + B represents the diagonal of a parallelogram with sides A and B. Vector
   subtraction A - B can be visualized as A + (-B), representing the vector from the
   tip of B to the tip of A when both start at the origin.
   
   Key Concepts:
   - Commutative: A + B = B + A
   - Associative: (A + B) + C = A + (B + C)
   - Triangle inequality: |A + B| ≤ |A| + |B|
   - Applications: Force composition, displacement in physics, gradient descent in optimization

3. UNIT VECTORS AND MAGNITUDES
   Theory: The magnitude (or norm) of a vector |A| = √(A·A) represents its length.
   A unit vector Â = A/|A| has magnitude 1 and points in the same direction as A.
   Unit vectors are fundamental for defining directions independently of scale.
   The circles drawn represent the locus of all points at distance |A| and |B| from
   the origin, illustrating the concept of vector magnitude geometrically.
   
   Key Concepts:
   - Normalization: converting any non-zero vector to unit length
   - Magnitude properties: |cA| = |c||A| for scalar c, |A| ≥ 0
   - Euclidean norm: |A| = √(Σ aᵢ²) in n dimensions
   - Applications: Direction fields, normalized features in ML, basis vectors

4. CROSS PRODUCT AREA AND ORTHOGONAL DECOMPOSITION
   Theory: In 2D, the cross product magnitude |A × B| = |A||B|sin(θ) equals the area
   of the parallelogram formed by vectors A and B. Any vector A can be uniquely
   decomposed into components parallel and perpendicular to another vector B:
   A = A_∥ + A_⊥, where A_∥ = proj_B(A) and A_⊥ = A - A_∥.
   
   Key Concepts:
   - 2D cross product: scalar value A_x·B_y - A_y·B_x
   - Orthogonal decomposition: fundamental for projections and Gram-Schmidt process
   - Area interpretation: |A × B| gives signed area (orientation matters)
   - Applications: Torque in physics, area calculations, rotation representations

Mathematical Foundation:
- All operations preserve linearity and can be extended to higher dimensions
- These geometric interpretations form the basis for linear transformations
- Inner product spaces generalize these concepts to abstract vector spaces
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

# --- Step 1: Define vectors (you can modify them) ---
A = np.array([3, 2])
B = np.array([2, 4])

# --- Step 2: Compute dot product and components ---
dot_product = np.dot(A, B)
norm_A = np.linalg.norm(A)
norm_B = np.linalg.norm(B)
cos_theta = dot_product / (norm_A * norm_B)
theta_deg = np.degrees(np.arccos(cos_theta))

# --- Step 3: Create a formatted table with calculations ---
data = {
    "Operation": [
        "Vector A",
        "Vector B",
        "A ⋅ B (Dot Product)",
        "|A| (Magnitude)",
        "|B| (Magnitude)",
        "cos(θ)",
        "θ (degrees)",
        "Geometric Meaning"
    ],
    "Value": [
        str(A),
        str(B),
        f"{dot_product:.2f}",
        f"{norm_A:.2f}",
        f"{norm_B:.2f}",
        f"{cos_theta:.4f}",
        f"{theta_deg:.2f}°",
        "Projection of A on B scaled by |B|"
    ]
}

df = pd.DataFrame(data)
print("\n" + "-" * 60)
print("   INNER PRODUCT CALCULATION AND GEOMETRIC INTERPRETATION")
print("-" * 60)
print(df.to_string(index=False))
print("-" * 60)

# --- Step 4: Geometric visualization with 4 subplots ---
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
origin = np.array([0, 0])

# Additional computations for other properties
cross_product_mag = np.abs(A[0] * B[1] - A[1] * B[0])  # 2D cross product magnitude
A_unit = A / norm_A
B_unit = B / norm_B
vector_sum = A + B
vector_diff = A - B

# Subplot 1: Original - Dot Product and Projection
ax1 = axes[0, 0]
ax1.quiver(*origin, *A, angles='xy', scale_units='xy', scale=1, color='r', width=0.006, label='Vector A')
ax1.quiver(*origin, *B, angles='xy', scale_units='xy', scale=1, color='b', width=0.006, label='Vector B')
proj_A_on_B = np.dot(A, B_unit) * B_unit
ax1.quiver(*origin, *proj_A_on_B, angles='xy', scale_units='xy', scale=1, color='g', width=0.006, label='Projection of A on B')
ax1.text(A[0] * 1.05, A[1] * 1.05, 'A', color='r', fontsize=10)
ax1.text(B[0] * 1.05, B[1] * 1.05, 'B', color='b', fontsize=10)
arc = np.linspace(0, np.arccos(cos_theta), 100)
ax1.plot(0.7 * np.cos(arc), 0.7 * np.sin(arc), 'k--', linewidth=1)
ax1.text(0.9 * np.cos(arc[-1]/2), 0.9 * np.sin(arc[-1]/2), f'{theta_deg:.1f}°', fontsize=9)
ax1.set_xlim(-1, max(A[0], B[0]) + 2)
ax1.set_ylim(-1, max(A[1], B[1]) + 2)
ax1.set_aspect('equal', adjustable='box')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=9)
ax1.set_title(f"Dot Product: A·B = {dot_product:.2f}", fontsize=11, fontweight='bold')

# Subplot 2: Vector Addition and Subtraction
ax2 = axes[0, 1]
ax2.quiver(*origin, *A, angles='xy', scale_units='xy', scale=1, color='r', width=0.006, label='A')
ax2.quiver(*origin, *B, angles='xy', scale_units='xy', scale=1, color='b', width=0.006, label='B')
ax2.quiver(*origin, *vector_sum, angles='xy', scale_units='xy', scale=1, color='purple', width=0.006, label='A + B')
ax2.quiver(*origin, *vector_diff, angles='xy', scale_units='xy', scale=1, color='orange', width=0.006, label='A - B')
# Parallelogram for vector addition
ax2.plot([A[0], vector_sum[0]], [A[1], vector_sum[1]], 'k--', alpha=0.3, linewidth=1)
ax2.plot([B[0], vector_sum[0]], [B[1], vector_sum[1]], 'k--', alpha=0.3, linewidth=1)
ax2.text(A[0] * 1.05, A[1] * 1.05, 'A', color='r', fontsize=10)
ax2.text(B[0] * 1.05, B[1] * 1.05, 'B', color='b', fontsize=10)
ax2.text(vector_sum[0] * 1.05, vector_sum[1] * 1.05, 'A+B', color='purple', fontsize=10)
ax2.set_xlim(-2, max(vector_sum[0], A[0], B[0]) + 2)
ax2.set_ylim(-2, max(vector_sum[1], A[1], B[1]) + 2)
ax2.set_aspect('equal', adjustable='box')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=9)
ax2.set_title(f"Vector Addition/Subtraction", fontsize=11, fontweight='bold')

# Subplot 3: Unit Vectors and Magnitudes
ax3 = axes[1, 0]
ax3.quiver(*origin, *A, angles='xy', scale_units='xy', scale=1, color='r', width=0.006, label=f'A (|A|={norm_A:.2f})')
ax3.quiver(*origin, *B, angles='xy', scale_units='xy', scale=1, color='b', width=0.006, label=f'B (|B|={norm_B:.2f})')
ax3.quiver(*origin, *A_unit, angles='xy', scale_units='xy', scale=1, color='pink', width=0.006, label='A_unit (unit A)')
ax3.quiver(*origin, *B_unit, angles='xy', scale_units='xy', scale=1, color='cyan', width=0.006, label='B_unit (unit B)')
# Draw circles showing magnitudes
circle_A = Circle((0, 0), float(norm_A), color='r', fill=False, linestyle=':', alpha=0.3)
circle_B = Circle((0, 0), float(norm_B), color='b', fill=False, linestyle=':', alpha=0.3)
ax3.add_patch(circle_A)
ax3.add_patch(circle_B)
ax3.set_xlim(-1, max(A[0], B[0]) + 2)
ax3.set_ylim(-1, max(A[1], B[1]) + 2)
ax3.set_aspect('equal', adjustable='box')
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend(fontsize=9)
ax3.set_title(f"Unit Vectors and Magnitudes", fontsize=11, fontweight='bold')

# Subplot 4: Cross Product (Area) and Orthogonal Components
ax4 = axes[1, 1]
ax4.quiver(*origin, *A, angles='xy', scale_units='xy', scale=1, color='r', width=0.006, label='A')
ax4.quiver(*origin, *B, angles='xy', scale_units='xy', scale=1, color='b', width=0.006, label='B')
# Orthogonal component of A with respect to B
orth_A = A - proj_A_on_B
ax4.quiver(*proj_A_on_B, *orth_A, angles='xy', scale_units='xy', scale=1, color='magenta', width=0.006, label='A⊥B (orthogonal)')
ax4.plot([proj_A_on_B[0], A[0]], [proj_A_on_B[1], A[1]], 'k--', alpha=0.5, linewidth=1)
# Draw parallelogram to show area
parallelogram = Polygon([origin, A, A + B, B], alpha=0.2, color='yellow', edgecolor='black', linewidth=1.5)
ax4.add_patch(parallelogram)
ax4.text(A[0] * 1.05, A[1] * 1.05, 'A', color='r', fontsize=10)
ax4.text(B[0] * 1.05, B[1] * 1.05, 'B', color='b', fontsize=10)
ax4.set_xlim(-1, max(A[0] + B[0], A[0], B[0]) + 2)
ax4.set_ylim(-1, max(A[1] + B[1], A[1], B[1]) + 2)
ax4.set_aspect('equal', adjustable='box')
ax4.grid(True, linestyle='--', alpha=0.6)
ax4.legend(fontsize=9)
ax4.set_title(f"Cross Product Area = {cross_product_mag:.2f}", fontsize=11, fontweight='bold')

plt.suptitle("Vector Properties and Operations", fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()
