"""
DOT PRODUCT APPLICATIONS ACROSS DISCIPLINES

This script demonstrates practical applications of the dot product (inner product)
in various fields with detailed calculations and visualizations.

APPLICATIONS COVERED:
1. Linear Algebra - Orthogonality Testing
2. Physics - Work Calculation
3. Optimization - Gradient Alignment
4. Signal Processing - Correlation
5. Machine Learning - Cosine Similarity (bonus)

Each application includes:
- Theoretical background
- Numerical calculations in table format
- Geometric visualization
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyBboxPatch, Rectangle, Ellipse
import matplotlib.transforms as transforms

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# APPLICATION 1: LINEAR ALGEBRA - ORTHOGONALITY TESTING
# ============================================================================
print("\n" + "="*80)
print("APPLICATION 1: LINEAR ALGEBRA - ORTHOGONALITY TESTING")
print("="*80)
print("Theory: Two vectors are orthogonal (perpendicular) if and only if A·B = 0")
print("This is fundamental for: basis vectors, Gram-Schmidt, QR decomposition, etc.")
print("-"*80)

# Test vectors: perpendicular and non-perpendicular cases
v1 = np.array([3, 4])
v2_perp = np.array([-4, 3])  # Perpendicular to v1
v2_non_perp = np.array([2, 2])  # Not perpendicular

dot_perp = np.dot(v1, v2_perp)
dot_non_perp = np.dot(v1, v2_non_perp)
angle_perp = np.degrees(np.arccos(dot_perp / (np.linalg.norm(v1) * np.linalg.norm(v2_perp))) if dot_perp != 0 else 90)
angle_non_perp = np.degrees(np.arccos(dot_non_perp / (np.linalg.norm(v1) * np.linalg.norm(v2_non_perp))))

data1 = {
    "Test Case": ["v1", "v2 (perpendicular)", "v2 (non-perpendicular)"],
    "Vector": [str(v1), str(v2_perp), str(v2_non_perp)],
    "v1 · v2": ["—", f"{dot_perp:.4f}", f"{dot_non_perp:.4f}"],
    "Angle (°)": ["—", "90.00°", f"{angle_non_perp:.2f}°"],
    "Orthogonal?": ["—", "YES ✓", "NO ✗"]
}
df1 = pd.DataFrame(data1)
print(df1.to_string(index=False))
print("-"*80)

# ============================================================================
# APPLICATION 2: PHYSICS - WORK CALCULATION
# ============================================================================
print("\n" + "="*80)
print("APPLICATION 2: PHYSICS - WORK CALCULATION")
print("="*80)
print("Theory: Work W = F·d = |F||d|cos(θ)")
print("Only the component of force along displacement does work.")
print("If F ⊥ d, then W = 0 (e.g., centripetal force in circular motion)")
print("-"*80)

# Force and displacement vectors (in Newtons and meters)
F = np.array([10, 5])  # Force in N
d = np.array([4, 2])   # Displacement in m
F_perp = np.array([-2, 4])  # Force perpendicular to d

work = np.dot(F, d)
work_perp = np.dot(F_perp, d)
norm_F = np.linalg.norm(F)
norm_d = np.linalg.norm(d)
theta_work = np.degrees(np.arccos(work / (norm_F * norm_d)))

# Parallel component of force
F_parallel = (np.dot(F, d) / np.dot(d, d)) * d
F_parallel_mag = np.linalg.norm(F_parallel)

data2 = {
    "Quantity": [
        "Force F",
        "Displacement d",
        "|F| (magnitude)",
        "|d| (magnitude)",
        "F·d (dot product)",
        "θ (angle)",
        "|F_parallel| (along d)",
        "Work W = F·d",
        "Perpendicular force",
        "Work (perp force)"
    ],
    "Value": [
        f"{F}",
        f"{d}",
        f"{norm_F:.3f} N",
        f"{norm_d:.3f} m",
        f"{work:.3f}",
        f"{theta_work:.2f}°",
        f"{F_parallel_mag:.3f} N",
        f"{work:.3f} J (Joules)",
        f"{F_perp}",
        f"{work_perp:.3f} J (no work)"
    ]
}
df2 = pd.DataFrame(data2)
print(df2.to_string(index=False))
print("-"*80)

# ============================================================================
# APPLICATION 3: OPTIMIZATION - GRADIENT ALIGNMENT
# ============================================================================
print("\n" + "="*80)
print("APPLICATION 3: OPTIMIZATION - GRADIENT ALIGNMENT")
print("="*80)
print("Theory: ∇f·d > 0 means direction d increases function f (ascent)")
print("        ∇f·d < 0 means direction d decreases function f (descent)")
print("        ∇f·d = 0 means direction d is tangent to level curve")
print("Used in: gradient descent, line search, directional derivatives")
print("-"*80)

# Gradient at a point (e.g., ∇f(x) for f(x,y) = x² + 2y²)
grad_f = np.array([4, 8])  # Gradient at point (2, 2)
d_ascent = np.array([2, 4])  # Direction aligned with gradient
d_descent = np.array([-1, -2])  # Opposite to gradient
d_tangent = np.array([2, -1])  # Perpendicular to gradient

alignment_ascent = np.dot(grad_f, d_ascent)
alignment_descent = np.dot(grad_f, d_descent)
alignment_tangent = np.dot(grad_f, d_tangent)

# Normalized directional derivatives
dir_deriv_ascent = alignment_ascent / np.linalg.norm(d_ascent)
dir_deriv_descent = alignment_descent / np.linalg.norm(d_descent)
dir_deriv_tangent = alignment_tangent / np.linalg.norm(d_tangent)

data3 = {
    "Direction": [
        "∇f (gradient)",
        "d₁ (ascent direction)",
        "d₂ (descent direction)",
        "d₃ (tangent direction)"
    ],
    "Vector": [
        f"{grad_f}",
        f"{d_ascent}",
        f"{d_descent}",
        f"{d_tangent}"
    ],
    "∇f·d": [
        "—",
        f"{alignment_ascent:.2f}",
        f"{alignment_descent:.2f}",
        f"{alignment_tangent:.2f}"
    ],
    "Directional Derivative": [
        "—",
        f"{dir_deriv_ascent:.3f}",
        f"{dir_deriv_descent:.3f}",
        f"{dir_deriv_tangent:.3f}"
    ],
    "Effect": [
        "—",
        "Function INCREASES ↑",
        "Function DECREASES ↓",
        "Function CONSTANT →"
    ]
}
df3 = pd.DataFrame(data3)
print(df3.to_string(index=False))
print("-"*80)

# ============================================================================
# APPLICATION 4: SIGNAL PROCESSING - CORRELATION
# ============================================================================
print("\n" + "="*80)
print("APPLICATION 4: SIGNAL PROCESSING - CORRELATION")
print("="*80)
print("Theory: Correlation measures similarity between signals using dot product")
print("Normalized correlation: ρ = (s₁·s₂)/(|s₁||s₂|) ∈ [-1, 1]")
print("Used in: template matching, pattern recognition, signal detection")
print("-"*80)

# Create sample signals
t = np.linspace(0, 2*np.pi, 50)
signal1 = np.sin(t)
signal2_similar = np.sin(t + 0.2)  # Slightly shifted
signal2_opposite = -np.sin(t)  # Opposite phase
signal2_different = np.cos(2*t)  # Different frequency

# Calculate correlations
corr_similar = np.dot(signal1, signal2_similar)
corr_opposite = np.dot(signal1, signal2_opposite)
corr_different = np.dot(signal1, signal2_different)

# Normalized correlations (correlation coefficient)
norm_s1 = np.linalg.norm(signal1)
norm_similar = corr_similar / (norm_s1 * np.linalg.norm(signal2_similar))
norm_opposite = corr_opposite / (norm_s1 * np.linalg.norm(signal2_opposite))
norm_different = corr_different / (norm_s1 * np.linalg.norm(signal2_different))

data4 = {
    "Signal Pair": [
        "s₁ (reference)",
        "s₂ (similar phase)",
        "s₃ (opposite phase)",
        "s₄ (different freq)"
    ],
    "Description": [
        "sin(t)",
        "sin(t + 0.2)",
        "-sin(t)",
        "cos(2t)"
    ],
    "s₁·s": [
        "—",
        f"{corr_similar:.3f}",
        f"{corr_opposite:.3f}",
        f"{corr_different:.3f}"
    ],
    "Normalized ρ": [
        "1.000",
        f"{norm_similar:.3f}",
        f"{norm_opposite:.3f}",
        f"{norm_different:.3f}"
    ],
    "Similarity": [
        "Self",
        "High ✓",
        "Inverse ✗",
        "Low ✗"
    ]
}
df4 = pd.DataFrame(data4)
print(df4.to_string(index=False))
print("-"*80)

# ============================================================================
# APPLICATION 5: MACHINE LEARNING - COSINE SIMILARITY (BONUS)
# ============================================================================
print("\n" + "="*80)
print("APPLICATION 5: MACHINE LEARNING - COSINE SIMILARITY (BONUS)")
print("="*80)
print("Theory: Cosine similarity = (A·B)/(|A||B|) measures angle between vectors")
print("Range: [-1, 1] where 1=identical direction, 0=orthogonal, -1=opposite")
print("Used in: NLP, recommendation systems, document similarity, clustering")
print("-"*80)

# Document vectors (TF-IDF like representation)
doc1 = np.array([3, 2, 0, 5, 0, 0, 2])  # Document 1 word counts
doc2 = np.array([2, 3, 1, 4, 0, 0, 1])  # Similar document
doc3 = np.array([0, 0, 5, 0, 4, 3, 0])  # Different topic
doc4 = np.array([3, 2, 0, 5, 0, 0, 2])  # Identical to doc1

cosine_sim_12 = np.dot(doc1, doc2) / (np.linalg.norm(doc1) * np.linalg.norm(doc2))
cosine_sim_13 = np.dot(doc1, doc3) / (np.linalg.norm(doc1) * np.linalg.norm(doc3))
cosine_sim_14 = np.dot(doc1, doc4) / (np.linalg.norm(doc1) * np.linalg.norm(doc4))

data5 = {
    "Document Pair": [
        "Doc 1 vs Doc 2",
        "Doc 1 vs Doc 3",
        "Doc 1 vs Doc 4"
    ],
    "Relationship": [
        "Similar topics",
        "Different topics",
        "Identical"
    ],
    "Cosine Similarity": [
        f"{cosine_sim_12:.4f}",
        f"{cosine_sim_13:.4f}",
        f"{cosine_sim_14:.4f}"
    ],
    "Angle (°)": [
        f"{np.degrees(np.arccos(cosine_sim_12)):.2f}°",
        f"{np.degrees(np.arccos(cosine_sim_13)):.2f}°",
        f"{np.degrees(np.arccos(cosine_sim_14)):.2f}°"
    ],
    "Interpretation": [
        "High similarity",
        "Low similarity",
        "Perfect match"
    ]
}
df5 = pd.DataFrame(data5)
print(df5.to_string(index=False))
print("-"*80)

# ============================================================================
# VISUALIZATION: Create comprehensive subplot figure
# ============================================================================
fig = plt.figure(figsize=(16, 12))
origin = np.array([0, 0])

# ============================================================================
# SUBPLOT 1: Orthogonality Testing
# ============================================================================
ax1 = plt.subplot(3, 2, 1)
ax1.quiver(*origin, *v1, angles='xy', scale_units='xy', scale=1, color='red', width=0.006, label='v₁')
ax1.quiver(*origin, *v2_perp, angles='xy', scale_units='xy', scale=1, color='green', width=0.006, label='v₂ (⊥)')
ax1.quiver(*origin, *v2_non_perp, angles='xy', scale_units='xy', scale=1, color='orange', width=0.006, label='v₃ (not ⊥)')

# Draw right angle marker for perpendicular vectors
square_size = 0.5
square = Rectangle((0, 0), square_size, square_size, fill=False, edgecolor='green', linewidth=1.5)
transform_mat = transforms.Affine2D().rotate(np.arctan2(v1[1], v1[0])) + ax1.transData
square.set_transform(transform_mat)
ax1.add_patch(square)

ax1.text(v1[0]*0.6, v1[1]*0.6, 'v₁', fontsize=11, color='red', fontweight='bold')
ax1.text(v2_perp[0]*0.6, v2_perp[1]*0.6, 'v₂', fontsize=11, color='green', fontweight='bold')
ax1.text(v2_non_perp[0]*1.1, v2_non_perp[1]*1.1, 'v₃', fontsize=11, color='orange', fontweight='bold')

ax1.set_xlim(-6, 6)
ax1.set_ylim(-2, 6)
ax1.set_aspect('equal', adjustable='box')
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper right', fontsize=9)
ax1.set_title('1. Orthogonality Testing\nv₁·v₂ = 0 (perpendicular) | v₁·v₃ ≠ 0', 
              fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel('x', fontsize=9)
ax1.set_ylabel('y', fontsize=9)

# Add annotation box
textstr = f'v₁·v₂ = {dot_perp:.2f} ✓\nv₁·v₃ = {dot_non_perp:.2f} ✗'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

# ============================================================================
# SUBPLOT 2: Physics - Work
# ============================================================================
ax2 = plt.subplot(3, 2, 2)
ax2.quiver(*origin, *d, angles='xy', scale_units='xy', scale=1, color='blue', width=0.006, 
           label=f'd (displacement)', linewidth=2)
ax2.quiver(*origin, *F, angles='xy', scale_units='xy', scale=1, color='red', width=0.006, 
           label='F (force)', linewidth=2)
ax2.quiver(*origin, *F_parallel, angles='xy', scale_units='xy', scale=1, color='green', 
           width=0.006, label='F_∥ (does work)', linestyle='--', linewidth=2)
ax2.quiver(*origin, *F_perp, angles='xy', scale_units='xy', scale=1, color='gray', 
           width=0.005, label='F_⊥ (no work)', alpha=0.6)

# Draw angle arc
angle_rad = np.arctan2(F[1], F[0])
angle_d_rad = np.arctan2(d[1], d[0])
arc_angles = np.linspace(angle_d_rad, angle_rad, 30)
arc_radius = 1.5
ax2.plot(arc_radius * np.cos(arc_angles), arc_radius * np.sin(arc_angles), 
         'k--', linewidth=1.5, alpha=0.7)
mid_angle = (angle_rad + angle_d_rad) / 2
ax2.text(2*np.cos(mid_angle), 2*np.sin(mid_angle), f'θ={theta_work:.1f}°', 
         fontsize=9, ha='center')

ax2.text(F[0]*1.1, F[1]*1.1, 'F', fontsize=11, color='red', fontweight='bold')
ax2.text(d[0]*1.1, d[1]*1.1, 'd', fontsize=11, color='blue', fontweight='bold')

ax2.set_xlim(-3, 12)
ax2.set_ylim(-3, 8)
ax2.set_aspect('equal', adjustable='box')
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='upper right', fontsize=9)
ax2.set_title(f'2. Physics: Work Calculation\nW = F·d = {work:.2f} J', 
              fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel('x (meters)', fontsize=9)
ax2.set_ylabel('y (meters)', fontsize=9)

# Add annotation
textstr = f'W = F·d = {work:.2f} J\nOnly F_∥ does work\n|F_∥| = {F_parallel_mag:.2f} N'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

# ============================================================================
# SUBPLOT 3: Optimization - Gradient Alignment
# ============================================================================
ax3 = plt.subplot(3, 2, 3)
scale_grad = 0.3
ax3.quiver(*origin, *(grad_f*scale_grad), angles='xy', scale_units='xy', scale=1, 
           color='red', width=0.008, label='∇f (gradient)', linewidth=2.5, headwidth=4)
ax3.quiver(*origin, *d_ascent, angles='xy', scale_units='xy', scale=1, 
           color='green', width=0.006, label='d₁ (ascent)', linewidth=2)
ax3.quiver(*origin, *d_descent, angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.006, label='d₂ (descent)', linewidth=2)
ax3.quiver(*origin, *d_tangent, angles='xy', scale_units='xy', scale=1, 
           color='purple', width=0.006, label='d₃ (tangent)', linewidth=2)

# Draw contour-like ellipse to show level curves
ellipse = Ellipse((0, 0), 8, 4, angle=np.degrees(np.arctan2(grad_f[1], grad_f[0])),
                  fill=False, edgecolor='gray', linestyle=':', linewidth=2, alpha=0.5)
ax3.add_patch(ellipse)

ax3.text(grad_f[0]*scale_grad*1.2, grad_f[1]*scale_grad*1.2, '∇f', 
         fontsize=11, color='red', fontweight='bold')
ax3.text(d_ascent[0]*1.15, d_ascent[1]*1.15, 'd₁', fontsize=10, color='green', fontweight='bold')
ax3.text(d_descent[0]*1.2, d_descent[1]*1.2, 'd₂', fontsize=10, color='blue', fontweight='bold')
ax3.text(d_tangent[0]*1.15, d_tangent[1]*1.15, 'd₃', fontsize=10, color='purple', fontweight='bold')

ax3.set_xlim(-3, 5)
ax3.set_ylim(-4, 6)
ax3.set_aspect('equal', adjustable='box')
ax3.grid(True, linestyle='--', alpha=0.4)
ax3.legend(loc='upper right', fontsize=9)
ax3.set_title('3. Gradient Alignment\nDirectional Derivatives via Dot Product', 
              fontsize=11, fontweight='bold', pad=10)
ax3.set_xlabel('x', fontsize=9)
ax3.set_ylabel('y', fontsize=9)

# Add annotation
textstr = f'∇f·d₁ = {alignment_ascent:.1f} (↑)\n∇f·d₂ = {alignment_descent:.1f} (↓)\n∇f·d₃ = {alignment_tangent:.1f} (→)'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

# ============================================================================
# SUBPLOT 4: Signal Processing - Correlation
# ============================================================================
ax4 = plt.subplot(3, 2, 4)
ax4.plot(t, signal1, 'b-', linewidth=2, label='s₁: sin(t)', alpha=0.8)
ax4.plot(t, signal2_similar, 'g--', linewidth=2, label='s₂: sin(t+0.2)', alpha=0.8)
ax4.plot(t, signal2_opposite, 'r:', linewidth=2, label='s₃: -sin(t)', alpha=0.8)
ax4.plot(t, signal2_different, 'm-.', linewidth=2, label='s₄: cos(2t)', alpha=0.6)

ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax4.grid(True, linestyle='--', alpha=0.4)
ax4.legend(loc='upper right', fontsize=9)
ax4.set_title('4. Signal Correlation\nDot Product Measures Similarity', 
              fontsize=11, fontweight='bold', pad=10)
ax4.set_xlabel('Time (t)', fontsize=9)
ax4.set_ylabel('Amplitude', fontsize=9)
ax4.set_xlim(0, 2*np.pi)
ax4.set_ylim(-1.5, 1.5)

# Add annotation
textstr = f'Correlations (ρ):\ns₁·s₂: {norm_similar:.3f} (high)\ns₁·s₃: {norm_opposite:.3f} (inverse)\ns₁·s₄: {norm_different:.3f} (low)'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
ax4.text(0.05, 0.05, textstr, transform=ax4.transAxes, fontsize=9,
         verticalalignment='bottom', bbox=props)

# ============================================================================
# SUBPLOT 5: ML - Cosine Similarity (Document Vectors)
# ============================================================================
ax5 = plt.subplot(3, 2, 5)
# Visualize document vectors in 2D projection (using first 2 dimensions)
doc1_2d = doc1[:2]
doc2_2d = doc2[:2]
doc3_2d = doc3[:2]

ax5.quiver(*origin, *doc1_2d, angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.008, label='Doc 1', linewidth=2.5, headwidth=4)
ax5.quiver(*origin, *doc2_2d, angles='xy', scale_units='xy', scale=1, 
           color='green', width=0.006, label='Doc 2 (similar)', linewidth=2)
ax5.quiver(*origin, *doc3_2d, angles='xy', scale_units='xy', scale=1, 
           color='red', width=0.006, label='Doc 3 (different)', linewidth=2)

# Draw angle arcs
angle1 = np.arctan2(doc1_2d[1], doc1_2d[0])
angle2 = np.arctan2(doc2_2d[1], doc2_2d[0])
angle3 = np.arctan2(doc3_2d[1], doc3_2d[0])

arc_12 = np.linspace(angle1, angle2, 20)
ax5.plot(1.2 * np.cos(arc_12), 1.2 * np.sin(arc_12), 'g--', linewidth=1.5, alpha=0.6)

ax5.text(doc1_2d[0]*1.15, doc1_2d[1]*1.15, 'D1', fontsize=11, color='blue', fontweight='bold')
ax5.text(doc2_2d[0]*1.15, doc2_2d[1]*1.15, 'D2', fontsize=11, color='green', fontweight='bold')
ax5.text(doc3_2d[0]*1.15, doc3_2d[1]*1.15, 'D3', fontsize=11, color='red', fontweight='bold')

ax5.set_xlim(-1, 4)
ax5.set_ylim(-1, 4)
ax5.set_aspect('equal', adjustable='box')
ax5.grid(True, linestyle='--', alpha=0.4)
ax5.legend(loc='upper right', fontsize=9)
ax5.set_title('5. ML: Cosine Similarity\n(2D Projection of Document Vectors)', 
              fontsize=11, fontweight='bold', pad=10)
ax5.set_xlabel('Feature dimension 1', fontsize=9)
ax5.set_ylabel('Feature dimension 2', fontsize=9)

# Add annotation
textstr = f'Cosine Similarity:\nD1↔D2: {cosine_sim_12:.3f}\nD1↔D3: {cosine_sim_13:.3f}\nD1↔D4: {cosine_sim_14:.3f}'
props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
ax5.text(0.05, 0.95, textstr, transform=ax5.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

# ============================================================================
# SUBPLOT 6: Summary Comparison
# ============================================================================
ax6 = plt.subplot(3, 2, 6)
ax6.axis('off')

summary_text = """
DOT PRODUCT APPLICATIONS SUMMARY

1. ORTHOGONALITY TESTING (Linear Algebra)
   • A·B = 0 ⟺ A ⊥ B (perpendicular)
   • Used in: QR decomposition, basis construction
   
2. WORK CALCULATION (Physics)
   • W = F·d = |F||d|cos(θ)
   • Only parallel component does work
   
3. GRADIENT ALIGNMENT (Optimization)
   • ∇f·d > 0: function increases along d
   • ∇f·d < 0: function decreases (gradient descent)
   
4. SIGNAL CORRELATION (Signal Processing)
   • ρ = (s₁·s₂)/(|s₁||s₂|) ∈ [-1,1]
   • Measures pattern similarity
   
5. COSINE SIMILARITY (Machine Learning)
   • cos(θ) = (A·B)/(|A||B|)
   • Document/feature vector comparison

KEY INSIGHT: The dot product is a fundamental operation
that measures "alignment" or "similarity" across diverse
applications in mathematics, physics, and computing.
"""

ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1))

# Add title to summary
ax6.text(0.5, 0.98, 'Applications Overview', transform=ax6.transAxes,
         fontsize=12, fontweight='bold', ha='center', va='top')

plt.suptitle('DOT PRODUCT APPLICATIONS ACROSS DISCIPLINES', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.99))
plt.show()

print("\n" + "="*80)
print("Visualization complete! All 5 applications demonstrated.")
print("="*80)
