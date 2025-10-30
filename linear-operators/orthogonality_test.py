"""
ORTHOGONALITY TESTING - LINEAR ALGEBRA APPLICATION

Theory:
Two vectors A and B are orthogonal (perpendicular) if and only if their dot product equals zero:
    A·B = 0 ⟺ A ⊥ B

This fundamental concept is crucial for:
- Constructing orthonormal bases (Gram-Schmidt process)
- QR decomposition
- Principal Component Analysis (PCA)
- Checking independence of vectors
- Defining perpendicularity in any inner product space

Geometric Interpretation:
When vectors are orthogonal, the angle between them is exactly 90°.
This can be verified using: θ = arccos(A·B / (|A||B|))
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms

# Define test vectors
v1 = np.array([3, 4])
v2_perpendicular = np.array([-4, 3])  # Constructed to be perpendicular: v1·v2 = 3(-4) + 4(3) = 0
v2_non_perpendicular = np.array([2, 2])  # Not perpendicular
v3_almost_perpendicular = np.array([-4.1, 3])  # Nearly perpendicular (small dot product)

# Calculate dot products and angles
def analyze_orthogonality(v1, v2):
    """Analyze orthogonality between two vectors"""
    dot_prod = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Handle exact zero case
    if np.abs(dot_prod) < 1e-10:
        angle_deg = 90.0
        is_orthogonal = True
    else:
        cos_theta = dot_prod / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1, 1)  # Numerical stability
        angle_deg = np.degrees(np.arccos(cos_theta))
        is_orthogonal = np.abs(dot_prod) < 1e-6
    
    return {
        'dot_product': dot_prod,
        'angle': angle_deg,
        'orthogonal': is_orthogonal,
        'norm_v1': norm_v1,
        'norm_v2': norm_v2
    }

# Analyze all test cases
result1 = analyze_orthogonality(v1, v2_perpendicular)
result2 = analyze_orthogonality(v1, v2_non_perpendicular)
result3 = analyze_orthogonality(v1, v3_almost_perpendicular)

# Create comprehensive table
print("\n" + "="*85)
print("ORTHOGONALITY TESTING - DOT PRODUCT APPLICATION")
print("="*85)
print("\nTheorem: A·B = 0 if and only if A ⊥ B (vectors are perpendicular)")
print("-"*85)

data = {
    "Test Case": [
        "v₁ (reference)",
        "v₂ (perpendicular)",
        "v₃ (not perpendicular)",
        "v₄ (almost perpendicular)"
    ],
    "Vector": [
        str(v1),
        str(v2_perpendicular),
        str(v2_non_perpendicular),
        str(v3_almost_perpendicular)
    ],
    "v₁·v": [
        "—",
        f"{result1['dot_product']:.6f}",
        f"{result2['dot_product']:.6f}",
        f"{result3['dot_product']:.6f}"
    ],
    "Magnitude": [
        f"{result1['norm_v1']:.3f}",
        f"{result1['norm_v2']:.3f}",
        f"{result2['norm_v2']:.3f}",
        f"{result3['norm_v2']:.3f}"
    ],
    "Angle θ": [
        "—",
        f"{result1['angle']:.2f}°",
        f"{result2['angle']:.2f}°",
        f"{result3['angle']:.2f}°"
    ],
    "Orthogonal?": [
        "—",
        "YES ✓" if result1['orthogonal'] else "NO ✗",
        "YES ✓" if result2['orthogonal'] else "NO ✗",
        "YES ✓" if result3['orthogonal'] else "NO ✗"
    ]
}

df = pd.DataFrame(data)
print(df.to_string(index=False))
print("-"*85)
print(f"\nKey Insight: |v₁·v₂| < ε (where ε ≈ 10⁻⁶) indicates orthogonality")
print("="*85)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
origin = np.array([0, 0])

# Plot 1: Perfect Orthogonality
ax1 = axes[0]
ax1.quiver(*origin, *v1, angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.008, label='v₁', linewidth=2.5, headwidth=4)
ax1.quiver(*origin, *v2_perpendicular, angles='xy', scale_units='xy', scale=1, 
           color='green', width=0.007, label='v₂ (perpendicular)', linewidth=2.5, headwidth=4)

# Draw right angle marker
square_size = 0.6
square = Rectangle((0, 0), square_size, square_size, fill=False, 
                   edgecolor='green', linewidth=2, linestyle='--')
angle_v1 = np.arctan2(v1[1], v1[0])
transform_mat = transforms.Affine2D().rotate(angle_v1) + ax1.transData
square.set_transform(transform_mat)
ax1.add_patch(square)

# Add labels
ax1.text(v1[0]*0.5, v1[1]*0.5, 'v₁', fontsize=12, color='blue', 
         fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax1.text(v2_perpendicular[0]*0.6, v2_perpendicular[1]*0.6, 'v₂', fontsize=12, 
         color='green', fontweight='bold', ha='center', 
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Add annotation box
textstr = f'v₁·v₂ = {result1["dot_product"]:.6f}\nθ = {result1["angle"]:.1f}°\nOrthogonal: YES ✓'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='green', linewidth=2)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=props, family='monospace')

ax1.set_xlim(-5, 5)
ax1.set_ylim(-2, 6)
ax1.set_aspect('equal', adjustable='box')
ax1.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
ax1.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax1.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax1.set_title('Perfect Orthogonality\n(Dot Product = 0)', 
              fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('y', fontsize=10)

# Plot 2: Non-Orthogonal Vectors
ax2 = axes[1]
ax2.quiver(*origin, *v1, angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.008, label='v₁', linewidth=2.5, headwidth=4)
ax2.quiver(*origin, *v2_non_perpendicular, angles='xy', scale_units='xy', scale=1, 
           color='red', width=0.007, label='v₃ (not perpendicular)', linewidth=2.5, headwidth=4)

# Draw angle arc
angle1 = np.arctan2(v1[1], v1[0])
angle2 = np.arctan2(v2_non_perpendicular[1], v2_non_perpendicular[0])
arc_angles = np.linspace(min(angle1, angle2), max(angle1, angle2), 30)
arc_radius = 1.5
ax2.plot(arc_radius * np.cos(arc_angles), arc_radius * np.sin(arc_angles), 
         'k--', linewidth=2, alpha=0.7)
mid_angle = (angle1 + angle2) / 2
ax2.text(2*np.cos(mid_angle), 2*np.sin(mid_angle), f'{result2["angle"]:.1f}°', 
         fontsize=11, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Add labels
ax2.text(v1[0]*0.5, v1[1]*0.5, 'v₁', fontsize=12, color='blue', 
         fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax2.text(v2_non_perpendicular[0]*1.2, v2_non_perpendicular[1]*1.2, 'v₃', fontsize=12, 
         color='red', fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# Add annotation box
textstr = f'v₁·v₃ = {result2["dot_product"]:.6f}\nθ = {result2["angle"]:.1f}°\nOrthogonal: NO ✗'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='red', linewidth=2)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=props, family='monospace')

ax2.set_xlim(-1, 5)
ax2.set_ylim(-1, 6)
ax2.set_aspect('equal', adjustable='box')
ax2.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
ax2.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax2.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax2.set_title('Non-Orthogonal Vectors\n(Dot Product ≠ 0)', 
              fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('y', fontsize=10)

# Plot 3: Almost Orthogonal (Numerical Considerations)
ax3 = axes[2]
ax3.quiver(*origin, *v1, angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.008, label='v₁', linewidth=2.5, headwidth=4)
ax3.quiver(*origin, *v3_almost_perpendicular, angles='xy', scale_units='xy', scale=1, 
           color='orange', width=0.007, label='v₄ (almost perpendicular)', linewidth=2.5, headwidth=4)

# Draw approximate right angle
square2 = Rectangle((0, 0), square_size, square_size, fill=False, 
                    edgecolor='orange', linewidth=2, linestyle=':')
transform_mat2 = transforms.Affine2D().rotate(angle_v1) + ax3.transData
square2.set_transform(transform_mat2)
ax3.add_patch(square2)

# Draw angle arc
angle3 = np.arctan2(v3_almost_perpendicular[1], v3_almost_perpendicular[0])
arc_angles3 = np.linspace(min(angle1, angle3), max(angle1, angle3), 30)
ax3.plot(arc_radius * np.cos(arc_angles3), arc_radius * np.sin(arc_angles3), 
         'k--', linewidth=1.5, alpha=0.5)
mid_angle3 = (angle1 + angle3) / 2
ax3.text(2*np.cos(mid_angle3), 2*np.sin(mid_angle3), f'{result3["angle"]:.2f}°', 
         fontsize=11, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Add labels
ax3.text(v1[0]*0.5, v1[1]*0.5, 'v₁', fontsize=12, color='blue', 
         fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax3.text(v3_almost_perpendicular[0]*0.6, v3_almost_perpendicular[1]*0.6, 'v₄', fontsize=12, 
         color='orange', fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='peachpuff', alpha=0.7))

# Add annotation box
textstr = f'v₁·v₄ = {result3["dot_product"]:.6f}\nθ = {result3["angle"]:.2f}°\n|dot product| < ε'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='orange', linewidth=2)
ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', bbox=props, family='monospace')

ax3.set_xlim(-5, 5)
ax3.set_ylim(-2, 6)
ax3.set_aspect('equal', adjustable='box')
ax3.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
ax3.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax3.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
ax3.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax3.set_title('Nearly Orthogonal\n(Small Dot Product)', 
              fontsize=12, fontweight='bold', pad=10)
ax3.set_xlabel('x', fontsize=10)
ax3.set_ylabel('y', fontsize=10)

plt.suptitle('Orthogonality Testing via Dot Product (Linear Algebra)', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()

print("\n✓ Visualization complete!")
