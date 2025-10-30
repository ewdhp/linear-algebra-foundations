"""
GRADIENT ALIGNMENT - OPTIMIZATION APPLICATION

Theory:
The directional derivative of a function f in direction d is given by:
    D_d f = ∇f·d = |∇f||d|cos(θ)

When d is a unit vector (|d| = 1):
    D_d f = ∇f·d̂

Interpretation:
- ∇f·d > 0: Moving in direction d INCREASES function f (ascent)
- ∇f·d < 0: Moving in direction d DECREASES function f (descent)
- ∇f·d = 0: Direction d is tangent to level curve (no change)
- Maximum increase: d aligned with ∇f (d = ∇f/|∇f|)
- Maximum decrease: d opposite to ∇f (d = -∇f/|∇f|)

Applications:
- Gradient Descent: minimize f by moving in direction -∇f
- Gradient Ascent: maximize f by moving in direction +∇f
- Line Search: find optimal step size along search direction
- Steepest Descent: always move in direction of -∇f
- Newton's Method: uses second-order information for direction
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

# Define a point and gradient (e.g., for f(x,y) = x² + 2y²)
# At point (2, 2): ∇f = (2x, 4y) = (4, 8)
point = np.array([2, 2])
grad_f = np.array([4, 8])

# Define various search directions
d_ascent = np.array([2, 4])           # Aligned with gradient
d_descent = np.array([-1, -2])        # Opposite to gradient
d_tangent = np.array([2, -1])         # Perpendicular to gradient
d_suboptimal = np.array([1, 0.5])     # Partial ascent
d_bad_descent = np.array([-0.5, 0.5]) # Poor descent direction

def analyze_gradient_alignment(gradient, direction):
    """Analyze how a direction aligns with gradient"""
    alignment = np.dot(gradient, direction)
    norm_grad = np.linalg.norm(gradient)
    norm_dir = np.linalg.norm(direction)
    
    # Directional derivative (normalized)
    dir_derivative = alignment / norm_dir if norm_dir > 0 else 0
    
    # Angle between gradient and direction
    if norm_grad > 0 and norm_dir > 0:
        cos_theta = alignment / (norm_grad * norm_dir)
        cos_theta = np.clip(cos_theta, -1, 1)
        angle_deg = np.degrees(np.arccos(cos_theta))
    else:
        angle_deg = 0
    
    # Determine effect on function
    if alignment > 0.1:
        effect = "INCREASES ↑"
        color = "green"
    elif alignment < -0.1:
        effect = "DECREASES ↓"
        color = "blue"
    else:
        effect = "CONSTANT →"
        color = "gray"
    
    return {
        'alignment': alignment,
        'dir_derivative': dir_derivative,
        'angle': angle_deg,
        'effect': effect,
        'color': color,
        'norm_dir': norm_dir
    }

# Analyze all directions
results = {
    'ascent': analyze_gradient_alignment(grad_f, d_ascent),
    'descent': analyze_gradient_alignment(grad_f, d_descent),
    'tangent': analyze_gradient_alignment(grad_f, d_tangent),
    'suboptimal': analyze_gradient_alignment(grad_f, d_suboptimal),
    'bad_descent': analyze_gradient_alignment(grad_f, d_bad_descent)
}

print("\n" + "="*95)
print("GRADIENT ALIGNMENT - DOT PRODUCT APPLICATION IN OPTIMIZATION")
print("="*95)
print("\nFormula: Directional Derivative = ∇f·d = |∇f||d|cos(θ)")
print("Principle: Dot product determines rate of change along a direction")
print("-"*95)

# Create comprehensive table
data = {
    "Direction": [
        "∇f (gradient)",
        "d₁ (ascent)",
        "d₂ (descent)",
        "d₃ (tangent)",
        "d₄ (suboptimal ascent)",
        "d₅ (poor descent)"
    ],
    "Vector": [
        str(grad_f),
        str(d_ascent),
        str(d_descent),
        str(d_tangent),
        str(d_suboptimal),
        str(d_bad_descent)
    ],
    "∇f·d": [
        "—",
        f"{results['ascent']['alignment']:.2f}",
        f"{results['descent']['alignment']:.2f}",
        f"{results['tangent']['alignment']:.2f}",
        f"{results['suboptimal']['alignment']:.2f}",
        f"{results['bad_descent']['alignment']:.2f}"
    ],
    "D_d f (normalized)": [
        "—",
        f"{results['ascent']['dir_derivative']:.4f}",
        f"{results['descent']['dir_derivative']:.4f}",
        f"{results['tangent']['dir_derivative']:.4f}",
        f"{results['suboptimal']['dir_derivative']:.4f}",
        f"{results['bad_descent']['dir_derivative']:.4f}"
    ],
    "Angle θ": [
        "—",
        f"{results['ascent']['angle']:.2f}°",
        f"{results['descent']['angle']:.2f}°",
        f"{results['tangent']['angle']:.2f}°",
        f"{results['suboptimal']['angle']:.2f}°",
        f"{results['bad_descent']['angle']:.2f}°"
    ],
    "Effect on f": [
        "—",
        results['ascent']['effect'],
        results['descent']['effect'],
        results['tangent']['effect'],
        results['suboptimal']['effect'],
        results['bad_descent']['effect']
    ]
}

df = pd.DataFrame(data)
print(df.to_string(index=False))
print("-"*95)
print(f"\nOptimization Rules:")
print(f"  • Maximize f: move in direction d where ∇f·d > 0 (best: d = +∇f)")
print(f"  • Minimize f: move in direction d where ∇f·d < 0 (best: d = -∇f)")
print(f"  • No change: ∇f·d = 0 (perpendicular to gradient)")
print("="*95)

# Visualization
fig = plt.figure(figsize=(16, 10))

# Main plot: Gradient field with contours
ax1 = plt.subplot(2, 2, (1, 3))

# Create meshgrid for contour plot (function f(x,y) = x² + 2y²)
x_range = np.linspace(-1, 5, 100)
y_range = np.linspace(-1, 5, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = X**2 + 2*Y**2

# Plot contours
levels = np.linspace(0, 50, 15)
contours = ax1.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.4, linewidths=1)
ax1.clabel(contours, inline=True, fontsize=8, fmt='%.0f')
contourf = ax1.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.2)

# Mark the point
ax1.plot(*point, 'ro', markersize=12, label=f'Point ({point[0]}, {point[1]})', zorder=5)

# Draw gradient vector (scaled for visibility)
grad_scale = 0.4
origin = point
ax1.quiver(*origin, *(grad_f*grad_scale), angles='xy', scale_units='xy', scale=1,
           color='red', width=0.012, label='∇f (gradient)', linewidth=3, 
           headwidth=5, headlength=6, zorder=4)

# Draw direction vectors from the point
arrow_scale = 0.8
directions = {
    'd₁ (ascent)': (d_ascent, 'green', results['ascent']),
    'd₂ (descent)': (d_descent, 'blue', results['descent']),
    'd₃ (tangent)': (d_tangent, 'purple', results['tangent']),
    'd₄ (sub-optimal)': (d_suboptimal, 'orange', results['suboptimal']),
    'd₅ (poor)': (d_bad_descent, 'brown', results['bad_descent'])
}

for label, (direction, color, result) in directions.items():
    ax1.quiver(*origin, *direction, angles='xy', scale_units='xy', scale=1,
               color=color, width=0.008, label=label, linewidth=2.5,
               headwidth=4, headlength=5, alpha=0.8, zorder=3)

# Add angle arc between gradient and ascent direction
angle_grad = np.arctan2(grad_f[1], grad_f[0])
angle_asc = np.arctan2(d_ascent[1], d_ascent[0])
arc_angles = np.linspace(min(angle_grad, angle_asc), max(angle_grad, angle_asc), 20)
arc_radius = 1.0
arc_x = origin[0] + arc_radius * np.cos(arc_angles)
arc_y = origin[1] + arc_radius * np.sin(arc_angles)
ax1.plot(arc_x, arc_y, 'k--', linewidth=1.5, alpha=0.6)

ax1.set_xlim(-0.5, 5)
ax1.set_ylim(-0.5, 5)
ax1.set_aspect('equal', adjustable='box')
ax1.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=2)
ax1.set_title('Gradient Field with Multiple Search Directions\nf(x,y) = x² + 2y²', 
              fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('y', fontsize=11)

# Add colorbar
cbar = plt.colorbar(contourf, ax=ax1)
cbar.set_label('Function Value f(x,y)', fontsize=10)

# Annotation box
textstr = f'At point ({point[0]}, {point[1]}):\n∇f = {grad_f}\n|∇f| = {np.linalg.norm(grad_f):.2f}'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='red', linewidth=2)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=props, family='monospace')

# Subplot 2: Directional Derivatives Comparison
ax2 = plt.subplot(2, 2, 2)

direction_names = ['d₁\n(ascent)', 'd₂\n(descent)', 'd₃\n(tangent)', 
                   'd₄\n(sub-opt)', 'd₅\n(poor)']
dir_derivatives = [
    results['ascent']['dir_derivative'],
    results['descent']['dir_derivative'],
    results['tangent']['dir_derivative'],
    results['suboptimal']['dir_derivative'],
    results['bad_descent']['dir_derivative']
]
colors_bar = ['green', 'blue', 'gray', 'orange', 'brown']

bars = ax2.bar(direction_names, dir_derivatives, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, val in zip(bars, dir_derivatives):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.3f}',
             ha='center', va='bottom' if height > 0 else 'top', 
             fontsize=10, fontweight='bold')

ax2.axhline(y=0, color='black', linewidth=2, alpha=0.8)
ax2.grid(True, axis='y', linestyle='--', alpha=0.4)
ax2.set_ylabel('Directional Derivative D_d f', fontsize=11, fontweight='bold')
ax2.set_title('Rate of Change Along Each Direction', fontsize=12, fontweight='bold', pad=10)
ax2.set_ylim(min(dir_derivatives)*1.3, max(dir_derivatives)*1.3)

# Add interpretation zones
ax2.axhspan(0, max(dir_derivatives)*1.3, alpha=0.1, color='green', label='Ascent')
ax2.axhspan(min(dir_derivatives)*1.3, 0, alpha=0.1, color='blue', label='Descent')
ax2.legend(loc='upper right', fontsize=9)

# Subplot 3: Angle Analysis
ax3 = plt.subplot(2, 2, 4)

angles = [
    results['ascent']['angle'],
    results['descent']['angle'],
    results['tangent']['angle'],
    results['suboptimal']['angle'],
    results['bad_descent']['angle']
]

# Create polar plot showing angles
theta_rad = np.radians(angles)
radii = np.abs(dir_derivatives)

# Plot on polar coordinates
ax3_polar = plt.subplot(2, 2, 4, projection='polar')
for i, (theta, radius, color, name) in enumerate(zip(theta_rad, radii, colors_bar, direction_names)):
    ax3_polar.plot([0, theta], [0, radius], color=color, linewidth=3, marker='o', 
                   markersize=8, label=name.replace('\n', ' '), alpha=0.8)

# Reference gradient at 0 degrees
grad_angle = 0  # We measure relative to gradient
ax3_polar.plot([0, grad_angle], [0, max(radii)*1.2], 'r--', linewidth=2, 
               label='∇f direction', alpha=0.6)

ax3_polar.set_ylim(0, max(radii)*1.3)
ax3_polar.set_theta_zero_location('E')
ax3_polar.set_theta_direction(1)
ax3_polar.set_title('Angular Relationship to Gradient\n(magnitude = |directional derivative|)', 
                    fontsize=11, fontweight='bold', pad=20, y=1.08)
ax3_polar.legend(loc='upper left', bbox_to_anchor=(1.15, 1.1), fontsize=8)
ax3_polar.grid(True, alpha=0.4)

plt.suptitle('Gradient Alignment and Directional Derivatives (Optimization)', 
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()

print("\n✓ Visualization complete!")
