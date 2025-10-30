"""
WORK CALCULATION - PHYSICS APPLICATION

Theory:
Work is the scalar product of force and displacement:
    W = F·d = |F||d|cos(θ)

Physical Interpretation:
- Only the component of force parallel to displacement does work
- If F ⊥ d (perpendicular), then W = 0 (e.g., centripetal force)
- Positive work: force aids motion (θ < 90°)
- Negative work: force opposes motion (θ > 90°)
- Maximum work: force aligned with motion (θ = 0°)

Units: Work is measured in Joules (J) = Newton·meter (N·m)

Applications:
- Calculating energy transfer
- Analyzing mechanical systems
- Understanding conservative vs non-conservative forces
- Power calculation: P = F·v (force dot velocity)
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Define force and displacement vectors (in SI units: N and m)
F = np.array([10, 5])     # Force in Newtons
d = np.array([4, 2])      # Displacement in meters
F_perp = np.array([-2, 4]) # Perpendicular force (no work)
F_oppose = np.array([-6, -3]) # Opposing force (negative work)

def calculate_work(force, displacement):
    """Calculate work and related quantities"""
    work = np.dot(force, displacement)
    norm_F = np.linalg.norm(force)
    norm_d = np.linalg.norm(displacement)
    
    # Calculate angle
    cos_theta = work / (norm_F * norm_d) if work != 0 else 0
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_deg = np.degrees(np.arccos(cos_theta))
    
    # Parallel and perpendicular components
    d_unit = displacement / norm_d
    F_parallel = np.dot(force, d_unit) * d_unit
    F_perp_component = force - F_parallel
    
    return {
        'work': work,
        'norm_F': norm_F,
        'norm_d': norm_d,
        'angle': theta_deg,
        'F_parallel': F_parallel,
        'F_perp': F_perp_component,
        'F_parallel_mag': np.linalg.norm(F_parallel)
    }

# Calculate for different scenarios
result1 = calculate_work(F, d)
result2 = calculate_work(F_perp, d)
result3 = calculate_work(F_oppose, d)

print("\n" + "="*90)
print("WORK CALCULATION - DOT PRODUCT APPLICATION IN PHYSICS")
print("="*90)
print("\nFormula: W = F·d = |F||d|cos(θ)")
print("Principle: Only the component of force along displacement does work")
print("-"*90)

# Create detailed table
data = {
    "Scenario": [
        "Case 1: Force aids motion",
        "Case 2: Force perpendicular",
        "Case 3: Force opposes motion"
    ],
    "Force F (N)": [
        str(F),
        str(F_perp),
        str(F_oppose)
    ],
    "Displacement d (m)": [
        str(d),
        str(d),
        str(d)
    ],
    "|F| (N)": [
        f"{result1['norm_F']:.3f}",
        f"{result2['norm_F']:.3f}",
        f"{result3['norm_F']:.3f}"
    ],
    "|d| (m)": [
        f"{result1['norm_d']:.3f}",
        f"{result2['norm_d']:.3f}",
        f"{result3['norm_d']:.3f}"
    ],
    "θ (angle)": [
        f"{result1['angle']:.2f}°",
        f"{result2['angle']:.2f}°",
        f"{result3['angle']:.2f}°"
    ],
    "|F_parallel| (N)": [
        f"{result1['F_parallel_mag']:.3f}",
        f"{result2['F_parallel_mag']:.6f}",
        f"{result3['F_parallel_mag']:.3f}"
    ],
    "Work W (J)": [
        f"{result1['work']:.3f}",
        f"{result2['work']:.6f}",
        f"{result3['work']:.3f}"
    ],
    "Interpretation": [
        "Positive (energy added)",
        "Zero (no work done)",
        "Negative (energy removed)"
    ]
}

df = pd.DataFrame(data)
print(df.to_string(index=False))
print("-"*90)
print(f"\nKey Insight: Work = (Force component along displacement) × (distance)")
print("             W = |F_parallel| × |d| = F·d")
print("="*90)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
origin = np.array([0, 0])

# Plot 1: Positive Work (Force aids motion)
ax1 = axes[0]
ax1.quiver(*origin, *d, angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.008, label='d (displacement)', linewidth=3, headwidth=4)
ax1.quiver(*origin, *F, angles='xy', scale_units='xy', scale=1, 
           color='red', width=0.008, label='F (force)', linewidth=3, headwidth=4)
ax1.quiver(*origin, *result1['F_parallel'], angles='xy', scale_units='xy', scale=1, 
           color='green', width=0.006, label='F_∥ (parallel)', linewidth=2.5, 
           linestyle='--', headwidth=3.5, alpha=0.8)

# Draw perpendicular component with dashed line
ax1.plot([result1['F_parallel'][0], F[0]], [result1['F_parallel'][1], F[1]], 
         'purple', linestyle=':', linewidth=2, alpha=0.6, label='F_⊥ (perpendicular)')

# Draw angle arc
angle_d = np.arctan2(d[1], d[0])
angle_F = np.arctan2(F[1], F[0])
arc_angles = np.linspace(angle_d, angle_F, 30)
arc_radius = 2
ax1.plot(arc_radius * np.cos(arc_angles), arc_radius * np.sin(arc_angles), 
         'k--', linewidth=1.5, alpha=0.7)
mid_angle = (angle_d + angle_F) / 2
ax1.text(2.5*np.cos(mid_angle), 2.5*np.sin(mid_angle), f'θ={result1["angle"]:.1f}°', 
         fontsize=10, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Labels
ax1.text(d[0]*1.15, d[1]*1.15, 'd', fontsize=13, color='blue', fontweight='bold')
ax1.text(F[0]*1.1, F[1]*1.1, 'F', fontsize=13, color='red', fontweight='bold')

# Annotation box
textstr = f'W = F·d = {result1["work"]:.2f} J\n|F| = {result1["norm_F"]:.2f} N\n|d| = {result1["norm_d"]:.2f} m\nθ = {result1["angle"]:.1f}°\n\nPositive Work ✓'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='green', linewidth=2)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9.5,
         verticalalignment='top', bbox=props, family='monospace')

ax1.set_xlim(-1, 12)
ax1.set_ylim(-1, 7)
ax1.set_aspect('equal', adjustable='box')
ax1.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
ax1.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax1.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax1.set_title('Case 1: Force Aids Motion\n(Positive Work)', 
              fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('x (meters)', fontsize=10)
ax1.set_ylabel('y (meters)', fontsize=10)

# Plot 2: Zero Work (Perpendicular Force)
ax2 = axes[1]
ax2.quiver(*origin, *d, angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.008, label='d (displacement)', linewidth=3, headwidth=4)
ax2.quiver(*origin, *F_perp, angles='xy', scale_units='xy', scale=1, 
           color='purple', width=0.008, label='F_⊥ (perpendicular force)', 
           linewidth=3, headwidth=4)

# Draw right angle indicator
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
square_size = 0.6
square = Rectangle((0, 0), square_size, square_size, fill=False, 
                   edgecolor='purple', linewidth=2, linestyle='--')
transform_mat = transforms.Affine2D().rotate(angle_d) + ax2.transData
square.set_transform(transform_mat)
ax2.add_patch(square)

# Labels
ax2.text(d[0]*1.15, d[1]*1.15, 'd', fontsize=13, color='blue', fontweight='bold')
ax2.text(F_perp[0]*1.2, F_perp[1]*1.1, 'F⊥', fontsize=13, color='purple', fontweight='bold')

# Annotation box
textstr = f'W = F·d = {result2["work"]:.6f} J\n|F_⊥| = {result2["norm_F"]:.2f} N\n|d| = {result2["norm_d"]:.2f} m\nθ = 90.0°\n\nNo Work Done ○'
props = dict(boxstyle='round', facecolor='lightgray', alpha=0.9, edgecolor='purple', linewidth=2)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=9.5,
         verticalalignment='top', bbox=props, family='monospace')

# Add explanatory text
ax2.text(0.5, 0.12, 'Centripetal force example:\nNo work in circular motion', 
         transform=ax2.transAxes, fontsize=9, ha='center', style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax2.set_xlim(-3, 6)
ax2.set_ylim(-1, 6)
ax2.set_aspect('equal', adjustable='box')
ax2.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
ax2.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax2.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax2.set_title('Case 2: Perpendicular Force\n(Zero Work)', 
              fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('x (meters)', fontsize=10)
ax2.set_ylabel('y (meters)', fontsize=10)

# Plot 3: Negative Work (Force opposes motion)
ax3 = axes[2]
ax3.quiver(*origin, *d, angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.008, label='d (displacement)', linewidth=3, headwidth=4)
ax3.quiver(*origin, *F_oppose, angles='xy', scale_units='xy', scale=1, 
           color='darkred', width=0.008, label='F (opposing force)', linewidth=3, headwidth=4)
ax3.quiver(*origin, *result3['F_parallel'], angles='xy', scale_units='xy', scale=1, 
           color='orange', width=0.006, label='F_∥ (parallel, opposite)', 
           linewidth=2.5, linestyle='--', headwidth=3.5, alpha=0.8)

# Draw angle arc
angle_F_oppose = np.arctan2(F_oppose[1], F_oppose[0])
if angle_F_oppose < angle_d:
    arc_angles3 = np.linspace(angle_F_oppose, angle_d, 30)
else:
    arc_angles3 = np.linspace(angle_d, angle_F_oppose, 30)
ax3.plot(arc_radius * np.cos(arc_angles3), arc_radius * np.sin(arc_angles3), 
         'k--', linewidth=1.5, alpha=0.7)
mid_angle3 = (angle_d + angle_F_oppose) / 2
ax3.text(2.5*np.cos(mid_angle3), 2.5*np.sin(mid_angle3), f'θ={result3["angle"]:.1f}°', 
         fontsize=10, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Labels
ax3.text(d[0]*1.15, d[1]*1.15, 'd', fontsize=13, color='blue', fontweight='bold')
ax3.text(F_oppose[0]*1.15, F_oppose[1]*1.15, 'F', fontsize=13, color='darkred', fontweight='bold')

# Annotation box
textstr = f'W = F·d = {result3["work"]:.2f} J\n|F| = {result3["norm_F"]:.2f} N\n|d| = {result3["norm_d"]:.2f} m\nθ = {result3["angle"]:.1f}°\n\nNegative Work ✗'
props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.9, edgecolor='darkred', linewidth=2)
ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=9.5,
         verticalalignment='top', bbox=props, family='monospace')

# Add explanatory text
ax3.text(0.5, 0.12, 'Friction/Drag force example:\nRemoves energy from system', 
         transform=ax3.transAxes, fontsize=9, ha='center', style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax3.set_xlim(-8, 5)
ax3.set_ylim(-4, 4)
ax3.set_aspect('equal', adjustable='box')
ax3.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
ax3.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax3.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax3.set_title('Case 3: Force Opposes Motion\n(Negative Work)', 
              fontsize=12, fontweight='bold', pad=10)
ax3.set_xlabel('x (meters)', fontsize=10)
ax3.set_ylabel('y (meters)', fontsize=10)

plt.suptitle('Work Calculation via Dot Product (Physics)', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()

print("\n✓ Visualization complete!")
