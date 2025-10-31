"""
TORQUE CALCULATION - PHYSICS APPLICATION OF CROSS PRODUCT

Theory:
Torque (τ) is the rotational analog of force, calculated as:
    τ = r × F
    
where:
- r is the position vector from the rotation axis to the point of force application
- F is the applied force vector
- × denotes the cross product

Magnitude:
    |τ| = |r||F|sin(θ)
    
where θ is the angle between r and F

Direction:
The direction of τ is perpendicular to both r and F, following the right-hand rule:
- Point fingers along r
- Curl them toward F
- Thumb points in the direction of τ

Physical Interpretation:
- |τ| measures the "turning effectiveness" of a force
- Maximum torque: when F ⊥ r (sin(90°) = 1)
- Zero torque: when F || r (sin(0°) = 0)
- Perpendicular component of F does all the rotational work
- Lever arm: perpendicular distance from axis to force line = |r|sin(θ)

Applications:
- Wrenches and tools (mechanical advantage)
- Door hinges and handles
- Steering wheels
- Motors and engines
- Gyroscopes
- Planetary orbital mechanics
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def cross_product_3d(a, b):
    """Calculate 3D cross product"""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

def calculate_torque(r, F):
    """Calculate torque and related quantities"""
    # Torque vector
    tau = cross_product_3d(r, F)
    
    # Magnitudes
    norm_r = np.linalg.norm(r)
    norm_F = np.linalg.norm(F)
    norm_tau = np.linalg.norm(tau)
    
    # Angle between r and F
    dot_prod = np.dot(r, F)
    if norm_r > 0 and norm_F > 0:
        cos_theta = dot_prod / (norm_r * norm_F)
        cos_theta = np.clip(cos_theta, -1, 1)
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)
        sin_theta = np.sin(theta_rad)
    else:
        theta_deg = 0
        sin_theta = 0
    
    # Lever arm (perpendicular distance)
    lever_arm = norm_r * sin_theta if norm_r > 0 else 0
    
    # Perpendicular component of force
    if norm_r > 0:
        F_perp = F - (dot_prod / (norm_r**2)) * r
        norm_F_perp = np.linalg.norm(F_perp)
    else:
        F_perp = F
        norm_F_perp = norm_F
    
    return {
        'tau': tau,
        'norm_tau': norm_tau,
        'norm_r': norm_r,
        'norm_F': norm_F,
        'theta': theta_deg,
        'lever_arm': lever_arm,
        'F_perp': F_perp,
        'norm_F_perp': norm_F_perp
    }

# Define several torque scenarios
scenarios = {
    'Door Handle (Optimal)': {
        'r': np.array([0.8, 0, 0]),      # 0.8m from hinge (perpendicular)
        'F': np.array([0, 50, 0]),       # 50N perpendicular force
        'description': 'Force perpendicular to door at handle'
    },
    'Door Handle (Suboptimal)': {
        'r': np.array([0.8, 0, 0]),
        'F': np.array([0, 35, 25]),      # Force at an angle
        'description': 'Force at 35° angle to door'
    },
    'Wrench on Bolt': {
        'r': np.array([0.25, 0, 0]),     # 25cm wrench
        'F': np.array([0, 80, 30]),      # 80N force with some vertical component
        'description': 'Wrench handle perpendicular to bolt'
    },
    'Lever (Long arm)': {
        'r': np.array([1.5, 0, 0]),      # 1.5m lever arm
        'F': np.array([0, 0, -30]),      # 30N downward force
        'description': 'Long lever with perpendicular force'
    },
    'No Torque (Parallel)': {
        'r': np.array([0.5, 0, 0]),
        'F': np.array([40, 0, 0]),       # Force parallel to r
        'description': 'Force parallel to position vector'
    }
}

# Calculate torque for all scenarios
results = {}
for name, scenario in scenarios.items():
    results[name] = calculate_torque(scenario['r'], scenario['F'])
    results[name]['description'] = scenario['description']

print("\n" + "="*105)
print("TORQUE CALCULATION - CROSS PRODUCT APPLICATION IN PHYSICS")
print("="*105)
print("\nFormula: τ = r × F  (cross product)")
print("Magnitude: |τ| = |r||F|sin(θ)")
print("Direction: Perpendicular to both r and F (right-hand rule)")
print("-"*105)

# Create comprehensive table
data = {
    "Scenario": list(scenarios.keys()),
    "Position r (m)": [str(sc['r']) for sc in scenarios.values()],
    "Force F (N)": [str(sc['F']) for sc in scenarios.values()],
    "|r| (m)": [f"{results[name]['norm_r']:.3f}" for name in scenarios.keys()],
    "|F| (N)": [f"{results[name]['norm_F']:.3f}" for name in scenarios.keys()],
    "Angle θ": [f"{results[name]['theta']:.2f}°" for name in scenarios.keys()],
    "Lever Arm (m)": [f"{results[name]['lever_arm']:.3f}" for name in scenarios.keys()],
    "Torque τ (N·m)": [str(np.round(results[name]['tau'], 3)) for name in scenarios.keys()],
    "|τ| (N·m)": [f"{results[name]['norm_tau']:.3f}" for name in scenarios.keys()]
}

df = pd.DataFrame(data)
print(df.to_string(index=False))
print("-"*105)
print("\nKey Insight: Maximum torque occurs when force is perpendicular to position vector (θ = 90°)")
print("             Torque = 0 when force is parallel to position vector (θ = 0° or 180°)")
print("="*105)

# Visualization
fig = plt.figure(figsize=(16, 12))

# Plot configurations for 2D views
plot_configs = [
    ('Door Handle (Optimal)', 1, 'Maximum Torque\n(F ⊥ r)'),
    ('Door Handle (Suboptimal)', 2, 'Reduced Torque\n(F at angle)'),
    ('Wrench on Bolt', 3, 'Wrench Application\n(3D force)'),
    ('Lever (Long arm)', 4, 'Long Lever Arm\n(Mechanical advantage)'),
    ('No Torque (Parallel)', 5, 'Zero Torque\n(F || r)')
]

# Create 2D projections for first 5 subplots
for scenario_name, idx, title in plot_configs:
    ax = plt.subplot(3, 3, idx)
    
    r = scenarios[scenario_name]['r']
    F = scenarios[scenario_name]['F']
    result = results[scenario_name]
    tau = result['tau']
    
    # Project to 2D (XY plane)
    origin = np.array([0, 0])
    r_2d = r[:2]
    F_2d = F[:2]
    
    # Draw rotation axis (point at origin)
    ax.plot(0, 0, 'ko', markersize=15, label='Rotation Axis', zorder=5)
    
    # Draw position vector
    ax.quiver(*origin, *r_2d, angles='xy', scale_units='xy', scale=1,
              color='blue', width=0.012, label='r (position)', linewidth=3,
              headwidth=5, headlength=0.08, zorder=3)
    
    # Draw force vector from end of r
    scale_F = 0.01  # Scale force for visualization
    ax.quiver(r_2d[0], r_2d[1], F_2d[0]*scale_F, F_2d[1]*scale_F,
              angles='xy', scale_units='xy', scale=1,
              color='red', width=0.012, label='F (force)', linewidth=3,
              headwidth=5, headlength=0.08, zorder=3)
    
    # Draw perpendicular component
    F_perp_2d = result['F_perp'][:2]
    ax.quiver(r_2d[0], r_2d[1], F_perp_2d[0]*scale_F, F_perp_2d[1]*scale_F,
              angles='xy', scale_units='xy', scale=1,
              color='green', width=0.008, label='F⊥ (perpendicular)', 
              linewidth=2, linestyle='--', headwidth=4, headlength=0.08,
              zorder=2, alpha=0.7)
    
    # Draw angle arc
    if result['theta'] > 1 and result['theta'] < 179:
        angle_r = np.arctan2(r_2d[1], r_2d[0])
        angle_F = np.arctan2(F_2d[1], F_2d[0])
        arc_radius = min(0.15, result['norm_r']*0.3)
        
        # Create arc from r to F
        if angle_F < angle_r:
            angles_arc = np.linspace(angle_F, angle_r, 30)
        else:
            angles_arc = np.linspace(angle_r, angle_F, 30)
        
        arc_x = r_2d[0] + arc_radius * np.cos(angles_arc)
        arc_y = r_2d[1] + arc_radius * np.sin(angles_arc)
        ax.plot(arc_x, arc_y, 'k--', linewidth=1.5, alpha=0.6)
    
    # Labels
    ax.text(r_2d[0]*0.5, r_2d[1]*0.5, 'r', fontsize=12, color='blue',
            fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Add torque direction indicator (into or out of page)
    if tau[2] > 0.1:
        marker = '⊙'  # Out of page
        color = 'purple'
        direction = 'out'
    elif tau[2] < -0.1:
        marker = '⊗'  # Into page
        color = 'darkviolet'
        direction = 'in'
    else:
        marker = '○'
        color = 'gray'
        direction = 'zero'
    
    ax.text(0.95, 0.95, f'τ: {marker}', transform=ax.transAxes,
            fontsize=24, ha='right', va='top', color=color, fontweight='bold')
    
    # Annotation box
    textstr = f'|τ| = {result["norm_tau"]:.2f} N·m\nθ = {result["theta"]:.1f}°\nLever arm = {result["lever_arm"]:.2f} m'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9,
                edgecolor='black', linewidth=1.5)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props, family='monospace')
    
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('x (meters)', fontsize=9)
    ax.set_ylabel('y (meters)', fontsize=9)
    
    # Set limits
    max_extent = max(np.abs(r_2d).max(), np.abs(F_2d).max()*scale_F) * 1.5
    ax.set_xlim(-0.2, max_extent + 0.5)
    ax.set_ylim(-0.3, max_extent + 0.3)

# Subplot 6: Torque Magnitude Comparison
ax6 = plt.subplot(3, 3, 6)
names_short = [name.split('(')[0].strip() for name in scenarios.keys()]
torque_mags = [results[name]['norm_tau'] for name in scenarios.keys()]
colors_bar = ['green', 'orange', 'blue', 'purple', 'red']

bars = ax6.barh(names_short, torque_mags, color=colors_bar, alpha=0.7,
                edgecolor='black', linewidth=2)

# Add value labels
for bar, val in zip(bars, torque_mags):
    width = bar.get_width()
    ax6.text(width + 1, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}',
             ha='left', va='center', fontsize=10, fontweight='bold')

ax6.axvline(x=0, color='black', linewidth=2)
ax6.grid(True, axis='x', linestyle=':', alpha=0.4)
ax6.set_xlabel('Torque Magnitude |τ| (N·m)', fontsize=10, fontweight='bold')
ax6.set_title('Torque Comparison\n(Effectiveness of Different Configurations)',
              fontsize=11, fontweight='bold', pad=10)

# Subplot 7: 3D Visualization of Door Handle (Optimal)
ax7 = plt.subplot(3, 3, 7, projection='3d')

scenario_name = 'Door Handle (Optimal)'
r = scenarios[scenario_name]['r']
F = scenarios[scenario_name]['F']
tau = results[scenario_name]['tau']

origin_3d = np.array([0, 0, 0])

# Draw position vector
ax7.quiver(*origin_3d, *r, color='blue', linewidth=3, arrow_length_ratio=0.15,
           label='r (position)', alpha=0.8)

# Draw force vector
scale_F = 0.01
ax7.quiver(r[0], r[1], r[2], F[0]*scale_F, F[1]*scale_F, F[2]*scale_F,
           color='red', linewidth=3, arrow_length_ratio=0.15,
           label='F (force)', alpha=0.8)

# Draw torque vector (perpendicular to both)
scale_tau = 0.02
ax7.quiver(*origin_3d, tau[0]*scale_tau, tau[1]*scale_tau, tau[2]*scale_tau,
           color='purple', linewidth=4, arrow_length_ratio=0.15,
           label='τ (torque)', alpha=0.9)

# Draw rotation axis
ax7.plot([0, 0], [0, 0], [-0.3, 0.3], 'k--', linewidth=2, alpha=0.5,
         label='Rotation axis')

# Draw plane formed by r and F
xx, yy = np.meshgrid(np.linspace(-0.2, 1, 10), np.linspace(-0.2, 1, 10))
z_scale = r[0] if r[0] != 0 else 1
zz = 0 * xx  # XY plane
ax7.plot_surface(xx, yy, zz, alpha=0.1, color='yellow')

ax7.set_xlabel('X (m)', fontsize=9)
ax7.set_ylabel('Y (m)', fontsize=9)
ax7.set_zlabel('Z (m)', fontsize=9)
ax7.set_title('3D View: τ = r × F\n(Right-Hand Rule)',
              fontsize=11, fontweight='bold', pad=10)
ax7.legend(loc='upper left', fontsize=8)
ax7.set_box_aspect([1,1,0.8])

# Subplot 8: Lever Arm Concept
ax8 = plt.subplot(3, 3, 8)

# Illustrate lever arm concept with Door Handle (Suboptimal)
scenario_name = 'Door Handle (Suboptimal)'
r = scenarios[scenario_name]['r']
F = scenarios[scenario_name]['F']
result = results[scenario_name]

r_2d = r[:2]
F_2d = F[:2]
origin = np.array([0, 0])

# Draw setup
ax8.plot(0, 0, 'ko', markersize=15, zorder=5)
ax8.quiver(*origin, *r_2d, angles='xy', scale_units='xy', scale=1,
          color='blue', width=0.012, linewidth=3,
          headwidth=5, headlength=0.08, zorder=3)

scale_F = 0.01
ax8.quiver(r_2d[0], r_2d[1], F_2d[0]*scale_F, F_2d[1]*scale_F,
          angles='xy', scale_units='xy', scale=1,
          color='red', width=0.012, linewidth=3,
          headwidth=5, headlength=0.08, zorder=3)

# Draw lever arm (perpendicular distance)
# Line of action of force
F_unit = F_2d / np.linalg.norm(F_2d)
# Point on force line
force_line_points = np.array([r_2d + t * F_unit for t in np.linspace(-0.5, 0.5, 100)])
ax8.plot(force_line_points[:, 0], force_line_points[:, 1], 'r--', 
        linewidth=1.5, alpha=0.4, label='Line of action of F')

# Perpendicular from origin to line of action
# This is the lever arm
perp_point = r_2d + np.dot(-r_2d, F_unit) * F_unit
ax8.plot([0, perp_point[0]], [0, perp_point[1]], 'g-', linewidth=3,
        label=f'Lever arm = {result["lever_arm"]:.3f} m')
ax8.plot(perp_point[0], perp_point[1], 'go', markersize=8)

# Draw right angle indicator at perpendicular point
perp_size = 0.05
perp_dir = (origin - perp_point) / np.linalg.norm(origin - perp_point)
perp_side1 = perp_point + perp_size * perp_dir
perp_side2 = perp_point + perp_size * F_unit
perp_corner = perp_side1 + perp_size * F_unit
square_x = [perp_point[0], perp_side1[0], perp_corner[0], perp_side2[0], perp_point[0]]
square_y = [perp_point[1], perp_side1[1], perp_corner[1], perp_side2[1], perp_point[1]]
ax8.plot(square_x, square_y, 'g-', linewidth=1.5)

ax8.set_aspect('equal', adjustable='box')
ax8.grid(True, linestyle='--', alpha=0.4)
ax8.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax8.set_title('Lever Arm Concept\n(Perpendicular Distance to Force Line)',
             fontsize=11, fontweight='bold', pad=10)
ax8.set_xlabel('x (meters)', fontsize=9)
ax8.set_ylabel('y (meters)', fontsize=9)
ax8.set_xlim(-0.2, 1.2)
ax8.set_ylim(-0.4, 1.0)

# Subplot 9: Right-Hand Rule Illustration
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

rule_text = """
RIGHT-HAND RULE FOR CROSS PRODUCT

τ = r × F  (torque = position × force)

Step-by-Step:
1. Point fingers along first vector (r)
2. Curl fingers toward second vector (F)
3. Thumb points in direction of result (τ)

Properties:
• Non-commutative: r × F ≠ F × r
• Anti-commutative: r × F = -(F × r)
• Perpendicular: τ ⊥ r and τ ⊥ F
• Magnitude: |τ| = |r||F|sin(θ)

Special Cases:
• θ = 90°: Maximum torque (sin(90°) = 1)
• θ = 0° or 180°: Zero torque (sin(0°) = 0)

Torque Direction Symbols (looking down):
⊙ = Out of page (counterclockwise rotation)
⊗ = Into page (clockwise rotation)

Physical Meaning:
Torque causes rotational acceleration around
an axis perpendicular to both r and F.

Applications:
• Tightening bolts (wrench)
• Opening doors (handle placement)
• Steering vehicles (wheel radius)
• Balancing seesaws (moment balance)
"""

ax9.text(0.1, 0.95, rule_text, transform=ax9.transAxes, fontsize=9.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.95,
                  edgecolor='blue', linewidth=2, pad=1.5))

plt.suptitle('Torque Calculation via Cross Product (Physics)',
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.99))
plt.show()

print("\n✓ Visualization complete!")
