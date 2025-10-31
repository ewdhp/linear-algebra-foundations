"""
ANGULAR MOMENTUM - PHYSICS APPLICATION OF CROSS PRODUCT

Theory:
Angular momentum (L) is the rotational analog of linear momentum, calculated as:
    L = r × p = r × mv
    
where:
- r is the position vector from the reference point/axis
- p = mv is the linear momentum vector
- m is mass, v is velocity

Magnitude:
    |L| = |r||p|sin(θ) = m|r||v|sin(θ)
    
where θ is the angle between r and v

Direction:
The direction of L is perpendicular to both r and v, following the right-hand rule

Physical Interpretation:
- L measures "rotational inertia" in motion
- Conserved in closed systems (no external torque)
- Maximum when v ⊥ r (circular orbit)
- Zero when v || r (radial motion)

Key Relations:
- Torque: τ = dL/dt (rate of change of angular momentum)
- Conservation: If Σ τ = 0, then L = constant
- For circular motion: L = mvr (when v ⊥ r)

Applications:
- Planetary orbits (Kepler's second law)
- Figure skating spins (arms in/out)
- Gyroscopes and stabilization
- Atomic physics (electron angular momentum)
- Satellite motion
- Conservation in collisions
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Wedge
from matplotlib.collections import PatchCollection

def cross_product_3d(a, b):
    """Calculate 3D cross product"""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

def calculate_angular_momentum(r, v, mass):
    """Calculate angular momentum and related quantities"""
    # Linear momentum
    p = mass * v
    
    # Angular momentum
    L = cross_product_3d(r, p)
    
    # Magnitudes
    norm_r = np.linalg.norm(r)
    norm_v = np.linalg.norm(v)
    norm_p = np.linalg.norm(p)
    norm_L = np.linalg.norm(L)
    
    # Angle between r and v
    dot_prod = np.dot(r, v)
    if norm_r > 0 and norm_v > 0:
        cos_theta = dot_prod / (norm_r * norm_v)
        cos_theta = np.clip(cos_theta, -1, 1)
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)
        sin_theta = np.sin(theta_rad)
    else:
        theta_deg = 0
        sin_theta = 0
    
    # Perpendicular component of velocity
    if norm_r > 0:
        v_perp = v - (dot_prod / (norm_r**2)) * r
        norm_v_perp = np.linalg.norm(v_perp)
    else:
        v_perp = v
        norm_v_perp = norm_v
    
    # For circular motion (when v ⊥ r)
    is_circular = np.abs(dot_prod) < 0.01 * norm_r * norm_v
    
    return {
        'L': L,
        'p': p,
        'norm_L': norm_L,
        'norm_r': norm_r,
        'norm_v': norm_v,
        'norm_p': norm_p,
        'theta': theta_deg,
        'v_perp': v_perp,
        'norm_v_perp': norm_v_perp,
        'is_circular': is_circular
    }

# Define several angular momentum scenarios
scenarios = {
    'Circular Orbit (Planet)': {
        'r': np.array([1.5e11, 0, 0]),        # 1.5e11 m (Earth-Sun distance)
        'v': np.array([0, 3e4, 0]),           # 30 km/s perpendicular
        'mass': 5.97e24,                       # Earth mass (kg)
        'description': 'Planet in circular orbit',
        'scale_r': 1e-11,
        'scale_v': 1e-4
    },
    'Elliptical Orbit (Comet)': {
        'r': np.array([2e11, 0, 0]),
        'v': np.array([1e4, 2e4, 0]),         # Velocity at angle
        'mass': 1e13,                          # Comet mass
        'description': 'Comet in elliptical orbit',
        'scale_r': 1e-11,
        'scale_v': 1e-4
    },
    'Ice Skater (Arms Out)': {
        'r': np.array([0.6, 0, 0]),           # 60cm arm radius
        'v': np.array([0, 3, 0]),             # 3 m/s tangential
        'mass': 2,                             # 2kg arm mass
        'description': 'Skater spinning with arms extended',
        'scale_r': 1,
        'scale_v': 0.2
    },
    'Ice Skater (Arms In)': {
        'r': np.array([0.2, 0, 0]),           # 20cm arm radius (pulled in)
        'v': np.array([0, 9, 0]),             # 9 m/s (faster, conserved L)
        'mass': 2,                             # Same mass
        'description': 'Skater with arms pulled in (faster spin)',
        'scale_r': 1,
        'scale_v': 0.2
    },
    'Satellite': {
        'r': np.array([4.2e7, 0, 0]),         # 42,000 km altitude
        'v': np.array([0, 3100, 0]),          # Geostationary velocity
        'mass': 5000,                          # 5000 kg satellite
        'description': 'Geostationary satellite',
        'scale_r': 1e-7,
        'scale_v': 1e-3
    }
}

# Calculate angular momentum for all scenarios
results = {}
for name, scenario in scenarios.items():
    results[name] = calculate_angular_momentum(
        scenario['r'], scenario['v'], scenario['mass']
    )
    results[name]['description'] = scenario['description']

print("\n" + "="*115)
print("ANGULAR MOMENTUM - CROSS PRODUCT APPLICATION IN PHYSICS")
print("="*115)
print("\nFormula: L = r × p = r × mv  (cross product)")
print("Magnitude: |L| = m|r||v|sin(θ)")
print("Direction: Perpendicular to both r and v (right-hand rule)")
print("-"*115)

# Create comprehensive table
data = {
    "Scenario": list(scenarios.keys()),
    "Mass (kg)": [f"{sc['mass']:.2e}" for sc in scenarios.values()],
    "|r| (m)": [f"{results[name]['norm_r']:.2e}" for name in scenarios.keys()],
    "|v| (m/s)": [f"{results[name]['norm_v']:.2e}" for name in scenarios.keys()],
    "Angle θ": [f"{results[name]['theta']:.2f}°" for name in scenarios.keys()],
    "|p| (kg·m/s)": [f"{results[name]['norm_p']:.2e}" for name in scenarios.keys()],
    "|L| (kg·m²/s)": [f"{results[name]['norm_L']:.2e}" for name in scenarios.keys()],
    "Circular?": ["Yes ✓" if results[name]['is_circular'] else "No ✗" 
                  for name in scenarios.keys()]
}

df = pd.DataFrame(data)
print(df.to_string(index=False))
print("-"*115)
print("\nConservation: If no external torque acts on system, L remains constant")
print("              Example: Ice skater pulling arms in → ω increases to keep L constant")
print("="*115)

# Visualization
fig = plt.figure(figsize=(16, 12))

# Plot configurations
plot_configs = [
    ('Circular Orbit (Planet)', 1, 'Circular Planetary Orbit\n(Maximum L for given r, v)'),
    ('Elliptical Orbit (Comet)', 2, 'Elliptical Comet Orbit\n(L varies with position)'),
    ('Ice Skater (Arms Out)', 3, 'Ice Skater: Arms Extended\n(Large r, moderate ω)'),
    ('Ice Skater (Arms In)', 4, 'Ice Skater: Arms Pulled In\n(Small r, large ω, same L)'),
    ('Satellite', 5, 'Geostationary Satellite\n(Circular orbit, constant L)')
]

# Create 2D projections for scenarios
for scenario_name, idx, title in plot_configs:
    ax = plt.subplot(3, 3, idx)
    
    r = scenarios[scenario_name]['r']
    v = scenarios[scenario_name]['v']
    mass = scenarios[scenario_name]['mass']
    result = results[scenario_name]
    L = result['L']
    
    # Scale factors for visualization
    scale_r = scenarios[scenario_name]['scale_r']
    scale_v = scenarios[scenario_name]['scale_v']
    
    # Project to 2D (XY plane)
    origin = np.array([0, 0])
    r_2d = r[:2] * scale_r
    v_2d = v[:2] * scale_v
    
    # Draw reference point (origin)
    ax.plot(0, 0, 'ko', markersize=15, label='Reference Point', zorder=5)
    
    # Draw position vector
    ax.quiver(*origin, *r_2d, angles='xy', scale_units='xy', scale=1,
              color='blue', width=0.012, label='r (position)', linewidth=3,
              headwidth=5, headlength=0.15, zorder=3)
    
    # Draw velocity vector from end of r
    ax.quiver(r_2d[0], r_2d[1], v_2d[0], v_2d[1],
              angles='xy', scale_units='xy', scale=1,
              color='red', width=0.012, label='v (velocity)', linewidth=3,
              headwidth=5, headlength=0.15, zorder=3)
    
    # Draw perpendicular component of velocity
    v_perp_2d = result['v_perp'][:2] * scale_v
    ax.quiver(r_2d[0], r_2d[1], v_perp_2d[0], v_perp_2d[1],
              angles='xy', scale_units='xy', scale=1,
              color='green', width=0.008, label='v⊥ (perpendicular)', 
              linewidth=2, linestyle='--', headwidth=4, headlength=0.15,
              zorder=2, alpha=0.7)
    
    # Draw particle
    ax.plot(r_2d[0], r_2d[1], 'o', color='orange', markersize=12,
            markeredgecolor='black', markeredgewidth=2, zorder=4,
            label=f'Particle (m={mass:.1e} kg)')
    
    # Draw circular path hint if circular
    if result['is_circular']:
        circle = Circle((0, 0), np.linalg.norm(r_2d), fill=False,
                       edgecolor='gray', linestyle=':', linewidth=2, alpha=0.5)
        ax.add_patch(circle)
    
    # Draw angle arc if not perpendicular
    if 5 < result['theta'] < 175:
        angle_r = np.arctan2(r_2d[1], r_2d[0])
        angle_v = np.arctan2(v_2d[1], v_2d[0])
        arc_radius = min(0.3, np.linalg.norm(r_2d)*0.2)
        
        angles_arc = np.linspace(angle_r, angle_v, 30)
        arc_x = r_2d[0] + arc_radius * np.cos(angles_arc)
        arc_y = r_2d[1] + arc_radius * np.sin(angles_arc)
        ax.plot(arc_x, arc_y, 'k--', linewidth=1.5, alpha=0.6)
    
    # Add angular momentum direction indicator
    if L[2] > 0:
        marker = '⊙'  # Out of page (counterclockwise)
        color = 'purple'
    elif L[2] < 0:
        marker = '⊗'  # Into page (clockwise)
        color = 'darkviolet'
    else:
        marker = '○'
        color = 'gray'
    
    ax.text(0.95, 0.95, f'L: {marker}', transform=ax.transAxes,
            fontsize=24, ha='right', va='top', color=color, fontweight='bold')
    
    # Annotation box
    textstr = f'|L| = {result["norm_L"]:.2e} kg·m²/s\nθ = {result["theta"]:.1f}°\n|r| = {result["norm_r"]:.2e} m'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9,
                edgecolor='black', linewidth=1.5)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='bottom', bbox=props, family='monospace')
    
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9)
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=8)
    ax.set_xlabel('x', fontsize=9)
    ax.set_ylabel('y', fontsize=9)
    
    # Set limits
    max_extent = max(np.linalg.norm(r_2d), np.linalg.norm(v_2d)) * 1.8
    ax.set_xlim(-max_extent*0.3, max_extent)
    ax.set_ylim(-max_extent*0.3, max_extent)

# Subplot 6: Angular Momentum Conservation (Ice Skater)
ax6 = plt.subplot(3, 3, 6)

# Compare arms out vs arms in
names = ['Arms Out', 'Arms In']
scenario_names = ['Ice Skater (Arms Out)', 'Ice Skater (Arms In)']
radii = [results[s]['norm_r'] for s in scenario_names]
velocities = [results[s]['norm_v'] for s in scenario_names]
angular_momenta = [results[s]['norm_L'] for s in scenario_names]

x = np.arange(len(names))
width = 0.25

# Normalize for visualization
radii_norm = [r / radii[0] for r in radii]
velocities_norm = [v / velocities[0] for v in velocities]
L_norm = [L / angular_momenta[0] for L in angular_momenta]

bars1 = ax6.bar(x - width, radii_norm, width, label='Radius (normalized)',
               alpha=0.8, color='blue', edgecolor='black', linewidth=1.5)
bars2 = ax6.bar(x, velocities_norm, width, label='Velocity (normalized)',
               alpha=0.8, color='red', edgecolor='black', linewidth=1.5)
bars3 = ax6.bar(x + width, L_norm, width, label='Angular Momentum L',
               alpha=0.8, color='purple', edgecolor='black', linewidth=1.5)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax6.set_ylabel('Normalized Value', fontsize=10, fontweight='bold')
ax6.set_title('Conservation of Angular Momentum\n(Ice Skater Example)',
             fontsize=11, fontweight='bold', pad=10)
ax6.set_xticks(x)
ax6.set_xticklabels(names)
ax6.legend(loc='upper right', fontsize=9)
ax6.grid(True, axis='y', linestyle=':', alpha=0.4)
ax6.axhline(y=1.0, color='purple', linewidth=2, linestyle='--', alpha=0.6)

# Add explanation
textstr = 'When r decreases (arms in),\nv increases to conserve L'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.9)
ax6.text(0.5, 0.8, textstr, transform=ax6.transAxes, fontsize=9,
         ha='center', va='center', bbox=props, style='italic')

# Subplot 7: Angular Momentum Magnitude Comparison
ax7 = plt.subplot(3, 3, 7)

all_scenarios = list(scenarios.keys())
L_magnitudes = [results[name]['norm_L'] for name in all_scenarios]
colors_bar = ['green', 'orange', 'blue', 'purple', 'red']

# Use log scale for better visualization
L_magnitudes_log = [np.log10(L) if L > 0 else 0 for L in L_magnitudes]

bars = ax7.barh(all_scenarios, L_magnitudes_log, color=colors_bar, alpha=0.7,
               edgecolor='black', linewidth=2)

# Add value labels
for bar, val, val_orig in zip(bars, L_magnitudes_log, L_magnitudes):
    width = bar.get_width()
    ax7.text(width + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val_orig:.1e}',
             ha='left', va='center', fontsize=9, fontweight='bold')

ax7.axvline(x=0, color='black', linewidth=2)
ax7.grid(True, axis='x', linestyle=':', alpha=0.4)
ax7.set_xlabel('log₁₀(|L|) [kg·m²/s]', fontsize=10, fontweight='bold')
ax7.set_title('Angular Momentum Comparison\n(Log Scale)',
             fontsize=11, fontweight='bold', pad=10)

# Subplot 8: Kepler's Second Law (Equal Areas)
ax8 = plt.subplot(3, 3, 8)

# Illustrate Kepler's second law with elliptical orbit
# When L is conserved, area swept per time is constant

theta_vals = np.linspace(0, 2*np.pi, 100)
a, b = 2.0, 1.2  # Semi-major and semi-minor axes
x_ellipse = a * np.cos(theta_vals)
y_ellipse = b * np.sin(theta_vals)

ax8.plot(x_ellipse, y_ellipse, 'b-', linewidth=2, label='Orbital path')
ax8.plot(0, 0, 'yo', markersize=20, markeredgecolor='orange', 
        markeredgewidth=2, label='Sun/Center', zorder=5)

# Draw two positions and swept areas
# Position 1 (near perihelion - closest)
angle1 = 0
r1 = np.array([a * np.cos(angle1), b * np.sin(angle1)])
ax8.plot(r1[0], r1[1], 'ro', markersize=10, markeredgecolor='black',
        markeredgewidth=2, zorder=4)
ax8.plot([0, r1[0]], [0, r1[1]], 'r-', linewidth=2, alpha=0.6)

# Small angular sweep at perihelion
delta_angle1 = 0.15
angles1 = np.linspace(angle1, angle1 + delta_angle1, 20)
r_sweep1 = np.array([[0, 0]] + [[a * np.cos(t), b * np.sin(t)] for t in angles1])
ax8.fill(r_sweep1[:, 0], r_sweep1[:, 1], 'red', alpha=0.3, 
        edgecolor='red', linewidth=2, label='Area 1 (short time)')

# Position 2 (near aphelion - farthest)
angle2 = np.pi
r2 = np.array([a * np.cos(angle2), b * np.sin(angle2)])
ax8.plot(r2[0], r2[1], 'go', markersize=10, markeredgecolor='black',
        markeredgewidth=2, zorder=4)
ax8.plot([0, r2[0]], [0, r2[1]], 'g-', linewidth=2, alpha=0.6)

# Larger angular sweep at aphelion (same area, longer time)
delta_angle2 = 0.4
angles2 = np.linspace(angle2, angle2 + delta_angle2, 20)
r_sweep2 = np.array([[0, 0]] + [[a * np.cos(t), b * np.sin(t)] for t in angles2])
ax8.fill(r_sweep2[:, 0], r_sweep2[:, 1], 'green', alpha=0.3,
        edgecolor='green', linewidth=2, label='Area 2 (same time)')

ax8.set_aspect('equal', adjustable='box')
ax8.grid(True, linestyle='--', alpha=0.4)
ax8.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax8.set_title("Kepler's Second Law\n(Equal areas in equal times)",
             fontsize=11, fontweight='bold', pad=10)
ax8.set_xlabel('x', fontsize=9)
ax8.set_ylabel('y', fontsize=9)
ax8.set_xlim(-2.5, 2.5)
ax8.set_ylim(-1.8, 1.8)

# Add explanation
textstr = 'L conserved → dA/dt = const\nPlanet moves faster\nwhen closer to sun'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.95)
ax8.text(0.98, 0.02, textstr, transform=ax8.transAxes, fontsize=8.5,
         ha='right', va='bottom', bbox=props, style='italic')

# Subplot 9: Theory and Applications
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

theory_text = """
ANGULAR MOMENTUM THEORY & APPLICATIONS

Definition:  L = r × p = m(r × v)

Properties:
• Vector quantity (magnitude + direction)
• Perpendicular to plane of motion
• Conserved when Σ τ_ext = 0

Conservation Examples:
1. Ice Skater
   L = Iω = constant
   Arms in → I↓ → ω↑ (spins faster)

2. Planetary Orbits
   L = mrv sin(θ) = constant
   Closer to sun → v↑ (Kepler's 2nd law)

3. Gyroscopes
   L resists change in orientation
   Used in navigation, stabilization

Relationship to Torque:
   τ = dL/dt
   No external torque → L conserved

Applications:
• Satellite attitude control
• Angular momentum wheels in spacecraft
• Figure skating and diving
• Quantum mechanics (orbital angular momentum)
• Precession of spinning tops
• Conservation in collisions (billiards)

Right-Hand Rule:
Curl fingers from r toward v,
thumb points along L direction
"""

ax9.text(0.08, 0.95, theory_text, transform=ax9.transAxes, fontsize=9.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.95,
                  edgecolor='blue', linewidth=2, pad=1.5))

plt.suptitle('Angular Momentum via Cross Product (Physics)',
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.99))
plt.show()

print("\n✓ Visualization complete!")
print("\nConservation Example: Ice skater pulling arms in")
print(f"  Arms Out:  L = {results['Ice Skater (Arms Out)']['norm_L']:.2f} kg·m²/s")
print(f"  Arms In:   L = {results['Ice Skater (Arms In)']['norm_L']:.2f} kg·m²/s")
print(f"  Ratio:     {results['Ice Skater (Arms In)']['norm_L'] / results['Ice Skater (Arms Out)']['norm_L']:.3f} (≈1, conserved!)")
