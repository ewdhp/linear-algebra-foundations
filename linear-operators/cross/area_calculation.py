"""
AREA CALCULATION - GEOMETRY APPLICATION OF CROSS PRODUCT

Theory:
The magnitude of the cross product of two vectors gives the area of
the parallelogram they span:
    Area(parallelogram) = |a × b|
    
For a triangle formed by vectors a and b:
    Area(triangle) = (1/2)|a × b|

In 2D (treating as 3D with z=0):
    a = (a_x, a_y, 0)
    b = (b_x, b_y, 0)
    a × b = (0, 0, a_x*b_y - a_y*b_x)
    |a × b| = |a_x*b_y - a_y*b_x|

Geometric Interpretation:
- The cross product magnitude equals the parallelogram area
- Sign indicates orientation (counterclockwise/clockwise)
- Works in any dimension when embedded in 3D
- Base × height formula emerges: |a||b|sin(θ)

Properties:
- Area is invariant under translation
- Area scales with vector magnitudes
- Maximum area when vectors are perpendicular
- Zero area when vectors are parallel (collinear)

Applications:
- Computer graphics (polygon rendering)
- Computational geometry (polygon areas)
- Geographic information systems (land parcel areas)
- Physics (flux through surfaces)
- Surveying and mapping
- Collision detection in games
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch
from matplotlib.collections import PatchCollection

def cross_product_2d(a, b):
    """Calculate 2D cross product (scalar result, z-component)"""
    return a[0]*b[1] - a[1]*b[0]

def cross_product_3d(a, b):
    """Calculate 3D cross product"""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

def calculate_areas(a, b):
    """Calculate parallelogram and triangle areas"""
    # Extend to 3D for cross product
    a_3d = np.array([a[0], a[1], 0])
    b_3d = np.array([b[0], b[1], 0])
    
    # Cross product
    cross = cross_product_3d(a_3d, b_3d)
    cross_2d = cross_product_2d(a, b)
    
    # Area of parallelogram
    area_parallelogram = np.abs(cross_2d)
    
    # Area of triangle
    area_triangle = area_parallelogram / 2
    
    # Magnitudes and angle
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a > 0 and norm_b > 0:
        cos_theta = np.dot(a, b) / (norm_a * norm_b)
        cos_theta = np.clip(cos_theta, -1, 1)
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)
        sin_theta = np.sin(theta_rad)
    else:
        theta_deg = 0
        sin_theta = 0
    
    # Height of parallelogram (perpendicular distance)
    height = norm_b * sin_theta if norm_a > 0 else 0
    
    # Verification: base × height
    area_verification = norm_a * height
    
    return {
        'area_parallelogram': area_parallelogram,
        'area_triangle': area_triangle,
        'norm_a': norm_a,
        'norm_b': norm_b,
        'theta': theta_deg,
        'height': height,
        'cross_2d': cross_2d,
        'area_verification': area_verification
    }

# Define various geometric shapes
shapes = {
    'Square': {
        'a': np.array([2, 0]),
        'b': np.array([0, 2]),
        'description': 'Perpendicular vectors (square)'
    },
    'Rectangle': {
        'a': np.array([4, 0]),
        'b': np.array([0, 2]),
        'description': 'Perpendicular vectors (rectangle)'
    },
    'Rhombus': {
        'a': np.array([3, 0]),
        'b': np.array([1.5, 2]),
        'description': 'Equal length vectors at angle'
    },
    'General Parallelogram': {
        'a': np.array([4, 1]),
        'b': np.array([1, 3]),
        'description': 'General case parallelogram'
    },
    'Acute Triangle': {
        'a': np.array([3, 1]),
        'b': np.array([1, 2.5]),
        'description': 'Triangle with acute angles'
    },
    'Nearly Parallel': {
        'a': np.array([5, 0]),
        'b': np.array([4.8, 0.5]),
        'description': 'Almost parallel (small area)'
    }
}

# Calculate areas for all shapes
results = {}
for name, shape in shapes.items():
    results[name] = calculate_areas(shape['a'], shape['b'])
    results[name]['description'] = shape['description']

print("\n" + "="*110)
print("AREA CALCULATION - CROSS PRODUCT APPLICATION IN GEOMETRY")
print("="*110)
print("\nFormula: Area(parallelogram) = |a × b| = |a||b|sin(θ)")
print("         Area(triangle) = (1/2)|a × b|")
print("In 2D: a × b = a_x*b_y - a_y*b_x (z-component)")
print("-"*110)

# Create comprehensive table
data = {
    "Shape": list(shapes.keys()),
    "Vector a": [str(s['a']) for s in shapes.values()],
    "Vector b": [str(s['b']) for s in shapes.values()],
    "|a|": [f"{results[name]['norm_a']:.3f}" for name in shapes.keys()],
    "|b|": [f"{results[name]['norm_b']:.3f}" for name in shapes.keys()],
    "Angle θ": [f"{results[name]['theta']:.2f}°" for name in shapes.keys()],
    "Height": [f"{results[name]['height']:.3f}" for name in shapes.keys()],
    "Parallelogram Area": [f"{results[name]['area_parallelogram']:.3f}" for name in shapes.keys()],
    "Triangle Area": [f"{results[name]['area_triangle']:.3f}" for name in shapes.keys()]
}

df = pd.DataFrame(data)
print(df.to_string(index=False))
print("-"*110)
print("\nVerification: Area = base × height = |a| × |b|sin(θ)")
print("              Cross product automatically computes this!")
print("="*110)

# Visualization
fig = plt.figure(figsize=(16, 12))

# Plot configurations for shapes
plot_configs = list(shapes.items())[:6]

# Create visualizations for each shape
for idx, (shape_name, shape_data) in enumerate(plot_configs, 1):
    ax = plt.subplot(3, 3, idx)
    
    a = shape_data['a']
    b = shape_data['b']
    result = results[shape_name]
    
    origin = np.array([0, 0])
    
    # Draw parallelogram
    vertices = np.array([
        origin,
        a,
        a + b,
        b,
        origin
    ])
    
    parallelogram = Polygon(vertices[:-1], alpha=0.3, facecolor='lightblue',
                           edgecolor='blue', linewidth=2.5)
    ax.add_patch(parallelogram)
    
    # Draw triangle (half of parallelogram)
    triangle = Polygon([origin, a, b], alpha=0.4, facecolor='lightcoral',
                      edgecolor='red', linewidth=2, linestyle='--')
    ax.add_patch(triangle)
    
    # Draw vectors
    ax.quiver(*origin, *a, angles='xy', scale_units='xy', scale=1,
              color='darkblue', width=0.012, label='a', linewidth=3,
              headwidth=5, headlength=0.2, zorder=3)
    
    ax.quiver(*origin, *b, angles='xy', scale_units='xy', scale=1,
              color='darkgreen', width=0.012, label='b', linewidth=3,
              headwidth=5, headlength=0.2, zorder=3)
    
    # Draw height line (perpendicular from b to a)
    # Project b onto a
    if result['norm_a'] > 0:
        a_unit = a / result['norm_a']
        proj_length = np.dot(b, a_unit)
        proj_point = proj_length * a_unit
        
        # Draw height
        ax.plot([b[0], proj_point[0]], [b[1], proj_point[1]], 
               'purple', linewidth=2.5, linestyle=':', 
               label=f'height = {result["height"]:.2f}', zorder=2)
        
        # Draw right angle marker
        perp_size = 0.15
        perp_vec = (b - proj_point) / np.linalg.norm(b - proj_point) if np.linalg.norm(b - proj_point) > 0 else np.array([0, 1])
        corner_points = [
            proj_point,
            proj_point + perp_size * a_unit,
            proj_point + perp_size * a_unit + perp_size * perp_vec,
            proj_point + perp_size * perp_vec,
            proj_point
        ]
        ax.plot([p[0] for p in corner_points], [p[1] for p in corner_points],
               'purple', linewidth=1.5)
    
    # Draw angle arc
    if 5 < result['theta'] < 175:
        angle_a = np.arctan2(a[1], a[0])
        angle_b = np.arctan2(b[1], b[0])
        arc_radius = min(0.5, result['norm_a']*0.25, result['norm_b']*0.25)
        
        angles_arc = np.linspace(angle_a, angle_b, 30)
        arc_x = arc_radius * np.cos(angles_arc)
        arc_y = arc_radius * np.sin(angles_arc)
        ax.plot(arc_x, arc_y, 'k--', linewidth=1.5, alpha=0.6)
        
        mid_angle = (angle_a + angle_b) / 2
        ax.text(arc_radius*1.5*np.cos(mid_angle), arc_radius*1.5*np.sin(mid_angle),
               f'{result["theta"]:.0f}°', fontsize=9, ha='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Labels for vectors
    ax.text(a[0]*0.5, a[1]*0.5, 'a', fontsize=12, color='darkblue',
           fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.text(b[0]*0.5, b[1]*0.5, 'b', fontsize=12, color='darkgreen',
           fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Annotation box
    textstr = f'Parallelogram: {result["area_parallelogram"]:.2f}\nTriangle: {result["area_triangle"]:.2f}\nFormula: {result["norm_a"]:.2f}×{result["height"]:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.95,
                edgecolor='black', linewidth=1.5)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=8.5,
           ha='right', va='bottom', bbox=props, family='monospace')
    
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_title(f'{shape_name}\n{shape_data["description"]}',
                fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('x', fontsize=9)
    ax.set_ylabel('y', fontsize=9)
    
    # Set limits
    max_x = max(abs(a[0]), abs(b[0]), abs(a[0]+b[0])) * 1.3
    max_y = max(abs(a[1]), abs(b[1]), abs(a[1]+b[1])) * 1.3
    ax.set_xlim(-0.5, max_x + 0.5)
    ax.set_ylim(-0.5, max_y + 0.5)

# Subplot 7: Area Comparison Bar Chart
ax7 = plt.subplot(3, 3, 7)

names_short = list(shapes.keys())
areas_para = [results[name]['area_parallelogram'] for name in names_short]
areas_tri = [results[name]['area_triangle'] for name in names_short]

x = np.arange(len(names_short))
width = 0.35

bars1 = ax7.bar(x - width/2, areas_para, width, label='Parallelogram',
               alpha=0.8, color='lightblue', edgecolor='blue', linewidth=2)
bars2 = ax7.bar(x + width/2, areas_tri, width, label='Triangle',
               alpha=0.8, color='lightcoral', edgecolor='red', linewidth=2)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax7.set_ylabel('Area (square units)', fontsize=10, fontweight='bold')
ax7.set_title('Area Comparison\n(Parallelogram vs Triangle)',
             fontsize=11, fontweight='bold', pad=10)
ax7.set_xticks(x)
ax7.set_xticklabels(names_short, rotation=15, ha='right', fontsize=8)
ax7.legend(loc='upper left', fontsize=9)
ax7.grid(True, axis='y', linestyle=':', alpha=0.4)

# Subplot 8: Complex Polygon (Shoelace Formula)
ax8 = plt.subplot(3, 3, 8)

# Define a complex polygon using multiple points
polygon_points = np.array([
    [1, 1],
    [4, 1.5],
    [5, 4],
    [3, 5],
    [0.5, 3.5],
    [1, 1]  # Close the polygon
])

# Calculate area using cross product (shoelace formula)
area_total = 0
for i in range(len(polygon_points) - 1):
    v1 = polygon_points[i]
    v2 = polygon_points[i + 1]
    area_total += cross_product_2d(v1, v2)
area_total = abs(area_total) / 2

# Draw polygon
poly = Polygon(polygon_points[:-1], alpha=0.4, facecolor='lightgreen',
              edgecolor='darkgreen', linewidth=3)
ax8.add_patch(poly)

# Draw and label vertices
for i, point in enumerate(polygon_points[:-1]):
    ax8.plot(point[0], point[1], 'ro', markersize=10, markeredgecolor='black',
            markeredgewidth=2, zorder=4)
    ax8.text(point[0], point[1], f' P{i+1}', fontsize=10, fontweight='bold',
            ha='left', va='bottom')

# Draw edges as vectors
for i in range(len(polygon_points) - 1):
    start = polygon_points[i]
    end = polygon_points[i + 1]
    ax8.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color='darkgreen',
                              lw=2, alpha=0.6))

ax8.set_aspect('equal', adjustable='box')
ax8.grid(True, linestyle='--', alpha=0.4)
ax8.set_title(f'Complex Polygon Area\n(Shoelace Formula: Area = {area_total:.2f})',
             fontsize=11, fontweight='bold', pad=10)
ax8.set_xlabel('x', fontsize=9)
ax8.set_ylabel('y', fontsize=9)
ax8.set_xlim(-0.5, 6)
ax8.set_ylim(0, 6)

# Add explanation
textstr = 'Area = (1/2)|Σ(x_i*y_{i+1} - x_{i+1}*y_i)|\nSum of cross products'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.95)
ax8.text(0.02, 0.98, textstr, transform=ax8.transAxes, fontsize=8.5,
        ha='left', va='top', bbox=props, family='monospace')

# Subplot 9: Theory and Formulas
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

theory_text = """
CROSS PRODUCT FOR AREA CALCULATION

2D Cross Product (Scalar):
  a × b = a_x*b_y - a_y*b_x

3D Cross Product (Vector):
  a × b = |  i    j    k  |
          | a_x  a_y  a_z |
          | b_x  b_y  b_z |

Area Formulas:
  Parallelogram: A = |a × b|
  Triangle:      A = (1/2)|a × b|
  
Alternative Form:
  A = |a||b|sin(θ)
  where θ is angle between vectors

Shoelace Formula (Polygon):
  For vertices (x₁,y₁),...,(x_n,y_n):
  A = (1/2)|Σ(x_i*y_{i+1} - x_{i+1}*y_i)|

Properties:
• Geometric: Area is magnitude of cross product
• Sign: Positive if CCW, negative if CW
• Zero: When vectors are parallel
• Maximum: When vectors are perpendicular

Applications:
• Computer Graphics: Polygon rendering
• GIS: Land area calculations
• Surveying: Irregular plot areas
• Physics: Flux calculations
• Collision Detection: Point in polygon
• Mesh generation: Triangle areas

Key Insight:
Cross product encodes both area AND
orientation in a single operation!
"""

ax9.text(0.08, 0.95, theory_text, transform=ax9.transAxes, fontsize=9.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.95,
                  edgecolor='blue', linewidth=2, pad=1.5))

plt.suptitle('Area Calculation via Cross Product (Geometry)',
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.99))
plt.show()

print("\n✓ Visualization complete!")
print("\nNote: Cross product provides both area magnitude and orientation")
print("      Positive: counterclockwise, Negative: clockwise")
