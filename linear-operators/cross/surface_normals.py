"""
SURFACE NORMALS - COMPUTER GRAPHICS APPLICATION OF CROSS PRODUCT

Theory:
In 3D graphics, the normal vector to a surface is perpendicular to that surface.
For a triangle or polygon face defined by vertices, the normal is computed using
the cross product of two edge vectors:
    
    n = (v1 - v0) × (v2 - v0)
    n̂ = n / |n|  (unit normal)

where v0, v1, v2 are vertices of the triangle

Direction:
The direction of the normal follows the right-hand rule and depends on vertex order:
- Counterclockwise (CCW) vertices → outward facing normal
- Clockwise (CW) vertices → inward facing normal

Physical Applications:
- Lighting calculations (Lambertian/diffuse reflection)
- Shading (Phong, Gouraud, flat shading)
- Backface culling (optimization)
- Collision detection
- Ray tracing (reflection/refraction)
- Normal mapping and bump mapping

Lighting Model:
The intensity of diffuse lighting depends on the dot product of the normal
and light direction:
    I = I_ambient + I_diffuse * max(0, n̂ · L̂)
    
where:
- n̂ is the unit surface normal
- L̂ is the unit vector toward the light source
- max(0, ...) ensures no negative lighting (surface facing away from light)
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource

def cross_product_3d(a, b):
    """Calculate 3D cross product"""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

def calculate_normal(v0, v1, v2):
    """Calculate normal vector for a triangle"""
    # Edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Normal (cross product)
    normal = cross_product_3d(edge1, edge2)
    
    # Normalize
    norm_magnitude = np.linalg.norm(normal)
    if norm_magnitude > 0:
        normal_unit = normal / norm_magnitude
    else:
        normal_unit = np.array([0, 0, 1])
    
    # Triangle area (half of parallelogram)
    area = norm_magnitude / 2
    
    return {
        'normal': normal,
        'normal_unit': normal_unit,
        'norm_magnitude': norm_magnitude,
        'area': area,
        'edge1': edge1,
        'edge2': edge2
    }

def calculate_lighting(normal_unit, light_direction, ambient=0.2, diffuse=0.8):
    """Calculate lighting intensity using Lambertian model"""
    # Ensure light direction is unit vector
    light_unit = light_direction / np.linalg.norm(light_direction)
    
    # Dot product (cosine of angle)
    cos_theta = np.dot(normal_unit, light_unit)
    
    # Clamp to [0, 1] - no negative lighting
    cos_theta = max(0, cos_theta)
    
    # Total intensity
    intensity = ambient + diffuse * cos_theta
    intensity = min(1.0, intensity)  # Clamp to max 1.0
    
    return {
        'intensity': intensity,
        'cos_theta': cos_theta,
        'angle': np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))
    }

# Define several triangular faces
faces = {
    'Face 1 (Front)': {
        'v0': np.array([0, 0, 0]),
        'v1': np.array([2, 0, 0]),
        'v2': np.array([1, 2, 0]),
        'description': 'Front-facing triangle (XY plane)'
    },
    'Face 2 (Top)': {
        'v0': np.array([0, 0, 1]),
        'v1': np.array([2, 0, 1]),
        'v2': np.array([1, 0, 3]),
        'description': 'Top face (angled upward)'
    },
    'Face 3 (Side)': {
        'v0': np.array([0, 0, 0]),
        'v1': np.array([0, 0, 2]),
        'v2': np.array([0, 2, 1]),
        'description': 'Side face (YZ plane)'
    },
    'Face 4 (Angled)': {
        'v0': np.array([1, 1, 0]),
        'v1': np.array([3, 1, 1]),
        'v2': np.array([2, 3, 1.5]),
        'description': 'Arbitrary angled face'
    }
}

# Light source direction (pointing toward light)
light_direction = np.array([1, 1, 1])  # Light from top-right-front
light_direction = light_direction / np.linalg.norm(light_direction)

# Calculate normals and lighting for all faces
results = {}
for name, face in faces.items():
    normal_data = calculate_normal(face['v0'], face['v1'], face['v2'])
    lighting_data = calculate_lighting(normal_data['normal_unit'], light_direction)
    
    results[name] = {**normal_data, **lighting_data, 'description': face['description']}

print("\n" + "="*115)
print("SURFACE NORMALS - CROSS PRODUCT APPLICATION IN COMPUTER GRAPHICS")
print("="*115)
print("\nFormula: n = (v1 - v0) × (v2 - v0)  where v0, v1, v2 are triangle vertices")
print("Unit Normal: n̂ = n / |n|")
print(f"Light Direction: {light_direction} (normalized)")
print("-"*115)

# Create comprehensive table
data = {
    "Face": list(faces.keys()),
    "Normal n": [str(np.round(results[name]['normal'], 3)) for name in faces.keys()],
    "Unit Normal n̂": [str(np.round(results[name]['normal_unit'], 3)) for name in faces.keys()],
    "Area": [f"{results[name]['area']:.3f}" for name in faces.keys()],
    "n̂·L̂": [f"{results[name]['cos_theta']:.4f}" for name in faces.keys()],
    "Angle to Light": [f"{results[name]['angle']:.2f}°" for name in faces.keys()],
    "Lighting Intensity": [f"{results[name]['intensity']:.3f}" for name in faces.keys()]
}

df = pd.DataFrame(data)
print(df.to_string(index=False))
print("-"*115)
print("\nLighting Model: I = I_ambient + I_diffuse × max(0, n̂·L̂)")
print("                Brightness depends on angle between normal and light direction")
print("="*115)

# Visualization
fig = plt.figure(figsize=(16, 12))

# Subplot 1-4: Individual face normals in 3D
for idx, (face_name, face_data) in enumerate(list(faces.items())[:4], 1):
    ax = fig.add_subplot(3, 3, idx, projection='3d')
    
    v0 = face_data['v0']
    v1 = face_data['v1']
    v2 = face_data['v2']
    result = results[face_name]
    
    # Draw triangle face
    vertices = [v0, v1, v2]
    tri = Poly3DCollection([vertices], alpha=result['intensity'], 
                          facecolors=plt.cm.viridis(result['intensity']),
                          edgecolors='black', linewidths=2)
    ax.add_collection3d(tri)
    
    # Draw vertices
    vertices_array = np.array(vertices)
    ax.scatter(vertices_array[:, 0], vertices_array[:, 1], vertices_array[:, 2],
              c='red', s=100, edgecolors='black', linewidths=2, zorder=5)
    
    # Label vertices
    for i, v in enumerate(vertices):
        ax.text(v[0], v[1], v[2], f'  v{i}', fontsize=9, fontweight='bold')
    
    # Draw edge vectors
    centroid = (v0 + v1 + v2) / 3
    ax.quiver(*v0, *result['edge1'], color='blue', linewidth=2,
             arrow_length_ratio=0.15, alpha=0.6, label='edge1')
    ax.quiver(*v0, *result['edge2'], color='green', linewidth=2,
             arrow_length_ratio=0.15, alpha=0.6, label='edge2')
    
    # Draw normal vector from centroid
    normal_scale = 0.7
    ax.quiver(*centroid, *(result['normal_unit']*normal_scale),
             color='red', linewidth=3, arrow_length_ratio=0.2,
             label='normal n̂', alpha=0.9)
    
    # Draw light direction
    light_start = centroid + np.array([0.5, 0.5, 0.5])
    ax.quiver(*light_start, *(light_direction*0.7),
             color='yellow', linewidth=2.5, arrow_length_ratio=0.15,
             label='light L̂', alpha=0.8)
    
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.set_title(f'{face_name}\nIntensity: {result["intensity"]:.2f}',
                fontsize=10, fontweight='bold', pad=5)
    
    # Set equal aspect ratio
    max_range = 3
    ax.set_xlim([0, max_range])
    ax.set_ylim([0, max_range])
    ax.set_zlim([0, max_range])

# Subplot 5: Lighting comparison
ax5 = plt.subplot(3, 3, 5)

face_names = list(faces.keys())
intensities = [results[name]['intensity'] for name in face_names]
angles = [results[name]['angle'] for name in face_names]

# Color bars by intensity
colors_bar = [plt.cm.viridis(intensity) for intensity in intensities]

bars = ax5.barh(face_names, intensities, color=colors_bar, alpha=0.8,
               edgecolor='black', linewidth=2)

# Add value labels
for bar, val, angle in zip(bars, intensities, angles):
    width = bar.get_width()
    ax5.text(width + 0.02, bar.get_y() + bar.get_height()/2,
             f'{val:.2f} ({angle:.0f}°)',
             ha='left', va='center', fontsize=9, fontweight='bold')

ax5.set_xlabel('Lighting Intensity', fontsize=10, fontweight='bold')
ax5.set_title('Lighting Intensity Comparison\n(Based on Normal-Light Angle)',
             fontsize=11, fontweight='bold', pad=10)
ax5.set_xlim(0, 1.2)
ax5.grid(True, axis='x', linestyle=':', alpha=0.4)

# Add color bar
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                           norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax5, orientation='horizontal', pad=0.15, aspect=20)
cbar.set_label('Intensity', fontsize=9)

# Subplot 6: Angle vs Intensity curve
ax6 = plt.subplot(3, 3, 6)

# Plot theoretical curve
angles_range = np.linspace(0, 180, 200)
intensities_theoretical = [0.2 + 0.8 * max(0, np.cos(np.radians(a))) 
                          for a in angles_range]

ax6.plot(angles_range, intensities_theoretical, 'b-', linewidth=3,
        label='Theoretical (Lambertian)', alpha=0.7)

# Plot actual face data points
for face_name in face_names:
    angle = results[face_name]['angle']
    intensity = results[face_name]['intensity']
    ax6.plot(angle, intensity, 'ro', markersize=12, markeredgecolor='black',
            markeredgewidth=2, zorder=5)
    ax6.annotate(face_name.split()[0] + ' ' + face_name.split()[1],
                (angle, intensity), xytext=(5, 5), textcoords='offset points',
                fontsize=8, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax6.axvline(x=90, color='gray', linestyle='--', linewidth=1.5, alpha=0.5,
           label='90° (perpendicular)')
ax6.grid(True, linestyle=':', alpha=0.4)
ax6.set_xlabel('Angle between Normal and Light (degrees)', fontsize=10)
ax6.set_ylabel('Lighting Intensity', fontsize=10)
ax6.set_title('Lambertian Lighting Model\nI = ambient + diffuse×cos(θ)',
             fontsize=11, fontweight='bold', pad=10)
ax6.set_xlim(0, 180)
ax6.set_ylim(0, 1.1)
ax6.legend(loc='upper right', fontsize=9)

# Subplot 7: Cube with multiple faces and normals
ax7 = fig.add_subplot(3, 3, 7, projection='3d')

# Define a simple cube
cube_vertices = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top
])

# Define cube faces (vertex indices, CCW winding)
cube_faces_indices = [
    [0, 1, 2, 3],  # Bottom
    [4, 5, 6, 7],  # Top
    [0, 1, 5, 4],  # Front
    [2, 3, 7, 6],  # Back
    [0, 3, 7, 4],  # Left
    [1, 2, 6, 5]   # Right
]

# Calculate normals for each face and shade
for face_indices in cube_faces_indices:
    face_verts = cube_vertices[face_indices]
    
    # Calculate normal
    v0, v1, v2 = face_verts[0], face_verts[1], face_verts[2]
    normal_data = calculate_normal(v0, v1, v2)
    lighting_data = calculate_lighting(normal_data['normal_unit'], light_direction)
    
    # Draw face with lighting
    poly = Poly3DCollection([face_verts], alpha=0.7,
                           facecolors=plt.cm.viridis(lighting_data['intensity']),
                           edgecolors='black', linewidths=2)
    ax7.add_collection3d(poly)
    
    # Draw normal from face center
    center = np.mean(face_verts, axis=0)
    ax7.quiver(*center, *(normal_data['normal_unit']*0.3),
              color='red', linewidth=2, arrow_length_ratio=0.3, alpha=0.8)

# Draw light direction
ax7.quiver(1.5, 1.5, 1.5, *light_direction, color='yellow', linewidth=3,
          arrow_length_ratio=0.2, label='Light', alpha=0.9)

ax7.set_xlabel('X', fontsize=9)
ax7.set_ylabel('Y', fontsize=9)
ax7.set_zlabel('Z', fontsize=9)
ax7.set_title('Cube with Face Normals\n(Different lighting per face)',
             fontsize=11, fontweight='bold', pad=10)
ax7.legend(loc='upper left', fontsize=9)
ax7.set_xlim([0, 2])
ax7.set_ylim([0, 2])
ax7.set_zlim([0, 2])

# Subplot 8: Backface culling illustration
ax8 = plt.subplot(3, 3, 8)

# 2D illustration of backface culling
ax8.text(0.5, 0.95, 'BACKFACE CULLING', transform=ax8.transAxes,
        fontsize=12, fontweight='bold', ha='center', va='top')

# Draw viewer and surface
viewer_pos = np.array([0.2, 0.5])
front_face_center = np.array([0.5, 0.7])
back_face_center = np.array([0.5, 0.3])

# Normals
front_normal = np.array([0, 0.15])
back_normal = np.array([0, -0.15])

# Draw surfaces
ax8.plot([0.3, 0.7], [0.7, 0.7], 'g-', linewidth=8, label='Front face (visible)',
        solid_capstyle='round')
ax8.plot([0.3, 0.7], [0.3, 0.3], 'r-', linewidth=8, label='Back face (culled)',
        solid_capstyle='round', alpha=0.3)

# Draw normals
ax8.arrow(front_face_center[0], front_face_center[1], front_normal[0], front_normal[1],
         head_width=0.03, head_length=0.02, fc='green', ec='darkgreen', linewidth=2)
ax8.arrow(back_face_center[0], back_face_center[1], back_normal[0], back_normal[1],
         head_width=0.03, head_length=0.02, fc='red', ec='darkred', linewidth=2)

# Draw viewer
ax8.plot(viewer_pos[0], viewer_pos[1], 'b*', markersize=25, 
        markeredgecolor='black', markeredgewidth=2, label='Viewer')

# Draw view rays
ax8.plot([viewer_pos[0], front_face_center[0]], 
        [viewer_pos[1], front_face_center[1]], 'g--', linewidth=1.5, alpha=0.6)
ax8.plot([viewer_pos[0], back_face_center[0]], 
        [viewer_pos[1], back_face_center[1]], 'r--', linewidth=1.5, alpha=0.3)

# Add annotations
ax8.text(0.85, 0.7, 'n̂·v < 0\n(facing viewer)', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax8.text(0.85, 0.3, 'n̂·v > 0\n(facing away)', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

ax8.set_xlim(0, 1)
ax8.set_ylim(0, 1)
ax8.set_aspect('equal')
ax8.axis('off')
ax8.legend(loc='lower left', fontsize=9, framealpha=0.95)

# Subplot 9: Theory and Applications
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

theory_text = """
SURFACE NORMALS IN COMPUTER GRAPHICS

Cross Product Application:
  n = (v1 - v0) × (v2 - v0)
  n̂ = n / |n|  (unit normal)

Vertex Ordering:
• CCW (counterclockwise) → outward normal
• CW (clockwise) → inward normal
• Consistent ordering is crucial!

Lighting Calculations:
Lambertian (Diffuse) Model:
  I = I_ambient + I_diffuse × max(0, n̂·L̂)
  
  n̂ = surface unit normal
  L̂ = unit vector to light
  max(0,...) = no negative lighting

Backface Culling:
  if (n̂·v̂ > 0): cull face
  where v̂ is view direction
  Optimization: don't render invisible faces

Applications:
• Real-time rendering (games, CAD)
• Ray tracing (reflections, shadows)
• Collision detection (surface contact)
• Normal mapping (detail without geometry)
• Mesh processing (smoothness, curvature)
• 3D printing (support structures)

Why Cross Product?
• Automatic perpendicularity
• Magnitude = parallelogram area
• Direction follows right-hand rule
• Fast computation (6 multiplications)
"""

ax9.text(0.08, 0.95, theory_text, transform=ax9.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.95,
                  edgecolor='blue', linewidth=2, pad=1.5))

plt.suptitle('Surface Normals via Cross Product (Computer Graphics)',
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.99))
plt.show()

print("\n✓ Visualization complete!")
print("\nNote: Normal vectors are essential for lighting, shading, and culling in 3D graphics")
