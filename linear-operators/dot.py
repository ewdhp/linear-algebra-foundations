import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# --- Step 4: Geometric visualization ---
fig, ax = plt.subplots(figsize=(7, 7))
origin = np.array([0, 0])

# Plot vectors
ax.quiver(*origin, *A, angles='xy', scale_units='xy', scale=1, color='r', label='Vector A')
ax.quiver(*origin, *B, angles='xy', scale_units='xy', scale=1, color='b', label='Vector B')

# Projection of A onto B
B_unit = B / norm_B
proj_A_on_B = np.dot(A, B_unit) * B_unit
ax.quiver(*origin, *proj_A_on_B, angles='xy', scale_units='xy', scale=1, color='g', label='Projection of A on B')

# Annotate vectors
ax.text(A[0] * 1.05, A[1] * 1.05, 'A', color='r', fontsize=12)
ax.text(B[0] * 1.05, B[1] * 1.05, 'B', color='b', fontsize=12)

# Draw angle arc
arc = np.linspace(0, np.arccos(cos_theta), 100)
ax.plot(0.7 * np.cos(arc), 0.7 * np.sin(arc), 'k--')
ax.text(0.9 * np.cos(arc[-1]/2), 0.9 * np.sin(arc[-1]/2), f'{theta_deg:.1f}°', fontsize=10)

# Formatting
ax.set_xlim(-1, max(A[0], B[0]) + 2)
ax.set_ylim(-1, max(A[1], B[1]) + 2)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
ax.set_title("Geometric Representation of the Dot Product")

plt.show()
