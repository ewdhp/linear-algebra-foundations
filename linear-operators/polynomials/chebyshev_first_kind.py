"""
CHEBYSHEV POLYNOMIALS (FIRST KIND) - ORTHOGONAL WITH WEIGHT 1/√(1-x²)

Theory:
Chebyshev polynomials of the first kind T_n(x) are orthogonal on the interval 
[-1, 1] with respect to the weight function w(x) = 1/√(1-x²).

Orthogonality Relation:
    ∫_{-1}^{1} T_m(x) T_n(x) / √(1-x²) dx = { 0       if m ≠ n
                                               { π       if m = n = 0
                                               { π/2     if m = n ≠ 0

Recurrence Relation:
    T_{n+1}(x) = 2x·T_n(x) - T_{n-1}(x)
    T_0(x) = 1
    T_1(x) = x

Trigonometric Definition:
    T_n(cos θ) = cos(nθ)
    or equivalently: T_n(x) = cos(n·arccos(x))

Applications:
- Approximation theory (minimax approximation)
- Numerical analysis (Chebyshev interpolation, spectral methods)
- Signal processing (Chebyshev filters)
- Polynomial approximation (minimizing maximum error)
- Computer graphics (efficient polynomial evaluation)

Properties:
- T_n(1) = 1 for all n
- T_n(-1) = (-1)^n
- Bounded: |T_n(x)| ≤ 1 for x ∈ [-1,1]
- Even/odd symmetry: T_n(-x) = (-1)^n T_n(x)
- Extrema: T_n has n+1 extrema in [-1,1] with values ±1
- Zeros: x_k = cos((2k+1)π/(2n)) for k=0,1,...,n-1
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import chebyt, eval_chebyt
from scipy.integrate import quad

def chebyshev_first_kind(n, x):
    """
    Generate Chebyshev polynomial of the first kind T_n(x) using recurrence relation
    """
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        T_prev_prev = np.ones_like(x)
        T_prev = x
        for k in range(2, n + 1):
            T_current = 2 * x * T_prev - T_prev_prev
            T_prev_prev = T_prev
            T_prev = T_current
        return T_current

def chebyshev_trigonometric(n, x):
    """
    Compute T_n(x) = cos(n·arccos(x)) for x ∈ [-1,1]
    """
    # Handle numerical precision issues
    x_clipped = np.clip(x, -1, 1)
    theta = np.arccos(x_clipped)
    return np.cos(n * theta)

def weight_function(x):
    """Weight function w(x) = 1/√(1-x²)"""
    # Avoid division by zero at endpoints
    x_safe = np.clip(x, -0.9999, 0.9999)
    return 1.0 / np.sqrt(1 - x_safe**2)

def compute_inner_product(n, m):
    """
    Compute inner product <T_n, T_m> = ∫_{-1}^{1} T_n(x) T_m(x) / √(1-x²) dx
    """
    def integrand(x):
        # Use a safe domain to avoid singularities
        return eval_chebyt(n, x) * eval_chebyt(m, x) / np.sqrt(1 - x**2)
    
    # Use change of variables: x = cos(θ), dx = -sin(θ)dθ
    # ∫_{-1}^{1} T_n(x) T_m(x) / √(1-x²) dx = ∫_{0}^{π} cos(nθ)cos(mθ) dθ
    def trig_integrand(theta):
        return np.cos(n * theta) * np.cos(m * theta)
    
    result, error = quad(trig_integrand, 0, np.pi)
    return result

def theoretical_norm_squared(n):
    """
    Theoretical value: ||T_n||² = π if n=0, π/2 if n≠0
    """
    return np.pi if n == 0 else np.pi / 2

# Generate Chebyshev polynomials up to degree 6
max_degree = 6
x = np.linspace(-1, 1, 500)

print("\n" + "="*100)
print("CHEBYSHEV POLYNOMIALS (FIRST KIND) - ORTHOGONAL WITH WEIGHT 1/√(1-x²)")
print("="*100)
print("\nWeight Function: w(x) = 1/√(1-x²)")
print("Orthogonality: ∫_{-1}^{1} T_m(x) T_n(x) / √(1-x²) dx = π·δ_{mn} (with π/2 for n≠0)")
print("Trigonometric: T_n(cos θ) = cos(nθ)")
print("-"*100)

# Compute and display polynomial expressions
poly_data = []
for n in range(max_degree + 1):
    # Get polynomial coefficients
    T_n_poly = chebyt(n)
    coeffs = T_n_poly.coef
    
    # Create expression string
    terms = []
    for i, c in enumerate(coeffs[::-1]):
        if abs(c) < 1e-10:
            continue
        power = len(coeffs) - 1 - i
        if power == 0:
            if abs(c - round(c)) < 1e-10:
                terms.append(f"{int(round(c))}")
            else:
                terms.append(f"{c:.4f}")
        elif power == 1:
            if abs(c - 1) < 1e-10:
                terms.append("x")
            elif abs(c + 1) < 1e-10:
                terms.append("-x")
            else:
                if abs(c - round(c)) < 1e-10:
                    terms.append(f"{int(round(c))}x")
                else:
                    terms.append(f"{c:.4f}x")
        else:
            if abs(c - 1) < 1e-10:
                terms.append(f"x^{power}")
            elif abs(c + 1) < 1e-10:
                terms.append(f"-x^{power}")
            else:
                if abs(c - round(c)) < 1e-10:
                    terms.append(f"{int(round(c))}x^{power}")
                else:
                    terms.append(f"{c:.4f}x^{power}")
    
    expression = " + ".join(terms).replace("+ -", "- ")
    if not expression:
        expression = "0"
    
    poly_data.append({
        "Degree n": n,
        "T_n(x)": expression,
        "T_n(-1)": int((-1)**n),
        "T_n(0)": f"{eval_chebyt(n, 0):.4f}",
        "T_n(1)": 1,
        "||T_n||²": f"{theoretical_norm_squared(n):.6f}"
    })

df_poly = pd.DataFrame(poly_data)
print("\nChebyshev Polynomials (First Kind) - first few degrees:")
print(df_poly.to_string(index=False))
print("-"*100)

# Test orthogonality by computing inner products
print("\nOrthogonality Test: Inner Products <T_m, T_n>")
print("(Should be 0 for m ≠ n, π for m=n=0, and π/2 for m=n≠0)")
print("-"*100)

orthogonality_data = []
for m in range(max_degree + 1):
    row_data = {"m": m}
    for n in range(max_degree + 1):
        inner_prod = compute_inner_product(m, n)
        if m == n:
            theoretical = theoretical_norm_squared(n)
            error = abs(inner_prod - theoretical) / theoretical * 100
            row_data[f"n={n}"] = f"{inner_prod:.6f}\n({error:.2f}% err)"
        else:
            row_data[f"n={n}"] = f"{inner_prod:.2e}"
    orthogonality_data.append(row_data)

df_ortho = pd.DataFrame(orthogonality_data)
print(df_ortho.to_string(index=False))
print("-"*100)

# Zeros (Chebyshev nodes)
print("\nChebyshev Zeros (Nodes): x_k = cos((2k+1)π/(2n))")
zeros_data = []
for n in range(1, min(7, max_degree + 1)):
    zeros = [np.cos((2*k + 1) * np.pi / (2*n)) for k in range(n)]
    zeros_str = ", ".join([f"{z:.6f}" for z in zeros])
    zeros_data.append({
        "Degree n": n,
        "Number of Zeros": n,
        "Zeros (Chebyshev nodes)": zeros_str
    })

df_zeros = pd.DataFrame(zeros_data)
print(df_zeros.to_string(index=False))
print("="*100)

# Statistical and mathematical properties
print("\nMathematical Properties:")
stats_data = []
for n in range(max_degree + 1):
    x_vals = np.linspace(-1, 1, 1000)
    T_n = chebyshev_first_kind(n, x_vals)
    
    # Number of extrema (should be n+1)
    # Extrema at x_k = cos(kπ/n) for k=0,1,...,n
    num_extrema = n + 1
    
    # Maximum absolute value (should be 1)
    max_val = np.max(np.abs(T_n))
    
    stats_data.append({
        "Degree n": n,
        "Zeros": n,
        "Extrema": num_extrema,
        "Max |T_n(x)|": f"{max_val:.6f}",
        "Parity": "Even" if n % 2 == 0 else "Odd",
        "Norm ||T_n||": f"{np.sqrt(theoretical_norm_squared(n)):.6f}"
    })

df_stats = pd.DataFrame(stats_data)
print(df_stats.to_string(index=False))
print("="*100)

# Visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: Chebyshev Polynomials (First Kind)
ax1 = plt.subplot(3, 3, 1)
colors = plt.cm.jet(np.linspace(0, 1, max_degree + 1))

for n in range(max_degree + 1):
    T_n = chebyshev_first_kind(n, x)
    ax1.plot(x, T_n, color=colors[n], linewidth=2.5, label=f'T_{n}(x)', alpha=0.8)

ax1.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax1.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
ax1.axhline(y=1, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
ax1.axhline(y=-1, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper left', fontsize=9, ncol=2)
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('T_n(x)', fontsize=10)
ax1.set_title('Chebyshev Polynomials (First Kind)\nT_n(x) on [-1,1]', 
              fontsize=11, fontweight='bold')
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1.2, 1.2)

# Plot 2: Weight Function
ax2 = plt.subplot(3, 3, 2)
x_interior = np.linspace(-0.99, 0.99, 500)
w = weight_function(x_interior)
ax2.plot(x_interior, w, 'r-', linewidth=3, label='w(x) = 1/√(1-x²)')
ax2.fill_between(x_interior, 0, w, alpha=0.3, color='red')
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='upper center', fontsize=10)
ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('w(x)', fontsize=10)
ax2.set_title('Weight Function\nw(x) = 1/√(1-x²)', 
              fontsize=11, fontweight='bold')
ax2.set_xlim(-1, 1)
ax2.set_ylim(0, 12)

# Plot 3: Trigonometric Representation
ax3 = plt.subplot(3, 3, 3)
theta = np.linspace(0, 2*np.pi, 500)
x_trig = np.cos(theta)

for n in range(min(5, max_degree + 1)):
    T_n_trig = np.cos(n * theta)
    ax3.plot(theta, T_n_trig, color=colors[n], linewidth=2, 
            label=f'cos({n}θ)', alpha=0.8)

ax3.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax3.grid(True, linestyle='--', alpha=0.4)
ax3.legend(loc='upper right', fontsize=8)
ax3.set_xlabel('θ (radians)', fontsize=10)
ax3.set_ylabel('T_n(cos θ) = cos(nθ)', fontsize=10)
ax3.set_title('Trigonometric Form\nT_n(cos θ) = cos(nθ)', 
              fontsize=11, fontweight='bold')
ax3.set_xlim(0, 2*np.pi)
ax3.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax3.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

# Plot 4: Orthogonality Matrix Heatmap
ax4 = plt.subplot(3, 3, 4)
ortho_matrix = np.zeros((max_degree + 1, max_degree + 1))
for m in range(max_degree + 1):
    for n in range(max_degree + 1):
        ortho_matrix[m, n] = compute_inner_product(m, n)

# Normalize for better visualization
ortho_matrix_norm = ortho_matrix / np.max(np.abs(ortho_matrix))

im = ax4.imshow(ortho_matrix_norm, cmap='RdYlGn', aspect='auto', 
                interpolation='nearest', vmin=-0.1, vmax=1)
ax4.set_xticks(range(max_degree + 1))
ax4.set_yticks(range(max_degree + 1))
ax4.set_xticklabels([f'n={i}' for i in range(max_degree + 1)])
ax4.set_yticklabels([f'm={i}' for i in range(max_degree + 1)])
ax4.set_xlabel('Polynomial degree n', fontsize=10)
ax4.set_ylabel('Polynomial degree m', fontsize=10)
ax4.set_title('Orthogonality Matrix\n<T_m, T_n> (normalized)', 
              fontsize=11, fontweight='bold')

# Add text annotations
for m in range(max_degree + 1):
    for n in range(max_degree + 1):
        text = ax4.text(n, m, f'{ortho_matrix_norm[m, n]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax4, label='Normalized Inner Product')

# Plot 5: Chebyshev Nodes (Zeros)
ax5 = plt.subplot(3, 3, 5)
n_nodes = 7
zeros = [np.cos((2*k + 1) * np.pi / (2*n_nodes)) for k in range(n_nodes)]
T_n_nodes = chebyshev_first_kind(n_nodes, x)

ax5.plot(x, T_n_nodes, 'b-', linewidth=3, label=f'T_{n_nodes}(x)')
ax5.axhline(y=0, color='k', linewidth=1, alpha=0.5)

# Mark zeros (Chebyshev nodes)
ax5.plot(zeros, np.zeros_like(zeros), 'ro', markersize=12, 
         markeredgecolor='black', markeredgewidth=2, label=f'{len(zeros)} nodes', zorder=5)

ax5.grid(True, linestyle='--', alpha=0.4)
ax5.legend(loc='upper left', fontsize=10)
ax5.set_xlabel('x', fontsize=10)
ax5.set_ylabel(f'T_{n_nodes}(x)', fontsize=10)
ax5.set_title(f'Chebyshev Nodes\nZeros of T_{n_nodes}(x)', 
              fontsize=11, fontweight='bold')
ax5.set_xlim(-1, 1)

# Plot 6: Extrema
ax6 = plt.subplot(3, 3, 6)
n_extrema = 5
T_n_ext = chebyshev_first_kind(n_extrema, x)
ax6.plot(x, T_n_ext, 'b-', linewidth=3, label=f'T_{n_extrema}(x)')

# Mark extrema at x_k = cos(kπ/n) for k=0,1,...,n
extrema_x = [np.cos(k * np.pi / n_extrema) for k in range(n_extrema + 1)]
extrema_y = [chebyshev_first_kind(n_extrema, np.array([ex]))[0] for ex in extrema_x]

ax6.plot(extrema_x, extrema_y, 'ro', markersize=12, 
         markeredgecolor='black', markeredgewidth=2, 
         label=f'{len(extrema_x)} extrema (±1)', zorder=5)

ax6.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax6.axhline(y=1, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
ax6.axhline(y=-1, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
ax6.grid(True, linestyle='--', alpha=0.4)
ax6.legend(loc='upper left', fontsize=9)
ax6.set_xlabel('x', fontsize=10)
ax6.set_ylabel(f'T_{n_extrema}(x)', fontsize=10)
ax6.set_title(f'Extrema of T_{n_extrema}(x)\nAll have |T_n| = 1', 
              fontsize=11, fontweight='bold')
ax6.set_xlim(-1, 1)
ax6.set_ylim(-1.2, 1.2)

# Plot 7: Even/Odd Symmetry
ax7 = plt.subplot(3, 3, 7)
n_even = 4
n_odd = 5
T_even = chebyshev_first_kind(n_even, x)
T_odd = chebyshev_first_kind(n_odd, x)

ax7.plot(x, T_even, 'b-', linewidth=2.5, label=f'T_{n_even}(x) (even)', alpha=0.8)
ax7.plot(x, T_odd, 'r-', linewidth=2.5, label=f'T_{n_odd}(x) (odd)', alpha=0.8)
ax7.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax7.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
ax7.grid(True, linestyle='--', alpha=0.4)
ax7.legend(loc='upper left', fontsize=9)
ax7.set_xlabel('x', fontsize=10)
ax7.set_ylabel('T_n(x)', fontsize=10)
ax7.set_title('Parity Symmetry\nT_n(-x) = (-1)^n·T_n(x)', 
              fontsize=11, fontweight='bold')
ax7.set_xlim(-1, 1)

# Plot 8: Norms
ax8 = plt.subplot(3, 3, 8)
degrees = np.arange(0, max_degree + 1)
norms = [np.sqrt(theoretical_norm_squared(n)) for n in degrees]

colors_norm = ['red' if n == 0 else 'blue' for n in degrees]
ax8.bar(degrees, norms, color=colors_norm, alpha=0.7, edgecolor='black', linewidth=1.5)
ax8.axhline(y=np.sqrt(np.pi), color='red', linewidth=2, linestyle='--', 
           label=f'||T_0|| = √π', alpha=0.7)
ax8.axhline(y=np.sqrt(np.pi/2), color='blue', linewidth=2, linestyle='--', 
           label=f'||T_n|| = √(π/2) (n≠0)', alpha=0.7)
ax8.grid(True, linestyle='--', alpha=0.4, axis='y')
ax8.legend(loc='upper right', fontsize=9)
ax8.set_xlabel('Degree n', fontsize=10)
ax8.set_ylabel('Norm', fontsize=10)
ax8.set_title('Norm of Chebyshev Polynomials\n||T_0||² = π, ||T_n||² = π/2 (n≠0)', 
              fontsize=11, fontweight='bold')
ax8.set_xticks(degrees)
ax8.set_ylim(0, 2.2)

# Plot 9: Theory Summary
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

theory_text = """
CHEBYSHEV POLYNOMIALS (FIRST KIND)

Definition & Recurrence:
  T₀(x) = 1
  T₁(x) = x
  T_{n+1}(x) = 2x·T_n(x) - T_{n-1}(x)

Trigonometric Form:
  T_n(cos θ) = cos(nθ)
  or T_n(x) = cos(n·arccos(x))

Orthogonality:
  ∫₋₁¹ T_m(x)T_n(x)/√(1-x²) dx = {π    if m=n=0
                                  {π/2  if m=n≠0
                                  {0    if m≠n

Weight Function:
  w(x) = 1/√(1-x²)  (Arcsine distribution)

Properties:
• Parity: T_n(-x) = (-1)^n T_n(x)
• Boundary: T_n(1) = 1, T_n(-1) = (-1)^n
• Bounded: |T_n(x)| ≤ 1 for x ∈ [-1,1]
• Extrema: n+1 extrema with |T_n| = 1
• Zeros: x_k = cos((2k+1)π/(2n)), k=0,...,n-1

Applications:
• Minimax polynomial approximation
• Chebyshev interpolation (spectral methods)
• Digital signal processing (filters)
• Numerical integration (Clenshaw-Curtis)
• Computer graphics

Key Insight:
Among all monic polynomials of degree n,
T_n(x)/2^(n-1) has the smallest maximum
absolute value on [-1,1]. This property
makes Chebyshev polynomials optimal for
approximation theory.
"""

ax9.text(0.05, 0.95, theory_text, transform=ax9.transAxes, fontsize=9.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.95,
                  edgecolor='blue', linewidth=2, pad=1.5))

plt.suptitle('Chebyshev Polynomials (First Kind) - Minimax Approximation', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.99))
plt.show()

print("\n✓ Visualization complete!")
print("\nNote: Chebyshev polynomials minimize maximum approximation error")
print("      Used extensively in numerical analysis and signal processing")
