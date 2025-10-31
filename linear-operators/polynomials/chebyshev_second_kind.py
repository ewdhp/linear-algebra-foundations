"""
CHEBYSHEV POLYNOMIALS (SECOND KIND) - ORTHOGONAL WITH WEIGHT √(1-x²)

Theory:
Chebyshev polynomials of the second kind U_n(x) are orthogonal on the interval 
[-1, 1] with respect to the weight function w(x) = √(1-x²), which is related 
to the Wigner semicircle distribution.

Orthogonality Relation:
    ∫_{-1}^{1} U_m(x) U_n(x) √(1-x²) dx = (π/2) · δ_{mn}
    
where δ_{mn} is the Kronecker delta (1 if m=n, 0 otherwise)

Recurrence Relation:
    U_{n+1}(x) = 2x·U_n(x) - U_{n-1}(x)
    U_0(x) = 1
    U_1(x) = 2x

Trigonometric Definition:
    U_n(cos θ) = sin((n+1)θ) / sin(θ)
    or equivalently for x ∈ [-1,1]:
    U_n(x) = sin((n+1)·arccos(x)) / sin(arccos(x)) = sin((n+1)·arccos(x)) / √(1-x²)

Relation to First Kind:
    dT_{n+1}/dx = (n+1)·U_n(x)
    U_n(x) = (1/(n+1)) · dT_{n+1}/dx

Applications:
- Numerical analysis (Chebyshev-Gauss quadrature of second kind)
- Random matrix theory (eigenvalue distributions)
- Antenna array theory (Chebyshev array design)
- Signal processing (filter design)
- Physics (Wigner semicircle distribution in random matrices)

Properties:
- U_n(1) = n + 1 for all n
- U_n(-1) = (-1)^n (n + 1)
- Even/odd symmetry: U_n(-x) = (-1)^n U_n(x)
- Zeros: x_k = cos(kπ/(n+1)) for k=1,2,...,n
- Not bounded like T_n: |U_n(x)| can grow with n
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import chebyu, eval_chebyu
from scipy.integrate import quad

def chebyshev_second_kind(n, x):
    """
    Generate Chebyshev polynomial of the second kind U_n(x) using recurrence relation
    """
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        U_prev_prev = np.ones_like(x)
        U_prev = 2 * x
        for k in range(2, n + 1):
            U_current = 2 * x * U_prev - U_prev_prev
            U_prev_prev = U_prev
            U_prev = U_current
        return U_current

def chebyshev_trigonometric(n, x):
    """
    Compute U_n(x) = sin((n+1)·arccos(x)) / √(1-x²) for x ∈ [-1,1]
    """
    # Handle numerical precision issues
    x_clipped = np.clip(x, -0.9999, 0.9999)
    theta = np.arccos(x_clipped)
    
    # U_n(cos θ) = sin((n+1)θ) / sin(θ)
    numerator = np.sin((n + 1) * theta)
    denominator = np.sin(theta)
    
    # Avoid division by zero
    result = np.where(np.abs(denominator) > 1e-10, numerator / denominator, n + 1)
    return result

def weight_function(x):
    """Weight function w(x) = √(1-x²) (Wigner semicircle distribution)"""
    x_safe = np.clip(x, -1, 1)
    return np.sqrt(1 - x_safe**2)

def compute_inner_product(n, m):
    """
    Compute inner product <U_n, U_m> = ∫_{-1}^{1} U_n(x) U_m(x) √(1-x²) dx
    """
    # Use change of variables: x = cos(θ), dx = -sin(θ)dθ
    # ∫_{-1}^{1} U_n(x) U_m(x) √(1-x²) dx = ∫_{0}^{π} sin((n+1)θ)sin((m+1)θ) dθ
    def trig_integrand(theta):
        return np.sin((n + 1) * theta) * np.sin((m + 1) * theta)
    
    result, error = quad(trig_integrand, 0, np.pi)
    return result

def theoretical_norm_squared(n):
    """
    Theoretical value: ||U_n||² = π/2
    """
    return np.pi / 2

# Generate Chebyshev polynomials of second kind up to degree 6
max_degree = 6
x = np.linspace(-1, 1, 500)

print("\n" + "="*100)
print("CHEBYSHEV POLYNOMIALS (SECOND KIND) - ORTHOGONAL WITH WEIGHT √(1-x²)")
print("="*100)
print("\nWeight Function: w(x) = √(1-x²)  (Wigner semicircle distribution)")
print("Orthogonality: ∫_{-1}^{1} U_m(x) U_n(x) √(1-x²) dx = (π/2) · δ_{mn}")
print("Trigonometric: U_n(cos θ) = sin((n+1)θ) / sin(θ)")
print("-"*100)

# Compute and display polynomial expressions
poly_data = []
for n in range(max_degree + 1):
    # Get polynomial coefficients
    U_n_poly = chebyu(n)
    coeffs = U_n_poly.coef
    
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
        "U_n(x)": expression,
        "U_n(-1)": int((-1)**n * (n + 1)),
        "U_n(0)": f"{eval_chebyu(n, 0):.4f}",
        "U_n(1)": n + 1,
        "||U_n||²": f"{theoretical_norm_squared(n):.6f}"
    })

df_poly = pd.DataFrame(poly_data)
print("\nChebyshev Polynomials (Second Kind) - first few degrees:")
print(df_poly.to_string(index=False))
print("-"*100)

# Test orthogonality by computing inner products
print("\nOrthogonality Test: Inner Products <U_m, U_n>")
print("(Should be 0 for m ≠ n, and π/2 for m = n)")
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

# Zeros
print("\nChebyshev Zeros (Nodes) for U_n: x_k = cos(kπ/(n+1))")
zeros_data = []
for n in range(1, min(7, max_degree + 1)):
    zeros = [np.cos(k * np.pi / (n + 1)) for k in range(1, n + 1)]
    zeros_str = ", ".join([f"{z:.6f}" for z in zeros])
    zeros_data.append({
        "Degree n": n,
        "Number of Zeros": n,
        "Zeros": zeros_str
    })

df_zeros = pd.DataFrame(zeros_data)
print(df_zeros.to_string(index=False))
print("="*100)

# Statistical and mathematical properties
print("\nMathematical Properties:")
stats_data = []
for n in range(max_degree + 1):
    x_vals = np.linspace(-1, 1, 1000)
    U_n = chebyshev_second_kind(n, x_vals)
    
    # Maximum absolute value
    max_val = np.max(np.abs(U_n))
    
    stats_data.append({
        "Degree n": n,
        "Zeros": n,
        "U_n(1)": n + 1,
        "U_n(-1)": int((-1)**n * (n + 1)),
        "Max |U_n(x)|": f"{max_val:.2f}",
        "Parity": "Even" if n % 2 == 0 else "Odd",
        "Norm ||U_n||": f"{np.sqrt(theoretical_norm_squared(n)):.6f}"
    })

df_stats = pd.DataFrame(stats_data)
print(df_stats.to_string(index=False))
print("="*100)

# Relation to first kind
print("\nRelation to Chebyshev Polynomials (First Kind):")
print("U_n(x) = (1/(n+1)) · dT_{n+1}/dx")
relation_data = []
for n in range(min(5, max_degree + 1)):
    x_test = 0.5
    U_n_val = eval_chebyu(n, x_test)
    
    relation_data.append({
        "Degree n": n,
        "U_n(0.5)": f"{U_n_val:.6f}",
        "Relation": f"U_{n} = (1/{n+1})·dT_{n+1}/dx"
    })

df_relation = pd.DataFrame(relation_data)
print(df_relation.to_string(index=False))
print("="*100)

# Visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: Chebyshev Polynomials (Second Kind)
ax1 = plt.subplot(3, 3, 1)
colors = plt.cm.rainbow(np.linspace(0, 1, max_degree + 1))

for n in range(max_degree + 1):
    U_n = chebyshev_second_kind(n, x)
    ax1.plot(x, U_n, color=colors[n], linewidth=2.5, label=f'U_{n}(x)', alpha=0.8)

ax1.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax1.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper left', fontsize=9, ncol=2)
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('U_n(x)', fontsize=10)
ax1.set_title('Chebyshev Polynomials (Second Kind)\nU_n(x) on [-1,1]', 
              fontsize=11, fontweight='bold')
ax1.set_xlim(-1, 1)
ax1.set_ylim(-8, 8)

# Plot 2: Weight Function (Semicircle)
ax2 = plt.subplot(3, 3, 2)
w = weight_function(x)
ax2.plot(x, w, 'r-', linewidth=3, label='w(x) = √(1-x²)')
ax2.fill_between(x, 0, w, alpha=0.3, color='red')
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='upper center', fontsize=10)
ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('w(x)', fontsize=10)
ax2.set_title('Weight Function (Wigner Semicircle)\nw(x) = √(1-x²)', 
              fontsize=11, fontweight='bold')
ax2.set_xlim(-1, 1)
ax2.set_ylim(0, 1.2)

# Plot 3: Weighted Polynomials
ax3 = plt.subplot(3, 3, 3)
for n in range(max_degree + 1):
    U_n = chebyshev_second_kind(n, x)
    weighted = U_n * w  # Weighted
    ax3.plot(x, weighted, color=colors[n], linewidth=2, label=f'U_{n}·w', alpha=0.7)

ax3.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax3.grid(True, linestyle='--', alpha=0.4)
ax3.legend(loc='upper left', fontsize=8, ncol=2)
ax3.set_xlabel('x', fontsize=10)
ax3.set_ylabel('U_n(x)·√(1-x²)', fontsize=10)
ax3.set_title('Weighted Chebyshev Polynomials\nU_n(x)·√(1-x²)', 
              fontsize=11, fontweight='bold')
ax3.set_xlim(-1, 1)

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
ax4.set_title('Orthogonality Matrix\n<U_m, U_n> (normalized)', 
              fontsize=11, fontweight='bold')

# Add text annotations
for m in range(max_degree + 1):
    for n in range(max_degree + 1):
        text = ax4.text(n, m, f'{ortho_matrix_norm[m, n]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax4, label='Normalized Inner Product')

# Plot 5: Trigonometric Representation
ax5 = plt.subplot(3, 3, 5)
theta = np.linspace(0, np.pi, 500)

for n in range(min(5, max_degree + 1)):
    # U_n(cos θ) = sin((n+1)θ) / sin(θ)
    U_n_trig = np.sin((n + 1) * theta) / np.sin(theta)
    ax5.plot(theta, U_n_trig, color=colors[n], linewidth=2, 
            label=f'sin({n+1}θ)/sin(θ)', alpha=0.8)

ax5.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax5.grid(True, linestyle='--', alpha=0.4)
ax5.legend(loc='upper right', fontsize=8)
ax5.set_xlabel('θ (radians)', fontsize=10)
ax5.set_ylabel('U_n(cos θ)', fontsize=10)
ax5.set_title('Trigonometric Form\nU_n(cos θ) = sin((n+1)θ)/sin(θ)', 
              fontsize=11, fontweight='bold')
ax5.set_xlim(0, np.pi)
ax5.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
ax5.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])

# Plot 6: Chebyshev Nodes (Zeros)
ax6 = plt.subplot(3, 3, 6)
n_nodes = 6
zeros = [np.cos(k * np.pi / (n_nodes + 1)) for k in range(1, n_nodes + 1)]
U_n_nodes = chebyshev_second_kind(n_nodes, x)

ax6.plot(x, U_n_nodes, 'b-', linewidth=3, label=f'U_{n_nodes}(x)')
ax6.axhline(y=0, color='k', linewidth=1, alpha=0.5)

# Mark zeros
ax6.plot(zeros, np.zeros_like(zeros), 'ro', markersize=12, 
         markeredgecolor='black', markeredgewidth=2, label=f'{len(zeros)} zeros', zorder=5)

ax6.grid(True, linestyle='--', alpha=0.4)
ax6.legend(loc='upper left', fontsize=10)
ax6.set_xlabel('x', fontsize=10)
ax6.set_ylabel(f'U_{n_nodes}(x)', fontsize=10)
ax6.set_title(f'Zeros of U_{n_nodes}(x)\nx_k = cos(kπ/{n_nodes+1})', 
              fontsize=11, fontweight='bold')
ax6.set_xlim(-1, 1)

# Plot 7: Boundary Values
ax7 = plt.subplot(3, 3, 7)
degrees = np.arange(0, max_degree + 1)
vals_at_minus_1 = [(-1)**n * (n + 1) for n in degrees]
vals_at_1 = [n + 1 for n in degrees]
vals_at_0 = [eval_chebyu(n, 0) for n in degrees]

ax7.plot(degrees, vals_at_minus_1, 'ro-', linewidth=2, markersize=8, 
         label='U_n(-1) = (-1)^n(n+1)', alpha=0.8)
ax7.plot(degrees, vals_at_1, 'bs-', linewidth=2, markersize=8, 
         label='U_n(1) = n+1', alpha=0.8)
ax7.plot(degrees, vals_at_0, 'g^-', linewidth=2, markersize=8, 
         label='U_n(0)', alpha=0.8)
ax7.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax7.grid(True, linestyle='--', alpha=0.4)
ax7.legend(loc='upper left', fontsize=9)
ax7.set_xlabel('Degree n', fontsize=10)
ax7.set_ylabel('Value', fontsize=10)
ax7.set_title('Boundary Values\nU_n at x = -1, 0, 1', 
              fontsize=11, fontweight='bold')
ax7.set_xticks(degrees)

# Plot 8: Comparison with First Kind
ax8 = plt.subplot(3, 3, 8)
from scipy.special import eval_chebyt

n_compare = 4
T_n = eval_chebyt(n_compare, x)
U_n = chebyshev_second_kind(n_compare, x)

ax8.plot(x, T_n, 'b-', linewidth=2.5, label=f'T_{n_compare}(x) (First Kind)', alpha=0.8)
ax8.plot(x, U_n, 'r-', linewidth=2.5, label=f'U_{n_compare}(x) (Second Kind)', alpha=0.8)
ax8.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax8.grid(True, linestyle='--', alpha=0.4)
ax8.legend(loc='upper left', fontsize=9)
ax8.set_xlabel('x', fontsize=10)
ax8.set_ylabel('Value', fontsize=10)
ax8.set_title(f'Comparison: T_{n_compare}(x) vs U_{n_compare}(x)\nFirst Kind vs Second Kind', 
              fontsize=11, fontweight='bold')
ax8.set_xlim(-1, 1)

# Plot 9: Theory Summary
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

theory_text = """
CHEBYSHEV POLYNOMIALS (SECOND KIND)

Definition & Recurrence:
  U₀(x) = 1
  U₁(x) = 2x
  U_{n+1}(x) = 2x·U_n(x) - U_{n-1}(x)

Trigonometric Form:
  U_n(cos θ) = sin((n+1)θ) / sin(θ)

Orthogonality:
  ∫₋₁¹ U_m(x)U_n(x)√(1-x²) dx = (π/2)·δ_{mn}

Weight Function:
  w(x) = √(1-x²)  (Wigner semicircle)

Relation to First Kind:
  U_n(x) = (1/(n+1))·dT_{n+1}/dx
  dT_{n+1}/dx = (n+1)·U_n(x)

Properties:
• Parity: U_n(-x) = (-1)^n U_n(x)
• Boundary: U_n(1) = n+1, U_n(-1) = (-1)^n(n+1)
• Zeros: x_k = cos(kπ/(n+1)), k=1,...,n
• NOT bounded like T_n (can grow with n)

Applications:
• Gauss-Chebyshev quadrature (2nd kind)
• Random matrix theory (eigenvalues)
• Antenna array theory
• Signal processing (filter design)
• Physics (Wigner semicircle distribution)

Key Insight:
Chebyshev polynomials of the second kind
arise as derivatives of the first kind and
are connected to the Wigner semicircle
distribution in random matrix theory.
"""

ax9.text(0.05, 0.95, theory_text, transform=ax9.transAxes, fontsize=9.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.95,
                  edgecolor='purple', linewidth=2, pad=1.5))

plt.suptitle('Chebyshev Polynomials (Second Kind) - Wigner Semicircle Weight', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.99))
plt.show()

print("\n✓ Visualization complete!")
print("\nNote: Chebyshev polynomials (second kind) are related to the first kind")
print("      via differentiation: U_n(x) = (1/(n+1))·dT_{n+1}/dx")
print("      They arise in random matrix theory and antenna array design.")
