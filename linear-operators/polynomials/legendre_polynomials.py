"""
LEGENDRE POLYNOMIALS - ORTHOGONAL WITH RESPECT TO UNIFORM DISTRIBUTION

Theory:
Legendre polynomials P_n(x) are orthogonal on the interval [-1, 1] with respect 
to the uniform weight function w(x) = 1.

Orthogonality Relation:
    ∫_{-1}^{1} P_m(x) P_n(x) dx = (2/(2n+1)) · δ_{mn}
    
where δ_{mn} is the Kronecker delta (1 if m=n, 0 otherwise)

Recurrence Relation:
    (n+1)P_{n+1}(x) = (2n+1)x·P_n(x) - n·P_{n-1}(x)
    P_0(x) = 1
    P_1(x) = x

Rodrigues' Formula:
    P_n(x) = (1/(2^n·n!)) · d^n/dx^n[(x²-1)^n]

Applications:
- Solving Laplace's equation in spherical coordinates
- Multipole expansion in electrostatics
- Legendre-Gauss quadrature (numerical integration)
- Spherical harmonics (quantum mechanics, geophysics)
- Approximation theory

Properties:
- P_n(1) = 1 for all n
- P_n(-1) = (-1)^n
- Even/odd symmetry: P_n(-x) = (-1)^n P_n(x)
- Bounded: |P_n(x)| ≤ 1 for x ∈ [-1,1]
- Norm: ||P_n||² = 2/(2n+1)
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import legendre, eval_legendre
from scipy.integrate import quad

def legendre_polynomial(n, x):
    """
    Generate Legendre polynomial P_n(x) using recurrence relation
    """
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        P_prev_prev = np.ones_like(x)
        P_prev = x
        for k in range(2, n + 1):
            P_current = ((2*k - 1) * x * P_prev - (k - 1) * P_prev_prev) / k
            P_prev_prev = P_prev
            P_prev = P_current
        return P_current

def compute_inner_product(n, m):
    """
    Compute inner product <P_n, P_m> = ∫_{-1}^{1} P_n(x) P_m(x) dx
    """
    def integrand(x):
        return eval_legendre(n, x) * eval_legendre(m, x)
    
    result, error = quad(integrand, -1, 1)
    return result

def theoretical_norm_squared(n):
    """
    Theoretical value: ||P_n||² = 2/(2n+1)
    """
    return 2.0 / (2*n + 1)

# Generate Legendre polynomials up to degree 6
max_degree = 6
x = np.linspace(-1, 1, 500)

print("\n" + "="*100)
print("LEGENDRE POLYNOMIALS - ORTHOGONAL WITH RESPECT TO UNIFORM DISTRIBUTION")
print("="*100)
print("\nWeight Function: w(x) = 1  (Uniform on [-1, 1])")
print("Orthogonality: ∫_{-1}^{1} P_m(x) P_n(x) dx = 2/(2n+1) · δ_{mn}")
print("-"*100)

# Compute and display polynomial expressions
poly_data = []
for n in range(max_degree + 1):
    # Get polynomial coefficients
    P_n_poly = legendre(n)
    coeffs = P_n_poly.coef
    
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
        "P_n(x)": expression,
        "P_n(-1)": int((-1)**n),
        "P_n(0)": f"{eval_legendre(n, 0):.4f}",
        "P_n(1)": 1,
        "||P_n||²": f"{theoretical_norm_squared(n):.6f}"
    })

df_poly = pd.DataFrame(poly_data)
print("\nLegendre Polynomials (first few degrees):")
print(df_poly.to_string(index=False))
print("-"*100)

# Test orthogonality by computing inner products
print("\nOrthogonality Test: Inner Products <P_m, P_n>")
print("(Should be 0 for m ≠ n, and 2/(2n+1) for m = n)")
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

# Statistical and mathematical properties
print("\nMathematical Properties:")
stats_data = []
for n in range(max_degree + 1):
    x_vals = np.linspace(-1, 1, 1000)
    P_n = legendre_polynomial(n, x_vals)
    
    # Number of zeros (roots)
    roots = np.roots(legendre(n).coef[::-1])
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    zeros_in_range = np.sum((real_roots >= -1) & (real_roots <= 1))
    
    # Maximum value
    max_val = np.max(np.abs(P_n))
    
    stats_data.append({
        "Degree n": n,
        "Number of Zeros": zeros_in_range,
        "Max |P_n(x)|": f"{max_val:.6f}",
        "Parity": "Even" if n % 2 == 0 else "Odd",
        "Norm ||P_n||": f"{np.sqrt(theoretical_norm_squared(n)):.6f}",
        "2/(2n+1)": f"{theoretical_norm_squared(n):.6f}"
    })

df_stats = pd.DataFrame(stats_data)
print(df_stats.to_string(index=False))
print("="*100)

# Special values table
print("\nSpecial Values of Legendre Polynomials:")
special_vals_data = []
special_points = [-1, -0.5, 0, 0.5, 1]
for n in range(min(6, max_degree + 1)):
    row = {"n": n}
    for x_val in special_points:
        val = eval_legendre(n, x_val)
        row[f"P_n({x_val})"] = f"{val:.6f}"
    special_vals_data.append(row)

df_special = pd.DataFrame(special_vals_data)
print(df_special.to_string(index=False))
print("="*100)

# Visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: Legendre Polynomials
ax1 = plt.subplot(3, 3, 1)
colors = plt.cm.plasma(np.linspace(0, 1, max_degree + 1))

for n in range(max_degree + 1):
    P_n = legendre_polynomial(n, x)
    ax1.plot(x, P_n, color=colors[n], linewidth=2.5, label=f'P_{n}(x)', alpha=0.8)

ax1.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax1.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
ax1.axhline(y=1, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
ax1.axhline(y=-1, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper left', fontsize=9, ncol=2)
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('P_n(x)', fontsize=10)
ax1.set_title('Legendre Polynomials P_n(x)\n(First 7 polynomials on [-1,1])', 
              fontsize=11, fontweight='bold')
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1.2, 1.2)

# Plot 2: Weight Function (Constant)
ax2 = plt.subplot(3, 3, 2)
w = np.ones_like(x)
ax2.plot(x, w, 'r-', linewidth=3, label='w(x) = 1')
ax2.fill_between(x, 0, w, alpha=0.3, color='red')
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('w(x)', fontsize=10)
ax2.set_title('Weight Function (Uniform)\nw(x) = 1 on [-1, 1]', 
              fontsize=11, fontweight='bold')
ax2.set_xlim(-1, 1)
ax2.set_ylim(0, 1.5)

# Plot 3: Orthogonality Matrix Heatmap
ax3 = plt.subplot(3, 3, 3)
ortho_matrix = np.zeros((max_degree + 1, max_degree + 1))
for m in range(max_degree + 1):
    for n in range(max_degree + 1):
        ortho_matrix[m, n] = compute_inner_product(m, n)

# Normalize for better visualization
ortho_matrix_norm = ortho_matrix / np.max(np.abs(ortho_matrix))

im = ax3.imshow(ortho_matrix_norm, cmap='RdYlGn', aspect='auto', 
                interpolation='nearest', vmin=-0.1, vmax=1)
ax3.set_xticks(range(max_degree + 1))
ax3.set_yticks(range(max_degree + 1))
ax3.set_xticklabels([f'n={i}' for i in range(max_degree + 1)])
ax3.set_yticklabels([f'm={i}' for i in range(max_degree + 1)])
ax3.set_xlabel('Polynomial degree n', fontsize=10)
ax3.set_ylabel('Polynomial degree m', fontsize=10)
ax3.set_title('Orthogonality Matrix\n<P_m, P_n> (normalized)', 
              fontsize=11, fontweight='bold')

# Add text annotations
for m in range(max_degree + 1):
    for n in range(max_degree + 1):
        text = ax3.text(n, m, f'{ortho_matrix_norm[m, n]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax3, label='Normalized Inner Product')

# Plot 4: Individual polynomial with zeros
ax4 = plt.subplot(3, 3, 4)
n_example = 5
P_example = legendre_polynomial(n_example, x)
ax4.plot(x, P_example, 'b-', linewidth=3, label=f'P_{n_example}(x)')
ax4.axhline(y=0, color='k', linewidth=1, alpha=0.5)

# Mark zeros
roots = np.roots(legendre(n_example).coef[::-1])
roots = roots[np.abs(roots.imag) < 1e-10].real
roots = roots[(roots >= -1) & (roots <= 1)]
ax4.plot(roots, np.zeros_like(roots), 'ro', markersize=12, 
         markeredgecolor='black', markeredgewidth=2, label=f'{len(roots)} zeros', zorder=5)

ax4.grid(True, linestyle='--', alpha=0.4)
ax4.legend(loc='upper left', fontsize=10)
ax4.set_xlabel('x', fontsize=10)
ax4.set_ylabel('P_5(x)', fontsize=10)
ax4.set_title(f'Legendre Polynomial P_{n_example}(x)\nwith {len(roots)} zeros marked', 
              fontsize=11, fontweight='bold')
ax4.set_xlim(-1, 1)

# Plot 5: Derivatives
ax5 = plt.subplot(3, 3, 5)
n_deriv = 3
P_n = legendre_polynomial(n_deriv, x)
# Compute derivative numerically
dP_n_dx = np.gradient(P_n, x)

ax5.plot(x, P_n, 'b-', linewidth=2.5, label=f'P_{n_deriv}(x)', alpha=0.8)
ax5.plot(x, dP_n_dx, 'r--', linewidth=2.5, label=f"P'_{n_deriv}(x)", alpha=0.8)
ax5.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax5.grid(True, linestyle='--', alpha=0.4)
ax5.legend(loc='upper left', fontsize=9)
ax5.set_xlabel('x', fontsize=10)
ax5.set_ylabel('Value', fontsize=10)
ax5.set_title(f'Legendre Polynomial and Derivative\nP_{n_deriv}(x) and P\'_{n_deriv}(x)', 
              fontsize=11, fontweight='bold')
ax5.set_xlim(-1, 1)

# Plot 6: Even/Odd Symmetry
ax6 = plt.subplot(3, 3, 6)
n_even = 4
n_odd = 5
P_even = legendre_polynomial(n_even, x)
P_odd = legendre_polynomial(n_odd, x)

ax6.plot(x, P_even, 'b-', linewidth=2.5, label=f'P_{n_even}(x) (even)', alpha=0.8)
ax6.plot(x, P_odd, 'r-', linewidth=2.5, label=f'P_{n_odd}(x) (odd)', alpha=0.8)
ax6.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax6.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
ax6.grid(True, linestyle='--', alpha=0.4)
ax6.legend(loc='upper left', fontsize=9)
ax6.set_xlabel('x', fontsize=10)
ax6.set_ylabel('P_n(x)', fontsize=10)
ax6.set_title('Parity Symmetry\nP_n(-x) = (-1)^n·P_n(x)', 
              fontsize=11, fontweight='bold')
ax6.set_xlim(-1, 1)

# Plot 7: Boundary Values
ax7 = plt.subplot(3, 3, 7)
degrees = np.arange(0, max_degree + 1)
vals_at_minus_1 = [(-1)**n for n in degrees]
vals_at_1 = [1 for n in degrees]
vals_at_0 = [eval_legendre(n, 0) for n in degrees]

ax7.plot(degrees, vals_at_minus_1, 'ro-', linewidth=2, markersize=8, 
         label='P_n(-1) = (-1)^n', alpha=0.8)
ax7.plot(degrees, vals_at_1, 'bs-', linewidth=2, markersize=8, 
         label='P_n(1) = 1', alpha=0.8)
ax7.plot(degrees, vals_at_0, 'g^-', linewidth=2, markersize=8, 
         label='P_n(0)', alpha=0.8)
ax7.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax7.grid(True, linestyle='--', alpha=0.4)
ax7.legend(loc='best', fontsize=9)
ax7.set_xlabel('Degree n', fontsize=10)
ax7.set_ylabel('Value', fontsize=10)
ax7.set_title('Boundary Values\nP_n at x = -1, 0, 1', 
              fontsize=11, fontweight='bold')
ax7.set_xticks(degrees)

# Plot 8: Norms
ax8 = plt.subplot(3, 3, 8)
norms = [np.sqrt(theoretical_norm_squared(n)) for n in degrees]
theoretical_curve = [np.sqrt(2.0/(2*n + 1)) for n in degrees]

ax8.plot(degrees, norms, 'mo-', linewidth=2.5, markersize=10, 
         label='||P_n|| = √(2/(2n+1))', alpha=0.8)
ax8.fill_between(degrees, 0, norms, alpha=0.3, color='magenta')
ax8.grid(True, linestyle='--', alpha=0.4)
ax8.legend(loc='upper right', fontsize=9)
ax8.set_xlabel('Degree n', fontsize=10)
ax8.set_ylabel('Norm', fontsize=10)
ax8.set_title('Norm of Legendre Polynomials\n||P_n|| = √(2/(2n+1))', 
              fontsize=11, fontweight='bold')
ax8.set_xticks(degrees)
ax8.set_ylim(0, 1.6)

# Plot 9: Theory Summary
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

theory_text = """
LEGENDRE POLYNOMIALS SUMMARY

Definition & Recurrence:
  P₀(x) = 1
  P₁(x) = x
  (n+1)P_{n+1}(x) = (2n+1)x·P_n(x) - n·P_{n-1}(x)

Orthogonality:
  ∫₋₁¹ P_m(x)P_n(x) dx = 2/(2n+1)·δ_{mn}

Weight Function:
  w(x) = 1  (Uniform on [-1, 1])

Rodrigues' Formula:
  P_n(x) = 1/(2^n·n!) · d^n/dx^n[(x²-1)^n]

Properties:
• Parity: P_n(-x) = (-1)^n P_n(x)
• Boundary: P_n(1) = 1, P_n(-1) = (-1)^n
• Bounded: |P_n(x)| ≤ 1 for x ∈ [-1,1]
• Zeros: n distinct real zeros in (-1, 1)
• Norm: ||P_n|| = √(2/(2n+1))

Applications:
• Laplace equation (spherical coords)
• Multipole expansion (electrostatics)
• Gauss-Legendre quadrature
• Spherical harmonics (quantum mechanics)
• Approximation theory

Key Insight:
Legendre polynomials form a complete
orthogonal basis for L²[-1,1], making them
ideal for solving PDEs with spherical
symmetry and numerical integration.
"""

ax9.text(0.05, 0.95, theory_text, transform=ax9.transAxes, fontsize=9.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95,
                  edgecolor='orange', linewidth=2, pad=1.5))

plt.suptitle('Legendre Polynomials - Orthogonal with Uniform Weight', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.99))
plt.show()

print("\n✓ Visualization complete!")
print("\nNote: Legendre polynomials are solutions to Legendre's differential equation:")
print("      d/dx[(1-x²)dP_n/dx] + n(n+1)P_n = 0")
print("      They arise naturally in problems with spherical symmetry.")
