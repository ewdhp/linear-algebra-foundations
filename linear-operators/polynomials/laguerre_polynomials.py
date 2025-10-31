"""
LAGUERRE POLYNOMIALS - ORTHOGONAL WITH RESPECT TO EXPONENTIAL DISTRIBUTION

Theory:
Laguerre polynomials L_n(x) are orthogonal on the interval [0, ∞) with respect 
to the weight function w(x) = e^(-x), which is the exponential distribution.

Orthogonality Relation:
    ∫_{0}^{∞} L_m(x) L_n(x) e^(-x) dx = δ_{mn}
    
where δ_{mn} is the Kronecker delta (1 if m=n, 0 otherwise)

Recurrence Relation:
    (n+1)L_{n+1}(x) = (2n+1-x)L_n(x) - n·L_{n-1}(x)
    L_0(x) = 1
    L_1(x) = 1 - x

Rodrigues' Formula:
    L_n(x) = (e^x/n!) · d^n/dx^n(x^n·e^(-x))

Applications:
- Quantum mechanics (hydrogen atom radial wavefunctions)
- Statistics (chi-squared and gamma distributions)
- Numerical analysis (Gauss-Laguerre quadrature)
- Physics (quantum harmonic oscillator in position space)
- Probability theory (moment generating functions)

Properties:
- L_n(0) = 1 for all n
- Zeros: n distinct positive real zeros
- Derivative: dL_n/dx = -L_n^(1)(x) (generalized Laguerre)
- Connection to gamma distribution: Γ(n+1) related to moments
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import laguerre, eval_laguerre, genlaguerre
from scipy.integrate import quad

def laguerre_polynomial(n, x):
    """
    Generate Laguerre polynomial L_n(x) using recurrence relation
    """
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 1 - x
    else:
        L_prev_prev = np.ones_like(x)
        L_prev = 1 - x
        for k in range(2, n + 1):
            L_current = ((2*k - 1 - x) * L_prev - (k - 1) * L_prev_prev) / k
            L_prev_prev = L_prev
            L_prev = L_current
        return L_current

def weight_function(x):
    """Exponential weight function w(x) = e^(-x)"""
    return np.exp(-x)

def compute_inner_product(n, m, x_max=50):
    """
    Compute inner product <L_n, L_m> = ∫_{0}^{∞} L_n(x) L_m(x) e^(-x) dx
    """
    def integrand(x):
        return eval_laguerre(n, x) * eval_laguerre(m, x) * np.exp(-x)
    
    result, error = quad(integrand, 0, x_max, limit=100)
    return result

def theoretical_norm_squared(n):
    """
    Theoretical value: ||L_n||² = 1
    """
    return 1.0

# Generate Laguerre polynomials up to degree 5
max_degree = 5
x = np.linspace(0, 15, 500)

print("\n" + "="*100)
print("LAGUERRE POLYNOMIALS - ORTHOGONAL WITH RESPECT TO EXPONENTIAL DISTRIBUTION")
print("="*100)
print("\nWeight Function: w(x) = e^(-x)  (Exponential distribution)")
print("Orthogonality: ∫_{0}^{∞} L_m(x) L_n(x) e^(-x) dx = δ_{mn}")
print("-"*100)

# Compute and display polynomial expressions
poly_data = []
for n in range(max_degree + 1):
    # Get polynomial coefficients
    L_n_poly = laguerre(n)
    coeffs = L_n_poly.coef
    
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
        "L_n(x)": expression,
        "L_n(0)": 1,
        "L_n(1)": f"{eval_laguerre(n, 1):.4f}",
        "L_n(5)": f"{eval_laguerre(n, 5):.4f}",
        "||L_n||²": f"{theoretical_norm_squared(n):.6f}"
    })

df_poly = pd.DataFrame(poly_data)
print("\nLaguerre Polynomials (first few degrees):")
print(df_poly.to_string(index=False))
print("-"*100)

# Test orthogonality by computing inner products
print("\nOrthogonality Test: Inner Products <L_m, L_n>")
print("(Should be 0 for m ≠ n, and 1 for m = n)")
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
    x_vals = np.linspace(0, 20, 2000)
    L_n = laguerre_polynomial(n, x_vals)
    
    # Number of zeros (roots)
    roots = np.roots(laguerre(n).coef[::-1])
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    zeros_in_range = np.sum(real_roots > 0)
    
    # Find smallest positive root
    if zeros_in_range > 0:
        positive_roots = real_roots[real_roots > 0]
        smallest_root = np.min(positive_roots)
    else:
        smallest_root = 0
    
    stats_data.append({
        "Degree n": n,
        "Number of Zeros": zeros_in_range,
        "Smallest Zero": f"{smallest_root:.4f}" if smallest_root > 0 else "N/A",
        "L_n(0)": 1,
        "Norm ||L_n||": f"{np.sqrt(theoretical_norm_squared(n)):.6f}"
    })

df_stats = pd.DataFrame(stats_data)
print(df_stats.to_string(index=False))
print("="*100)

# Connection to Gamma distribution
print("\nConnection to Gamma Distribution:")
print("The Laguerre polynomials are related to moments of the Gamma distribution.")
print("Gamma(k, θ) has PDF: f(x) = (1/Γ(k)θ^k) x^(k-1) e^(-x/θ)")
print("For θ=1, this simplifies to the Exponential distribution when k=1.")
print("-"*100)

gamma_data = []
for n in range(max_degree + 1):
    # Expected value: E[L_n(X)] for X ~ Exp(1)
    def moment_integrand(x):
        return eval_laguerre(n, x) * np.exp(-x)
    
    expected_val, _ = quad(moment_integrand, 0, 50, limit=100)
    
    gamma_data.append({
        "Degree n": n,
        "E[L_n(X)] for X~Exp(1)": f"{expected_val:.6f}",
        "Theoretical": "1.0000" if n == 0 else "0.0000"
    })

df_gamma = pd.DataFrame(gamma_data)
print(df_gamma.to_string(index=False))
print("="*100)

# Visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: Laguerre Polynomials
ax1 = plt.subplot(3, 3, 1)
colors = plt.cm.coolwarm(np.linspace(0, 1, max_degree + 1))

for n in range(max_degree + 1):
    L_n = laguerre_polynomial(n, x)
    ax1.plot(x, L_n, color=colors[n], linewidth=2.5, label=f'L_{n}(x)', alpha=0.8)

ax1.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax1.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper right', fontsize=9, ncol=2)
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('L_n(x)', fontsize=10)
ax1.set_title('Laguerre Polynomials L_n(x)\n(First 6 polynomials on [0, 15])', 
              fontsize=11, fontweight='bold')
ax1.set_xlim(0, 15)
ax1.set_ylim(-5, 8)

# Plot 2: Weight Function (Exponential)
ax2 = plt.subplot(3, 3, 2)
w = weight_function(x)
ax2.plot(x, w, 'r-', linewidth=3, label='w(x) = e^(-x)')
ax2.fill_between(x, 0, w, alpha=0.3, color='red')
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('w(x)', fontsize=10)
ax2.set_title('Weight Function (Exponential)\nw(x) = e^(-x)', 
              fontsize=11, fontweight='bold')
ax2.set_xlim(0, 15)

# Plot 3: Weighted Polynomials
ax3 = plt.subplot(3, 3, 3)
for n in range(max_degree + 1):
    L_n = laguerre_polynomial(n, x)
    weighted = L_n * np.sqrt(w)  # Weighted for visualization
    ax3.plot(x, weighted, color=colors[n], linewidth=2, label=f'L_{n}·√w', alpha=0.7)

ax3.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax3.grid(True, linestyle='--', alpha=0.4)
ax3.legend(loc='upper right', fontsize=8, ncol=2)
ax3.set_xlabel('x', fontsize=10)
ax3.set_ylabel('L_n(x)·√w(x)', fontsize=10)
ax3.set_title('Weighted Laguerre Polynomials\nL_n(x)·√(e^(-x))', 
              fontsize=11, fontweight='bold')
ax3.set_xlim(0, 15)

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
ax4.set_title('Orthogonality Matrix\n<L_m, L_n> (normalized)', 
              fontsize=11, fontweight='bold')

# Add text annotations
for m in range(max_degree + 1):
    for n in range(max_degree + 1):
        text = ax4.text(n, m, f'{ortho_matrix_norm[m, n]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax4, label='Normalized Inner Product')

# Plot 5: Individual polynomial with zeros
ax5 = plt.subplot(3, 3, 5)
n_example = 4
L_example = laguerre_polynomial(n_example, x)
ax5.plot(x, L_example, 'b-', linewidth=3, label=f'L_{n_example}(x)')
ax5.axhline(y=0, color='k', linewidth=1, alpha=0.5)

# Mark zeros
roots = np.roots(laguerre(n_example).coef[::-1])
roots = roots[np.abs(roots.imag) < 1e-10].real
roots = roots[(roots > 0) & (roots < 15)]
ax5.plot(roots, np.zeros_like(roots), 'ro', markersize=12, 
         markeredgecolor='black', markeredgewidth=2, label=f'{len(roots)} zeros', zorder=5)

ax5.grid(True, linestyle='--', alpha=0.4)
ax5.legend(loc='upper right', fontsize=10)
ax5.set_xlabel('x', fontsize=10)
ax5.set_ylabel('L_4(x)', fontsize=10)
ax5.set_title(f'Laguerre Polynomial L_{n_example}(x)\nwith {len(roots)} zeros marked', 
              fontsize=11, fontweight='bold')
ax5.set_xlim(0, 15)

# Plot 6: Gamma Distribution Connection
ax6 = plt.subplot(3, 3, 6)
from scipy.stats import gamma as gamma_dist

x_gamma = np.linspace(0, 15, 500)
# Plot Gamma distributions with different shape parameters
shapes = [1, 2, 3, 5]
gamma_colors = plt.cm.Set1(np.linspace(0, 1, len(shapes)))

for k, color in zip(shapes, gamma_colors):
    gamma_pdf = gamma_dist.pdf(x_gamma, k, scale=1)
    ax6.plot(x_gamma, gamma_pdf, color=color, linewidth=2.5, 
            label=f'Γ(k={k}, θ=1)', alpha=0.8)

ax6.grid(True, linestyle='--', alpha=0.4)
ax6.legend(loc='upper right', fontsize=9)
ax6.set_xlabel('x', fontsize=10)
ax6.set_ylabel('PDF', fontsize=10)
ax6.set_title('Gamma Distributions\n(Related to Laguerre polynomials)', 
              fontsize=11, fontweight='bold')
ax6.set_xlim(0, 15)

# Plot 7: Hydrogen Atom Radial Wavefunctions
ax7 = plt.subplot(3, 3, 7)
# Simplified radial wavefunctions (without normalization constants)
# R_nl(r) ∝ r^l e^(-r/n) L_{n-l-1}^(2l+1)(2r/n)

r = np.linspace(0, 15, 500)
# For simplicity, showing n=3, l=0, 1, 2 states
n_qm = 3
radial_colors = plt.cm.tab10(np.linspace(0, 1, 3))

for l in range(3):
    # Generalized Laguerre polynomial
    rho = 2*r/n_qm
    L_gen = eval_laguerre(n_qm - l - 1, rho)
    R_nl = (rho**l) * np.exp(-rho/2) * L_gen
    # Normalize for visualization
    R_nl = R_nl / np.max(np.abs(R_nl))
    
    ax7.plot(r, R_nl, color=radial_colors[l], linewidth=2.5, 
            label=f'R_{n_qm},{l}(r)', alpha=0.8)

ax7.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax7.grid(True, linestyle='--', alpha=0.4)
ax7.legend(loc='upper right', fontsize=9)
ax7.set_xlabel('r (atomic units)', fontsize=10)
ax7.set_ylabel('R_{nl}(r) (normalized)', fontsize=10)
ax7.set_title('Hydrogen Atom Radial Wavefunctions\n(Using Laguerre polynomials)', 
              fontsize=11, fontweight='bold')
ax7.set_xlim(0, 15)

# Plot 8: Values at x=0
ax8 = plt.subplot(3, 3, 8)
degrees = np.arange(0, max_degree + 1)
vals_at_0 = [1 for n in degrees]  # L_n(0) = 1 for all n
vals_at_1 = [eval_laguerre(n, 1) for n in degrees]
vals_at_5 = [eval_laguerre(n, 5) for n in degrees]

ax8.plot(degrees, vals_at_0, 'ro-', linewidth=2, markersize=8, 
         label='L_n(0) = 1', alpha=0.8)
ax8.plot(degrees, vals_at_1, 'bs-', linewidth=2, markersize=8, 
         label='L_n(1)', alpha=0.8)
ax8.plot(degrees, vals_at_5, 'g^-', linewidth=2, markersize=8, 
         label='L_n(5)', alpha=0.8)
ax8.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax8.grid(True, linestyle='--', alpha=0.4)
ax8.legend(loc='best', fontsize=9)
ax8.set_xlabel('Degree n', fontsize=10)
ax8.set_ylabel('Value', fontsize=10)
ax8.set_title('Polynomial Values\nL_n at x = 0, 1, 5', 
              fontsize=11, fontweight='bold')
ax8.set_xticks(degrees)

# Plot 9: Theory Summary
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

theory_text = """
LAGUERRE POLYNOMIALS SUMMARY

Definition & Recurrence:
  L₀(x) = 1
  L₁(x) = 1 - x
  (n+1)L_{n+1}(x) = (2n+1-x)L_n(x) - n·L_{n-1}(x)

Orthogonality:
  ∫₀^∞ L_m(x)L_n(x)e^(-x) dx = δ_{mn}

Weight Function:
  w(x) = e^(-x)  (Exponential distribution)

Rodrigues' Formula:
  L_n(x) = e^x/n! · d^n/dx^n(x^n·e^(-x))

Properties:
• L_n(0) = 1 for all n
• Zeros: n distinct positive real zeros
• Norm: ||L_n|| = 1
• Connection to Gamma distribution
• Generalized form: L_n^(α)(x)

Applications:
• Hydrogen atom (radial wavefunctions)
• Quantum mechanics (harmonic oscillator)
• Gauss-Laguerre quadrature
• Chi-squared & Gamma distributions
• Moment generating functions

Key Insight:
Laguerre polynomials naturally arise in
quantum mechanics (hydrogen atom) and are
the orthogonal basis for functions with
exponential decay (weight e^(-x)).
"""

ax9.text(0.05, 0.95, theory_text, transform=ax9.transAxes, fontsize=9.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.95,
                  edgecolor='green', linewidth=2, pad=1.5))

plt.suptitle('Laguerre Polynomials - Orthogonal with Exponential Weight', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.99))
plt.show()

print("\n✓ Visualization complete!")
print("\nNote: Laguerre polynomials are essential in quantum mechanics")
print("      Hydrogen atom radial wavefunctions: R_nl(r) ∝ L_{n-l-1}^(2l+1)(2r/n)·e^(-r/n)")
