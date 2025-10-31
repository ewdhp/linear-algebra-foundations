"""
HERMITE POLYNOMIALS - ORTHOGONAL WITH RESPECT TO GAUSSIAN DISTRIBUTION

Theory:
Hermite polynomials H_n(x) are orthogonal with respect to the weight function
w(x) = e^(-x²), which is the Gaussian (normal) distribution with zero mean.

Orthogonality Relation:
    ∫_{-∞}^{∞} H_m(x) H_n(x) e^(-x²) dx = √π · 2^n · n! · δ_{mn}
    
where δ_{mn} is the Kronecker delta (1 if m=n, 0 otherwise)

Recurrence Relation:
    H_{n+1}(x) = 2x·H_n(x) - 2n·H_{n-1}(x)
    H_0(x) = 1
    H_1(x) = 2x

Applications:
- Quantum mechanics (harmonic oscillator wavefunctions)
- Probability theory (Hermite expansion)
- Signal processing (Gaussian derivatives)
- Physics (quantum field theory)
- Statistics (Edgeworth series)

Properties:
- Even/odd symmetry: H_n(-x) = (-1)^n H_n(x)
- Derivatives: dH_n/dx = 2n·H_{n-1}(x)
- Rodrigues' formula: H_n(x) = (-1)^n e^(x²) d^n/dx^n(e^(-x²))
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import hermite, eval_hermite
from scipy.integrate import quad

def hermite_polynomial(n, x):
    """
    Generate Hermite polynomial H_n(x) using recurrence relation
    """
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        H_prev_prev = np.ones_like(x)
        H_prev = 2 * x
        for k in range(2, n + 1):
            H_current = 2 * x * H_prev - 2 * (k - 1) * H_prev_prev
            H_prev_prev = H_prev
            H_prev = H_current
        return H_current

def weight_function(x):
    """Gaussian weight function w(x) = e^(-x²)"""
    return np.exp(-x**2)

def compute_inner_product(n, m, x_range=(-5, 5), num_points=1000):
    """
    Compute inner product <H_n, H_m> = ∫ H_n(x) H_m(x) e^(-x²) dx
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    H_n = hermite_polynomial(n, x)
    H_m = hermite_polynomial(m, x)
    w = weight_function(x)
    
    # Numerical integration using trapezoidal rule
    integrand = H_n * H_m * w
    inner_product = np.trapz(integrand, x)
    
    return inner_product

def theoretical_norm(n):
    """
    Theoretical value: ||H_n||² = √π · 2^n · n!
    """
    import math
    return np.sqrt(np.pi) * (2**n) * math.factorial(n)

# Generate Hermite polynomials up to degree 5
max_degree = 5
x = np.linspace(-3, 3, 500)

print("\n" + "="*100)
print("HERMITE POLYNOMIALS - ORTHOGONAL WITH RESPECT TO GAUSSIAN DISTRIBUTION")
print("="*100)
print("\nWeight Function: w(x) = e^(-x²)  (Gaussian with zero mean)")
print("Orthogonality: ∫_{-∞}^{∞} H_m(x) H_n(x) e^(-x²) dx = √π · 2^n · n! · δ_{mn}")
print("-"*100)

# Compute and display polynomial expressions
poly_data = []
for n in range(max_degree + 1):
    # Get polynomial coefficients
    H_n_poly = hermite(n)
    coeffs = H_n_poly.coef
    
    # Create expression string
    terms = []
    for i, c in enumerate(coeffs[::-1]):
        if abs(c) < 1e-10:
            continue
        power = len(coeffs) - 1 - i
        if power == 0:
            terms.append(f"{int(c)}")
        elif power == 1:
            if c == 1:
                terms.append("x")
            elif c == -1:
                terms.append("-x")
            else:
                terms.append(f"{int(c)}x")
        else:
            if c == 1:
                terms.append(f"x^{power}")
            elif c == -1:
                terms.append(f"-x^{power}")
            else:
                terms.append(f"{int(c)}x^{power}")
    
    expression = " + ".join(terms).replace("+ -", "- ")
    
    poly_data.append({
        "Degree n": n,
        "H_n(x)": expression,
        "H_n(0)": int(hermite_polynomial(n, np.array([0]))[0]),
        "H_n(1)": int(hermite_polynomial(n, np.array([1]))[0]),
        "Norm² (theoretical)": f"{theoretical_norm(n):.4f}"
    })

df_poly = pd.DataFrame(poly_data)
print("\nHermite Polynomials (first few degrees):")
print(df_poly.to_string(index=False))
print("-"*100)

# Test orthogonality by computing inner products
print("\nOrthogonality Test: Inner Products <H_m, H_n>")
print("(Should be 0 for m ≠ n, and √π·2^n·n! for m = n)")
print("-"*100)

orthogonality_data = []
for m in range(max_degree + 1):
    row_data = {"m": m}
    for n in range(max_degree + 1):
        inner_prod = compute_inner_product(m, n)
        if m == n:
            theoretical = theoretical_norm(n)
            error = abs(inner_prod - theoretical) / theoretical * 100
            row_data[f"n={n}"] = f"{inner_prod:.2f}\n({error:.1f}% err)"
        else:
            row_data[f"n={n}"] = f"{inner_prod:.4f}"
    orthogonality_data.append(row_data)

df_ortho = pd.DataFrame(orthogonality_data)
print(df_ortho.to_string(index=False))
print("-"*100)

# Statistical properties
print("\nStatistical Properties:")
stats_data = []
for n in range(max_degree + 1):
    x_vals = np.linspace(-4, 4, 1000)
    H_n = hermite_polynomial(n, x_vals)
    w = weight_function(x_vals)
    
    # Weighted mean
    weighted_mean = np.trapz(x_vals * H_n**2 * w, x_vals) / np.trapz(H_n**2 * w, x_vals)
    
    # Number of zeros (roots)
    sign_changes = np.sum(np.diff(np.sign(H_n)) != 0)
    
    stats_data.append({
        "Degree n": n,
        "Number of Zeros": sign_changes,
        "Max |H_n(x)|": f"{np.max(np.abs(H_n)):.2f}",
        "Parity": "Even" if n % 2 == 0 else "Odd",
        "Norm ||H_n||": f"{np.sqrt(theoretical_norm(n)):.4f}"
    })

df_stats = pd.DataFrame(stats_data)
print(df_stats.to_string(index=False))
print("="*100)

# Visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: Hermite Polynomials
ax1 = plt.subplot(3, 3, 1)
colors = plt.cm.viridis(np.linspace(0, 1, max_degree + 1))

for n in range(max_degree + 1):
    H_n = hermite_polynomial(n, x)
    ax1.plot(x, H_n, color=colors[n], linewidth=2.5, label=f'H_{n}(x)', alpha=0.8)

ax1.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax1.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper left', fontsize=9, ncol=2)
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('H_n(x)', fontsize=10)
ax1.set_title('Hermite Polynomials H_n(x)\n(First 6 polynomials)', 
              fontsize=11, fontweight='bold')
ax1.set_xlim(-3, 3)
ax1.set_ylim(-30, 30)

# Plot 2: Weight Function (Gaussian)
ax2 = plt.subplot(3, 3, 2)
w = weight_function(x)
ax2.plot(x, w, 'r-', linewidth=3, label='w(x) = e^(-x²)')
ax2.fill_between(x, 0, w, alpha=0.3, color='red')
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('w(x)', fontsize=10)
ax2.set_title('Weight Function (Gaussian)\nw(x) = e^(-x²)', 
              fontsize=11, fontweight='bold')
ax2.set_xlim(-3, 3)

# Plot 3: Weighted Polynomials
ax3 = plt.subplot(3, 3, 3)
for n in range(max_degree + 1):
    H_n = hermite_polynomial(n, x)
    weighted = H_n * np.sqrt(w)  # Weighted for visualization
    ax3.plot(x, weighted, color=colors[n], linewidth=2, label=f'H_{n}·√w', alpha=0.7)

ax3.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax3.grid(True, linestyle='--', alpha=0.4)
ax3.legend(loc='upper left', fontsize=8, ncol=2)
ax3.set_xlabel('x', fontsize=10)
ax3.set_ylabel('H_n(x)·√w(x)', fontsize=10)
ax3.set_title('Weighted Hermite Polynomials\nH_n(x)·√(e^(-x²))', 
              fontsize=11, fontweight='bold')
ax3.set_xlim(-3, 3)

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
ax4.set_title('Orthogonality Matrix\n<H_m, H_n> (normalized)', 
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
H_example = hermite_polynomial(n_example, x)
ax5.plot(x, H_example, 'b-', linewidth=3, label=f'H_{n_example}(x)')
ax5.axhline(y=0, color='k', linewidth=1, alpha=0.5)

# Mark zeros
roots = np.roots(hermite(n_example).coef[::-1])
roots = roots[np.abs(roots.imag) < 1e-10].real
roots = roots[(roots > -3) & (roots < 3)]
ax5.plot(roots, np.zeros_like(roots), 'ro', markersize=12, 
         markeredgecolor='black', markeredgewidth=2, label=f'{len(roots)} zeros', zorder=5)

ax5.grid(True, linestyle='--', alpha=0.4)
ax5.legend(loc='upper left', fontsize=10)
ax5.set_xlabel('x', fontsize=10)
ax5.set_ylabel('H_4(x)', fontsize=10)
ax5.set_title(f'Hermite Polynomial H_{n_example}(x)\nwith {len(roots)} zeros marked', 
              fontsize=11, fontweight='bold')
ax5.set_xlim(-3, 3)

# Plot 6: Quantum Harmonic Oscillator Connection
ax6 = plt.subplot(3, 3, 6)
x_qho = np.linspace(-4, 4, 500)
for n in range(4):
    # Quantum harmonic oscillator wavefunction: ψ_n(x) ∝ H_n(x) e^(-x²/2)
    H_n = hermite_polynomial(n, x_qho)
    psi_n = H_n * np.exp(-x_qho**2 / 2)
    # Normalize
    norm = np.sqrt(np.trapz(psi_n**2, x_qho))
    psi_n = psi_n / norm
    
    ax6.plot(x_qho, psi_n + n, color=colors[n], linewidth=2.5, 
            label=f'ψ_{n}(x)', alpha=0.8)
    ax6.axhline(y=n, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)

ax6.grid(True, linestyle='--', alpha=0.4)
ax6.legend(loc='upper right', fontsize=9)
ax6.set_xlabel('x (position)', fontsize=10)
ax6.set_ylabel('ψ_n(x) + n (shifted)', fontsize=10)
ax6.set_title('Quantum Harmonic Oscillator\nWavefunctions (using Hermite)', 
              fontsize=11, fontweight='bold')
ax6.set_xlim(-4, 4)

# Plot 7: Derivatives
ax7 = plt.subplot(3, 3, 7)
n_deriv = 3
H_n = hermite_polynomial(n_deriv, x)
# dH_n/dx = 2n * H_{n-1}(x)
dH_n_dx = 2 * n_deriv * hermite_polynomial(n_deriv - 1, x)

ax7.plot(x, H_n, 'b-', linewidth=2.5, label=f'H_{n_deriv}(x)', alpha=0.8)
ax7.plot(x, dH_n_dx, 'r--', linewidth=2.5, label=f"H'_{n_deriv}(x) = {2*n_deriv}·H_{n_deriv-1}(x)", alpha=0.8)
ax7.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax7.grid(True, linestyle='--', alpha=0.4)
ax7.legend(loc='upper left', fontsize=9)
ax7.set_xlabel('x', fontsize=10)
ax7.set_ylabel('Value', fontsize=10)
ax7.set_title(f'Derivative Property\ndH_n/dx = 2n·H_{{n-1}}(x)', 
              fontsize=11, fontweight='bold')
ax7.set_xlim(-3, 3)

# Plot 8: Even/Odd Symmetry
ax8 = plt.subplot(3, 3, 8)
n_even = 4
n_odd = 3
H_even = hermite_polynomial(n_even, x)
H_odd = hermite_polynomial(n_odd, x)

ax8.plot(x, H_even, 'b-', linewidth=2.5, label=f'H_{n_even}(x) (even)', alpha=0.8)
ax8.plot(x, H_odd, 'r-', linewidth=2.5, label=f'H_{n_odd}(x) (odd)', alpha=0.8)
ax8.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax8.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
ax8.grid(True, linestyle='--', alpha=0.4)
ax8.legend(loc='upper left', fontsize=9)
ax8.set_xlabel('x', fontsize=10)
ax8.set_ylabel('H_n(x)', fontsize=10)
ax8.set_title('Parity Symmetry\nH_n(-x) = (-1)^n·H_n(x)', 
              fontsize=11, fontweight='bold')
ax8.set_xlim(-3, 3)

# Plot 9: Theory Summary
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

theory_text = """
HERMITE POLYNOMIALS SUMMARY

Definition & Recurrence:
  H₀(x) = 1
  H₁(x) = 2x
  H_{n+1}(x) = 2x·H_n(x) - 2n·H_{n-1}(x)

Orthogonality:
  ∫₋∞^∞ H_m(x)H_n(x)e^(-x²) dx = √π·2^n·n!·δ_{mn}

Weight Function:
  w(x) = e^(-x²)  (Gaussian, zero mean)

Properties:
• Parity: H_n(-x) = (-1)^n H_n(x)
• Derivative: dH_n/dx = 2n·H_{n-1}(x)
• Zeros: n real, distinct zeros
• Rodrigues: H_n(x)=(-1)^n e^(x²) d^n/dx^n(e^(-x²))

Applications:
• Quantum mechanics (harmonic oscillator)
• Probability theory (Gaussian expansions)
• Signal processing (edge detection)
• Physics (quantum field theory)
• Numerical analysis (Gauss-Hermite quadrature)

Key Insight:
Hermite polynomials are the natural basis
for functions with Gaussian weight, making
them essential in quantum mechanics and
probability theory.
"""

ax9.text(0.05, 0.95, theory_text, transform=ax9.transAxes, fontsize=9.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.95,
                  edgecolor='blue', linewidth=2, pad=1.5))

plt.suptitle('Hermite Polynomials - Orthogonal with Gaussian Weight', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.99))
plt.show()

print("\n✓ Visualization complete!")
print("\nNote: Hermite polynomials are fundamental in quantum mechanics")
print("      Quantum harmonic oscillator eigenfunctions: ψ_n(x) ∝ H_n(x)·e^(-x²/2)")
