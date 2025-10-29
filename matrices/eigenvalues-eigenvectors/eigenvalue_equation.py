"""
Eigenvalue Equation: Av = λv

This script demonstrates the fundamental concepts of eigenvalues and eigenvectors:
1. Understanding Av = λv
2. Geometric interpretation: directions preserved under transformation
3. Eigenvalues as scaling factors
4. Finding eigenvalues and eigenvectors
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# Set matplotlib backend for display
import matplotlib
import os
import warnings

# Suppress Qt/Wayland warnings on GNOME
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'
warnings.filterwarnings('ignore', category=UserWarning)

matplotlib.use('Qt5Agg')  # Use Qt5Agg backend


def compute_eigenvalues_eigenvectors(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of matrix A.
    
    Args:
        A: Square matrix
        
    Returns:
        eigenvalues: Array of eigenvalues
        eigenvectors: Matrix where each column is an eigenvector
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors


def verify_eigenvalue_equation(A: np.ndarray, eigenvalue: float, eigenvector: np.ndarray) -> bool:
    """
    Verify that Av = λv holds for the given eigenvalue and eigenvector.
    
    Args:
        A: Matrix
        eigenvalue: Eigenvalue λ
        eigenvector: Eigenvector v
        
    Returns:
        True if equation holds (within numerical precision)
    """
    Av = A @ eigenvector
    lambda_v = eigenvalue * eigenvector
    
    print(f"\nVerifying Av = λv:")
    print(f"A·v = {Av}")
    print(f"λ·v = {lambda_v}")
    print(f"Difference: {np.linalg.norm(Av - lambda_v):.10f}")
    
    return np.allclose(Av, lambda_v)


def visualize_transformation_2d(A: np.ndarray, eigenvalues: np.ndarray, 
                                eigenvectors: np.ndarray, axes=None, title_prefix=""):
    """
    Visualize how matrix A transforms space and preserves eigenvector directions.
    
    Args:
        A: 2x2 transformation matrix
        eigenvalues: Eigenvalues of A
        eigenvectors: Eigenvectors of A
        axes: Optional axes to plot on (for subplot)
        title_prefix: Optional prefix for title
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        standalone = True
    else:
        standalone = False
    
    # Create a grid of vectors to show transformation
    x = np.linspace(-2, 2, 8)
    y = np.linspace(-2, 2, 8)
    X, Y = np.meshgrid(x, y)
    
    # Original vectors
    U = X.flatten()
    V = Y.flatten()
    
    # Transformed vectors
    transformed = A @ np.array([U, V])
    U_trans = transformed[0]
    V_trans = transformed[1]
    
    # Plot original space
    ax1 = axes[0]
    ax1.quiver(U, V, U, V, alpha=0.3, color='blue', width=0.003)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.set_title(f'{title_prefix}Original Space', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Plot eigenvectors in original space
    for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        if np.isreal(eigenval):
            eigenvec_real = np.real(eigenvec)
            color = ['red', 'green'][i % 2]
            ax1.arrow(0, 0, eigenvec_real[0]*2, eigenvec_real[1]*2,
                     head_width=0.15, head_length=0.15, fc=color, ec=color,
                     linewidth=2.5, label=f'v{i+1} (λ={eigenval:.2f})')
    
    ax1.legend(loc='upper right')
    
    # Plot transformed space
    ax2 = axes[1]
    ax2.quiver(np.zeros_like(U), np.zeros_like(V), U_trans, V_trans, 
              alpha=0.3, color='blue', width=0.003)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.set_title(f'{title_prefix}Transformed Space (A·v)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # Plot transformed eigenvectors (scaled by eigenvalue)
    for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        if np.isreal(eigenval):
            eigenvec_real = np.real(eigenvec)
            transformed_eigenvec = A @ eigenvec_real
            color = ['red', 'green'][i % 2]
            ax2.arrow(0, 0, transformed_eigenvec[0]*2, transformed_eigenvec[1]*2,
                     head_width=0.15, head_length=0.15, fc=color, ec=color,
                     linewidth=2.5, label=f'A·v{i+1} = λ{i+1}·v{i+1}')
    
    ax2.legend(loc='upper right')
    
    if standalone:
        plt.suptitle(f'Eigenvector Directions Preserved Under Transformation\nMatrix A = {A}',
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()


def demonstrate_scaling_factor(A: np.ndarray, eigenvalue: float, eigenvector: np.ndarray, ax=None, title_prefix=""):
    """
    Demonstrate how eigenvalue acts as a scaling factor.
    
    Args:
        A: Matrix
        eigenvalue: Eigenvalue λ
        eigenvector: Eigenvector v
        ax: Optional axis to plot on (for subplot)
        title_prefix: Optional prefix for title
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        standalone = True
    else:
        standalone = False
    
    # Normalize eigenvector for visualization
    v_norm = eigenvector / np.linalg.norm(eigenvector)
    
    # Original eigenvector
    ax.arrow(0, 0, v_norm[0], v_norm[1],
            head_width=0.1, head_length=0.1, fc='blue', ec='blue',
            linewidth=3, label=f'v (eigenvector)', zorder=5)
    
    # Transformed eigenvector (scaled by eigenvalue)
    Av = A @ v_norm
    ax.arrow(0, 0, Av[0], Av[1],
            head_width=0.1, head_length=0.1, fc='red', ec='red',
            linewidth=3, label=f'A·v = λ·v (λ={eigenvalue:.2f})', zorder=5)
    
    # Show the scaling visually with dashed line
    if abs(eigenvalue) > 1:
        ax.plot([v_norm[0], Av[0]], [v_norm[1], Av[1]], 
               'k--', alpha=0.5, linewidth=2, label=f'Scaled by factor {eigenvalue:.2f}')
    
    # Regular vector for comparison (not an eigenvector)
    regular_vec = np.array([0.8, 0.6])
    A_regular = A @ regular_vec
    
    ax.arrow(0, 0, regular_vec[0], regular_vec[1],
            head_width=0.08, head_length=0.08, fc='green', ec='green',
            linewidth=2, label='u (regular vector)', alpha=0.6, zorder=4)
    
    ax.arrow(0, 0, A_regular[0], A_regular[1],
            head_width=0.08, head_length=0.08, fc='orange', ec='orange',
            linewidth=2, label='A·u (direction changes)', alpha=0.6, zorder=4)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend(loc='best', fontsize=9)
    ax.set_title(f'{title_prefix}Eigenvalue as Scaling Factor: Av = λv\nDirection preserved, regular vector changes',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    if standalone:
        plt.tight_layout()
        plt.show()


def find_eigenvalues_characteristic_polynomial(A: np.ndarray):
    """
    Demonstrate finding eigenvalues using the characteristic polynomial det(A - λI) = 0.
    
    Args:
        A: Square matrix
    """
    print("\n" + "="*70)
    print("FINDING EIGENVALUES: Characteristic Polynomial Method")
    print("="*70)
    
    n = A.shape[0]
    print(f"\nMatrix A:")
    print(A)
    
    print(f"\nTo find eigenvalues, solve: det(A - λI) = 0")
    print(f"where I is the {n}×{n} identity matrix\n")
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    
    print(f"Eigenvalues (λ):")
    for i, eigenval in enumerate(eigenvalues):
        if np.isreal(eigenval):
            print(f"  λ{i+1} = {np.real(eigenval):.4f}")
        else:
            print(f"  λ{i+1} = {eigenval:.4f}")
    
    # Verify by computing det(A - λI) for each eigenvalue
    print(f"\nVerification: det(A - λI) should ≈ 0 for each eigenvalue")
    for i, eigenval in enumerate(eigenvalues):
        det_val = np.linalg.det(A - eigenval * np.eye(n))
        print(f"  λ{i+1}: det(A - {eigenval:.4f}·I) = {abs(det_val):.10f}")


def demonstrate_example_1():
    """
    Example 1: Simple 2×2 matrix with clear geometric interpretation.
    Returns visualization data for combined display.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Scaling Transformation")
    print("="*70)
    
    # A simple diagonal matrix (scales x by 2, y by 3)
    A = np.array([[2.0, 0.0],
                  [0.0, 3.0]])
    
    print(f"\nMatrix A (scaling transformation):")
    print(A)
    
    eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(A)
    
    print(f"\nEigenvalues:")
    for i, eigenval in enumerate(eigenvalues):
        print(f"  λ{i+1} = {eigenval:.4f}")
    
    print(f"\nEigenvectors:")
    for i, eigenvec in enumerate(eigenvectors.T):
        print(f"  v{i+1} = {eigenvec}")
    
    # Verify eigenvalue equation for each pair
    for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        print(f"\n--- Eigenpair {i+1} ---")
        verify_eigenvalue_equation(A, eigenval, eigenvec)
    
    return A, eigenvalues, eigenvectors


def demonstrate_example_2():
    """
    Example 2: Rotation + Scaling matrix.
    Returns visualization data for combined display.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Rotation + Scaling")
    print("="*70)
    
    # Matrix that rotates and scales
    theta = np.pi / 6  # 30 degrees
    scale = 1.5
    A = scale * np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
    
    print(f"\nMatrix A (rotation by 30° + scaling by {scale}):")
    print(A)
    
    eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(A)
    
    print(f"\nEigenvalues:")
    for i, eigenval in enumerate(eigenvalues):
        print(f"  λ{i+1} = {eigenval:.4f}")
    
    print(f"\nEigenvectors:")
    for i, eigenvec in enumerate(eigenvectors.T):
        print(f"  v{i+1} = {eigenvec}")
    
    if np.allclose(eigenvalues.imag, 0):
        print("\nNote: Real eigenvalues exist even though this includes rotation!")
        return A, eigenvalues, eigenvectors
    else:
        print("\nNote: Complex eigenvalues (no real eigenvectors for pure rotation)")
        return None, None, None


def demonstrate_example_3():
    """
    Example 3: Shear transformation.
    Returns visualization data for combined display.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Shear Transformation")
    print("="*70)
    
    # Shear matrix
    A = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    
    print(f"\nMatrix A (shear transformation):")
    print(A)
    
    eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(A)
    
    print(f"\nEigenvalues:")
    for i, eigenval in enumerate(eigenvalues):
        print(f"  λ{i+1} = {eigenval:.4f}")
    
    print(f"\nEigenvectors:")
    for i, eigenvec in enumerate(eigenvectors.T):
        print(f"  v{i+1} = {eigenvec}")
    
    # Verify
    for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        print(f"\n--- Eigenpair {i+1} ---")
        verify_eigenvalue_equation(A, eigenval, eigenvec)
    
    return A, eigenvalues, eigenvectors


def demonstrate_example_4():
    """
    Example 4: Symmetric matrix (guaranteed real eigenvalues).
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Symmetric Matrix")
    print("="*70)
    
    # Symmetric matrix (common in ML/AI - covariance matrices)
    A = np.array([[3.0, 1.0],
                  [1.0, 3.0]])
    
    print(f"\nMatrix A (symmetric):")
    print(A)
    print(f"Note: Symmetric matrices always have real eigenvalues!")
    
    eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(A)
    
    print(f"\nEigenvalues:")
    for i, eigenval in enumerate(eigenvalues):
        print(f"  λ{i+1} = {eigenval:.4f}")
    
    print(f"\nEigenvectors (orthogonal for symmetric matrices):")
    for i, eigenvec in enumerate(eigenvectors.T):
        print(f"  v{i+1} = {eigenvec}")
    
    # Check orthogonality
    dot_product = np.dot(eigenvectors[:, 0], eigenvectors[:, 1])
    print(f"\nOrthogonality check: v1 · v2 = {dot_product:.10f} (should be ≈ 0)")
    
    # Verify
    for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        print(f"\n--- Eigenpair {i+1} ---")
        verify_eigenvalue_equation(A, eigenval, eigenvec)
    
    # Find eigenvalues using characteristic polynomial
    find_eigenvalues_characteristic_polynomial(A)
    
    # Visualize
    visualize_transformation_2d(A, eigenvalues, eigenvectors)


def main():
    """
    Main function to run all demonstrations.
    """
    print("\n" + "="*70)
    print("EIGENVALUE EQUATION: Av = λv")
    print("Understanding Eigenvalues and Eigenvectors")
    print("="*70)
    
    print("\nKey Concepts:")
    print("1. Av = λv: Matrix A transforms eigenvector v by scaling it by λ")
    print("2. Direction Preserved: Eigenvectors maintain their direction")
    print("3. Scaling Factor: Eigenvalue λ determines the scaling amount")
    print("4. Finding Eigenvalues: Solve det(A - λI) = 0")
    
    # Run examples
    demonstrate_example_1()
    
    print("\n" + "="*70)
    input("Press Enter to continue to Example 2...")
    plt.close('all')  # Close previous figures
    
    demonstrate_example_2()
    
    print("\n" + "="*70)
    input("Press Enter to continue to Example 3...")
    plt.close('all')
    
    demonstrate_example_3()
    
    print("\n" + "="*70)
    input("Press Enter to continue to Example 4...")
    plt.close('all')
    
    demonstrate_example_4()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nKey Takeaways:")
    print("• Eigenvectors are special directions that are only scaled (not rotated)")
    print("• Eigenvalues tell us the scaling factor")
    print("• Regular vectors change direction under transformation")
    print("• Symmetric matrices have real eigenvalues and orthogonal eigenvectors")
    print("• This forms the foundation for PCA, spectral methods, and stability analysis")
    print("="*70)


if __name__ == "__main__":
    main()
