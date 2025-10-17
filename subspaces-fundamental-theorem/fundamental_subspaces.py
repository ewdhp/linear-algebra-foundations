#!/usr/bin/env python3
"""
Subspaces & Fundamental Theorem of Linear Algebra
==================================================

This module demonstrates the four fundamental subspaces and the rank-nullity theorem
with comprehensive implementations and visualizations for ML/AI applications.

Mathematical Theory:
-------------------

For an m×n matrix A with rank r:

1. FOUR FUNDAMENTAL SUBSPACES:
   • Column Space (Range/Image): C(A) ⊂ ℝᵐ, dim = r
     - All possible outputs: {Ax : x ∈ ℝⁿ}
     
   • Null Space (Kernel): N(A) ⊂ ℝⁿ, dim = n - r
     - All vectors mapping to zero: {x : Ax = 0}
     
   • Row Space: C(Aᵀ) ⊂ ℝⁿ, dim = r  
     - Orthogonal complement of null space
     
   • Left Null Space: N(Aᵀ) ⊂ ℝᵐ, dim = m - r
     - Orthogonal complement of column space

2. RANK-NULLITY THEOREM:
   nullity(A) + rank(A) = n (number of columns)
   
   Interpretation: Input dimension = dimension lost + dimension preserved

3. ORTHOGONALITY RELATIONS:
   • Row(A) ⊥ Null(A)
   • Col(A) ⊥ Null(Aᵀ)
   
4. DECOMPOSITIONS:
   • ℝⁿ = Row(A) ⊕ Null(A)    (Domain)
   • ℝᵐ = Col(A) ⊕ Null(Aᵀ)   (Codomain)

Author: ML/AI Education Project
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from typing import Tuple, List, Optional
import warnings


class FundamentalSubspaces:
    """
    Compute and analyze the four fundamental subspaces of a matrix.
    
    This class provides methods to:
    - Compute null space, column space, row space, and left null space
    - Verify the rank-nullity theorem
    - Check orthogonality relations
    - Visualize subspaces (for low dimensions)
    - Apply to ML/AI problems
    """
    
    def __init__(self, A: np.ndarray, tolerance: float = 1e-10):
        """
        Initialize with a matrix A.
        
        Args:
            A: Input matrix (m × n)
            tolerance: Numerical tolerance for rank computation
        """
        self.A = np.array(A, dtype=float)
        self.m, self.n = self.A.shape
        self.tolerance = tolerance
        
        # Compute rank using SVD (most numerically stable)
        self.rank = self._compute_rank()
        self.nullity = self.n - self.rank
        
    def _compute_rank(self) -> int:
        """
        Compute matrix rank using SVD.
        
        Rank = number of singular values > tolerance
        """
        s = np.linalg.svd(self.A, compute_uv=False)
        return np.sum(s > self.tolerance)
    
    def null_space(self) -> np.ndarray:
        """
        Compute the NULL SPACE (KERNEL) of A.
        
        Mathematical Definition:
        N(A) = {x ∈ ℝⁿ : Ax = 0}
        
        Method: Use SVD decomposition A = UΣVᵀ
        - Right singular vectors corresponding to zero singular values
        - These form an orthonormal basis for N(A)
        
        Returns:
            Orthonormal basis for null space (n × nullity matrix)
            Empty array if nullity = 0 (injective transformation)
        """
        # Compute full SVD
        U, s, Vt = np.linalg.svd(self.A, full_matrices=True)
        
        # Find indices of near-zero singular values
        # Note: s has length min(m, n), but Vt has shape (n, n)
        rank_from_sv = np.sum(s > self.tolerance)
        
        # The last (n - rank) rows of Vt form the null space basis
        if rank_from_sv < self.n:
            null_basis = Vt[rank_from_sv:, :].T
        else:
            null_basis = np.empty((self.n, 0))
        
        return null_basis
    
    def column_space(self) -> np.ndarray:
        """
        Compute the COLUMN SPACE (RANGE/IMAGE) of A.
        
        Mathematical Definition:
        C(A) = {Ax : x ∈ ℝⁿ} = span{a₁, a₂, ..., aₙ}
        where aᵢ are the columns of A
        
        Method: Use QR decomposition or SVD
        - Left singular vectors corresponding to non-zero singular values
        - These form an orthonormal basis for C(A)
        
        Returns:
            Orthonormal basis for column space (m × rank matrix)
        """
        # Use SVD for orthonormal basis
        U, s, Vt = np.linalg.svd(self.A, full_matrices=True)
        
        # Rank from singular values
        rank_from_sv = np.sum(s > self.tolerance)
        
        # First rank columns of U form column space basis
        col_basis = U[:, :rank_from_sv]
        
        return col_basis
    
    def row_space(self) -> np.ndarray:
        """
        Compute the ROW SPACE of A.
        
        Mathematical Definition:
        Row(A) = C(Aᵀ) = span{r₁, r₂, ..., rₘ}
        where rᵢ are the rows of A
        
        Property: Row(A) ⊥ N(A) (orthogonal complement)
        
        Method: Column space of Aᵀ
        - Right singular vectors corresponding to non-zero singular values
        
        Returns:
            Orthonormal basis for row space (n × rank matrix)
        """
        # Use SVD
        U, s, Vt = np.linalg.svd(self.A, full_matrices=True)
        
        # Rank from singular values
        rank_from_sv = np.sum(s > self.tolerance)
        
        # First rank rows of Vt (transposed) form row space basis
        row_basis = Vt[:rank_from_sv, :].T
        
        return row_basis
    
    def left_null_space(self) -> np.ndarray:
        """
        Compute the LEFT NULL SPACE of A.
        
        Mathematical Definition:
        N(Aᵀ) = {y ∈ ℝᵐ : Aᵀy = 0}
        
        Property: N(Aᵀ) ⊥ C(A) (orthogonal complement)
        
        Returns:
            Orthonormal basis for left null space (m × (m-rank) matrix)
        """
        # Use SVD
        U, s, Vt = np.linalg.svd(self.A, full_matrices=True)
        
        # Find indices of zero singular values (considering we might have m > n or m < n)
        if len(s) < self.m:
            # More rows than columns, last (m - n) left singular vectors are in left null space
            left_null_start = len(s)
            left_null_basis = U[:, left_null_start:]
            
            # Also include any zero singular values within the first n
            null_mask = s <= self.tolerance
            if np.any(null_mask):
                additional_basis = U[:, null_mask]
                left_null_basis = np.hstack([additional_basis, left_null_basis])
        else:
            null_mask = s <= self.tolerance
            left_null_basis = U[:, null_mask]
        
        return left_null_basis
    
    def verify_rank_nullity_theorem(self) -> bool:
        """
        Verify the RANK-NULLITY THEOREM:
        
        nullity(A) + rank(A) = n (number of columns)
        
        This is the fundamental theorem connecting dimensions of subspaces.
        
        Returns:
            True if theorem holds (should always be True)
        """
        return self.nullity + self.rank == self.n
    
    def verify_orthogonality(self) -> dict:
        """
        Verify ORTHOGONALITY RELATIONS between fundamental subspaces:
        
        1. Row(A) ⊥ N(A)     - Row space orthogonal to null space
        2. C(A) ⊥ N(Aᵀ)      - Column space orthogonal to left null space
        
        Returns:
            Dictionary with orthogonality check results
        """
        results = {}
        
        # Get bases
        null_basis = self.null_space()
        col_basis = self.column_space()
        row_basis = self.row_space()
        left_null_basis = self.left_null_space()
        
        # Check Row(A) ⊥ N(A)
        if row_basis.size > 0 and null_basis.size > 0:
            inner_product_1 = row_basis.T @ null_basis
            max_inner_1 = np.max(np.abs(inner_product_1))
            results['row_null_orthogonal'] = max_inner_1 < self.tolerance
            results['row_null_max_inner'] = max_inner_1
        else:
            results['row_null_orthogonal'] = True
            results['row_null_max_inner'] = 0.0
        
        # Check C(A) ⊥ N(Aᵀ)
        if col_basis.size > 0 and left_null_basis.size > 0:
            inner_product_2 = col_basis.T @ left_null_basis
            max_inner_2 = np.max(np.abs(inner_product_2))
            results['col_leftnull_orthogonal'] = max_inner_2 < self.tolerance
            results['col_leftnull_max_inner'] = max_inner_2
        else:
            results['col_leftnull_orthogonal'] = True
            results['col_leftnull_max_inner'] = 0.0
        
        return results
    
    def dimension_summary(self) -> dict:
        """
        Get summary of dimensions of all subspaces.
        
        Returns:
            Dictionary with dimension information
        """
        return {
            'matrix_shape': (self.m, self.n),
            'rank': self.rank,
            'nullity': self.nullity,
            'dim_column_space': self.rank,
            'dim_null_space': self.nullity,
            'dim_row_space': self.rank,
            'dim_left_null_space': self.m - self.rank,
            'rank_nullity_verified': self.verify_rank_nullity_theorem()
        }
    
    def is_in_column_space(self, b: np.ndarray) -> Tuple[bool, float]:
        """
        Check if vector b is in the column space of A.
        
        Equivalently: Check if Ax = b has a solution.
        
        Args:
            b: Vector to test (m-dimensional)
            
        Returns:
            (is_in_col_space, residual_norm)
        """
        b = np.array(b).flatten()
        
        # Project b onto column space
        col_basis = self.column_space()
        if col_basis.size > 0:
            projection = col_basis @ (col_basis.T @ b)
            residual = b - projection
            residual_norm = np.linalg.norm(residual)
        else:
            residual_norm = np.linalg.norm(b)
        
        return residual_norm < self.tolerance, residual_norm
    
    def project_onto_column_space(self, b: np.ndarray) -> np.ndarray:
        """
        Project vector b onto the column space of A.
        
        This gives the best approximation of b in C(A).
        Used in least squares: minimize ||Ax - b||
        
        Args:
            b: Vector to project (m-dimensional)
            
        Returns:
            Projection of b onto C(A)
        """
        b = np.array(b).flatten()
        col_basis = self.column_space()
        
        if col_basis.size > 0:
            projection = col_basis @ (col_basis.T @ b)
        else:
            projection = np.zeros_like(b)
        
        return projection


def demonstrate_fundamental_subspaces():
    """
    Demonstrate the four fundamental subspaces with examples
    """
    print("🔬 FUNDAMENTAL SUBSPACES DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Example 1: Simple 3×3 matrix with rank 2
    print("📐 EXAMPLE 1: Rank-Deficient 3×3 Matrix")
    print("-" * 40)
    
    A1 = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9]
    ])
    
    print("Matrix A:")
    print(A1)
    print()
    
    fs1 = FundamentalSubspaces(A1)
    
    # Dimension summary
    dims = fs1.dimension_summary()
    print("📊 DIMENSION ANALYSIS:")
    print(f"  Matrix shape: {dims['matrix_shape']}")
    print(f"  Rank: {dims['rank']}")
    print(f"  Nullity: {dims['nullity']}")
    print(f"  Rank-Nullity Theorem: {dims['nullity']} + {dims['rank']} = {fs1.n} ✓" if dims['rank_nullity_verified'] else "  ✗ Rank-Nullity check failed!")
    print()
    
    # Four fundamental subspaces
    print("🌐 FOUR FUNDAMENTAL SUBSPACES:")
    print()
    
    null_basis = fs1.null_space()
    print("1. NULL SPACE N(A) - dim =", dims['dim_null_space'])
    if null_basis.size > 0:
        print("   Basis vectors (columns):")
        print(null_basis)
        print("   Verification: A × (null space basis) =")
        print(A1 @ null_basis)
    else:
        print("   Trivial null space: {0}")
    print()
    
    col_basis = fs1.column_space()
    print("2. COLUMN SPACE C(A) - dim =", dims['dim_column_space'])
    print("   Basis vectors (columns):")
    print(col_basis)
    print()
    
    row_basis = fs1.row_space()
    print("3. ROW SPACE C(Aᵀ) - dim =", dims['dim_row_space'])
    print("   Basis vectors (columns):")
    print(row_basis)
    print()
    
    left_null_basis = fs1.left_null_space()
    print("4. LEFT NULL SPACE N(Aᵀ) - dim =", dims['dim_left_null_space'])
    if left_null_basis.size > 0:
        print("   Basis vectors (columns):")
        print(left_null_basis)
    else:
        print("   Trivial left null space: {0}")
    print()
    
    # Verify orthogonality
    ortho_check = fs1.verify_orthogonality()
    print("⊥ ORTHOGONALITY VERIFICATION:")
    print(f"  Row(A) ⊥ N(A): {ortho_check['row_null_orthogonal']} (max inner product: {ortho_check['row_null_max_inner']:.2e})")
    print(f"  C(A) ⊥ N(Aᵀ): {ortho_check['col_leftnull_orthogonal']} (max inner product: {ortho_check['col_leftnull_max_inner']:.2e})")
    print()
    print()
    
    # Example 2: ML Application - Feature Redundancy
    print("🤖 EXAMPLE 2: ML Application - Feature Redundancy Detection")
    print("-" * 60)
    
    # Feature matrix where one feature is redundant
    # Features: [f1, f2, f3] where f3 = f1 + 2*f2
    X = np.array([
        [1, 2, 5],    # sample 1: f3 = 1 + 2*2 = 5
        [2, 3, 8],    # sample 2: f3 = 2 + 2*3 = 8
        [3, 1, 5],    # sample 3: f3 = 3 + 2*1 = 5
        [4, 4, 12]    # sample 4: f3 = 4 + 2*4 = 12
    ])
    
    print("Feature Matrix X (samples × features):")
    print(X)
    print("Note: Feature 3 = Feature 1 + 2×Feature 2")
    print()
    
    fs2 = FundamentalSubspaces(X.T)  # Transpose to analyze features
    dims2 = fs2.dimension_summary()
    
    print(f"Feature space analysis (features as rows):")
    print(f"  Number of features: {fs2.m}")
    print(f"  Effective rank: {dims2['rank']}")
    print(f"  Redundant dimensions: {dims2['nullity']}")
    print()
    
    # Find redundant feature combination
    null_basis2 = fs2.null_space()
    if null_basis2.size > 0:
        print("Redundancy relationship (null space basis):")
        print(null_basis2)
        print()
        print("Interpretation: This shows the linear combination of features that equals zero.")
        coeffs = null_basis2[:, 0]
        print(f"Approximately: {coeffs[0]:.3f}×f1 + {coeffs[1]:.3f}×f2 + {coeffs[2]:.3f}×f3 = 0")
        print(f"Which means: f3 ≈ {-coeffs[0]/coeffs[2]:.3f}×f1 + {-coeffs[1]/coeffs[2]:.3f}×f2")
    print()
    print()
    
    # Example 3: Solvability of Linear Systems
    print("🎯 EXAMPLE 3: Solvability of Linear Systems")
    print("-" * 60)
    
    A3 = np.array([
        [1, 2],
        [2, 4],
        [3, 6]
    ])
    
    print("Matrix A:")
    print(A3)
    print()
    
    fs3 = FundamentalSubspaces(A3)
    
    # Test different b vectors
    b_solvable = np.array([2, 4, 6])      # In column space (b = 2×col1)
    b_not_solvable = np.array([1, 2, 4])  # Not in column space
    
    is_solvable1, residual1 = fs3.is_in_column_space(b_solvable)
    is_solvable2, residual2 = fs3.is_in_column_space(b_not_solvable)
    
    print(f"Test 1: b = {b_solvable}")
    print(f"  In column space? {is_solvable1} (residual: {residual1:.2e})")
    print(f"  Ax = b is {'SOLVABLE' if is_solvable1 else 'NOT SOLVABLE'}")
    print()
    
    print(f"Test 2: b = {b_not_solvable}")
    print(f"  In column space? {is_solvable2} (residual: {residual2:.2e})")
    print(f"  Ax = b is {'SOLVABLE' if is_solvable2 else 'NOT SOLVABLE (use least squares)'}")
    
    if not is_solvable2:
        projection = fs3.project_onto_column_space(b_not_solvable)
        print(f"  Best approximation (projection): {projection}")
    print()


if __name__ == "__main__":
    print("🌟 Fundamental Subspaces & Rank-Nullity Theorem Explorer 🌟")
    print("This script demonstrates the four fundamental subspaces of linear algebra")
    print("and their applications to Machine Learning and AI problems.")
    print()
    
    try:
        demonstrate_fundamental_subspaces()
        
        print("\n" + "=" * 60)
        print("🎓 KEY TAKEAWAYS:")
        print("=" * 60)
        print("• Every matrix has FOUR fundamental subspaces")
        print("• Rank-Nullity Theorem: nullity + rank = n (columns)")
        print("• Row space ⊥ Null space")
        print("• Column space ⊥ Left null space")
        print("• Null space reveals redundant features in ML")
        print("• Column space determines solvability of Ax = b")
        print("• These concepts are essential for understanding:")
        print("  - Feature selection and dimensionality reduction")
        print("  - Information loss in neural networks")
        print("  - Least squares and regression")
        print("  - PCA and SVD")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
    
    print(f"\n✨ Thanks for exploring fundamental subspaces! ✨")
