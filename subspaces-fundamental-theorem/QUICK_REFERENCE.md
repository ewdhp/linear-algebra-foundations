# Subspaces & Fundamental Theorem - Quick Reference

## 🎯 Core Concepts Summary

### The Four Fundamental Subspaces

For an **m × n** matrix **A** with rank **r**:

| Subspace | Notation | Dimension | Lives in | Definition |
|----------|----------|-----------|----------|------------|
| **Column Space** | C(A) or Im(A) | r | ℝᵐ | {Ax : x ∈ ℝⁿ} |
| **Null Space** | N(A) or Ker(A) | n - r | ℝⁿ | {x : Ax = 0} |
| **Row Space** | C(Aᵀ) | r | ℝⁿ | span of rows of A |
| **Left Null Space** | N(Aᵀ) | m - r | ℝᵐ | {y : Aᵀy = 0} |

### Rank-Nullity Theorem

```
nullity(A) + rank(A) = n (number of columns)
```

**Interpretation**: Input dimension = dimension lost (kernel) + dimension preserved (image)

### Orthogonality Relations

```
Row(A) ⊥ Null(A)        (in ℝⁿ)
Col(A) ⊥ Null(Aᵀ)       (in ℝᵐ)
```

### Decompositions

```
ℝⁿ = Row(A) ⊕ Null(A)   (Domain split)
ℝᵐ = Col(A) ⊕ Null(Aᵀ)  (Codomain split)
```

## 🤖 ML/AI Applications Quick Guide

### 1. Feature Selection
- **Problem**: Which features are redundant?
- **Solution**: Features in null space don't affect output
- **Method**: Compute N(Xᵀ) where X is feature matrix

### 2. Neural Network Analysis
- **Problem**: Information loss through layers
- **Solution**: Rank of weight matrix = max info preserved
- **Method**: rank(W) tells you effective dimension

### 3. Linear Regression
- **Problem**: When does Ax = b have solutions?
- **Solution**: b must be in C(A)
- **Method**: Check if b ⊥ N(Aᵀ)

### 4. Dimensionality Reduction
- **Problem**: Reduce data dimension
- **Solution**: Project onto column space
- **Method**: Use C(A) for optimal low-rank approximation

### 5. Constraint Satisfaction
- **Problem**: Find all solutions to homogeneous system
- **Solution**: Null space gives complete solution set
- **Method**: x = x_particular + N(A)

## 💻 Python Quick Start

```python
from fundamental_subspaces import FundamentalSubspaces
import numpy as np

# Your matrix
A = np.array([[1, 2, 3],
              [2, 4, 6],
              [3, 6, 9]])

# Create analyzer
fs = FundamentalSubspaces(A)

# Get dimensions
print(f"Rank: {fs.rank}")
print(f"Nullity: {fs.nullity}")

# Compute subspaces
null_basis = fs.null_space()       # Kernel
col_basis = fs.column_space()      # Range/Image
row_basis = fs.row_space()         # Row space
left_null = fs.left_null_space()   # Left null

# Verify rank-nullity theorem
print(fs.verify_rank_nullity_theorem())  # True

# Check if b is in column space
b = np.array([2, 4, 6])
is_solvable, residual = fs.is_in_column_space(b)
print(f"Ax = b solvable: {is_solvable}")

# Project onto column space (least squares)
projection = fs.project_onto_column_space(b)
```

## 📐 Common Patterns

### Pattern 1: Checking Redundancy
```python
# Feature matrix (samples × features)
X = np.array([[...]])

# Analyze features
fs = FundamentalSubspaces(X.T)  # Transpose!
effective_features = fs.rank
redundant_dims = fs.nullity

if fs.nullity > 0:
    print("Redundant features detected!")
    null_basis = fs.null_space()
    # null_basis shows linear dependencies
```

### Pattern 2: Solvability Check
```python
# Check if Ax = b has solution
A = np.array([[...]])
b = np.array([...])

fs = FundamentalSubspaces(A)
solvable, residual = fs.is_in_column_space(b)

if solvable:
    # Solve Ax = b exactly
    x = np.linalg.solve(A, b)
else:
    # Use least squares
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
```

### Pattern 3: Finding All Solutions
```python
# For Ax = b with non-unique solutions
fs = FundamentalSubspaces(A)

# Particular solution
x_particular = np.linalg.lstsq(A, b, rcond=None)[0]

# General solution
null_basis = fs.null_space()
# x = x_particular + c₁×n₁ + c₂×n₂ + ... (where nᵢ are null basis vectors)
```

## 🔑 Key Insights

1. **Rank tells you everything**: 
   - dim(C(A)) = r
   - dim(N(A)) = n - r
   - dim(Row(A)) = r
   - dim(N(Aᵀ)) = m - r

2. **Orthogonality is fundamental**:
   - Every vector in Row(A) is perpendicular to every vector in N(A)
   - This enables projection and decomposition

3. **Solvability condition**:
   - Ax = b solvable ⟺ b ∈ C(A) ⟺ b ⊥ N(Aᵀ)

4. **Information loss**:
   - nullity(A) = dimensions lost in transformation
   - rank(A) = dimensions preserved

5. **Feature redundancy**:
   - If N(Xᵀ) ≠ {0}, features are linearly dependent
   - Null space vectors show exact redundancy relationships

## 📊 Visual Summary

```
           Input Space ℝⁿ                    Output Space ℝᵐ
    ┌─────────────────────────┐         ┌──────────────────────┐
    │                         │         │                      │
    │  Row Space              │   A     │   Column Space       │
    │  (dim = r)              │  ───>   │   (dim = r)          │
    │                         │         │                      │
    │─────────────────────────│         │──────────────────────│
    │                         │         │                      │
    │  Null Space             │   A     │   Left Null Space    │
    │  (dim = n-r)           │  ───>   │   (dim = m-r)        │
    │  maps to 0              │         │                      │
    └─────────────────────────┘         └──────────────────────┘
           ⊥                                      ⊥
```

## 🎓 Study Tips

1. **Always check rank first** - it determines all dimensions
2. **Visualize in 2D/3D** - helps build geometric intuition
3. **Verify orthogonality** - row space ⊥ null space is key
4. **Practice with examples** - work through small matrices by hand
5. **Connect to ML** - every concept has practical applications

## 📚 Next Steps

After mastering these concepts:
- **Eigenvalues**: Eigenspaces are invariant subspaces
- **SVD**: Gives optimal bases for all four subspaces
- **PCA**: Projects data onto principal subspace
- **Least Squares**: Projects onto column space

---

**Remember**: Understanding these subspaces is THE foundation for advanced linear algebra and ML!
