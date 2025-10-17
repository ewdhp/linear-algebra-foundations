# Normed Spaces

## 📖 What You'll Learn

Normed spaces provide a rigorous mathematical framework for measuring the "size" or "length" of vectors in any vector space. Understanding norms is essential for optimization, regularization, and measuring distances in machine learning applications.

## 📂 Subtopics

This section contains the following specialized normed spaces:

1. **[Hilbert Spaces](./hilbert-spaces/)** - Complete inner product spaces with perfect geometry
   - **[Kernel Methods & RKHS](./hilbert-spaces/kernel-methods-rkhs/)** - SVMs, Gaussian Processes, and Neural Tangent Kernels
2. **[Banach Spaces](./banach-spaces/)** - Complete normed spaces for optimization and neural operators

### Core Concepts

1. **Vector Norms**
   - Definition: A function ‖·‖ : V → ℝ satisfying three axioms
   - Non-negativity: ‖x‖ ≥ 0, with ‖x‖ = 0 iff x = 0
   - Homogeneity: ‖αx‖ = |α|‖x‖ for any scalar α
   - Triangle inequality: ‖x + y‖ ≤ ‖x‖ + ‖y‖

2. **Common Norms (Lp Norms)**
   - **L1 Norm (Manhattan)**: ‖x‖₁ = Σ|xᵢ|
   - **L2 Norm (Euclidean)**: ‖x‖₂ = √(Σxᵢ²)
   - **L∞ Norm (Maximum)**: ‖x‖∞ = max|xᵢ|
   - **General Lp Norm**: ‖x‖ₚ = (Σ|xᵢ|ᵖ)^(1/p)

3. **Norm Equivalence**
   - All norms in finite-dimensional spaces are equivalent
   - Relationships between different norms
   - Choosing norms for specific applications

4. **Matrix Norms**
   - Induced norms: ‖A‖ = sup{‖Ax‖ / ‖x‖ : x ≠ 0}
   - Frobenius norm: ‖A‖_F = √(ΣΣaᵢⱼ²)
   - Spectral norm: ‖A‖₂ = largest singular value
   - Nuclear norm: sum of singular values

5. **Unit Balls**
   - Geometric interpretation of different norms
   - Unit ball in L1: diamond/cross-polytope
   - Unit ball in L2: sphere
   - Unit ball in L∞: hypercube

6. **Normed Vector Spaces**
   - Structure: (V, ‖·‖)
   - Metric induced by norm: d(x,y) = ‖x - y‖
   - Convergence and continuity in normed spaces
   - Cauchy sequences and completeness

7. **Dual Norms**
   - Definition: ‖z‖* = sup{⟨x,z⟩ : ‖x‖ ≤ 1}
   - Relationship: (Lp)* = Lq where 1/p + 1/q = 1
   - Applications in optimization duality

## 🤖 Machine Learning Applications

### Regularization
- **L1 Regularization (Lasso)**:
  - Penalty term: λ‖w‖₁ = λΣ|wᵢ|
  - Promotes sparsity (many weights become exactly zero)
  - Feature selection built into optimization
  - Applications: compressed sensing, sparse models
  
- **L2 Regularization (Ridge)**:
  - Penalty term: λ‖w‖₂² = λΣwᵢ²
  - Shrinks all weights proportionally
  - Prevents overfitting by limiting model complexity
  - Applications: neural networks, linear regression
  
- **Elastic Net**: Combines L1 and L2: α‖w‖₁ + β‖w‖₂²
  
- **Group Lasso**: ΣΣ‖w_g‖₂ for structured sparsity

### Loss Functions
- **Mean Absolute Error (L1 Loss)**: ‖y - ŷ‖₁
  - Robust to outliers
  - Used in robust regression
  
- **Mean Squared Error (L2 Loss)**: ‖y - ŷ‖₂²
  - Penalizes large errors more heavily
  - Smooth gradients for optimization
  - Standard in regression problems
  
- **Huber Loss**: Combines L1 and L2 properties
  
- **Hinge Loss**: max(0, 1 - yf(x)) for SVM

### Distance Metrics & Similarity
- **Cosine Similarity**: Related to L2 norm
- **Manhattan Distance**: L1 distance for grid-like spaces
- **Euclidean Distance**: L2 distance standard metric
- **Chebyshev Distance**: L∞ distance
- **k-NN Classification**: Choice of norm affects neighborhoods

### Neural Network Training
- **Weight Decay**: L2 penalty on network weights
- **Dropout**: Implicitly enforces norm constraints
- **Batch Normalization**: Normalizing activations
- **Gradient Clipping**: Constraining gradient norms
- **Lipschitz Constraints**: Bounding operator norms for stability

### Optimization
- **Gradient Descent**: Direction opposite to gradient norm
- **Proximal Methods**: Utilizing specific norm structures
- **Projected Gradient**: Constraining to norm balls
- **ADMM**: Exploiting separable norm penalties

### Model Compression
- **Pruning**: Removing small-norm weights
- **Quantization**: Reducing precision while controlling norm
- **Low-rank Approximation**: Nuclear norm minimization
- **Knowledge Distillation**: Matching output distributions

## 📊 Topics Covered

### Theoretical Foundations
- Axiomatic definition of norms
- Norm equivalence theorems
- Completion of normed spaces
- Hölder and Minkowski inequalities
- Banach space theory basics

### Computational Methods
- Computing various Lp norms
- Efficient matrix norm computation
- Proximal operators for common norms
- Subgradients for non-smooth norms
- Projection onto norm balls

### Advanced Concepts
- Dual norms and conjugate functions
- Quotient norms
- Operator norms
- Nuclear and trace norms
- Schatten norms

### Geometric Intuition
- Visualizing unit balls in different norms
- Sparsity patterns from L1 geometry
- Smoothness from L2 geometry
- Contour plots and level sets

## 💻 What's Included

- **Comprehensive Theory**: Mathematical foundations and properties
- **Visual Intuition**: 2D/3D visualizations of different norm balls
- **Python Implementations**: NumPy and SciPy code for:
  - Computing various Lp norms
  - Implementing regularized regression
  - Visualizing norm geometries
  - Proximal operators for optimization
  - Matrix norms and spectral properties
- **Interactive Examples**: Comparing different norms and regularization
- **ML Applications**: Real-world examples with datasets
- **Exercises**: From basic norm computations to advanced regularization
- **Code Patterns**: Common ML implementations using norms

## 🎯 Learning Outcomes

By the end of this section, you will be able to:

✅ Define and compute various vector and matrix norms  
✅ Understand geometric differences between L1, L2, and L∞ norms  
✅ Apply L1 regularization for sparse models (Lasso)  
✅ Apply L2 regularization to prevent overfitting (Ridge)  
✅ Choose appropriate loss functions for ML tasks  
✅ Implement regularized optimization algorithms  
✅ Understand why L1 promotes sparsity geometrically  
✅ Use norms for distance metrics and similarity measures  
✅ Apply gradient clipping and weight decay in neural networks  
✅ Analyze model behavior through norm-based constraints  

## 📐 Mathematical Summary

### Key Definitions

**Lp Norms:**
```
‖x‖₁ = Σ|xᵢ|                     (L1/Manhattan)
‖x‖₂ = √(Σxᵢ²)                   (L2/Euclidean)
‖x‖ₚ = (Σ|xᵢ|ᵖ)^(1/p)            (General Lp)
‖x‖∞ = max|xᵢ|                   (L∞/Maximum)
```

**Dual Norm Relationship:**
```
If 1/p + 1/q = 1, then (Lp)* = Lq

Examples:
(L1)* = L∞
(L2)* = L2 (self-dual)
```

**Matrix Norms:**
```
‖A‖₂ = σ_max(A)                  (Spectral norm)
‖A‖_F = √(ΣΣaᵢⱼ²) = √(tr(AᵀA))   (Frobenius)
‖A‖* = Σσᵢ                       (Nuclear norm)
```

**Regularized Loss:**
```
L(w) = Error(w) + λR(w)

Ridge:  R(w) = ‖w‖₂²
Lasso:  R(w) = ‖w‖₁
Elastic Net: R(w) = α‖w‖₁ + β‖w‖₂²
```

## 🔬 Example Problems

### Problem 1: Computing Different Norms
Given vector x = [3, -4, 0, 2]:
- Compute ‖x‖₁, ‖x‖₂, ‖x‖∞
- Sketch unit balls for each norm in 2D
- Explain geometric differences

### Problem 2: Lasso vs Ridge Regression
Given dataset with correlated features:
- Implement L1-regularized regression (Lasso)
- Implement L2-regularized regression (Ridge)
- Compare sparsity patterns
- Analyze feature selection behavior

### Problem 3: Optimal Regularization Parameter
For a regression problem:
- Use cross-validation to select λ
- Plot validation error vs λ
- Compare L1 and L2 regularization paths
- Visualize weight trajectories

### Problem 4: Gradient Clipping
In neural network training:
- Implement gradient norm clipping
- Compare training with/without clipping
- Analyze effect on convergence
- Prevent exploding gradients

## 📚 Prerequisites

- Understanding of vectors and vector spaces
- Basic calculus and optimization
- Linear algebra fundamentals
- Familiarity with regression and loss functions

## 🚀 Next Steps

After mastering normed spaces, explore the specialized subtopics:

1. **[Hilbert Spaces](./hilbert-spaces/)** - Adding inner products to normed spaces
   - **[Kernel Methods & RKHS](./hilbert-spaces/kernel-methods-rkhs/)** - Applications to SVMs and GPs
2. **[Banach Spaces](./banach-spaces/)** - Complete normed spaces and functional analysis
3. **Optimization Theory** - Convex analysis and proximal methods
4. **Advanced Regularization** - Nuclear norms, structured sparsity
5. **Compressed Sensing** - L1 minimization for sparse recovery

## 🎓 Why This Matters for ML/AI

Understanding norms is crucial because:

- **Regularization prevents overfitting** - Essential for generalization
- **Different norms encode different prior beliefs** - L1 for sparsity, L2 for smoothness
- **Loss functions are norms** - Choosing the right metric for your problem
- **Optimization algorithms depend on geometry** - Norm structure affects convergence
- **Model compression uses norm constraints** - Pruning and quantization
- **Stability analysis requires operator norms** - Understanding model sensitivity
- **Distance metrics define neighborhoods** - Critical for k-NN, clustering

Norms are the foundation for understanding how to measure and constrain models!

## 📖 Recommended Reading

- *Functional Analysis* by Walter Rudin (Chapter 1)
- *Convex Optimization* by Boyd & Vandenberghe (Chapter 3)
- *Deep Learning* by Goodfellow et al. (Chapter 7.1)
- *The Elements of Statistical Learning* by Hastie et al. (Chapter 3.4)
- *High-Dimensional Statistics* by Wainwright (Chapter 2)

## 🔗 Related Topics

- [Vectors](../vectors/) - Basic operations and dot products
- [Matrices](../matrices/) - Matrix norms and transformations
- [Hilbert Spaces](./hilbert-spaces/) - Inner product spaces with norms
- [Banach Spaces](./banach-spaces/) - Complete normed spaces
- [Linear Operators](../linear-operators/) - Operators on normed spaces
