# Hilbert Spaces

## 📖 What You'll Learn

Hilbert spaces are complete inner product spaces where geometry works perfectly—orthogonality, projections, and angles are all well-defined. They provide the mathematical foundation for many modern machine learning techniques, especially kernel methods and infinite-dimensional learning theory.

## 📂 Subtopics

This section contains the following specialized application:

1. **[Kernel Methods & RKHS](./kernel-methods-rkhs/)** - Reproducing Kernel Hilbert Spaces for SVMs, Gaussian Processes, and Neural Tangent Kernels

### Core Concepts

1. **Inner Product Spaces**
   - Definition: Vector space with inner product ⟨·,·⟩ satisfying:
     - Linearity: ⟨αx + βy, z⟩ = α⟨x,z⟩ + β⟨y,z⟩
     - Symmetry: ⟨x,y⟩ = ⟨y,x⟩ (or conjugate symmetry for complex spaces)
     - Positive definiteness: ⟨x,x⟩ ≥ 0, with equality iff x = 0
   - Induced norm: ‖x‖ = √⟨x,x⟩
   - Examples: ℝⁿ with dot product, L²[a,b] with ∫f(x)g(x)dx

2. **Hilbert Space Definition**
   - Complete inner product space
   - Completeness: Every Cauchy sequence converges
   - Combines algebraic structure (vector space) with geometric structure (inner product) and topological structure (completeness)
   - Finite-dimensional: Every inner product space is a Hilbert space
   - Infinite-dimensional: L², ℓ², function spaces

3. **Orthogonality & Orthonormal Bases**
   - Orthogonal vectors: ⟨x,y⟩ = 0
   - Orthonormal basis: {eᵢ} where ⟨eᵢ,eⱼ⟩ = δᵢⱼ
   - Gram-Schmidt orthogonalization
   - Expansion: x = Σ⟨x,eᵢ⟩eᵢ (generalized Fourier series)
   - Parseval's identity: ‖x‖² = Σ|⟨x,eᵢ⟩|²

4. **Projection Theorem**
   - Every x ∈ H has unique decomposition: x = y + z
   - y ∈ M (closed subspace), z ⊥ M
   - y = P_M(x) is the orthogonal projection
   - Minimizes distance: ‖x - y‖ = min{‖x - m‖ : m ∈ M}
   - Applications: least squares, best approximation

5. **Riesz Representation Theorem**
   - Every continuous linear functional f on H has form f(x) = ⟨x,y⟩ for unique y ∈ H
   - Establishes correspondence between H and its dual H*
   - Hilbert spaces are self-dual
   - Foundation for kernel methods

6. **Reproducing Kernels**
   - Kernel function: K : X × X → ℝ
   - Positive definiteness: ΣᵢⱼcᵢcⱼK(xᵢ,xⱼ) ≥ 0
   - Reproducing property: f(x) = ⟨f, K(x,·)⟩_H
   - Moore-Aronszajn theorem: K ↔ unique RKHS

7. **Operators on Hilbert Spaces**
   - Bounded linear operators: ‖T‖ = sup{‖Tx‖ : ‖x‖ ≤ 1}
   - Adjoint operator: ⟨Tx,y⟩ = ⟨x,T*y⟩
   - Self-adjoint (Hermitian): T = T*
   - Compact operators: map bounded sets to relatively compact sets
   - Spectral theorem for compact self-adjoint operators

## 🤖 Machine Learning Applications

### Reproducing Kernel Hilbert Space (RKHS)
- **Kernel Trick**: Implicitly map to high/infinite-dimensional space
- **Support Vector Machines (SVMs)**:
  - Optimization in RKHS
  - Maximum margin classification
  - Kernel functions: linear, polynomial, RBF/Gaussian
  - Decision function: f(x) = ΣαᵢyᵢK(x,xᵢ) + b
  
- **Gaussian Processes**:
  - Prior over functions in RKHS
  - Kernel defines covariance structure
  - Bayesian inference for regression/classification
  - Uncertainty quantification

### Neural Tangent Kernel (NTK)
- **Infinite-Width Neural Networks**:
  - Limit as width → ∞ yields kernel method
  - Training dynamics become linear in function space
  - Deterministic kernel characterizing network architecture
  - Connection between deep learning and kernel methods
  
- **Theoretical Deep Learning**:
  - Convergence guarantees for gradient descent
  - Generalization bounds via RKHS theory
  - Understanding neural network function spaces
  - Lazy training regime analysis

### Kernel Methods
- **Kernel PCA**: Nonlinear dimensionality reduction in RKHS
- **Kernel Ridge Regression**: Regularized regression with kernels
- **Kernel k-Means**: Clustering in feature space
- **Spectral Clustering**: Graph kernels and eigenvectors
- **Multiple Kernel Learning**: Combining multiple kernels

### Function Approximation
- **Best Approximation**: Projection onto finite-dimensional subspaces
- **Universal Approximation**: Dense subspaces in function spaces
- **Spline Interpolation**: Minimizing smoothness functionals
- **Radial Basis Functions**: Kernel-based interpolation

### Representation Learning
- **Feature Maps**: φ : X → H where K(x,y) = ⟨φ(x),φ(y)⟩
- **Implicit Feature Spaces**: Never compute φ explicitly
- **Kernel Embeddings**: Distributions in RKHS
- **Maximum Mean Discrepancy (MMD)**: Distance between distributions

### Optimization in Hilbert Spaces
- **Variational Methods**: Minimizing functionals
- **Gradient Descent in Function Space**: Infinite-dimensional optimization
- **Regularization**: Penalties on RKHS norm
- **Representer Theorem**: Optimal solutions in span of kernel functions

## 📊 Topics Covered

### Theoretical Foundations
- Inner product axioms and properties
- Cauchy-Schwarz and triangle inequalities
- Completeness and convergence
- Orthonormal bases and expansions
- Projection theorem and applications
- Riesz representation theorem
- Duality of Hilbert spaces

### Kernel Theory
- Positive definite kernels
- Mercer's theorem
- Moore-Aronszajn theorem
- Reproducing property
- Common kernels: RBF, polynomial, Matérn
- Kernel composition and construction
- Kernel embeddings

### Computational Methods
- Gram-Schmidt orthogonalization
- QR decomposition
- Computing projections
- Kernel matrix computation
- Eigendecomposition of kernel matrices
- Nyström approximation
- Random Fourier features

### Advanced Concepts
- Spectral theorem for compact operators
- Weak and strong convergence
- Separable Hilbert spaces
- Sobolev spaces
- Bergman spaces
- Hardy spaces

## 💻 What's Included

- **Comprehensive Theory**: Inner products, completeness, and kernels
- **Visual Intuition**: Geometric interpretations of projections and orthogonality
- **Python Implementations**: NumPy, SciPy, scikit-learn code for:
  - Computing inner products and norms
  - Gram-Schmidt orthogonalization
  - Orthogonal projections
  - Kernel matrix construction
  - SVM with various kernels
  - Gaussian Process regression
  - Kernel PCA
  - Neural Tangent Kernel computation
- **Interactive Examples**: Visualizing RKHS and kernel methods
- **ML Applications**: Real datasets with kernel methods
- **Exercises**: Theory and implementation problems
- **Mathematical Derivations**: Proofs of key theorems

## 🎯 Learning Outcomes

By the end of this section, you will be able to:

✅ Understand inner products and Hilbert space structure  
✅ Apply orthogonal projections for best approximation  
✅ Work with reproducing kernels and RKHS  
✅ Implement Support Vector Machines with kernel trick  
✅ Apply Gaussian Processes for regression/classification  
✅ Understand Neural Tangent Kernel theory  
✅ Choose appropriate kernels for different problems  
✅ Compute kernel matrices and solve kernel methods  
✅ Apply representer theorem for optimization  
✅ Connect infinite-dimensional theory to finite implementations  

## 📐 Mathematical Summary

### Key Definitions

**Inner Product Properties:**
```
⟨x,y⟩ = ⟨y,x⟩                    (Symmetry)
⟨αx + βy, z⟩ = α⟨x,z⟩ + β⟨y,z⟩   (Linearity)
⟨x,x⟩ ≥ 0, = 0 iff x = 0        (Positive definiteness)

Induced norm: ‖x‖ = √⟨x,x⟩
```

**Cauchy-Schwarz Inequality:**
```
|⟨x,y⟩| ≤ ‖x‖·‖y‖

Equality iff x and y are linearly dependent
```

**Projection Theorem:**
```
For closed subspace M ⊂ H:
x = P_M(x) + (x - P_M(x))
where P_M(x) ∈ M and (x - P_M(x)) ⊥ M

P_M(x) = argmin{‖x - m‖ : m ∈ M}
```

**Reproducing Property:**
```
For RKHS H with kernel K:
f(x) = ⟨f, K(x,·)⟩_H  for all f ∈ H

‖K(x,·)‖²_H = K(x,x)
```

**Representer Theorem:**
```
For regularized optimization:
min_f L(f) + λ‖f‖²_H

Optimal solution: f*(x) = Σᵢαᵢ K(x,xᵢ)
```

**Common Kernels:**
```
Linear:     K(x,y) = ⟨x,y⟩
Polynomial: K(x,y) = (⟨x,y⟩ + c)^d
RBF/Gaussian: K(x,y) = exp(-‖x-y‖²/(2σ²))
Matérn:     K(x,y) = (2^(1-ν)/Γ(ν))(√(2ν)r/ℓ)^ν K_ν(√(2ν)r/ℓ)
```

## 🔬 Example Problems

### Problem 1: Orthogonal Projection
Given vectors in ℝ³:
- Find orthogonal projection onto a subspace
- Compute residual and verify orthogonality
- Apply to least squares problem

### Problem 2: Kernel SVM
For a 2D classification dataset:
- Implement SVM with RBF kernel
- Visualize decision boundary
- Compare with linear SVM
- Analyze support vectors

### Problem 3: Gaussian Process Regression
Given noisy observations:
- Fit GP with different kernels
- Predict with uncertainty bounds
- Compare kernel choices
- Visualize posterior distributions

### Problem 4: Neural Tangent Kernel
For a simple neural network:
- Compute NTK at initialization
- Compare finite-width vs infinite-width
- Analyze training dynamics
- Verify kernel method correspondence

## 📚 Prerequisites

- Inner products and dot products
- Normed spaces and metric spaces
- Linear algebra fundamentals
- Calculus and analysis
- Basic optimization theory
- Understanding of SVMs (helpful)

## 🚀 Next Steps

After mastering Hilbert spaces, explore:

1. **Banach Spaces** - Spaces without inner products
2. **Operator Theory** - Advanced study of linear operators
3. **Spectral Methods** - Eigenvalue problems and decompositions
4. **Functional Analysis** - General theory of function spaces
5. **Stochastic Processes** - Random elements in Hilbert spaces
6. **Differential Geometry** - Manifolds and tangent spaces

## 🎓 Why This Matters for ML/AI

Understanding Hilbert spaces is essential because:

- **Kernel methods are built on RKHS theory** - SVMs, GPs, kernel PCA
- **Provides infinite-dimensional perspective** - Functions as points in space
- **Geometric intuition extends to function spaces** - Angles, projections, distances
- **Theoretical foundation for deep learning** - NTK, function space view
- **Best approximation theory** - Why and how learning works
- **Regularization has geometric meaning** - RKHS norm as smoothness
- **Bridges finite and infinite dimensions** - From data to functions

Hilbert spaces unite geometry, analysis, and probability for ML theory!

## 📖 Recommended Reading

- *Functional Analysis* by Walter Rudin (Chapters 4, 12)
- *Learning with Kernels* by Schölkopf & Smola
- *Gaussian Processes for Machine Learning* by Rasmussen & Williams
- *An Introduction to RKHS* by Berlinet & Thomas-Agnan
- *Elements of Statistical Learning* by Hastie et al. (Chapter 5)
- *Neural Tangent Kernel* papers by Jacot et al.

## 🔗 Related Topics

- [Normed Spaces](../) - Foundation with norms
- [Kernel Methods & RKHS](./kernel-methods-rkhs/) - Detailed kernel theory
- [Banach Spaces](../banach-spaces/) - More general complete normed spaces
- [Linear Operators](../../linear-operators/) - Transformations on Hilbert spaces
- [Spectral Methods](../../matrices/spectral-methods/) - Eigenvalue decompositions
