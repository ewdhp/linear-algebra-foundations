# Spectral Methods & Decomposition

## 📖 What You'll Learn

Spectral methods analyze the eigenstructure (spectrum) of matrices and operators to understand their behavior, stability, and dimensionality. These techniques are fundamental to numerous machine learning applications, from Principal Component Analysis and spectral clustering to analyzing the stability of neural networks and understanding graph-based learning.

### Core Concepts

1. **Eigenvalues & Eigenvectors Review**
   - **Eigenvalue equation**: Av = λv
   - v is eigenvector, λ is eigenvalue
   - Geometric interpretation: direction preserved under transformation
   - Algebraic multiplicity vs geometric multiplicity
   - Characteristic polynomial: det(A - λI) = 0

2. **Spectral Theorem**
   - **For symmetric matrices**: A = QΛQᵀ
   - Q orthogonal (columns are orthonormal eigenvectors)
   - Λ diagonal (eigenvalues)
   - All eigenvalues real, eigenvectors orthogonal
   - Extends to self-adjoint operators on Hilbert spaces

3. **Spectrum of an Operator**
   - **Spectrum**: σ(A) = {λ : A - λI not invertible}
   - Point spectrum: eigenvalues
   - Continuous spectrum: approximate eigenvalues
   - Residual spectrum: other non-invertible values
   - Spectral radius: ρ(A) = max{|λ| : λ ∈ σ(A)}

4. **Eigendecomposition**
   - Diagonalizable matrices: A = PDP⁻¹
   - P columns are eigenvectors
   - D diagonal matrix of eigenvalues
   - Not all matrices diagonalizable
   - Jordan canonical form for general case

5. **Spectral Properties**
   - **Trace**: tr(A) = Σλᵢ
   - **Determinant**: det(A) = Πλᵢ
   - **Rank**: Number of nonzero eigenvalues
   - **Condition number**: κ(A) = |λ_max|/|λ_min|
   - **Matrix powers**: Aⁿ = QΛⁿQᵀ

6. **Rayleigh Quotient**
   - **Definition**: R(x) = (xᵀAx)/(xᵀx)
   - Bounds eigenvalues: λ_min ≤ R(x) ≤ λ_max
   - Stationary points are eigenvectors
   - Variational characterization of eigenvalues
   - Power method and inverse iteration

7. **Matrix Functions**
   - **Definition via spectral decomposition**: f(A) = Qf(Λ)Qᵀ
   - Examples: exp(A), log(A), A^(1/2)
   - Applications: solving differential equations
   - Matrix exponential for dynamics

8. **Spectral Clustering**
   - Graph Laplacian: L = D - A
   - Eigenvalues measure connectivity
   - Eigenvectors reveal cluster structure
   - Normalized Laplacian: L_norm = I - D⁻¹/²AD⁻¹/²

9. **Stability Analysis**
   - Largest eigenvalue magnitude determines stability
   - |λ_max| < 1: stable discrete dynamics
   - Re(λ_max) < 0: stable continuous dynamics
   - Spectral norm bounds operator behavior

## 🤖 Machine Learning Applications

### Principal Component Analysis (PCA)
- **Eigendecomposition of Covariance**:
  - Compute covariance matrix: C = (1/n)XᵀX
  - Eigendecomposition: C = QΛQᵀ
  - Principal components: eigenvectors (columns of Q)
  - Variance explained: eigenvalues (diagonal of Λ)
  
- **Dimensionality Reduction**:
  - Project onto top k eigenvectors
  - Maximize variance preserved
  - Minimize reconstruction error
  - Cumulative variance ratio for k selection
  
- **Applications**:
  - Feature extraction
  - Data visualization
  - Denoising
  - Compression
  - Preprocessing for other algorithms

### Spectral Clustering
- **Graph-Based Clustering**:
  - Construct similarity graph
  - Compute graph Laplacian L = D - A
  - Find k smallest eigenvalues/eigenvectors
  - Cluster eigenvector rows (k-means)
  
- **Advantages**:
  - Captures non-convex clusters
  - Based on connectivity, not distance
  - Theoretically grounded (graph cuts)
  - Works with arbitrary similarity measures
  
- **Applications**:
  - Image segmentation
  - Community detection
  - Bioinformatics
  - Document clustering

### Graph Neural Networks (GNNs)
- **Spectral Graph Convolution**:
  - Fourier transform on graphs via eigenvectors
  - Graph Laplacian eigenbasis
  - Convolution in spectral domain: f̂ * ĝ
  - ChebNet: Chebyshev polynomial approximation
  
- **Graph Signal Processing**:
  - Signals on vertices/edges
  - Frequency analysis via eigenvalues
  - Low-pass/high-pass filters
  - Graph wavelets
  
- **Applications**:
  - Node classification
  - Link prediction
  - Graph classification
  - Molecular property prediction

### Neural Network Analysis
- **Hessian Eigenspectrum**:
  - Second derivatives of loss
  - Eigenvalues indicate curvature
  - Large eigenvalues: sharp minima
  - Small eigenvalues: flat directions
  
- **Gradient Dynamics**:
  - Spectral analysis of weight matrices
  - Initialization and learning dynamics
  - Mode connectivity
  - Loss landscape geometry
  
- **Stability & Generalization**:
  - Spectral norm of weight matrices
  - Lipschitz bounds
  - PAC-Bayes via spectral complexity
  - Sharpness-aware minimization

### Regularization via Low-Rank Approximation
- **Low-Rank Projection**:
  - Truncate small eigenvalues
  - Reduces model complexity
  - Regularization via spectral truncation
  - Nuclear norm minimization
  
- **Matrix Completion**:
  - Netflix problem: low-rank assumption
  - Minimize rank subject to observations
  - Convex relaxation via nuclear norm
  - Alternating minimization
  
- **Tensor Decomposition**:
  - Higher-order SVD (HOSVD)
  - CP decomposition via eigenvalues
  - Tucker decomposition
  - Neural network compression

### Kernel Methods & Spectral Learning
- **Kernel PCA**:
  - Eigendecomposition of kernel matrix
  - Nonlinear principal components
  - Mercer's theorem: K = Σλᵢφᵢφᵢᵀ
  
- **Gaussian Processes**:
  - Covariance kernel eigendecomposition
  - Karhunen-Loève expansion
  - Sparse approximations via low-rank
  
- **Spectral Learning**:
  - Learning latent variable models
  - Method of moments via eigendecomposition
  - Hidden Markov Models, mixture models

### Time Series & Dynamics
- **Dynamic Mode Decomposition (DMD)**:
  - Data-driven spectral analysis
  - Eigenvalues: growth rates and frequencies
  - Eigenvectors: spatial patterns
  - Forecasting and control
  
- **Koopman Operator Theory**:
  - Linear representation of nonlinear dynamics
  - Spectral decomposition for predictions
  - Modal analysis

## 📊 Topics Covered

### Theoretical Foundations
- Eigenvalue problem formulation
- Spectral theorem and diagonalization
- Jordan canonical form
- Perturbation theory for eigenvalues
- Variational principles (min-max theorem)
- Spectral mapping theorem
- Gerschgorin circle theorem

### Computational Methods
- Power method and inverse iteration
- QR algorithm
- Jacobi method for symmetric matrices
- Lanczos and Arnoldi methods
- Divide and conquer algorithms
- Randomized eigendecomposition
- Sparse eigensolvers

### Spectral Analysis Techniques
- Rayleigh quotient iteration
- Spectral gap analysis
- Spectrum visualization
- Eigenvalue sensitivity
- Condition number estimation
- Spectral radius computation

### Advanced Topics
- Generalized eigenvalue problems
- Matrix pencils (A - λB)
- Quadratic eigenvalue problems
- Polynomial eigenvalue problems
- Structured eigenvalue problems
- Infinite-dimensional spectral theory

## 💻 What's Included

- **Comprehensive Theory**: Eigenvalues, spectral theorem, and applications
- **Visual Intuition**: Eigenvector directions and spectral decomposition
- **Python Implementations**: NumPy, SciPy, scikit-learn, PyTorch code for:
  - Computing eigenvalues/eigenvectors
  - PCA from scratch and with sklearn
  - Spectral clustering implementation
  - Graph Laplacian computation
  - Spectral graph convolution
  - Hessian eigenspectrum analysis
  - Low-rank approximations
  - Dynamic mode decomposition
  - Matrix exponential computation
  - Rayleigh quotient iteration
- **Interactive Examples**: Visualizing spectral methods
- **ML Applications**: Real datasets for PCA, clustering, GNNs
- **Exercises**: Theory, proofs, and implementations
- **Performance Analysis**: Scalability and computational complexity

## 🎯 Learning Outcomes

By the end of this section, you will be able to:

✅ Compute eigenvalues and eigenvectors efficiently  
✅ Apply PCA for dimensionality reduction  
✅ Implement spectral clustering for non-convex clusters  
✅ Use graph Laplacian for GNN architectures  
✅ Analyze neural network stability via eigenspectra  
✅ Apply low-rank approximation for regularization  
✅ Understand spectral properties of matrices/operators  
✅ Use Rayleigh quotient for optimization  
✅ Perform spectral analysis of dynamics  
✅ Choose appropriate spectral methods for problems  

## 📐 Mathematical Summary

### Key Definitions

**Eigenvalue Problem:**
```
Av = λv  (v ≠ 0)

Characteristic polynomial: det(A - λI) = 0
```

**Spectral Theorem (Symmetric):**
```
For symmetric A ∈ ℝⁿˣⁿ:
A = QΛQᵀ

Where:
- Q orthogonal: QᵀQ = I
- Λ = diag(λ₁, ..., λₙ)
- All λᵢ ∈ ℝ
- Columns of Q orthonormal eigenvectors
```

**Rayleigh Quotient:**
```
R(x) = (xᵀAx)/(xᵀx)

Properties:
- λ_min ≤ R(x) ≤ λ_max
- R(vᵢ) = λᵢ (at eigenvectors)
- ∇R(x) = 0 at eigenvectors
```

**PCA via Eigendecomposition:**
```
Covariance: C = (1/n)XᵀX

Eigendecomposition: C = QΛQᵀ

Projection: Z = XQ_k (first k columns)

Variance explained: Σλᵢ / Σλⱼ (i ≤ k, all j)
```

**Graph Laplacian:**
```
L = D - A

Where:
- A: adjacency matrix
- D: degree matrix (diagonal)

Normalized Laplacian:
L_norm = I - D⁻¹/²AD⁻¹/²
```

**Matrix Functions:**
```
If A = QΛQᵀ, then:

f(A) = Qf(Λ)Qᵀ = Q diag(f(λ₁),...,f(λₙ)) Qᵀ

Examples:
exp(A) = Q exp(Λ) Qᵀ
A^(1/2) = Q Λ^(1/2) Qᵀ
```

**Stability Conditions:**
```
Discrete dynamics: xₜ₊₁ = Axₜ
Stable if ρ(A) < 1 (spectral radius)

Continuous dynamics: dx/dt = Ax
Stable if max Re(λᵢ) < 0
```

## 🔬 Example Problems

### Problem 1: PCA Implementation
For high-dimensional dataset:
- Compute covariance matrix
- Eigendecomposition
- Determine number of components (elbow plot)
- Project data to lower dimensions
- Visualize and interpret principal components
- Reconstruct and compute error

### Problem 2: Spectral Clustering
For non-convex clusters:
- Construct k-NN or RBF similarity graph
- Compute graph Laplacian
- Find smallest eigenvalues/eigenvectors
- Apply k-means to eigenvectors
- Compare with k-means in original space
- Visualize clusters

### Problem 3: GNN Spectral Convolution
For citation network:
- Compute graph Laplacian
- Eigendecomposition
- Implement spectral convolution layer
- Train for node classification
- Compare with spatial methods
- Analyze learned filters

### Problem 4: Neural Network Hessian
For trained neural network:
- Compute Hessian (or approximate)
- Eigendecomposition of Hessian
- Visualize eigenvalue spectrum
- Identify sharp vs flat directions
- Relate to generalization
- Compare different minima

## 📚 Prerequisites

- Eigenvalues and eigenvectors
- Linear algebra fundamentals
- Matrix calculus
- Optimization basics
- Graph theory (for GNNs)
- Basic neural networks

## 🚀 Next Steps

After mastering spectral methods, explore:

1. **Random Matrix Theory** - Large-scale eigenvalue distributions
2. **Tensor Decompositions** - Higher-order spectral methods
3. **Operator Theory** - Spectral theory in infinite dimensions
4. **Quantum Computing** - Eigenvalues in quantum mechanics
5. **Compressed Sensing** - Spectral methods for sparse recovery
6. **Manifold Learning** - Spectral embeddings (Laplacian eigenmaps)

## 🎓 Why This Matters for ML/AI

Understanding spectral methods is essential because:

- **PCA is ubiquitous** - Most widely used dimensionality reduction
- **Spectral clustering captures complex structure** - Non-convex clusters
- **GNNs use spectral convolutions** - Graph-based deep learning
- **Eigenspectrum reveals network properties** - Stability, generalization
- **Low-rank regularization prevents overfitting** - Matrix completion
- **Kernel methods rely on eigendecomposition** - Kernel PCA, GPs
- **Dynamics analysis uses eigenvalues** - Stability and convergence
- **Theoretical understanding via spectrum** - Condition number, rank

Spectral methods are the bridge between linear algebra and ML applications!

## 📖 Recommended Reading

- *Matrix Computations* by Golub & Van Loan (Chapters 7-8)
- *Spectral Graph Theory* by Chung
- *Deep Learning* by Goodfellow et al. (Chapter 2.7, 5.8)
- *Pattern Recognition and Machine Learning* by Bishop (Chapter 12)
- *Elements of Statistical Learning* by Hastie et al. (Chapter 14.5)
- *Spectral Methods for Data Science* by Bandeira & van Handel
- *Graph Representation Learning* by Hamilton

## 🔗 Related Topics

- [Eigenvalues & Eigenvectors](../eigenvalues-eigenvectors/) - Foundation
- [SVD](../svd/) - Related decomposition
- [Matrices](../) - Parent topic
- [Kernel Methods & RKHS](../../normed-spaces/hilbert-spaces/kernel-methods-rkhs/) - Kernel eigendecomposition
- [Linear Operators](../../linear-operators/) - Operator spectrum
- [Subspaces & Fundamental Theorem](../../subspaces-fundamental-theorem/) - Eigenspaces
