# Eigenvalues & Eigenvectors

## 📖 What You'll Learn

Solve Av = λv to reveal principal directions and scaling factors. Eigendecomposition is one of the most powerful tools in linear algebra with profound applications in ML/AI:

### Core Concepts

1. **The Eigenvalue Equation**
   - Understanding Av = λv
   - Geometric interpretation: directions preserved under transformation
   - Eigenvalues as scaling factors
   - Finding eigenvalues and eigenvectors

2. **Characteristic Polynomial**
   - det(A - λI) = 0
   - Computing eigenvalues from the characteristic equation
   - Algebraic and geometric multiplicity

3. **Eigendecomposition**
   - Diagonalization: A = PDP⁻¹
   - Conditions for diagonalizability
   - Symmetric matrices and orthogonal eigendecomposition
   - Spectral theorem

4. **Matrix Powers and Functions**
   - Computing Aⁿ using eigendecomposition
   - Matrix exponential and logarithm
   - Applications in dynamical systems

5. **Eigenspaces**
   - Null space and eigenspace relationship
   - Basis of eigenvectors
   - Invariant subspaces

## 🤖 Machine Learning Applications

### PCA for Dimensionality Reduction
- **Covariance Matrix Eigendecomposition**: C = QΛQᵀ
  - Eigenvectors: Principal components (directions of maximum variance)
  - Eigenvalues: Amount of variance explained by each component
- **Dimensionality Reduction**: Project data onto top k eigenvectors
  - Reduced representation: Z = XQₖ
  - Variance retained: Σλᵢ (i≤k) / Σλⱼ (all j)
- **Feature Extraction**: Transform correlated features to uncorrelated principal components
- **Data Compression**: Store data in lower dimensions
- **Noise Reduction**: Discard components with small eigenvalues
- **Visualization**: Project high-dimensional data to 2D/3D
- **Preprocessing**: Whitening data for neural networks

### Spectral Clustering (Graph Laplacian)
- **Graph Laplacian Matrix**: L = D - A
  - D: degree matrix (diagonal)
  - A: adjacency matrix
  - Properties: Symmetric, positive semi-definite
- **Eigenvalue 0**: Multiplicity = number of connected components
- **Fiedler Vector**: Second smallest eigenvalue's eigenvector
  - Reveals cluster structure
  - Bisection of graph
- **k-way Clustering**: Use k smallest eigenvalues' eigenvectors
- **Applications**:
  - Image segmentation
  - Community detection in social networks
  - Document clustering
  - Graph partitioning
- **Normalized Laplacian**: L_norm = I - D⁻¹/²AD⁻¹/²

### Stability Analysis in Recurrent Neural Networks (RNNs)
- **Gradient Flow**: Eigenvalues of weight matrices determine stability
- **Vanishing Gradients**: |λ_max| < 1
  - Gradients decay exponentially through time
  - Long-term dependencies not learned
- **Exploding Gradients**: |λ_max| > 1
  - Gradients grow exponentially
  - Training instability
- **Critical Initialization**: Initialize with spectral radius ≈ 1
- **Jacobian Analysis**: ∂h_t/∂h_{t-1} eigenvalues
- **Long Short-Term Memory (LSTM)**: Designed to address eigenvalue problems
- **Solutions**:
  - Gradient clipping
  - Orthogonal weight initialization
  - Skip connections (ResNets)
  - Spectral normalization

### Other Key Applications
- **Markov Chains & PageRank**: Dominant eigenvector of transition matrix
- **Recommendation Systems**: Matrix factorization via eigendecomposition
- **Quantum Mechanics**: Schrödinger equation solutions
- **Vibration Analysis**: Natural frequencies as eigenvalues
- **Control Theory**: System stability via eigenvalue placement

## 📊 Topics Covered

- Computing eigenvalues and eigenvectors
- Symmetric and Hermitian matrices
- Real vs. complex eigenvalues
- Positive definite matrices
- Spectral radius
- Rayleigh quotient
- Power iteration method
- QR algorithm
- Applications to differential equations
- Stability analysis

## 💻 What's Included

- In-depth theoretical explanations
- Geometric visualizations
- NumPy and SciPy implementations
- PCA implementation from scratch
- Real datasets for dimensionality reduction
- Graph spectral analysis examples
- RNN stability analysis
- Interactive notebooks

## 🎯 Learning Outcomes

By the end of this section, you will be able to:
- Calculate eigenvalues and eigenvectors efficiently
- Implement PCA for dimensionality reduction
- Analyze stability of iterative systems
- Apply spectral methods to graph problems
- Understand the geometry of linear transformations
- Debug gradient issues in deep networks
- Interpret principal components in real data

## 📚 Prerequisites

- Solid understanding of matrices
- Matrix multiplication and determinants
- Basic understanding of optimization
- Python and NumPy proficiency

## 🚀 Next Steps

After mastering eigendecomposition, you'll learn about **Singular Value Decomposition (SVD)**, which extends these concepts to non-square matrices and provides even more powerful tools for ML/AI applications.
