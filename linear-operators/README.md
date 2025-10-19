# Linear Operators

## 📖 What You'll Learn

Linear operators are mappings between vector spaces , function spaces that preserve the linear structure. They are fundamental to understanding transformations in machine learning, from simple matrix operations to complex differential operators in physics-informed ML and attention mechanisms in transformers.

### Core Concepts

1. **Linear Operator Definition**
   - Map T : V → W between vector spaces
   - Linearity: T(αx + βy) = αT(x) + βT(y)
   - Preserves vector space structure
   - Examples: matrices, differentiation, integration, convolution

2. **Bounded vs Unbounded Operators**
   - **Bounded**: ‖T‖ = sup{‖Tx‖ : ‖x‖ ≤ 1} < ∞
   - Equivalent to continuity in normed spaces
   - All linear operators in finite dimensions are bounded
   - **Unbounded**: ‖T‖ = ∞ (e.g., differentiation)
   - Domain and densely defined operators

3. **Matrix Representation**
   - Finite-dimensional: T ↔ matrix A
   - Change of basis transformations
   - Coordinate-free vs coordinate view
   - Relationship: T(v) = Av

4. **Operator Norms**
   - **Spectral norm**: ‖A‖₂ = σ_max(A)
   - **Frobenius norm**: ‖A‖_F = √(Σᵢⱼaᵢⱼ²)
   - **Induced norms**: ‖A‖_{p→q} = sup{‖Ax‖_q : ‖x‖_p ≤ 1}
   - **Operator norm**: sup{‖Tx‖_W : ‖x‖_V ≤ 1}

5. **Adjoint Operators**
   - **Definition**: ⟨Tx,y⟩_W = ⟨x,T*y⟩_V
   - In finite-dim: T* ↔ Aᵀ (real) or A† (complex)
   - Properties: (T*)* = T, (ST)* = T*S*
   - Self-adjoint (Hermitian): T = T*
   - Normal operators: TT* = T*T

6. **Special Classes of Operators**
   - **Projection operators**: P² = P
   - **Isometries**: ‖Tx‖ = ‖x‖
   - **Unitary/Orthogonal**: T*T = TT* = I
   - **Positive operators**: ⟨Tx,x⟩ ≥ 0
   - **Compact operators**: Map bounded sets to relatively compact sets

7. **Composition and Invertibility**
   - Composition: (S ∘ T)(x) = S(T(x))
   - Inverse operator: T⁻¹ exists if T bijective
   - Pseudoinverse: T† for non-invertible operators
   - Fredholm operators: Finite-dimensional kernel and cokernel

8. **Differential Operators**
   - Derivative: D[f] = f'
   - Gradient: ∇f = (∂f/∂x₁, ..., ∂f/∂xₙ)
   - Laplacian: Δf = ∇²f = Σ∂²f/∂xᵢ²
   - Divergence and curl
   - General form: L = Σaₐ(x)Dᵃ

9. **Integral Operators**
   - **Integral transform**: (Kf)(x) = ∫K(x,y)f(y)dy
   - Kernel K(x,y) determines operator
   - Examples: Fourier, Laplace, convolution
   - Often compact operators

## 🤖 Machine Learning Applications

### Convolutional Neural Networks (CNNs)
- **Convolution as Linear Operator**:
  - (f * g)(x) = ∫f(x-y)g(y)dy
  - Translation-equivariant operator
  - Toeplitz/circulant matrix structure
  - Local receptive fields
  
- **CNN Architectures**:
  - Stacked convolution operators
  - Pooling as operator composition
  - Stride and dilation parameters
  - Operator norm and Lipschitz bounds
  
- **Analysis**:
  - Frequency domain interpretation
  - Stability to deformations
  - Multi-scale representations
  - Scattering transforms

### Attention Mechanisms & Transformers
- **Attention as Operator**:
  - Query-Key-Value formulation
  - Attention(Q,K,V) = softmax(QKᵀ/√d)V
  - Linear operator in function space
  - Self-attention: operating on same space
  
- **Multi-Head Attention**:
  - Parallel operator decomposition
  - Multiple representation subspaces
  - Operator composition
  
- **Transformers**:
  - Sequence of attention operators
  - Position encodings
  - Residual connections preserve gradient flow
  - Layer normalization as operator

### Graph Neural Networks (GNNs)
- **Graph Operators**:
  - Adjacency matrix A as operator
  - Graph Laplacian: L = D - A
  - Normalized Laplacian: L_norm = I - D⁻¹/²AD⁻¹/²
  
- **Message Passing**:
  - Convolution on graphs: Operator on graph signal space
  - Spectral methods: Eigendecomposition of Laplacian
  - Spatial methods: Local aggregation operators
  
- **Applications**:
  - Node classification
  - Graph classification
  - Link prediction
  - Spectral clustering

### Physics-Informed Machine Learning
- **Differential Operators**:
  - PDEs as operator equations: Lu = f
  - Neural networks approximate solution operator
  - Physics-informed loss: ‖Lu - f‖²
  
- **Physics-Informed Neural Networks (PINNs)**:
  - Incorporate differential operators in loss
  - Automatic differentiation for derivatives
  - Boundary and initial conditions
  - Inverse problems: learning operators
  
- **Neural Operators**:
  - DeepONet: Learn operator G : u → s
  - Fourier Neural Operator (FNO)
  - Resolution-invariant architectures
  - Applications: fluid dynamics, climate modeling

### Optimization & Gradient Methods
- **Gradient as Operator**:
  - ∇f maps parameters to tangent space
  - Jacobian and Hessian as linear operators
  - Chain rule: composition of operators
  
- **Preconditioners**:
  - Modifying metric via operator
  - Natural gradient: Fisher information operator
  - Second-order methods: Inverse Hessian
  
- **Operator Splitting**:
  - ADMM: Alternating operators
  - Proximal operators
  - Mirror descent

### Kernel Methods
- **Kernel Operator**:
  - K : L²(X) → L²(X)
  - Integral operator with kernel K(x,y)
  - Compact operator in RKHS
  
- **Applications**:
  - SVMs: Optimization over kernel operator
  - Gaussian Processes: Covariance operator
  - Kernel PCA: Eigendecomposition of operator

## 📊 Topics Covered

### Theoretical Foundations
- Linearity axioms and properties
- Bounded vs unbounded operators
- Domain, range, kernel (null space)
- Continuity and boundedness equivalence
- Operator topologies
- Spectral theory basics

### Matrix Operators
- Matrix as linear operator
- Change of basis
- Similar matrices and conjugation
- Matrix decompositions (SVD, QR, LU)
- Kronecker and Hadamard products

### Functional Operators
- Differential operators on function spaces
- Integral operators and kernels
- Convolution operators
- Fourier and wavelet transforms
- Green's functions

### Computational Methods
- Computing operator norms
- Adjoint computation
- Eigenvalue problems for operators
- Numerical methods for PDEs
- Automatic differentiation
- Finite element methods

### Advanced Concepts
- Spectral theorem for operators
- Compact operators and Fredholm theory
- Resolvent and spectrum
- Semigroups and evolution operators
- Unbounded self-adjoint operators
- Distribution theory

## 💻 What's Included

- **Comprehensive Theory**: Operator definitions, properties, and theorems
- **Visual Intuition**: Geometric interpretations and transformations
- **Python Implementations**: NumPy, PyTorch, JAX code for:
  - Matrix operators and compositions
  - Convolution operators (1D, 2D, 3D)
  - Attention mechanisms
  - Graph Laplacian operators
  - Differential operators via autodiff
  - Neural operator architectures
  - Physics-informed neural networks
  - Spectral methods
- **Interactive Examples**: Visualizing operator actions
- **ML Applications**: CNNs, transformers, GNNs, PINNs
- **Exercises**: Theory and implementation problems
- **Mathematical Derivations**: Proofs and operator calculus

## 🎯 Learning Outcomes

By the end of this section, you will be able to:

✅ Understand linear operators and their properties  
✅ Compute operator norms and adjoints  
✅ Implement convolution operators for CNNs  
✅ Design attention mechanisms as operators  
✅ Apply graph Laplacian operators in GNNs  
✅ Use differential operators in physics-informed ML  
✅ Understand operator composition in deep learning  
✅ Analyze stability via operator norms  
✅ Work with kernel operators in RKHS  
✅ Apply operator theory to ML architectures  

## 📐 Mathematical Summary

### Key Definitions

**Linear Operator:**
```
T : V → W is linear if:
T(αx + βy) = αT(x) + βT(y)  for all α,β ∈ ℝ, x,y ∈ V
```

**Operator Norm:**
```
‖T‖ = sup{‖Tx‖_W : ‖x‖_V ≤ 1}
     = sup{‖Tx‖_W / ‖x‖_V : x ≠ 0}
```

**Adjoint Operator:**
```
⟨Tx, y⟩_W = ⟨x, T*y⟩_V  for all x ∈ V, y ∈ W

For matrices: T* = Aᵀ (real) or A† (complex conjugate transpose)
```

**Composition:**
```
(S ∘ T)(x) = S(T(x))

(S ∘ T)* = T* ∘ S*
‖S ∘ T‖ ≤ ‖S‖·‖T‖
```

**Common Operators:**
```
Convolution:     (f * g)(x) = ∫f(x-y)g(y)dy
Differential:    D[f] = df/dx
Laplacian:       Δf = ∇²f = Σ∂²f/∂xᵢ²
Attention:       Attn(Q,K,V) = softmax(QKᵀ/√d)V
Graph Laplacian: L = D - A
```

**Special Properties:**
```
Projection:      P² = P
Isometry:        ‖Tx‖ = ‖x‖
Unitary:         T*T = TT* = I
Self-adjoint:    T = T*
Positive:        ⟨Tx,x⟩ ≥ 0
```

## 🔬 Example Problems

### Problem 1: Convolution Operator
Implement 2D convolution:
- As matrix operation (Toeplitz)
- Using FFT (frequency domain)
- Compare computational complexity
- Analyze translation equivariance

### Problem 2: Attention Mechanism
Design multi-head attention:
- Implement Q, K, V projections
- Compute attention weights
- Analyze as linear operator
- Visualize attention patterns

### Problem 3: Graph Laplacian
For a graph dataset:
- Compute adjacency and Laplacian
- Find eigenvalues and eigenvectors
- Implement spectral graph convolution
- Apply to node classification

### Problem 4: Physics-Informed NN
Solve 1D heat equation:
- Define differential operator loss
- Train neural network to satisfy PDE
- Enforce boundary conditions
- Compare with analytical solution

## 📚 Prerequisites

- Linear algebra fundamentals
- Vector spaces and bases
- Matrix operations
- Calculus and differentiation
- Inner products (for adjoints)
- Basic functional analysis

## 🚀 Next Steps

After mastering linear operators, explore:

1. **Spectral Theory** - Eigenvalues and spectral decomposition
2. **Operator Semigroups** - Time evolution and dynamics
3. **Nonlinear Operators** - Beyond linear maps
4. **Operator Algebras** - C*-algebras and von Neumann algebras
5. **Quantum Computing** - Unitary operators and quantum gates
6. **Control Theory** - Controllability and observability operators

## 🎓 Why This Matters for ML/AI

Understanding linear operators is essential because:

- **Neural networks are compositions of operators** - Layers as transformations
- **Convolution is a fundamental operator** - CNNs dominate computer vision
- **Attention mechanisms are operators** - Transformers revolutionized NLP
- **Graph methods use Laplacian operators** - GNNs for relational data
- **Physics-informed ML needs differential operators** - Scientific computing
- **Optimization uses gradient operators** - First and second-order methods
- **Spectral methods decompose operators** - Understanding network behavior
- **Operator theory unifies ML architectures** - Common mathematical framework

Linear operators are the language of transformations in ML!

## 📖 Recommended Reading

- *Functional Analysis* by Walter Rudin (Chapters 4, 12, 13)
- *Linear Operators* by Dunford & Schwartz
- *Deep Learning* by Goodfellow et al. (Chapters 9, 10, 12)
- *Attention is All You Need* by Vaswani et al.
- *Graph Representation Learning* by Hamilton
- *Physics-Informed Neural Networks* by Karniadakis et al.
- *Fourier Neural Operator* by Li et al.

## 🔗 Related Topics

- [Matrices](../matrices/) - Finite-dimensional operators
- [Hilbert Spaces](../normed-spaces/hilbert-spaces/) - Inner product and adjoints
- [Banach Spaces](../normed-spaces/banach-spaces/) - General setting for operators
- [Spectral Methods](../matrices/spectral-methods/) - Eigenvalue decomposition
- [Kernel Methods & RKHS](../normed-spaces/hilbert-spaces/kernel-methods-rkhs/) - Integral operators
