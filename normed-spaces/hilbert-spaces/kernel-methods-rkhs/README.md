# Kernel Methods & RKHS

## 📖 What You'll Learn

Kernel methods provide a powerful framework for learning in high or infinite-dimensional spaces without explicit computation. The kernel trick, combined with Reproducing Kernel Hilbert Space (RKHS) theory, enables linear methods to capture complex nonlinear patterns and forms the theoretical foundation for Support Vector Machines, Gaussian Processes, and modern deep learning theory.

### Core Concepts

1. **Kernel Functions**
   - **Definition**: K : X × X → ℝ
   - Symmetric: K(x,y) = K(y,x)
   - Positive semidefinite: ΣᵢⱼcᵢcⱼK(xᵢ,xⱼ) ≥ 0 for all c
   - Measures similarity between points
   - Examples: linear, polynomial, RBF, Matérn

2. **The Kernel Trick**
   - **Implicit feature map**: K(x,y) = ⟨φ(x),φ(y)⟩_H
   - Never compute φ(x) explicitly
   - Work directly with kernel values
   - Linear methods in high/infinite dimensions
   - Computational efficiency: O(n²) vs O(nd) or worse

3. **Reproducing Kernel Hilbert Space (RKHS)**
   - Hilbert space H of functions on X
   - **Reproducing property**: f(x) = ⟨f, K(x,·)⟩_H
   - Evaluation functional is continuous
   - **Moore-Aronszajin theorem**: K ↔ unique RKHS
   - Norm: ‖f‖²_H measures smoothness/complexity

4. **Feature Maps**
   - **Explicit maps**: φ : X → H where K(x,y) = ⟨φ(x),φ(y)⟩
   - Linear kernel: φ(x) = x
   - Polynomial: φ(x) contains all monomials
   - RBF: φ maps to infinite dimensions
   - Random features: approximate feature maps

5. **Common Kernels**
   - **Linear**: K(x,y) = ⟨x,y⟩
   - **Polynomial**: K(x,y) = (⟨x,y⟩ + c)^d
   - **RBF/Gaussian**: K(x,y) = exp(-‖x-y‖²/(2σ²))
   - **Laplacian**: K(x,y) = exp(-‖x-y‖/σ)
   - **Matérn**: Flexible smoothness
   - **String kernels**: For sequences
   - **Graph kernels**: For structured data

6. **Kernel Properties**
   - **Closure under operations**:
     - Sum: K₁ + K₂ is kernel
     - Product: K₁ · K₂ is kernel
     - Scaling: αK is kernel (α > 0)
     - Limit: pointwise limit of kernels
   - **Universality**: Dense in C(X) for compact X
   - **Characteristic kernels**: Unique mean embedding

7. **Representer Theorem**
   - **Statement**: For regularized loss minimization:
     min_f L(f) + λ‖f‖²_H
   - **Solution**: f*(x) = Σᵢαᵢ K(x,xᵢ)
   - Finite-dimensional optimization despite infinite-dimensional space
   - Foundation for kernel methods

8. **Mercer's Theorem**
   - Kernel expansion: K(x,y) = Σᵢλᵢφᵢ(x)φᵢ(y)
   - Eigenvalues λᵢ ≥ 0
   - Eigenfunctions φᵢ form orthonormal basis
   - Connection to integral operators

## 🤖 Machine Learning Applications

### Support Vector Machines (SVMs)
- **Binary Classification**:
  - max-margin principle in RKHS
  - Decision function: f(x) = Σᵢαᵢyᵢ K(x,xᵢ) + b
  - Only support vectors have αᵢ ≠ 0
  - Kernel trick: never compute φ explicitly
  
- **Kernel Selection**:
  - Linear: when data linearly separable
  - Polynomial: for interaction terms
  - RBF: universal approximator, most common
  - Domain-specific: string, graph kernels
  
- **Extensions**:
  - Multi-class SVM
  - Regression (SVR)
  - One-class SVM for anomaly detection
  - Structured output prediction

### Gaussian Processes (GPs)
- **Prior over Functions**:
  - GP(μ, K): distribution over functions
  - Kernel K defines covariance structure
  - Mean function μ (often zero)
  - Infinite-dimensional generalization of Gaussian
  
- **Posterior Inference**:
  - Closed-form posterior given data
  - Predictive mean and variance
  - Uncertainty quantification
  - Hyperparameter learning (marginal likelihood)
  
- **Applications**:
  - Regression with uncertainty
  - Bayesian optimization
  - Experimental design (active learning)
  - Time series modeling
  - Spatial statistics (kriging)

### Kernel PCA & Manifold Learning
- **Kernel PCA**:
  - PCA in feature space φ(X)
  - Eigendecomposition of kernel matrix
  - Nonlinear dimensionality reduction
  - Capturing nonlinear structure
  
- **Spectral Methods**:
  - Spectral clustering: kernel k-means
  - Graph Laplacian eigenmaps
  - Diffusion maps
  - Isomap with geodesic kernels
  
- **Applications**:
  - Feature extraction
  - Visualization
  - Denoising
  - Preprocessing for other methods

### Neural Tangent Kernel (NTK)
- **Infinite-Width Neural Networks**:
  - Limit as width → ∞ yields deterministic kernel
  - Training dynamics linearize
  - Convergence to global minimum (under conditions)
  
- **NTK Definition**:
  - Θ(x,x') = ⟨∂f(x;θ₀)/∂θ, ∂f(x';θ₀)/∂θ⟩
  - Characterizes network at initialization
  - Architecture-dependent kernel
  
- **Theoretical Insights**:
  - Connects deep learning to kernel methods
  - Generalization bounds via RKHS theory
  - Lazy training regime
  - Feature learning vs kernel regime
  
- **Practical Implications**:
  - Understanding neural network behavior
  - Architecture design principles
  - Initialization strategies
  - Width-depth tradeoffs

### Kernel Ridge Regression (KRR)
- **Formulation**:
  - min_f Σᵢ(yᵢ - f(xᵢ))² + λ‖f‖²_H
  - Solution: f(x) = Σᵢαᵢ K(x,xᵢ)
  - α = (K + λI)⁻¹y
  
- **Properties**:
  - Closed-form solution
  - Regularization prevents overfitting
  - Kernel trick for nonlinearity
  - Equivalent to Gaussian Process regression (with Gaussian likelihood)

### Multiple Kernel Learning (MKL)
- **Combining Kernels**:
  - K = Σᵢwᵢ Kᵢ (linear combination)
  - Learn weights wᵢ along with model
  - Captures different aspects/scales
  
- **Applications**:
  - Multi-modal data fusion
  - Multi-task learning
  - Feature selection via kernel weights
  - Interpretability

### Other Applications
- **Kernel Mean Embedding**:
  - Embed distributions in RKHS
  - Maximum Mean Discrepancy (MMD)
  - Two-sample testing
  - Generative models
  
- **Structured Prediction**:
  - Output kernels for structured outputs
  - Max-margin structured prediction
  - Conditional random fields
  
- **Transfer Learning**:
  - Domain adaptation via kernel methods
  - Optimal transport kernels

## 📊 Topics Covered

### Theoretical Foundations
- Positive definite kernels
- RKHS theory and construction
- Representer theorem (proof and applications)
- Mercer's theorem and eigenexpansion
- Kernel composition and closure properties
- Universality and characteristic kernels
- Regularization in RKHS

### Kernel Design
- Standard kernels (linear, polynomial, RBF)
- Stationary and isotropic kernels
- Domain-specific kernels
- Composite kernels
- Learned kernels
- Data-dependent kernels
- String and graph kernels

### Computational Methods
- Kernel matrix computation
- Eigendecomposition techniques
- Nyström approximation
- Random Fourier features
- Sketching methods
- Sparse approximations
- Low-rank factorization

### Advanced Topics
- Kernel embeddings of distributions
- Operator-valued kernels
- Vector-valued RKHS
- Kernel two-sample tests
- Kernel independence tests
- Conditional kernels
- Deep kernel learning

## 💻 What's Included

- **Comprehensive Theory**: RKHS foundations and kernel properties
- **Visual Intuition**: Feature space transformations and decision boundaries
- **Python Implementations**: scikit-learn, GPy, JAX code for:
  - Computing various kernels
  - Kernel matrix operations
  - SVM with multiple kernels
  - Gaussian Process regression
  - Kernel PCA
  - Neural Tangent Kernel computation
  - Multiple kernel learning
  - Random Fourier features
  - Kernel mean embeddings
- **Interactive Examples**: Visualizing kernels and feature spaces
- **ML Applications**: Classification, regression, clustering with kernels
- **Exercises**: Theory, proofs, and implementations
- **Benchmark Datasets**: Comparisons across methods

## 🎯 Learning Outcomes

By the end of this section, you will be able to:

✅ Understand positive definite kernels and RKHS theory  
✅ Apply the kernel trick to linear methods  
✅ Implement Support Vector Machines with various kernels  
✅ Use Gaussian Processes for regression with uncertainty  
✅ Perform kernel PCA and spectral clustering  
✅ Compute and analyze Neural Tangent Kernels  
✅ Design appropriate kernels for different problems  
✅ Apply representer theorem for optimization  
✅ Use kernel approximation techniques for scalability  
✅ Understand deep learning through kernel lens  

## 📐 Mathematical Summary

### Key Definitions

**Positive Definite Kernel:**
```
K : X × X → ℝ is positive definite if:
1. K(x,y) = K(y,x)  (symmetric)
2. ΣᵢⱼcᵢcⱼK(xᵢ,xⱼ) ≥ 0  for all n, {xᵢ}, {cᵢ}
```

**Reproducing Property:**
```
For RKHS H with kernel K:

f(x) = ⟨f, K(x,·)⟩_H  for all f ∈ H

K(x,y) = ⟨K(x,·), K(y,·)⟩_H
```

**Representer Theorem:**
```
min_{f∈H} Σᵢℓ(yᵢ, f(xᵢ)) + λ‖f‖²_H

Solution: f*(x) = Σᵢαᵢ K(x,xᵢ)

Reduces to finite-dimensional: min_α L(α)
```

**Common Kernels:**
```
Linear:      K(x,y) = ⟨x,y⟩
Polynomial:  K(x,y) = (⟨x,y⟩ + c)^d
RBF:         K(x,y) = exp(-‖x-y‖²/(2σ²))
Matérn:      K(x,y) = (2^(1-ν)/Γ(ν))(√(2ν)r)^ν K_ν(√(2ν)r)
Laplacian:   K(x,y) = exp(-‖x-y‖/σ)
```

**SVM Decision Function:**
```
f(x) = sign(Σᵢαᵢyᵢ K(x,xᵢ) + b)

Where α solves:
max_α Σᵢαᵢ - ½ΣᵢⱼαᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
s.t. Σᵢαᵢyᵢ = 0, 0 ≤ αᵢ ≤ C
```

**Gaussian Process:**
```
Prior: f ~ GP(μ, K)

Posterior given data (X,y):
f(x*) | X,y,x* ~ N(μ*, σ²*)

μ* = K(x*,X)(K(X,X) + σ²I)⁻¹y
σ²* = K(x*,x*) - K(x*,X)(K(X,X) + σ²I)⁻¹K(X,x*)
```

**Neural Tangent Kernel:**
```
Θ(x,x') = lim_{width→∞} ⟨∂f(x;θ)/∂θ, ∂f(x';θ)/∂θ⟩

Training dynamics:
df/dt = -∇_f L = -Θ(y - f)
```

## 🔬 Example Problems

### Problem 1: Kernel SVM
For XOR problem (non-linearly separable):
- Visualize data and linear decision boundary failure
- Apply RBF kernel SVM
- Tune hyperparameters (C, σ)
- Visualize decision boundary in input space
- Identify support vectors

### Problem 2: Gaussian Process Regression
Given sparse noisy observations:
- Fit GP with different kernels
- Predict with uncertainty estimates
- Compare Matérn, RBF, periodic kernels
- Optimize hyperparameters (marginal likelihood)
- Visualize predictive distribution

### Problem 3: Kernel PCA
For nonlinear manifold (e.g., Swiss roll):
- Apply kernel PCA with RBF kernel
- Compare with linear PCA
- Visualize in reduced dimensions
- Analyze eigenvalue spectrum
- Reconstruct data (pre-image problem)

### Problem 4: Neural Tangent Kernel
For a simple neural network:
- Implement NTK computation
- Compare finite vs infinite width
- Analyze training dynamics
- Compare NTK regression vs neural network
- Study effect of depth on NTK

## 📚 Prerequisites

- Linear algebra fundamentals
- Hilbert spaces and inner products
- Optimization theory
- Probability and statistics (for GPs)
- Familiarity with SVMs (helpful)
- Neural networks (for NTK)

## 🚀 Next Steps

After mastering kernel methods, explore:

1. **Deep Kernel Learning** - Combining kernels with deep learning
2. **Gaussian Process Variations** - Sparse GPs, deep GPs, neural processes
3. **Operator Learning** - Neural operators and PDE solving
4. **Optimal Transport** - Wasserstein distances and kernels
5. **Causal Inference** - Kernel methods for causality
6. **Approximate Inference** - Scalable kernel methods

## 🎓 Why This Matters for ML/AI

Understanding kernel methods is crucial because:

- **Kernel trick enables nonlinear learning** - With computational efficiency
- **RKHS provides theoretical foundation** - Generalization bounds, representer theorem
- **SVMs remain competitive** - Especially for small/medium datasets
- **Gaussian Processes offer uncertainty** - Critical for decision-making
- **NTK connects to deep learning** - Understanding neural networks theoretically
- **Universal approximation without explicit features** - Powerful abstraction
- **Scalable via approximations** - Random features, Nyström methods
- **Applicable to structured data** - Graphs, strings, images

Kernel methods bridge classical ML and modern deep learning!

## 📖 Recommended Reading

- *Learning with Kernels* by Schölkopf & Smola
- *Gaussian Processes for Machine Learning* by Rasmussen & Williams
- *An Introduction to RKHS* by Berlinet & Thomas-Agnan
- *Kernel Methods for Pattern Analysis* by Shawe-Taylor & Cristianini
- *Neural Tangent Kernel* by Jacot et al. (NeurIPS 2018)
- *Deep Learning Book* by Goodfellow et al. (Chapter 5.7)
- *Elements of Statistical Learning* by Hastie et al. (Chapters 5, 6, 14)

## 🔗 Related Topics

- [Hilbert Spaces](../) - Foundation for RKHS
- [Normed Spaces](../../) - General norm theory
- [Linear Operators](../../../linear-operators/) - Kernel as integral operator
- [SVD](../../../matrices/svd/) - Kernel matrix decomposition
- [Spectral Methods](../../../matrices/spectral-methods/) - Eigendecomposition applications
