# Linear Algebra Foundations for ML/AI

Welcome to the **Linear Algebra Foundations** repository! This comprehensive guide covers the essential linear algebra concepts that every Machine Learning and AI practitioner must know.

## 📚 Contents

This repository is organized into core topics with logical hierarchies:

### Core Topics

### 1. [Vectors](./vectors/)
- Represent data points, parameters, and embeddings
- Key concepts: dot product, norm, projection, orthogonality
- ML applications: cosine similarity for embeddings (e.g., word vectors)

### 2. [Matrices](./matrices/)
- Represent linear transformations, datasets, and neural network weight layers
- Key operations: multiplication, inverse, transpose, determinant
- ML applications: weight matrices in linear/fully connected layers
  
  **Subtopics:**
  - **[Eigenvalues & Eigenvectors](./matrices/eigenvalues-eigenvectors/)**: Solve Av = λv; directions preserved under transformation
  - **[Singular Value Decomposition (SVD)](./matrices/svd/)**: Factorization A = UΣVᵀ for low-rank approximation
  - **[Spectral Methods & Decomposition](./matrices/spectral-methods/)**: Eigenstructure analysis for PCA, spectral clustering, GNNs

### 3. [Subspaces & Fundamental Theorem](./subspaces-fundamental-theorem/)
- Understand the four fundamental subspaces: null space, column space, row space, left null space
- Master the rank-nullity theorem: nullity + rank = dimension
- Key concepts: kernel, image, orthogonal complements, solvability
- ML applications: feature redundancy, information loss in neural networks, dimensionality analysis

### 4. [Tensor Decompositions](./tensor-decompositions/)
- Generalization of matrices to multi-dimensional arrays
- Examples: CP decomposition, Tucker decomposition
- ML applications: compressing deep networks, multi-modal data representation

### Advanced Topics

### 5. [Normed Spaces](./normed-spaces/)
- Generalization of vector length with L1, L2, and Lp norms
- Understanding norm geometry and properties
- ML applications: Regularization (L1/Lasso, L2/Ridge), loss functions, distance metrics
  
  **Subtopics:**
  - **[Hilbert Spaces](./normed-spaces/hilbert-spaces/)**: Complete inner product spaces with perfect geometry
    - **[Kernel Methods & RKHS](./normed-spaces/hilbert-spaces/kernel-methods-rkhs/)**: SVMs, Gaussian Processes, Neural Tangent Kernel
  - **[Banach Spaces](./normed-spaces/banach-spaces/)**: Complete normed spaces for optimization and neural operators

### 6. [Linear Operators](./linear-operators/)
- Maps between vector or function spaces
- Bounded operators, adjoints, and special classes
- ML applications: Convolution (CNNs), attention mechanisms (transformers), differential operators

## 🎯 Learning Path

Each topic folder contains:
- Detailed explanations and theory
- Practical examples and code implementations
- Exercises and projects
- Real-world ML/AI applications

## 🚀 Getting Started

Navigate to any topic folder to begin learning. It's recommended to follow the order listed above, as concepts build upon each other.

## 📖 Prerequisites

- Basic understanding of mathematics
- Familiarity with Python (for code examples)
- Interest in Machine Learning and AI

## 🤝 Contributing

Contributions are welcome! Feel free to submit pull requests or open issues for improvements.

## 📝 License

This repository is for educational purposes.
