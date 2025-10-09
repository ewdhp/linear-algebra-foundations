# Singular Value Decomposition (SVD)

## 📖 What You'll Learn

SVD is one of the most important matrix factorizations in ML/AI, generalizing eigendecomposition to any matrix:

### Core Concepts

1. **SVD Factorization**
   - Understanding A = UΣVᵀ
   - Left singular vectors (U): output space orthonormal basis
   - Singular values (Σ): importance/scaling factors (diagonal)
   - Right singular vectors (Vᵀ): input space orthonormal basis

2. **Geometric Interpretation**
   - SVD as a sequence of transformations: rotation → scaling → rotation
   - How any linear transformation can be decomposed
   - Orthogonal bases for domain and codomain
   - Principal directions of stretching

3. **Computing SVD**
   - Relationship to eigendecomposition (AᵀA and AAᵀ)
   - Numerical algorithms for SVD
   - Computational complexity
   - Thin vs. full SVD

4. **Singular Values**
   - Ordering (largest to smallest)
   - Relationship to matrix rank
   - Condition number and numerical stability
   - Energy/information content

5. **Matrix Approximation**
   - Truncated SVD
   - Eckart-Young theorem (best low-rank approximation)
   - Reconstruction error
   - Choosing number of components

## 🤖 Machine Learning Applications

### Low-Rank Approximation
- **Model Compression**: Reducing neural network size while preserving performance
- **Image Compression**: Representing images with fewer parameters
- **Data Denoising**: Filtering noise by removing small singular values
- **Memory Efficiency**: Storing large matrices compactly

### Latent Factor Models
- **Collaborative Filtering**: Matrix completion for recommendation systems (Netflix Prize)
- **Topic Modeling**: Latent Semantic Analysis (LSA)
- **Feature Extraction**: Finding latent representations
- **Missing Data Imputation**: Reconstructing incomplete matrices

### Other Key Applications
- **Natural Language Processing**: Document-term matrices, semantic similarity
- **Computer Vision**: Face recognition (Eigenfaces), image processing
- **Recommender Systems**: User-item interaction matrices
- **Genomics**: Gene expression analysis
- **Audio Processing**: Signal separation and compression
- **Deep Learning**: Weight initialization, regularization, analysis

## 📊 Topics Covered

- Full SVD decomposition
- Compact/thin SVD
- Truncated SVD for approximation
- Relationship between SVD and eigendecomposition
- Moore-Penrose pseudoinverse
- Matrix rank and nullspace
- Frobenius norm and optimal approximation
- Randomized SVD for large matrices
- Incremental SVD
- Applications to least squares problems

## 💻 What's Included

- Comprehensive theoretical foundations
- Visual intuition and animations
- NumPy and SciPy implementations
- Image compression demos
- Recommendation system from scratch
- Latent Semantic Analysis example
- Model compression techniques
- Comparative analysis with PCA
- Real-world datasets and applications

## 🎯 Learning Outcomes

By the end of this section, you will be able to:
- Compute and interpret SVD decomposition
- Implement matrix compression algorithms
- Build recommendation systems using SVD
- Apply SVD to dimensionality reduction problems
- Compress neural network models
- Understand when to use SVD vs. eigendecomposition
- Optimize SVD computations for large datasets
- Analyze and debug matrix-based algorithms

## 📚 Prerequisites

- Strong understanding of matrices
- Knowledge of eigenvalues and eigenvectors
- Matrix norms and properties
- Python with NumPy/SciPy

## 🚀 Next Steps

After mastering SVD, you'll explore **Tensor Decompositions**, which extend these matrix concepts to higher-dimensional data structures commonly found in modern deep learning architectures.
