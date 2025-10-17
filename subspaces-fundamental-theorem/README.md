# Subspaces & Fundamental Theorem

## 📖 What You'll Learn

Understanding subspaces and the fundamental theorem of linear algebra is crucial for mastering how linear transformations work and applying them effectively in ML/AI. This section covers the deep structural properties of matrices and linear transformations.

### Core Concepts

1. **Vector Subspaces**
   - Definition: A subset of a vector space closed under addition and scalar multiplication
   - Examples: lines through origin, planes through origin, solution sets
   - Span and linear combinations
   - Basis and dimension of subspaces

2. **Kernel (Null Space)**
   - **Definition**: Kernel(A) = Null(A) = {x : Ax = 0}
   - All vectors that map to the zero vector
   - Finding the null space using Gaussian elimination
   - Basis for the null space
   - **Geometric Interpretation**: Directions that get "crushed" to zero
   - **Dimension**: nullity(A) = dim(Null(A))

3. **Image (Column Space / Range)**
   - **Definition**: Image(A) = Col(A) = Range(A) = {Ax : x ∈ ℝⁿ}
   - All possible output vectors of the transformation
   - Span of column vectors
   - Basis for the column space
   - **Geometric Interpretation**: The subspace that A maps onto
   - **Dimension**: rank(A) = dim(Col(A))

4. **Row Space**
   - **Definition**: Row(A) = Col(Aᵀ) = span of row vectors
   - Orthogonal complement of the null space
   - Row rank = column rank (rank theorem)
   - Finding basis using row reduction

5. **Left Null Space**
   - **Definition**: Null(Aᵀ) = {y : Aᵀy = 0}
   - Orthogonal complement of the column space
   - Connections to solvability of Ax = b

6. **The Rank-Nullity Theorem**
   - **Fundamental Theorem**: dim(Null(A)) + dim(Col(A)) = n (number of columns)
   - Also written as: nullity(A) + rank(A) = n
   - **Interpretation**: Input dimension = dimension lost + dimension preserved
   - Conservation of dimension principle

7. **Four Fundamental Subspaces**
   - For an m×n matrix A:
     1. **Column Space**: Col(A) ⊂ ℝᵐ, dimension = rank(A)
     2. **Null Space**: Null(A) ⊂ ℝⁿ, dimension = nullity(A) = n - rank(A)
     3. **Row Space**: Row(A) = Col(Aᵀ) ⊂ ℝⁿ, dimension = rank(A)
     4. **Left Null Space**: Null(Aᵀ) ⊂ ℝᵐ, dimension = m - rank(A)
   - **Orthogonality Relations**:
     - Row(A) ⊥ Null(A)
     - Col(A) ⊥ Null(Aᵀ)

8. **Orthogonal Complements**
   - Definition and properties
   - Decomposition: ℝⁿ = Row(A) ⊕ Null(A)
   - Applications to projection and least squares

## 🤖 Machine Learning Applications

### Feature Selection & Redundancy Detection
- **Identifying Redundant Features**: Features in the null space don't contribute to output
- **Feature Engineering**: Understanding which combinations of features are linearly dependent
- **Dimensionality Analysis**: Using rank to determine effective feature count
- **Multicollinearity Detection**: Finding nearly-dependent features in regression

### Linear Regression & Underdetermined Systems
- **Normal Equations**: Understanding when Aᵀ Ax = Aᵀ b has solutions
- **Ridge Regression**: Adding regularization when null space is non-trivial
- **Least Squares**: Column space determines which b can be fit exactly
- **Minimum Norm Solutions**: Finding solutions in the row space
- **Underdetermined Systems**: When nullity > 0, infinite solutions exist

### Neural Networks & Information Loss
- **Layer Compression**: Null space reveals information lost through layers
- **Bottleneck Analysis**: Rank determines maximum information flow
- **Gradient Flow**: Understanding vanishing gradients through null space analysis
- **Weight Initialization**: Avoiding singular weight matrices
- **Network Pruning**: Identifying redundant neurons/connections

### Dimensionality Reduction
- **PCA Foundation**: Null space of centered data covariance
- **Rank Reduction**: Projecting data onto column space
- **Data Compression**: Removing null space components
- **Latent Space Models**: Image/range as latent representations

### System Solvability
- **Consistent Systems**: b ∈ Col(A) determines if Ax = b has solutions
- **Homogeneous Systems**: Null space gives all solutions to Ax = 0
- **General Solutions**: Particular solution + null space basis
- **Constraint Satisfaction**: Feasibility analysis in optimization

### Other Key Applications
- **Recommender Systems**: Understanding missing data patterns
- **Image Processing**: Kernel methods and null space filtering
- **Graph Neural Networks**: Analyzing graph Laplacian null space
- **Control Theory**: Controllability and observability analysis
- **Signal Processing**: Identifying signal subspace vs noise subspace

## 📊 Topics Covered

### Theoretical Foundations
- Vector space axioms and subspace tests
- Linear independence and dependence
- Basis and dimension
- Span and generating sets
- Direct sum decompositions

### Computational Methods
- Gaussian elimination for null space
- Row reduction for row space
- QR decomposition for column space
- Rank computation algorithms
- Orthogonal projection methods

### Advanced Concepts
- Fundamental theorem of linear algebra (complete statement)
- Fredholm alternative theorem
- Pseudoinverse and generalized inverses
- Quotient spaces
- Invariant subspaces

### Geometric Intuition
- Visualizing subspaces in 2D/3D
- Transformations and their effect on subspaces
- Orthogonal decompositions
- Projection onto subspaces

## 💻 What's Included

- **Comprehensive Theory**: Mathematical definitions, theorems, and proofs
- **Visual Intuition**: 2D/3D visualizations of subspaces and transformations
- **Python Implementations**: NumPy and SciPy code for:
  - Computing null space (using SVD, QR, or row reduction)
  - Finding column space basis
  - Computing rank and nullity
  - Verifying orthogonality of fundamental subspaces
  - Projecting onto subspaces
- **Interactive Examples**: Step-by-step computations with explanations
- **ML Applications**: Real-world examples with datasets
- **Exercises**: Problems ranging from basic to advanced
- **Proofs and Derivations**: Understanding why theorems work

## 🎯 Learning Outcomes

By the end of this section, you will be able to:

✅ Define and identify all four fundamental subspaces of a matrix  
✅ Compute null space, column space, row space, and left null space  
✅ Apply the rank-nullity theorem to analyze linear transformations  
✅ Determine when linear systems have solutions (solvability conditions)  
✅ Understand information loss in neural network layers  
✅ Identify redundant features in datasets  
✅ Apply subspace concepts to regression and optimization  
✅ Visualize and interpret geometric meaning of subspaces  
✅ Use orthogonal decompositions for projections  
✅ Debug and analyze ML models using linear algebra insights  

## 📐 Mathematical Summary

### Key Equations

**Rank-Nullity Theorem:**
```
nullity(A) + rank(A) = n (number of columns)
```

**Four Fundamental Subspaces:**
```
For A ∈ ℝᵐˣⁿ:

1. Column Space: C(A) ⊂ ℝᵐ, dim = r
2. Null Space:   N(A) ⊂ ℝⁿ, dim = n - r  
3. Row Space:    C(Aᵀ) ⊂ ℝⁿ, dim = r
4. Left Null:    N(Aᵀ) ⊂ ℝᵐ, dim = m - r

where r = rank(A)
```

**Orthogonality:**
```
C(Aᵀ) ⊥ N(A)   (Row space perpendicular to null space)
C(A) ⊥ N(Aᵀ)   (Column space perpendicular to left null space)
```

**Decompositions:**
```
ℝⁿ = C(Aᵀ) ⊕ N(A)   (Domain decomposition)
ℝᵐ = C(A) ⊕ N(Aᵀ)   (Codomain decomposition)
```

## 🔬 Example Problems

### Problem 1: Computing Fundamental Subspaces
Given matrix:
```
A = [1  2  3]
    [2  4  6]
    [3  6  9]
```

Find: null space, column space, row space, rank, and nullity.

### Problem 2: Feature Redundancy
Given feature matrix X with n features, determine:
- Which features are linearly dependent?
- Minimum number of features needed?
- Which features to keep/remove?

### Problem 3: Solvability Analysis
For system Ax = b, determine:
- Under what conditions on b does a solution exist?
- If solvable, is the solution unique?
- If not unique, describe the solution set.

### Problem 4: Neural Network Layer Analysis
Given weight matrix W in a neural layer:
- What is the maximum information preserved?
- What information is lost (null space)?
- How to initialize to avoid singularity?

## 📚 Prerequisites

- Understanding of vectors and vector operations
- Matrix multiplication and basic matrix operations
- Systems of linear equations
- Gaussian elimination / row reduction
- Familiarity with the concepts of span and linear independence

## 🚀 Next Steps

After mastering subspaces and the fundamental theorem, you'll have the foundation to deeply understand:

1. **Eigenvalues & Eigenvectors** - Invariant subspaces and eigenspaces
2. **SVD** - Optimal bases for the four fundamental subspaces
3. **Orthogonal Projections** - Least squares and best approximations
4. **Matrix Factorizations** - LU, QR decompositions
5. **Advanced ML Concepts** - PCA, kernel methods, manifold learning

## 🎓 Why This Matters for ML/AI

Understanding subspaces is essential because:

- **Every linear transformation can be understood through its subspaces**
- **The fundamental theorem connects all matrix properties**
- **ML models rely on understanding what information is preserved vs lost**
- **Feature engineering requires recognizing dependencies and redundancies**
- **Debugging neural networks needs understanding rank deficiency**
- **Optimization algorithms depend on understanding constraint spaces**
- **Dimensionality reduction techniques are built on subspace projections**

This is arguably the most important conceptual foundation in applied linear algebra!

## 📖 Recommended Reading

- *Linear Algebra and Its Applications* by Gilbert Strang (Chapter 3)
- *Introduction to Linear Algebra* by Gilbert Strang (Chapter 4)
- *Deep Learning* by Goodfellow et al. (Chapter 2.3-2.6)
- MIT OpenCourseWare: 18.06 Linear Algebra lectures by Gilbert Strang

## 🔗 Related Topics

- [Vectors](../vectors/) - Foundation for understanding subspaces
- [Matrices](../matrices/) - Linear transformations and operations
- [Eigenvalues & Eigenvectors](../eigenvalues-eigenvectors/) - Eigenspaces as invariant subspaces
- [SVD](../svd/) - Orthonormal bases for fundamental subspaces
