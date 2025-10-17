# Kernel (Null Space) Applications in Machine Learning

## 🎯 Overview

In ML, the **kernel** (null space) reveals crucial information about:
1. **Invariances** - What changes don't affect the output
2. **Redundancies** - Which features are unnecessary
3. **Constraints** - What transformations preserve structure
4. **Ambiguities** - Multiple inputs producing same output

---

## 1. 🧬 Feature Engineering & Selection

### A. Detecting Multicollinearity

**Problem**: Correlated features cause instability in regression models.

```python
import numpy as np
from fundamental_subspaces import FundamentalSubspaces

# Feature matrix with correlated features
X = np.array([
    [1, 2, 3, 5],   # sample 1: age, income, debt, score
    [2, 4, 6, 10],  # sample 2
    [3, 6, 9, 15],  # sample 3
    [4, 8, 12, 20]  # sample 4
])

# Transpose: we want relationships between features
fs = FundamentalSubspaces(X.T)

print(f"Number of features: {X.shape[1]}")
print(f"Effective features: {fs.rank}")
print(f"Redundant dimensions: {fs.nullity}")

if fs.nullity > 0:
    null_basis = fs.null_space()
    print("\nLinear dependencies detected:")
    for i, vec in enumerate(null_basis.T):
        print(f"Dependency {i+1}: {vec}")
        # Shows: [a, b, c, d] means a*f1 + b*f2 + c*f3 + d*f4 ≈ 0
```

**Application**: Remove redundant features before training to improve:
- Model stability
- Training speed
- Interpretability
- Regularization effectiveness

### B. Feature Compression

**Problem**: High-dimensional data is expensive to process.

```python
def compress_features(X, target_rank):
    """Keep only informative feature combinations."""
    fs = FundamentalSubspaces(X.T)
    
    # Project onto row space (effective features)
    U, s, Vt = np.linalg.svd(X.T, full_matrices=False)
    
    # Keep top target_rank singular values
    X_compressed = X @ Vt[:target_rank].T
    
    print(f"Original: {X.shape[1]} features")
    print(f"Compressed: {target_rank} features")
    print(f"Information retained: {np.sum(s[:target_rank]**2) / np.sum(s**2):.2%}")
    
    return X_compressed

# Example: 1000 features → 50 effective features
# X_train_compressed = compress_features(X_train, target_rank=50)
```

---

## 2. 🧠 Neural Networks

### A. Weight Matrix Analysis

**Problem**: Understanding information flow through layers.

```python
class LayerAnalyzer:
    """Analyze information loss in neural network layers."""
    
    def __init__(self, weight_matrix):
        self.W = weight_matrix
        self.fs = FundamentalSubspaces(weight_matrix)
    
    def analyze_layer(self):
        """Compute information flow statistics."""
        input_dim = self.W.shape[1]
        output_dim = self.W.shape[0]
        effective_dim = self.fs.rank
        info_loss_dim = self.fs.nullity
        
        return {
            'input_dimension': input_dim,
            'output_dimension': output_dim,
            'effective_dimension': effective_dim,
            'information_loss_dimension': info_loss_dim,
            'information_preservation_ratio': effective_dim / input_dim,
            'bottleneck': effective_dim < min(input_dim, output_dim)
        }
    
    def get_invariant_directions(self):
        """Get input directions that produce zero output."""
        return self.fs.null_space()
    
    def get_output_span(self):
        """Get achievable output subspace."""
        return self.fs.column_space()

# Example: Analyze a trained layer
# W = model.layer1.weight.detach().numpy()
# analyzer = LayerAnalyzer(W)
# stats = analyzer.analyze_layer()
# print(f"This layer loses {stats['information_loss_dimension']} dimensions")
```

### B. Gradient Flow Diagnosis

**Problem**: Vanishing gradients in deep networks.

```python
def diagnose_gradient_flow(gradients_list):
    """
    Check if gradients are flowing through the network.
    
    gradients_list: List of gradient matrices for each layer
    """
    for i, grad in enumerate(gradients_list):
        fs = FundamentalSubspaces(grad)
        
        print(f"\nLayer {i}:")
        print(f"  Gradient rank: {fs.rank}/{min(grad.shape)}")
        print(f"  Null space dim: {fs.nullity}")
        
        if fs.nullity > grad.shape[1] * 0.5:
            print(f"  ⚠️ WARNING: Over 50% of gradient directions are zero!")
            print(f"  → Consider: BatchNorm, ResNet, or reduce depth")
        
        # Check singular values
        U, s, Vt = np.linalg.svd(grad, full_matrices=False)
        condition_number = s[0] / s[-1] if s[-1] > 1e-10 else np.inf
        
        if condition_number > 1e6:
            print(f"  ⚠️ WARNING: Ill-conditioned gradient (κ = {condition_number:.2e})")
```

### C. Autoencoder Bottleneck Analysis

**Problem**: Is the bottleneck actually learning a compressed representation?

```python
def analyze_autoencoder_bottleneck(encoder_weights, decoder_weights):
    """
    Check if autoencoder bottleneck is using all dimensions.
    """
    # Analyze encoder
    fs_enc = FundamentalSubspaces(encoder_weights)
    
    # Analyze decoder
    fs_dec = FundamentalSubspaces(decoder_weights)
    
    bottleneck_dim = encoder_weights.shape[0]
    
    print("Encoder Analysis:")
    print(f"  Bottleneck dimension: {bottleneck_dim}")
    print(f"  Effective dimension: {fs_enc.rank}")
    print(f"  Wasted dimensions: {fs_enc.nullity}")
    
    print("\nDecoder Analysis:")
    print(f"  Uses {fs_dec.rank}/{bottleneck_dim} dimensions from bottleneck")
    
    # Check if encoder null space overlaps with decoder row space
    enc_null = fs_enc.null_space()
    dec_row = fs_dec.row_space()
    
    # These should be approximately orthogonal for efficient encoding
    overlap = np.linalg.norm(enc_null.T @ dec_row)
    print(f"\nEncoder-Decoder coupling: {overlap:.6f}")
    if overlap > 0.1:
        print("⚠️ Encoder null space overlaps with decoder inputs")
        print("→ Bottleneck could be more efficient!")
```

---

## 3. 📊 Dimensionality Reduction

### A. PCA with Explicit Null Space

**Problem**: Understanding what information PCA discards.

```python
class PCAWithKernel:
    """PCA that explicitly tracks discarded information."""
    
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit_transform(self, X):
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance
        cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
        
        # SVD of covariance
        fs = FundamentalSubspaces(cov)
        U, s, Vt = np.linalg.svd(cov)
        
        # Principal components (what we keep)
        self.components_ = Vt[:self.n_components]
        
        # Null components (what we discard)
        self.null_components_ = Vt[self.n_components:]
        
        # Variance explained
        total_var = np.sum(s)
        kept_var = np.sum(s[:self.n_components])
        lost_var = np.sum(s[self.n_components:])
        
        print(f"Kept variance: {kept_var/total_var:.2%}")
        print(f"Lost variance: {lost_var/total_var:.2%}")
        print(f"Null space dimension: {len(self.null_components_)}")
        
        # Transform
        return X_centered @ self.components_.T
    
    def get_reconstruction_error(self, X):
        """Project onto null space to see what's lost."""
        X_centered = X - self.mean_
        
        # Project onto null space
        null_projection = X_centered @ self.null_components_.T @ self.null_components_
        
        return np.linalg.norm(null_projection, axis=1)

# Usage:
# pca = PCAWithKernel(n_components=50)
# X_reduced = pca.fit_transform(X)
# errors = pca.get_reconstruction_error(X)  # What each sample lost
```

---

## 4. 🎯 Regularization & Constraints

### A. Understanding L2 Regularization Effect

**Problem**: How does regularization affect the null space?

```python
def analyze_regularization_effect(X, y, lambda_values):
    """
    Show how regularization shrinks the null space.
    """
    print("Effect of L2 regularization on solution space:\n")
    
    for lam in lambda_values:
        # Ridge regression: (X^T X + λI)^-1 X^T y
        XtX = X.T @ X
        regularized = XtX + lam * np.eye(XtX.shape[0])
        
        fs_original = FundamentalSubspaces(XtX)
        fs_regularized = FundamentalSubspaces(regularized)
        
        print(f"λ = {lam}:")
        print(f"  Original null space dim: {fs_original.nullity}")
        print(f"  Regularized null space dim: {fs_regularized.nullity}")
        print(f"  Rank improvement: {fs_regularized.rank - fs_original.rank}")
        print()

# Example:
# analyze_regularization_effect(X_train, y_train, [0, 0.1, 1, 10])
```

### B. Constrained Optimization

**Problem**: Finding solutions satisfying linear constraints.

```python
def constrained_solution(A, b, C, d):
    """
    Solve: minimize ||Ax - b||²
    Subject to: Cx = d
    
    Uses null space of C to parameterize feasible solutions.
    """
    fs_C = FundamentalSubspaces(C)
    
    # Step 1: Find particular solution to Cx = d
    x_particular = np.linalg.lstsq(C, d, rcond=None)[0]
    
    # Step 2: Get null space of C (feasible directions)
    N_C = fs_C.null_space()
    
    print(f"Constraint dimension: {C.shape[0]}")
    print(f"Degrees of freedom: {fs_C.nullity}")
    
    # Step 3: Solve reduced problem
    # x = x_particular + N_C @ z
    # min ||A(x_particular + N_C @ z) - b||²
    
    A_reduced = A @ N_C
    b_reduced = b - A @ x_particular
    
    z_opt = np.linalg.lstsq(A_reduced, b_reduced, rcond=None)[0]
    x_opt = x_particular + N_C @ z_opt
    
    # Verify constraints
    constraint_error = np.linalg.norm(C @ x_opt - d)
    print(f"Constraint satisfaction error: {constraint_error:.2e}")
    
    return x_opt
```

---

## 5. 🔍 Anomaly Detection

### A. Null Space Anomaly Detection

**Problem**: Detect when data doesn't follow expected patterns.

```python
class NullSpaceAnomalyDetector:
    """
    Detect anomalies by checking if they lie in the null space.
    Normal data should NOT be in the null space.
    """
    
    def __init__(self):
        self.null_space = None
        self.threshold = None
    
    def fit(self, X_normal):
        """Learn null space from normal data."""
        # Compute transformation that normal data goes through
        # For example, X @ W where W is learned weights
        
        # Here we use covariance as example
        cov = np.cov(X_normal.T)
        fs = FundamentalSubspaces(cov)
        
        # Get null space (directions with zero variance in normal data)
        self.null_space = fs.null_space()
        
        # Compute threshold
        projections = self._project_to_null(X_normal)
        self.threshold = np.percentile(projections, 95)
        
        print(f"Null space dimension: {self.null_space.shape[1] if self.null_space.size > 0 else 0}")
        print(f"Detection threshold: {self.threshold:.6f}")
    
    def _project_to_null(self, X):
        """Project samples onto null space."""
        if self.null_space.size == 0:
            return np.zeros(X.shape[0])
        
        # Projection magnitude indicates "nullness"
        projection = X @ self.null_space @ self.null_space.T
        return np.linalg.norm(projection, axis=1)
    
    def predict(self, X):
        """Detect anomalies."""
        projections = self._project_to_null(X)
        
        # Anomalies have significant null space component
        is_anomaly = projections > self.threshold
        
        return is_anomaly, projections

# Usage:
# detector = NullSpaceAnomalyDetector()
# detector.fit(X_train_normal)
# anomalies, scores = detector.predict(X_test)
```

---

## 6. 🎨 Generative Models

### A. GAN Mode Collapse Detection

**Problem**: Is the generator producing diverse outputs?

```python
def detect_mode_collapse(generated_samples, n_bins=50):
    """
    Use null space to detect if generator is stuck in low-dimensional manifold.
    """
    # Compute covariance of generated samples
    cov = np.cov(generated_samples.T)
    fs = FundamentalSubspaces(cov)
    
    effective_dim = fs.rank
    total_dim = generated_samples.shape[1]
    
    print(f"Generated sample dimension: {total_dim}")
    print(f"Effective dimension: {effective_dim}")
    print(f"Null dimension (collapsed modes): {fs.nullity}")
    
    # Analyze singular value spectrum
    U, s, Vt = np.linalg.svd(cov)
    
    # Mode collapse indicator: many near-zero singular values
    s_normalized = s / np.sum(s)
    entropy = -np.sum(s_normalized * np.log(s_normalized + 1e-10))
    max_entropy = np.log(len(s))
    
    print(f"\nSpectrum entropy: {entropy:.2f} / {max_entropy:.2f}")
    print(f"Entropy ratio: {entropy/max_entropy:.2%}")
    
    if entropy/max_entropy < 0.5:
        print("⚠️ WARNING: Possible mode collapse detected!")
        print("→ Generator may be producing low-diversity samples")
        
        # Show which directions are collapsed
        null_basis = fs.null_space()
        if null_basis.size > 0:
            print(f"\nCollapsed directions (null space basis):")
            print(null_basis[:5])  # Show first 5 vectors
```

### B. Variational Autoencoder (VAE) Posterior Collapse

**Problem**: Is the VAE using the latent space?

```python
def diagnose_vae_posterior_collapse(z_samples, reconstruction_weights):
    """
    Check if VAE is ignoring latent variables.
    """
    # Analyze latent code usage
    fs_z = FundamentalSubspaces(np.cov(z_samples.T))
    
    print("Latent Space Analysis:")
    print(f"  Latent dimension: {z_samples.shape[1]}")
    print(f"  Active latent dimensions: {fs_z.rank}")
    print(f"  Inactive dimensions: {fs_z.nullity}")
    
    # Check if decoder uses all latent dimensions
    fs_dec = FundamentalSubspaces(reconstruction_weights)
    
    print("\nDecoder Analysis:")
    print(f"  Decoder uses {fs_dec.rank} latent dimensions")
    
    # Posterior collapse indicator
    if fs_z.nullity > z_samples.shape[1] * 0.3:
        print("\n⚠️ WARNING: Posterior collapse detected!")
        print("→ Over 30% of latent dimensions unused")
        print("→ Try: KL annealing, stronger decoder, β-VAE")
```

---

## 7. 🎓 Interpretability

### A. Feature Importance via Null Space

**Problem**: Which features actually matter for predictions?

```python
def feature_importance_via_kernel(model_weights, feature_names):
    """
    Features in null space don't affect output → low importance.
    """
    fs = FundamentalSubspaces(model_weights)
    
    # Get null space
    null_basis = fs.null_space()
    
    if null_basis.size == 0:
        print("All features are important (no null space)")
        return
    
    # Compute each feature's null space contribution
    importance_scores = np.zeros(len(feature_names))
    
    for i in range(len(feature_names)):
        # How much does feature i contribute to null space?
        feature_vec = np.zeros(len(feature_names))
        feature_vec[i] = 1.0
        
        null_projection = null_basis @ (null_basis.T @ feature_vec)
        importance_scores[i] = 1.0 - np.linalg.norm(null_projection)
    
    # Normalize
    importance_scores = importance_scores / np.sum(importance_scores)
    
    # Sort and display
    sorted_indices = np.argsort(importance_scores)[::-1]
    
    print("\nFeature Importance Ranking:")
    for i, idx in enumerate(sorted_indices[:10]):
        print(f"{i+1}. {feature_names[idx]}: {importance_scores[idx]:.4f}")
```

---

## 🔑 Key Takeaways

1. **Null Space = Invariance**: Directions in null space don't affect output
2. **Rank = Information**: Effective dimensions used by transformation
3. **Multicollinearity**: Null space reveals redundant features
4. **Bottlenecks**: Null space dimension = information loss
5. **Regularization**: Shrinks null space to ensure unique solutions
6. **Anomalies**: Can be detected by unusual null space projections
7. **Mode Collapse**: Large null space indicates low diversity
8. **Interpretability**: Null space features have zero importance

---

## 📚 Further Reading

- **PCA**: Null space = discarded principal components
- **SVD**: Provides optimal bases for null space and column space
- **Kernel Methods**: Different concept - kernel functions in SVM
- **Null Space Pursuit**: Sparse coding technique
- **Grassmann Manifolds**: Geometry of subspaces for ML

---

**Remember**: Every time you see low rank or singular matrices in ML, think about what the null space represents - it's often the key to understanding what the model is (or isn't) learning!
