"""
SIGNAL CORRELATION - SIGNAL PROCESSING APPLICATION

Theory:
Signal correlation measures similarity between signals using the dot product:
    Correlation: R = s₁·s₂ = Σ(s₁[i] × s₂[i])
    
Normalized Correlation (Correlation Coefficient):
    ρ = (s₁·s₂) / (|s₁||s₂|) = cos(θ)
    where ρ ∈ [-1, 1]

Interpretation:
- ρ ≈ +1: Signals are highly similar (same phase/pattern)
- ρ ≈ 0: Signals are uncorrelated (orthogonal)
- ρ ≈ -1: Signals are inversely correlated (opposite phase)

Applications:
- Template Matching: find patterns in signals
- Pattern Recognition: identify specific waveforms
- Radar/Sonar: detect echoes and measure distance
- Audio Processing: speech recognition, music analysis
- Communications: signal detection in noise
- Cross-Correlation: find time delays between signals
- Auto-Correlation: detect periodicity and self-similarity
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate time vector
t = np.linspace(0, 4*np.pi, 200)

# Create various signals
signal_ref = np.sin(t)                          # Reference signal
signal_similar = np.sin(t + 0.2)                # Similar (slight phase shift)
signal_opposite = -np.sin(t)                    # Opposite phase
signal_shifted = np.sin(t - np.pi/2)           # 90° phase shift (cosine)
signal_different = np.sin(3*t)                  # Different frequency
signal_noisy = np.sin(t) + 0.3*np.random.randn(len(t))  # Noisy version
signal_square = np.sign(np.sin(t))              # Square wave (different shape)
signal_uncorrelated = np.random.randn(len(t))   # Random noise

def calculate_correlation(signal1, signal2):
    """Calculate correlation metrics between two signals"""
    # Raw correlation (dot product)
    correlation = np.dot(signal1, signal2)
    
    # Magnitudes
    norm1 = np.linalg.norm(signal1)
    norm2 = np.linalg.norm(signal2)
    
    # Normalized correlation (correlation coefficient)
    if norm1 > 0 and norm2 > 0:
        corr_coeff = correlation / (norm1 * norm2)
    else:
        corr_coeff = 0
    
    # Angle between signals
    corr_coeff_clipped = np.clip(corr_coeff, -1, 1)
    angle_deg = np.degrees(np.arccos(corr_coeff_clipped))
    
    # Similarity interpretation
    if corr_coeff > 0.8:
        similarity = "High Positive"
        color = "green"
    elif corr_coeff > 0.3:
        similarity = "Moderate Positive"
        color = "lightgreen"
    elif corr_coeff > -0.3:
        similarity = "Low/Uncorrelated"
        color = "gray"
    elif corr_coeff > -0.8:
        similarity = "Moderate Negative"
        color = "orange"
    else:
        similarity = "High Negative"
        color = "red"
    
    return {
        'correlation': correlation,
        'norm1': norm1,
        'norm2': norm2,
        'corr_coeff': corr_coeff,
        'angle': angle_deg,
        'similarity': similarity,
        'color': color
    }

# Calculate correlations for all test signals
signals = {
    'Similar (phase shift)': signal_similar,
    'Opposite phase': signal_opposite,
    '90° phase shift': signal_shifted,
    'Different frequency': signal_different,
    'Noisy version': signal_noisy,
    'Square wave': signal_square,
    'Random noise': signal_uncorrelated
}

results = {}
for name, signal in signals.items():
    results[name] = calculate_correlation(signal_ref, signal)

print("\n" + "="*100)
print("SIGNAL CORRELATION - DOT PRODUCT APPLICATION IN SIGNAL PROCESSING")
print("="*100)
print("\nFormula: ρ = (s₁·s₂) / (|s₁||s₂|)  where ρ ∈ [-1, 1]")
print("Principle: Dot product measures similarity/alignment between signal vectors")
print("-"*100)

# Create comprehensive table
data = {
    "Signal Pair": ["Reference: sin(t)"] + list(signals.keys()),
    "Description": [
        "s₁ (reference)",
        "s₂: sin(t+0.2)",
        "s₃: -sin(t)",
        "s₄: sin(t-π/2)",
        "s₅: sin(3t)",
        "s₆: sin(t) + noise",
        "s₇: square wave",
        "s₈: random"
    ],
    "s₁·s": ["—"] + [f"{r['correlation']:.3f}" for r in results.values()],
    "Correlation ρ": ["1.000"] + [f"{r['corr_coeff']:.4f}" for r in results.values()],
    "Angle θ": ["0.00°"] + [f"{r['angle']:.2f}°" for r in results.values()],
    "Similarity": ["Self"] + [r['similarity'] for r in results.values()]
}

df = pd.DataFrame(data)
print(df.to_string(index=False))
print("-"*100)
print("\nInterpretation Guide:")
print("  ρ ≈ +1.0: Highly correlated (similar signals)")
print("  ρ ≈  0.0: Uncorrelated (orthogonal signals)")
print("  ρ ≈ -1.0: Anti-correlated (opposite signals)")
print("="*100)

# Visualization
fig = plt.figure(figsize=(16, 12))

# Define plot configurations
plot_configs = [
    ('Similar (phase shift)', signal_similar, 1, 'Slight Phase Shift\n(High Correlation)'),
    ('Opposite phase', signal_opposite, 2, 'Opposite Phase\n(Perfect Anti-correlation)'),
    ('90° phase shift', signal_shifted, 3, '90° Phase Shift\n(Orthogonal)'),
    ('Different frequency', signal_different, 4, 'Different Frequency\n(Low Correlation)'),
    ('Noisy version', signal_noisy, 5, 'Noisy Signal\n(Good Correlation)'),
    ('Square wave', signal_square, 6, 'Different Waveform\n(Moderate Correlation)')
]

# Create subplots
for name, signal, idx, title in plot_configs:
    ax = plt.subplot(3, 3, idx)
    result = results[name]
    
    # Plot signals
    ax.plot(t, signal_ref, 'b-', linewidth=2, label='s₁: sin(t)', alpha=0.7)
    ax.plot(t, signal, color=result['color'], linestyle='--', linewidth=2, 
            label=f's: {name[:20]}', alpha=0.8)
    
    # Add zero line
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Formatting
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_xlim(0, 4*np.pi)
    ax.set_ylim(-2, 2)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    
    if idx in [5, 6]:
        ax.set_xlabel('Time', fontsize=9)
    if idx in [1, 4]:
        ax.set_ylabel('Amplitude', fontsize=9)
    
    # Add annotation box
    textstr = f'ρ = {result["corr_coeff"]:.4f}\nθ = {result["angle"]:.1f}°'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, 
                 edgecolor=result['color'], linewidth=2)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')

# Subplot 7: Correlation Coefficients Bar Chart
ax7 = plt.subplot(3, 3, 7)
names_short = ['Similar', 'Opposite', '90° shift', 'Diff freq', 'Noisy', 'Square', 'Random']
corr_values = [results[name]['corr_coeff'] for name in signals.keys()]
colors_bars = [results[name]['color'] for name in signals.keys()]

bars = ax7.barh(names_short, corr_values, color=colors_bars, alpha=0.7, 
                edgecolor='black', linewidth=1.5)

# Add value labels
for bar, val in zip(bars, corr_values):
    width = bar.get_width()
    ax7.text(width, bar.get_y() + bar.get_height()/2,
             f' {val:.3f}',
             ha='left' if width > 0 else 'right',
             va='center', fontsize=9, fontweight='bold')

ax7.axvline(x=0, color='black', linewidth=2)
ax7.axvline(x=1, color='green', linewidth=1, linestyle='--', alpha=0.5, label='Perfect +')
ax7.axvline(x=-1, color='red', linewidth=1, linestyle='--', alpha=0.5, label='Perfect -')
ax7.grid(True, axis='x', linestyle=':', alpha=0.4)
ax7.set_xlabel('Correlation Coefficient ρ', fontsize=10, fontweight='bold')
ax7.set_title('Correlation Comparison', fontsize=11, fontweight='bold', pad=10)
ax7.set_xlim(-1.2, 1.2)
ax7.legend(loc='lower right', fontsize=8)

# Subplot 8: Scatter Plot (2D projection using first 2 points as axes)
ax8 = plt.subplot(3, 3, 8)
# Use first two time points to create a 2D projection
x_coords = [signal_ref[0], signal_ref[1]]
all_signals_list = [signal_ref] + list(signals.values())
signal_names_list = ['Reference'] + names_short

for i, (sig, name) in enumerate(zip(all_signals_list, signal_names_list)):
    y_coords = [sig[0], sig[1]]
    if name == 'Reference':
        ax8.scatter(x_coords, y_coords, s=200, marker='o', c='blue', 
                   edgecolors='black', linewidths=2, label=name, zorder=5)
    else:
        result = results[list(signals.keys())[i-1]]
        ax8.scatter(y_coords[0], y_coords[1], s=150, marker='s', 
                   c=result['color'], edgecolors='black', linewidths=1.5,
                   label=name, alpha=0.7, zorder=3)

ax8.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax8.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
ax8.grid(True, linestyle='--', alpha=0.4)
ax8.set_aspect('equal', adjustable='box')
ax8.legend(loc='upper left', fontsize=7, framealpha=0.9, ncol=2)
ax8.set_title('Signal Space Visualization\n(2D projection)', 
              fontsize=10, fontweight='bold', pad=8)
ax8.set_xlabel('Sample 1', fontsize=9)
ax8.set_ylabel('Sample 2', fontsize=9)

# Subplot 9: Cross-correlation example
ax9 = plt.subplot(3, 3, 9)

# Compute cross-correlation for time-shifted signal
shifted_signal = np.sin(t - 0.5)  # Time shifted version
lags = np.arange(-50, 51)
cross_corr = []

for lag in lags:
    if lag < 0:
        s1 = signal_ref[:lag]
        s2 = shifted_signal[-lag:]
    elif lag > 0:
        s1 = signal_ref[lag:]
        s2 = shifted_signal[:-lag]
    else:
        s1 = signal_ref
        s2 = shifted_signal
    
    # Normalize by length
    if len(s1) > 0:
        corr_val = np.dot(s1, s2) / len(s1)
    else:
        corr_val = 0
    cross_corr.append(corr_val)

ax9.plot(lags, cross_corr, 'b-', linewidth=2, marker='o', markersize=3)
max_corr_idx = np.argmax(cross_corr)
max_lag = lags[max_corr_idx]
ax9.axvline(x=max_lag, color='red', linestyle='--', linewidth=2, 
            label=f'Peak at lag={max_lag}')
ax9.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax9.grid(True, linestyle=':', alpha=0.4)
ax9.set_xlabel('Time Lag', fontsize=9)
ax9.set_ylabel('Cross-Correlation', fontsize=9)
ax9.set_title('Cross-Correlation Function\n(Time Delay Detection)', 
              fontsize=10, fontweight='bold', pad=8)
ax9.legend(loc='upper right', fontsize=8)

plt.suptitle('Signal Correlation via Dot Product (Signal Processing)', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.99))
plt.show()

print("\n✓ Visualization complete!")
