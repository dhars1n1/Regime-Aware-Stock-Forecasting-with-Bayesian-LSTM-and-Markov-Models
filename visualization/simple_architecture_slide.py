"""
Simplified architecture diagram optimized for presentation slides
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

def create_simple_architecture():
    """Create a clean, simple architecture diagram for slides"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Color scheme - professional and clean
    colors = {
        'input': '#3498DB',      # Blue
        'lstm': '#2ECC71',       # Green  
        'dropout': '#F39C12',    # Orange
        'dense': '#9B59B6',      # Purple
        'output': '#E74C3C'      # Red
    }
    
    # Title
    ax.text(5, 9.5, 'Regime-Aware Bayesian LSTM', 
            fontsize=24, fontweight='bold', ha='center')
    
    # Simplified layer representation
    layers = [
        # (y_pos, layer_type, main_text, dimension_text)
        (8.2, 'input', 'Input Features\n(22 features × 60 timesteps)', '(Batch, 60, 22)'),
        (7.0, 'lstm', 'LSTM Layer 1\n64 units', '(Batch, 60, 64)'),
        (6.1, 'dropout', 'Dropout 30%', 'MC Uncertainty'),
        (5.2, 'lstm', 'LSTM Layer 2\n32 units', '(Batch, 32)'),
        (4.3, 'dropout', 'Dropout 30%', 'MC Uncertainty'),
        (3.4, 'dense', 'Dense 32 → 16', 'Feature Learning'),
        (2.5, 'dropout', 'Dropout 30%', 'MC Uncertainty'),
        (1.6, 'output', 'Output\nLog Return', '(Batch, 1)')
    ]
    
    # Draw layers
    for i, (y_pos, layer_type, main_text, detail_text) in enumerate(layers):
        # Determine box width based on layer type
        if layer_type == 'dropout':
            width, height = 3, 0.5
            alpha = 0.7
            linestyle = '--'
        else:
            width, height = 5, 0.7
            alpha = 0.9
            linestyle = '-'
        
        # Main layer box
        rect = FancyBboxPatch((5 - width/2, y_pos - height/2), width, height,
                            boxstyle="round,pad=0.05", 
                            facecolor=colors[layer_type], 
                            edgecolor='black', linewidth=2, 
                            linestyle=linestyle, alpha=alpha)
        ax.add_patch(rect)
        
        # Layer text
        ax.text(5, y_pos, main_text, 
                fontsize=12, fontweight='bold', ha='center', va='center', color='white')
        
        # Detail text on the right
        ax.text(7.8, y_pos, detail_text, 
                fontsize=10, ha='left', va='center', style='italic')
        
        # Add arrows
        if i < len(layers) - 1:
            next_y = layers[i + 1][0]
            arrow = patches.FancyArrowPatch((5, y_pos - height/2 - 0.05), 
                                          (5, next_y + 0.35),
                                          arrowstyle='->', mutation_scale=20, 
                                          color='#2C3E50', linewidth=2)
            ax.add_patch(arrow)
    
    # Key features callout
    features_box = patches.FancyBboxPatch((0.2, 6), 2.5, 2.5, boxstyle="round,pad=0.1",
                                        facecolor='#ECF0F1', edgecolor='#34495E', linewidth=2)
    ax.add_patch(features_box)
    
    feature_text = """🎯 KEY FEATURES:
    
• Regime Probabilities
• Market Data (OHLCV)
• Technical Indicators  
• Macro Variables
• Lagged Returns"""
    
    ax.text(1.45, 7.25, feature_text, fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Output callout
    output_box = patches.FancyBboxPatch((7.3, 0.5), 2.5, 1.8, boxstyle="round,pad=0.1",
                                      facecolor='#FADBD8', edgecolor='#E74C3C', linewidth=2)
    ax.add_patch(output_box)
    
    output_text = """📊 OUTPUT:
    
• Mean Prediction
• Uncertainty (±σ)
• 95% Confidence Interval
• Regime-Aware Confidence"""
    
    ax.text(8.55, 1.4, output_text, fontsize=10, ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_uncertainty_explanation():
    """Create a diagram explaining Monte Carlo Dropout"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(6, 5.5, 'Monte Carlo Dropout for Uncertainty Quantification', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Input
    input_box = patches.Rectangle((0.5, 2.5), 1.5, 1, facecolor='#3498DB', alpha=0.8)
    ax.add_patch(input_box)
    ax.text(1.25, 3, 'Input\nSequence', fontsize=11, ha='center', va='center', 
            color='white', fontweight='bold')
    
    # Multiple forward passes
    for i in range(3):
        y_offset = i * 0.8 - 0.8
        
        # Neural network representation
        network_box = patches.Rectangle((3, 2.5 + y_offset), 3, 0.6, 
                                      facecolor='#2ECC71', alpha=0.7)
        ax.add_patch(network_box)
        ax.text(4.5, 2.8 + y_offset, f'Forward Pass {i+1}\n(Different Dropout)', 
                fontsize=9, ha='center', va='center', color='white', fontweight='bold')
        
        # Arrow from input
        arrow1 = patches.FancyArrowPatch((2.1, 3), (2.9, 2.8 + y_offset),
                                       arrowstyle='->', mutation_scale=15, 
                                       color='#2C3E50', linewidth=1.5)
        ax.add_patch(arrow1)
        
        # Prediction
        pred_box = patches.Rectangle((7, 2.5 + y_offset), 1.5, 0.6, 
                                   facecolor='#E74C3C', alpha=0.7)
        ax.add_patch(pred_box)
        
        # Sample predictions for visualization
        sample_preds = ['+2.1%', '+2.4%', '+2.7%']
        ax.text(7.75, 2.8 + y_offset, sample_preds[i], 
                fontsize=10, ha='center', va='center', color='white', fontweight='bold')
        
        # Arrow to prediction
        arrow2 = patches.FancyArrowPatch((6.1, 2.8 + y_offset), (6.9, 2.8 + y_offset),
                                       arrowstyle='->', mutation_scale=15, 
                                       color='#2C3E50', linewidth=1.5)
        ax.add_patch(arrow2)
    
    # Dots for more passes
    ax.text(4.5, 1.2, '⋮\n(97 more passes)', fontsize=12, ha='center', va='center')
    ax.text(7.75, 1.2, '⋮\n(97 more predictions)', fontsize=12, ha='center', va='center')
    
    # Final statistics
    stats_box = patches.Rectangle((9.5, 1.5), 2, 3, facecolor='#9B59B6', alpha=0.8)
    ax.add_patch(stats_box)
    
    stats_text = """STATISTICS:
    
Mean: +2.34%
Std: ±1.56%

95% CI:
[-0.72%, +5.40%]"""
    
    ax.text(10.5, 3, stats_text, fontsize=10, ha='center', va='center', 
            color='white', fontweight='bold')
    
    # Arrow to statistics
    arrow3 = patches.FancyArrowPatch((8.6, 3), (9.4, 3),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='#2C3E50', linewidth=2)
    ax.add_patch(arrow3)
    
    plt.tight_layout()
    return fig

def create_regime_comparison():
    """Create a diagram showing regime-dependent uncertainty"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Colors for regimes
    regime_colors = {'Crisis': '#E74C3C', 'Normal': '#F39C12', 'Bull': '#2ECC71'}
    
    # Sample data for each regime
    regimes = [
        ('Crisis', [1.8, 2.1, 2.4, 1.9, 2.3, 2.0, 2.2], 2.1, 0.8),
        ('Normal', [2.2, 2.3, 2.4, 2.25, 2.35], 2.3, 0.3),
        ('Bull', [2.4, 2.45, 2.42, 2.46, 2.43], 2.44, 0.15)
    ]
    
    axes = [ax1, ax2, ax3]
    
    for i, (regime_name, predictions, mean_pred, uncertainty) in enumerate(regimes):
        ax = axes[i]
        color = regime_colors[regime_name]
        
        # Plot predictions as histogram
        ax.hist(predictions, bins=5, alpha=0.7, color=color, edgecolor='black')
        ax.axvline(mean_pred, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_pred:.2f}%')
        
        # Add uncertainty shading
        ax.axvspan(mean_pred - uncertainty, mean_pred + uncertainty, 
                  alpha=0.3, color=color, label=f'±{uncertainty:.2f}% (1σ)')
        
        ax.set_title(f'{regime_name} Market\nUncertainty: ±{uncertainty:.2f}%', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Regime-Dependent Uncertainty in Bayesian LSTM Predictions', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def save_slide_diagrams():
    """Save presentation-ready diagrams"""
    
    os.makedirs("visualization", exist_ok=True)
    
    # Simple architecture
    print("Creating simple architecture diagram...")
    fig1 = create_simple_architecture()
    fig1.savefig("visualization/simple_lstm_architecture.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig1.savefig("visualization/simple_lstm_architecture.pdf", 
                bbox_inches='tight', facecolor='white')
    
    # Uncertainty explanation
    print("Creating uncertainty explanation diagram...")
    fig2 = create_uncertainty_explanation()
    fig2.savefig("visualization/monte_carlo_dropout_explanation.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig("visualization/monte_carlo_dropout_explanation.pdf", 
                bbox_inches='tight', facecolor='white')
    
    # Regime comparison
    print("Creating regime uncertainty comparison...")
    fig3 = create_regime_comparison()
    fig3.savefig("visualization/regime_uncertainty_comparison.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig3.savefig("visualization/regime_uncertainty_comparison.pdf", 
                bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    print("\n🎯 Presentation diagrams ready!")
    print("Files saved in visualization/:")
    print("  • simple_lstm_architecture.png/pdf")
    print("  • monte_carlo_dropout_explanation.png/pdf") 
    print("  • regime_uncertainty_comparison.png/pdf")

if __name__ == "__main__":
    save_slide_diagrams()
