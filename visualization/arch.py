"""
Generate clean architecture diagram for Regime-Aware Bayesian LSTM
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_clean_lstm_architecture():
    """Create a clean, simple architecture diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#3498DB',      # Blue
        'lstm': '#2ECC71',       # Green  
        'dense': '#9B59B6',      # Purple
        'output': '#E74C3C',     # Red
        'mc_dropout': '#F39C12'  # Orange
    }
    
    # Title
    ax.text(5, 11.5, 'Regime-Aware Bayesian LSTM Architecture', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Layer definitions
    layers = [
        # (y_pos, height, width, layer_type, main_text, dimension_text)
        (10.0, 0.8, 6, 'input', 'Input Features', '(batch_size, 60, 25)'),
        (8.8, 0.8, 6, 'lstm', 'LSTM Layer 1', '(batch_size, 60, 64)'),
        (7.6, 0.8, 6, 'lstm', 'LSTM Layer 2', '(batch_size, 32)'),
        (6.4, 0.6, 5, 'dense', 'Dense Layer 1', '(batch_size, 32)'),
        (5.4, 0.6, 5, 'dense', 'Dense Layer 2', '(batch_size, 16)'),
        (4.4, 0.6, 5, 'output', 'Output Layer', '(batch_size, 1)'),
        (3.0, 0.95, 6, 'mc_dropout', 'Monte Carlo Dropout', 'Multiple predictions with dropout active for uncertainty')
    ]
    
    # Draw layers
    for i, (y_pos, height, width, layer_type, main_text, detail_text) in enumerate(layers):
        x_center = 5
        
        # Main layer box
        rect = FancyBboxPatch((x_center - width/2, y_pos - height/2), width, height,
                            boxstyle="round,pad=0.05", 
                            facecolor=colors[layer_type], 
                            edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        
        # Layer name
        ax.text(x_center, y_pos + 0.1, main_text, 
                fontsize=12, fontweight='bold', ha='center', va='center', color='white')
        
        # Dimensions/details
        ax.text(x_center, y_pos - 0.15, detail_text, 
                fontsize=10, ha='center', va='center', color='white', style='italic')
        
        # Add arrows between layers (except for the last one)
        if i < len(layers) - 1:
            next_y = layers[i + 1][0]
            next_height = layers[i + 1][1]
            arrow = patches.FancyArrowPatch((x_center, y_pos - height/2 - 0.05), 
                                          (x_center, next_y + next_height/2 + 0.05),
                                          arrowstyle='->', mutation_scale=20, 
                                          color='#2C3E50', linewidth=2)
            ax.add_patch(arrow)
    
    # Add layer descriptions on the right
    descriptions = [
        "60 timesteps × 25 features\n(Market + Technical + Regime)",
        "64 units, return_sequences=True\nLearns temporal patterns",
        "32 units, return_sequences=False\nLearns high-level patterns", 
        "32 units, ReLU activation\nFeature processing",
        "16 units, ReLU activation\nFinal feature extraction",
        "1 unit, Linear activation\nPredicted log return",
        "Averaging results and\ncalculating uncertainty"
    ]
    
    for i, (y_pos, _, _, _, _, _) in enumerate(layers):
        ax.text(8.2, y_pos, descriptions[i], 
                fontsize=9, ha='left', va='center', color='#2C3E50')
    
#     # Add 25 features specification
#     features_text = """25 Input Features:
# • Market Data (6): OHLCV, VIX
# • Technical Indicators (8): RSI, MACD, etc.
# • Macroeconomic (3): CPI, Unemployment, FedFunds
# • Lagged Returns (4): return_lag_1,2,3,5
# • Regime Probabilities (3): Crisis, Normal, Bull
# • Sentiment (1): sentiment_score"""
    
#     ax.text(0.5, 8.5, features_text, 
#             fontsize=9, ha='left', va='top', color='#2C3E50',
#             bbox=dict(boxstyle="round,pad=0.3", facecolor='#ECF0F1', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_monte_carlo_explanation():
    """Create a simple Monte Carlo Dropout explanation"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(6, 5.5, 'Monte Carlo Dropout Process', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Single prediction
    ax.text(2, 4.5, 'Single Prediction', fontsize=12, fontweight='bold', ha='center')
    single_box = patches.Rectangle((1, 3.5), 2, 0.8, facecolor='#E74C3C', alpha=0.8)
    ax.add_patch(single_box)
    ax.text(2, 3.9, 'Return: +2.34%', fontsize=10, ha='center', va='center', 
            color='white', fontweight='bold')
    
    # Arrow
    arrow1 = patches.FancyArrowPatch((3.2, 3.9), (4.8, 3.9),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='#2C3E50', linewidth=2)
    ax.add_patch(arrow1)
    
    # Multiple predictions with MC Dropout
    ax.text(7, 4.5, '100 MC Dropout Predictions', fontsize=12, fontweight='bold', ha='center')
    
    # Show sample predictions
    sample_preds = ['+2.1%', '+2.4%', '+2.7%', '+1.9%', '+2.5%']
    for i, pred in enumerate(sample_preds):
        pred_box = patches.Rectangle((5.5 + i*0.6, 3.5), 0.5, 0.3, 
                                   facecolor='#F39C12', alpha=0.7)
        ax.add_patch(pred_box)
        ax.text(5.75 + i*0.6, 3.65, pred, fontsize=8, ha='center', va='center', 
                color='white', fontweight='bold')
    
    ax.text(7.5, 3.1, '... (95 more)', fontsize=9, ha='center', style='italic')
    
    # Arrow
    arrow2 = patches.FancyArrowPatch((8.5, 3.9), (9.5, 3.9),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='#2C3E50', linewidth=2)
    ax.add_patch(arrow2)
    
    # Final result
    ax.text(10.5, 4.5, 'Final Result', fontsize=12, fontweight='bold', ha='center')
    result_box = patches.Rectangle((9.8, 3.2), 1.4, 1.4, facecolor='#9B59B6', alpha=0.8)
    ax.add_patch(result_box)
    
    result_text = """Mean: +2.34%
Std: ±1.56%
95% CI:
[-0.72%, +5.40%]"""
    
    ax.text(10.5, 3.9, result_text, fontsize=9, ha='center', va='center', 
            color='white', fontweight='bold')
    
    # Process explanation
    process_text = """Monte Carlo Dropout Process:
1. Make prediction with dropout active (different neurons masked each time)
2. Repeat 100 times to get distribution of predictions
3. Calculate mean (best estimate) and standard deviation (uncertainty)
4. Generate confidence intervals for risk assessment"""
    
    ax.text(6, 2, process_text, fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#F8F9FA', alpha=0.9))
    
    plt.tight_layout()
    return fig

def save_diagrams():
    """Save the clean diagrams"""
    import os
    
    # Create visualization directory
    os.makedirs("visualization", exist_ok=True)
    
    # Create and save architecture diagram
    print("Creating clean LSTM architecture diagram...")
    fig1 = create_clean_lstm_architecture()
    fig1.savefig("visualization/clean_lstm_architecture.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig1.savefig("visualization/clean_lstm_architecture.pdf", 
                bbox_inches='tight', facecolor='white')
    print("✅ Clean architecture diagram saved")
    
    # Create and save Monte Carlo explanation
    print("Creating Monte Carlo explanation...")
    fig2 = create_monte_carlo_explanation()
    fig2.savefig("visualization/monte_carlo_explanation.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig("visualization/monte_carlo_explanation.pdf", 
                bbox_inches='tight', facecolor='white')
    print("✅ Monte Carlo explanation saved")
    
    plt.show()
    
    print("\n📊 Clean diagrams ready for presentation!")
    print("Files saved:")
    print("  • visualization/arch.png/pdf")
    print("  • visualization/arch0.png/pdf")

if __name__ == "__main__":
    save_diagrams()