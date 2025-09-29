"""
Master script to generate all architecture diagrams for the Bayesian LSTM project
"""
import os
import sys
import matplotlib.pyplot as plt

def run_all_diagram_generators():
    """Run all diagram generation scripts"""
    
    print("🎨 Generating Architecture Diagrams for Regime-Aware Bayesian LSTM")
    print("=" * 70)
    
    try:
        # Import and run detailed architecture diagrams
        print("\n📊 Generating detailed technical diagrams...")
        from architecture_diagram import save_diagrams
        save_diagrams()
        
        print("\n🎯 Generating presentation-ready diagrams...")
        from simple_architecture_slide import save_slide_diagrams
        save_slide_diagrams()
        
        print("\n✅ All diagrams generated successfully!")
        print("\n📁 Files created in visualization/ folder:")
        
        # List all generated files
        viz_files = [
            "bayesian_lstm_architecture.png",
            "bayesian_lstm_architecture.pdf", 
            "regime_flow_diagram.png",
            "regime_flow_diagram.pdf",
            "simple_lstm_architecture.png",
            "simple_lstm_architecture.pdf",
            "monte_carlo_dropout_explanation.png",
            "monte_carlo_dropout_explanation.pdf",
            "regime_uncertainty_comparison.png",
            "regime_uncertainty_comparison.pdf"
        ]
        
        for i, filename in enumerate(viz_files, 1):
            print(f"  {i:2d}. {filename}")
        
        print("\n🎓 Ready for presentation!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("pip install matplotlib numpy")
        
    except Exception as e:
        print(f"❌ Error generating diagrams: {e}")
        return False
    
    return True

def show_diagram_descriptions():
    """Show descriptions of each generated diagram"""
    
    descriptions = {
        "Detailed Technical Diagrams": {
            "bayesian_lstm_architecture": "Complete layer-by-layer architecture with dimensions, parameters, and technical details",
            "regime_flow_diagram": "Shows how regime probabilities flow through the network over time"
        },
        "Presentation Diagrams": {
            "simple_lstm_architecture": "Clean, slide-ready architecture overview", 
            "monte_carlo_dropout_explanation": "Step-by-step explanation of uncertainty quantification",
            "regime_uncertainty_comparison": "Visual comparison of uncertainty across different market regimes"
        }
    }
    
    print("\n📖 Diagram Descriptions:")
    print("=" * 50)
    
    for category, diagrams in descriptions.items():
        print(f"\n🎯 {category}:")
        for name, desc in diagrams.items():
            print(f"  • {name}: {desc}")

if __name__ == "__main__":
    # Create visualization directory
    os.makedirs("visualization", exist_ok=True)
    
    # Generate all diagrams
    success = run_all_diagram_generators()
    
    if success:
        show_diagram_descriptions()
        
        print("\n💡 Usage Tips:")
        print("  • Use PNG files for PowerPoint/Google Slides")
        print("  • Use PDF files for LaTeX/academic papers") 
        print("  • All diagrams are high-resolution (300 DPI)")
        print("  • Simple versions are optimized for presentations")
        print("  • Detailed versions show technical specifications")
    else:
        print("\n❌ Diagram generation failed. Check error messages above.")
