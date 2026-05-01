
import sys
import os
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from plot_academic import plot_swift_test_template
except ImportError:
    print("Error: Could not import plot_academic. Make sure it is in code/misc/")
    sys.exit(1)

def main():
    print("GTA-PS Swift Test Case")
    print("-" * 30)
    
    # Render the template plot
    try:
        print("Rendering swift test palette template...")
        output_path = plot_swift_test_template()
        
        if output_path.exists():
            print(f"SUCCESS: Plot saved to {output_path}")
            print(f"File size: {output_path.stat().st_size} bytes")
        else:
            print(f"FAILURE: Output path {output_path} does not exist.")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during swift test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
