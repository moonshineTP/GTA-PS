
import sys
import numpy as np

def main():
    print("Hello from compiled GTA-PS dummy script!")
    print(f"Python version: {sys.version}")
    print(f"Numpy version: {np.__version__}")
    a = np.array([1, 2, 3])
    print(f"Numpy array: {a}")

if __name__ == "__main__":
    main()
