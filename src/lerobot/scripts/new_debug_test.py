import sys
import time

def add_numbers(a, b):
    print(f"Adding {a} and {b}")
    return a + b

def main():
    print("Starting new debug test...")
    
    # Good place for a breakpoint
    x = 5
    y = 3
    result = add_numbers(x, y)
    
    print(f"Result: {result}")

if __name__ == "__main__":
    main() 