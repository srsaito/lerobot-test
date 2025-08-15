import sys
import time

print("Debug test starting...")  # Print to stdout
print("Debug test starting...", file=sys.stderr)  # Print to stderr

time.sleep(2)  # Pause for 2 seconds

print("If you see this, the script is running!")
print("If you see this, the script is running!", file=sys.stderr)

def main():
    print("Debug test starting...", file=sys.stderr)
    x = 1
    y = 2
    result = x + y
    print(f"Result: {result}", file=sys.stderr)

if __name__ == "__main__":
    main() 