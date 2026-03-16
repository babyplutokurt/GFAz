import sys
import os
from collections import defaultdict

def analyze_gfa(file_path):
    line_counts = defaultdict(int)
    byte_sizes = defaultdict(int)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    print(f"Analyzing GFA file: {file_path} ...")
    
    total_lines = 0
    total_bytes = 0

    try:
        # Read file in binary mode to accurately measure byte size
        with open(file_path, 'rb') as f:
            for line in f:
                line_len = len(line)
                
                if line_len == 0 or line.isspace():
                    continue
                
                # GFA is ASCII/UTF-8 compatible for the first character
                record_type = chr(line[0])
                
                # Valid GFA record types based on the specification
                if record_type in '#HSLJCPW':
                    line_counts[record_type] += 1
                    byte_sizes[record_type] += line_len
                else:
                    # In case of unrecognized lines
                    line_counts['UNKNOWN'] += 1
                    byte_sizes['UNKNOWN'] += line_len
                
                total_lines += 1
                total_bytes += line_len
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)

    # Format the byte size output clearly
    def format_bytes(b):
        if b < 1024:
            return f"{b} B"
        elif b < 1024**2:
            return f"{b / 1024:.2f} KB"
        elif b < 1024**3:
            return f"{b / 1024**2:.2f} MB"
        else:
            return f"{b / 1024**3:.2f} GB"

    print("-" * 75)
    print(f"{'Record Type':<15} | {'Line Count':<15} | {'Byte Size':<20} | {'Byte Size (Raw)'}")
    print("-" * 75)
    
    record_types_order = ['#', 'H', 'S', 'L', 'J', 'C', 'P', 'W', 'UNKNOWN']
    
    for rt in record_types_order:
        if line_counts[rt] > 0 or byte_sizes[rt] > 0:
            count = line_counts[rt]
            size = byte_sizes[rt]
            print(f"{rt:<15} | {count:<15} | {format_bytes(size):<20} | {size} bytes")
            
    print("-" * 75)
    print(f"{'Total':<15} | {total_lines:<15} | {format_bytes(total_bytes):<20} | {total_bytes} bytes")
    print("-" * 75)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gfa_analyzer.py <path_to_gfa_file>")
        sys.exit(1)
        
    gfa_file = sys.argv[1]
    analyze_gfa(gfa_file)
