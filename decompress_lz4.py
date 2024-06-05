import lz4.frame
import sys

def decompress_lz4(input_file, output_file):
    with open(input_file, 'rb') as compressed_file:
        compressed_data = compressed_file.read()
    decompressed_data = lz4.frame.decompress(compressed_data)
    with open(output_file, 'wb') as decompressed_file:
        decompressed_file.write(decompressed_data)
    print(f"Decompressed {input_file} to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python decompress_lz4.py <input_file.lz4> <output_file>")
        sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
decompress_lz4(input_file, output_file)

