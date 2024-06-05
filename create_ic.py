from kbmod import ImageCollection
import time
import logging
import os

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description='Converts a list of URIs to a KBMOD ImageCollection.')

# Required argument for the input file of URIs
parser.add_argument('--target_uris_file', type=str, help='The file containing the URIs of the images to be processed.')

# Required for the output file of the ImageCollection
parser.add_argument('--ic_output_file', type=str, help='The file to save the ImageCollection to.')

# Required for the output file of the ImageCollection
parser.add_argument('--uris_base_dir', type=str, help='A base directory of our target URIs, useful if the target URIs file and associated URIs have been moved between machines')

# Optional argument for debug printing
parser.add_argument('--silence', action='store_true', help="Don't print debug messages.")

args = parser.parse_args()
if args.target_uris_file is None:
    print("Please provide a file containing the URIs of the images to be processed.")
    exit(1)
if args.ic_output_file is None:
    print("Please provide a file to save the ImageCollection to.")
    exit(1)
DEBUG = True
if args.silence:
    logging.basicConfig(level=logging.DEBUG)
    DEBUG = False

# A function that wraps prints with a timestamp if DEBUG is True
# Accepts multiple arguments just like Python's print()
def print_debug(*msg):
    if DEBUG:
        print(time.strftime("%H:%M:%S", time.localtime()), *msg)

print_debug("Loading up URIs")
# Load the list of images from our saved file "sample_uris.txt"
uris = []
with open(args.target_uris_file) as f:
    for l in f.readlines():
        if not l.startswith("#"):
            # Ignore commented metadata
            uris.append(l)

if args.uris_base_dir is not None:
    print_debug("Using URIs base dir", args.uris_base_dir)
    if not os.path.isdir(args.uris_base_dir):
        raise ValueError("URIS base dir is not a valid directory", args.uris_base_dir)

# Clean up the URI strings
for i in range(len(uris)):
    file_prefix = "file://"
    curr = uris[i].replace('%23', '#').strip()
    if curr.startswith(file_prefix):
        curr = curr[len(file_prefix):]
    if args.uris_base_dir is not None:
       curr = os.path.join(args.uris_base_dir, curr.lstrip(os.path.sep))
    uris[i] = curr


print_debug("Creating ImageCollection")
# Create an ImageCollection object from the list of URIs
ic = ImageCollection.fromTargets(uris)
print_debug("ImageCollection created")

ic.write(args.ic_output_file, format='ascii.ecsv')

