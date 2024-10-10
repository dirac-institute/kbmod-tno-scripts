import kbmod
from kbmod import ImageCollection
from kbmod.work_unit import WorkUnit
from kbmod import ImageStack
import kbmod.reprojection_utils as reprojection_utils
import kbmod.reprojection as reprojection
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import EarthLocation
import astropy.time
import numpy as np
import logging
import os
import time

# Parse command line arguments
import argparse

parser = argparse.ArgumentParser(description='Loads a KBMOD WorkUnit for running a KBMOD search.')

# Required for the input file of the ImageCollection and the output directory
parser.add_argument('--wu_input_file', type=str, help='The file containing the WorkUnit to be processed.')
parser.add_argument('--result_dir', type=str, help='The directory to save the results to.')
parser.add_argument('--search_config', type=str, help='The file path fo the search configuration file')

# Parser argument for if the WorkUnit is sharded. You should provide the path of the head fits file.
parser.add_argument('--sharded', action='store_true', help="Treats the WorkUnit as sharded if true")

# Parser argument for if the WorkUnit is sharded
parser.add_argument('--downsample_n', type=int, help="Downsamples the WorkUnit to every nth Image")

# Optional argument for debug print_debuging
parser.add_argument('--silence', action='store_true', help="Don't print debug messages.")

args = parser.parse_args()

if args.wu_input_file is None:
    print("Please provide a file containing the ImageCollection to be processed.")
    exit(1)
if args.result_dir is None:
    print("Please provide a directory to save the results to.")
    exit(1)

DEBUG = True
if args.silence:
    DEBUG = False
else:
    logging.basicConfig(level=logging.DEBUG)

# A function that wraps print_debugs with a timestamp if DEBUG is True
# Accepts multiple arguments just like Python's print_debug()
def print_debug(*msg):
    if DEBUG:
        print(time.strftime("%H:%M:%S", time.localtime()), *msg)


print_debug("Loading workunit from file")
wu = None
if not args.sharded:
    wu = WorkUnit.from_fits(args.wu_input_file)
else:
    # Load a sharded WorkUnit from a directory and the name of the head file
    wu_dir = os.path.dirname(args.wu_input_file)
    wu_filename = os.path.basename(args.wu_input_file)
    wu = WorkUnit.from_sharded_fits(wu_filename, wu_dir, lazy=False)

if args.downsample_n is not None:
    if args.downsample_n < 1:
        print("Please provide a valid downsample_n argument.")
        exit(1)
    # Downsample the WorkUnit to every nth image
    print(f"Downsampling WorkUnit to every {args.downsample_n}th image.")
    old_stack = wu.im_stack
    new_images = []
    for i in range(old_stack.img_count()):
        if i % args.downsample_n == 0:
            # Only include the image in the stack if it is the nth image
            new_images.append(old_stack.get_single_image(i))
    # Create a new ImageStack downsampling to only the nth images.
    wu.im_stack = ImageStack(new_images)
    print(f"Downsampled WorkUnit has {wu.im_stack.img_count()} images")

print_debug("Loaded work unit")
if args.search_config is not None:
    # Load a search configuration, otherwise use the one loaded with the work unit
    wu.config = kbmod.configuration.SearchConfiguration.from_file(args.search_config)


config = wu.config

# Modify the work unit results to be what is specified in command line args
input_parameters = {
    "res_filepath": args.result_dir,
    "result_filename": os.path.join(args.result_dir, "full_results.ecsv"),
}
config.set_multiple(input_parameters)

# Save the search config in the results directory for record keeeping
config.to_file(os.path.join(args.result_dir, "search_config.yaml"))
wu.config = config

print_debug("Running KBMOD search")
res = kbmod.run_search.SearchRunner().run_search_from_work_unit(wu)

print_debug("Search complete")
print_debug(res)

print_debug("writing results table")
res.write_table(os.path.join(args.result_dir, "results.ecsv"))
