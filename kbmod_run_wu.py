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
wu = WorkUnit.from_fits(args.wu_input_file)

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
