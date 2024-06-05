import kbmod

from kbmod import ImageCollection
from kbmod.work_unit import WorkUnit
import kbmod.reprojection_utils as reprojection_utils
from kbmod.reprojection_utils import transform_wcses_to_ebd

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

parser = argparse.ArgumentParser(description='Converts a KBMOD ImageCollection to a Reprojected WorkUnit.')

# Required for the input file of the ImageCollection and the output directory
parser.add_argument('--ic_input_file', type=str, help='The file containing the ImageCollection to be processed.')
parser.add_argument('--result_dir', type=str, help='The directory to save the results to.')
parser.add_argument('--search_config', type=str, help='The file path fo the search configuration file')

# Required heliocentric guess distance in AU
parser.add_argument('--guess_dist', type=float, help='The heliocentric distance to use for the barycentric correction.')

# Required arguments for our selected patch
parser.add_argument('--patch_corners', type=str, help='The corners of the patch in RA, Dec coordinates. Or a plaintext file with this specified in a single line beginning with "#patch_center_coords="')
parser.add_argument('--image_width', type=int, help='The width of the image for our patch in pixels.')
parser.add_argument('--image_height', type=int, help='The height of the image for our patch in pixels.')

parser.add_argument('--n_workers', type=int, help='The max number of workers when parallelizing reprojection. Must be in [1, 64]. ')

# Optional argument for debug print_debuging
parser.add_argument('--silence', action='store_true', help="Don't print debug messages.")

args = parser.parse_args()
if args.ic_input_file is None:
    print("Please provide a file containing the ImageCollection to be processed.")
    exit(1)
if args.result_dir is None:
    print("Please provide a directory to save the results to.")
    exit(1)
if args.search_config is None:
    print("Please provide a search configuration file.")
    exit(1)
if args.patch_corners is None:
    print("Please provide the corners of the patch in RA, Dec coordinates.")
    exit(1)
if args.guess_dist is None:
    print("Please provide a heliocentric distance to use for the barycentric correction.")
    exit(1)
DEBUG = True
if args.silence:
    DEBUG = False
else:
    logging.basicConfig(level=logging.DEBUG)

n_workers = 8
if args.n_workers is not None and args.n_workers >=1 and args.n_workers <= 65:
    n_workers = args.n_workers

# A function that wraps print_debugs with a timestamp if DEBUG is True
# Accepts multiple arguments just like Python's print_debug()
def print_debug(*msg):
    if DEBUG:
        print(time.strftime("%H:%M:%S", time.localtime()), *msg)

if not os.path.exists(args.result_dir):
    # Create the directory
    print_debug("creating results directory:", str(args.result_dir))
    os.makedirs(args.result_dir)

patch_corners = args.patch_corners
if os.path.exists(args.patch_corners):
    found_corners = False
    with open(args.patch_corners, 'r') as f:
        lines = f.readlines()
        for l in lines:
            patch_header = "#patch_box="
            if l.startswith(patch_header):
                patch_corners = eval(l[len(patch_header):])
                found_corners = True
                break
    if not found_corners:
        raise ValueError("Did not read patch corners from file:", args.patch_corners)
else:
    patch_corners = eval(args.patch_corners), # TODO less dangerous way to evaluate patch corners
print_debug("Patch corners:", patch_corners)


def create_wcs_from_corners(corners, image_width, image_height, save_fits=False, filename='output.fits', pixel_scale=None, verbose=True):
    """
    Create a WCS object given the RA, Dec coordinates of the four corners of the image
    and the dimensions of the image in pixels. Optionally save as a FITS file.

    Parameters:
    corners (list of lists): [[RA1, Dec1], [RA2, Dec2], [RA3, Dec3], [RA4, Dec4]]
    image_width (int): Width of the image in pixels
    image_height (int): Height of the image in pixels
    save_fits (bool): If True, save the WCS to a FITS file
    filename (str): The name of the FITS file to save

    Returns:
    WCS: The World Coordinate System object for the image

    5/6/2024 COC + ChatGPT4
    """
    # TODO switch to https://docs.astropy.org/en/stable/api/astropy.wcs.utils.fit_wcs_from_points.html
    # Extract the corners
    corners = list(set(corners)) # eliminate duplicate coords in case someone passes the repeat 1st corner used for plotting 6/5/2024 COC
    if len(corners) != 4:
        raise ValueError(f'There should be four (4) corners. We saw: {corners}')
    ra = [corner[0] for corner in corners]
    dec = [corner[1] for corner in corners]
    # Calculate the central position (average of the coordinates)
    center_ra = np.mean(ra)
    center_dec = np.mean(dec)
    # Calculate the pixel scale in degrees per pixel
    if pixel_scale is not None:
        pixel_scale_ra = pixel_scale/60/60
        pixel_scale_dec = pixel_scale/60/60
    else:
        ra_range = (max(ra) - min(ra)) * np.cos(np.radians(center_dec))  # Adjust RA difference for declination
        dec_range = max(dec) - min(dec)
        pixel_scale_ra = ra_range / image_width
        pixel_scale_dec = dec_range / image_height
    if verbose: print_debug(f'Saw (RA,Dec) pixel scales ({pixel_scale_ra*60*60},{pixel_scale_dec*60*60})"/pixel. User-supplied: {pixel_scale}"/pixel.')
    # Initialize a WCS object with 2 axes (RA and Dec)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [image_width / 2, image_height / 2]
    wcs.wcs.crval = [center_ra, center_dec]
    wcs.wcs.cdelt = [-pixel_scale_ra, pixel_scale_dec]  # RA pixel scale might need to be negative (convention)
    # Rotation matrix, assuming no rotation
    wcs.wcs.pc = [[1, 0], [0, 1]]
    # Define coordinate frame and projection
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.array_shape = (image_height, image_width)
    if save_fits:
        # Create a new FITS file with the WCS information and a dummy data array
        hdu = fits.PrimaryHDU(data=np.ones((image_height, image_width)), header=wcs.to_header())
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)
        print_debug(f"Saved FITS file with WCS to {filename}")
    return wcs

# moving the patch generation up so it can fail early if it will fail 6/5/2024 COC
print_debug("Creating WCS from patch")

# Create a WCS object for the patch. This will be our common reprojection WCS
patch_wcs = create_wcs_from_corners(
    patch_corners,
    args.image_width,
    args.image_height, 
    save_fits=False)

ic = ImageCollection.read(args.ic_input_file, format='ascii.ecsv')
print_debug("ImageCollection read, create work unit")

orig_wu = ic.toWorkUnit(config=kbmod.configuration.SearchConfiguration.from_file(args.search_config))

# TODO re-enable
#print_debug("saving original work unit")
#orig_wu.to_fits(os.path.join(args.result_dir, "orig_wu.fits"), overwrite=True)

print_debug("Creating Barycentric WorkUnit")
# Find the EBD (estimated barycentric distance) WCS for each image
imgs = orig_wu.im_stack
point_on_earth = EarthLocation(1814303.74553723, -5214365.7436216, -3187340.56598756, unit='m')
ebd_per_image_wcs, geocentric_dists = transform_wcses_to_ebd(
    orig_wu._per_image_wcs,
    imgs.get_single_image(0).get_width(),
    imgs.get_single_image(0).get_height(),
    args.guess_dist, 
    [astropy.time.Time(img.get_obstime(), format='mjd') for img in imgs.get_images()],
    point_on_earth, 
    npoints=10, 
    seed=None,
)

if len(orig_wu._per_image_wcs) != len(ebd_per_image_wcs):
    raise ValueError("Number of barycentric WCS objects does not match number of images")
# Construct a WorkUnit with the EBD WCS and provenance data
ebd_wu = WorkUnit(
    im_stack=orig_wu.im_stack,
    config=orig_wu.config,
    per_image_wcs=orig_wu._per_image_wcs,
    per_image_ebd_wcs=ebd_per_image_wcs,
    heliocentric_distance=args.guess_dist,
    geocentric_distances=geocentric_dists,
)

print_debug("Reprojecting WorkUnit")

# Reproject to a common WCS using the WCS for our patch
reprojected_wu = reprojection.reproject_work_unit(ebd_wu, patch_wcs, frame="ebd", max_parallel_processes=n_workers)
print_debug("Reprojected WorkUnit created")
print_debug("saving reprojected work unit")
reprojected_wu.to_fits(os.path.join(args.result_dir, "reprojected_wu.fits"))
