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
import glob
import time

# A function that wraps print_debugs with a timestamp if DEBUG is True
# Accepts multiple arguments just like Python's print_debug()
def print_debug(*msg):
    if DEBUG:
        print(time.strftime("%H:%M:%S", time.localtime()), *msg)

def get_params_from_uri_file(uri_file, verbose=False):
    """
    Get parameters we place into URI file as comments at the top.
    Example start of URI file (6/6/2024 COC):
    #desired_dates=['2019-04-02', '2019-05-07']
    #dist_au=42.0
    #patch_size=[20, 20]
    #patch_id=5845
    #patch_center_coords=(216.49999999999997, -13.500000000000005)
    #patch_box=[[216.33333333333331, -13.666666666666671], [216.33333333333331, -13.333333333333337], [216.66666666666666, -13.333333333333337], [216.66666666666666, -13.666666666666671], [216.33333333333331, -13.666666666666671]]
    /gscratch/dirac/DEEP/repo/DEEP/20190507/A0b/science#step6/20240425T145342Z/differenceExp/20190508/VR/VR_DECam_c0007_6300.0_2600.0/855719/differenceExp_DECam_VR_VR_DECam_c0007_6300_0_2600_0_855719_S12_DEEP_20190507_A0b_scienceHASHstep6_20240425T145342Z.fits
    6/6/2024 COC
    """
    results = {}
    with open(uri_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '': continue # deal with rogue blank lines or invisible double line endings
            if not line.startswith('#'): break # comments section is done
            line = line.lstrip('#').split('=')
            lhs = line[0].strip()
            rhs = line[1].strip()
            try:
                rhs = eval(rhs)
            except ValueError:
                if verbose:
                    print_debug(f'Unable to eval() {lhs} field with value {rhs}.')
                continue
            results[lhs] = rhs
    return results


if __name__ == '__main__':
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Converts a KBMOD ImageCollection to a Reprojected WorkUnit.')
    
    # Required for the input file of the ImageCollection and the output directory
    parser.add_argument('--ic_input_file', type=str, help='The file containing the ImageCollection to be processed.')
    
    # URI file that is output by Region Search
    parser.add_argument('--uri_file', type=str, help='The original URI file used to assemble the ImageCollection. Optional, but user must supply patch corners and guess_dist if uri_file is not specified.')
    
    parser.add_argument('--result_dir', type=str, help='The directory to save the results to.')
    parser.add_argument('--search_config', type=str, help='The file path fo the search configuration file')
    
    # Required heliocentric guess distance in AU
    parser.add_argument('--guess_dist', type=float, help='The heliocentric distance (in au) to use for the barycentric correction. If none (the default), we extract this from the URI file (line beginning #dist_au).', default=None)
    
    # Required arguments for our selected patch
    parser.add_argument('--patch_corners', type=str, help='The corners of the patch in RA, Dec coordinates. If omitted, we derive this from the URI file from a single line beginning with "#patch_center_coords="', default=None)
    parser.add_argument('--image_width', type=int, help='The width of the image for our patch in pixels. Leave blank to derive from the URI file (but user must supply pixel scale then.)', default=None)
    parser.add_argument('--image_height', type=int, help='The height of the image for our patch in pixels. Leave blank to derive from the URI file (but user must supply pixel scale then.)', default=None)
    
    parser.add_argument('--pixel_scale', type=float, help='The pixel scale (in arcseconds per pixel) to use for calculating the patch dimensions in pixels. Optional if patch dimensions are manually specified.', default=None)
    
    parser.add_argument('--n_workers', type=int, help='The max number of workers when parallelizing reprojection. Must be in [1, 64]. Default is 8.', default=8)
    
    # Optional argument for debug print_debuging
    parser.add_argument('--silence', action='store_true', help="Don't print debug messages.")
#   parser.add_argument('--verbose', type=bool, help='Print more messages. Default is False.', default=False)
    
    args = parser.parse_args()
    
    DEBUG = True
    if args.silence:
        DEBUG = False
    else:
        logging.basicConfig(level=logging.DEBUG)
    
    if args.uri_file != None:
        uri_params = get_params_from_uri_file(uri_file=args.uri_file, verbose=DEBUG)
    else:
        uri_params = {}
    print_debug(f'Saw uri_params={uri_params}')
    
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
        if "patch_box" not in uri_params:
            print("Please provide the corners of the patch in decimal degree (RA, Dec) coordinates, or pass the values through the URI file in the #patch_corners line.")
            exit(1)
        patch_corners = uri_params['patch_box']
    else:
        patch_corners = eval(args.patch_corners) # TODO less dangerous way to evaluate patch corners
    print_debug("Patch corners:", patch_corners)
    
    if args.guess_dist is None:
        if 'dist_au' not in uri_params:
            print("Please provide a heliocentric distance (in au) to use for the barycentric correction, or specify a URI file that has the #dist_au line.")
            exit(1)
        guess_dist = uri_params['dist_au']
    else:
        guess_dist = args.guess_dist
        
    if args.n_workers is not None and args.n_workers >=1 and args.n_workers <= 65:
        n_workers = args.n_workers
    
    if not os.path.exists(args.result_dir):
        # Create the directory
        print_debug("creating results directory:", str(args.result_dir))
        os.makedirs(args.result_dir)

# Moved this section up and handling through URI file logic now 6/7/2024 COC
#   patch_corners = args.patch_corners
#   if os.path.exists(args.patch_corners):
#       found_corners = False
#       with open(args.patch_corners, 'r') as f:
#           lines = f.readlines()
#           for l in lines:
#               patch_header = "#patch_box="
#               if l.startswith(patch_header):
#                   patch_corners = eval(l[len(patch_header):])
#                   found_corners = True
#                   break
#       if not found_corners:
#           raise ValueError("Did not read patch corners from file:", args.patch_corners)
#   else:
#       patch_corners = eval(args.patch_corners), # TODO less dangerous way to evaluate patch corners
#   print_debug("Patch corners:", patch_corners)

def patch_arcmin_to_pixels(patch_size_arcmin, pixel_scale_arcsec_per_pix, verbose=True):
    """
    Take an array of two dimensions, in arcminutes, and convert this to pixels using the supplied pixel scale (in arcseconds per pixel).
    6/6/2024 COC
    """
    x_pixels = int(np.ceil( (patch_size_arcmin[0]*60)/pixel_scale_arcsec_per_pix ))
    y_pixels = int(np.ceil( (patch_size_arcmin[1]*60)/pixel_scale_arcsec_per_pix ))
    patch_pixels = [ x_pixels, y_pixels ]
    if verbose:
        print_debug(f'Derived patch_pixels = {patch_pixels} from patch_size_arcmin={patch_size_arcmin} and pixel_scale_arcsec_per_pix={pixel_scale_arcsec_per_pix}.')
    return x_pixels, y_pixels

if __name__ == '__main__':
    # initial pixel scale pass
    pixel_scale = args.pixel_scale
    if pixel_scale == None:
        if 'pixel_scale' in uri_params:
            pixel_scale = uri_params['pixel_scale']
    #
    # handle image dimensions
    image_width = args.image_width
    image_height = args.image_height
    if image_width == None or image_height == None:
        if 'patch_box' not in uri_params:
            raise KeyError(f'Must supply image dimensions (image_width, image_height) or #patch_size= must be in a specified URI file.')
        if pixel_scale == None:
            raise KeyError(f'When patch pixel dimensions are not specifified, the user must supply a pixel scale via the command line or the uri file.')
        image_width, image_height = patch_arcmin_to_pixels(patch_size_arcmin=uri_params['patch_size'], pixel_scale_arcsec_per_pix=pixel_scale, verbose=DEBUG)
    print_debug(f'(image_width, image_height) is ({image_width}, {image_height}).')



def create_wcs_from_corners(corners=None, image_width=None, image_height=None, save_fits=False, filename='output.fits', pixel_scale=None, verbose=True):
    """
    Create a WCS object given the RA, Dec coordinates of the four corners of the image
    and the dimensions of the image in pixels. Optionally save as a FITS file.

    Parameters:
    corners (list of lists): [[RA1, Dec1], [RA2, Dec2], [RA3, Dec3], [RA4, Dec4]]
    image_width (int): Width of the image in pixels; if none, pixel_scale will be used to determine size.
    image_height (int): Height of the image in pixels; if none, pixel_scale will be used to determine size.
    save_fits (bool): If True, save the WCS to a FITS file
    filename (str): The name of the FITS file to save
    pixel_scale (float): The pixel scale (in units of arcseconds per pixel); used if image_width or image_height is None.
    verbose (bool): Print more messages.

    Returns:
    WCS: The World Coordinate System object for the image

    5/6/2024 COC + ChatGPT4
    """
    # TODO switch to https://docs.astropy.org/en/stable/api/astropy.wcs.utils.fit_wcs_from_points.html
    # Extract the corners
    if verbose: print_debug(f'At the start, corners={corners}, type={type(corners)}')
    if type(corners) == type((0,1)) or len(corners) == 1:
        corners = corners[0]
        print_debug(f'After un-tuple corners are {corners}.')
    corners = np.unique(corners, axis=0) # eliminate duplicate coords in case someone passes the repeat 1st corner used for plotting 6/5/2024 COC/DO
    if len(corners) != 4:
        raise ValueError(f'There should be four (4) corners. We saw: {corners}')
    ra = [corner[0] for corner in corners]
    dec = [corner[1] for corner in corners]
    #
    # Calculate the central position (average of the coordinates)
    center_ra = np.mean(ra)
    center_dec = np.mean(dec)
    #
    # Calculate the pixel scale in degrees per pixel
    if pixel_scale is not None:
        pixel_scale_ra = pixel_scale/60/60
        pixel_scale_dec = pixel_scale/60/60
    else:
        ra_range = (max(ra) - min(ra)) # * np.cos(np.radians(center_dec))  # Adjust RA difference for declination; do not use cos(), results in incorrect pixel scale 6/6/2024 COC
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

if __name__ == '__main__':
    # moving the patch generation up so it can fail early if it will fail 6/5/2024 COC
    print_debug(f"Creating WCS from patch")
    
    # Create a WCS object for the patch. This will be our common reprojection WCS
    patch_wcs = create_wcs_from_corners(
        patch_corners,
        image_width,
        image_height,
        pixel_scale=pixel_scale,
        save_fits=False)
    
    ic = ImageCollection.read(args.ic_input_file, format='ascii.ecsv')
    print_debug(f"ImageCollection read from {args.ic_input_file}, creating work unit next...")

    last_time = time.time()
    orig_wu = ic.toWorkUnit(config=kbmod.configuration.SearchConfiguration.from_file(args.search_config))
    elapsed = round(time.time() - last_time,1)
    print_debug(f"{elapsed} seconds to create WorkUnit.")
    
    # TODO re-enable; reenabled 6/7/2024 COC
    orig_wu_file = os.path.join(args.result_dir, "orig_wu.fits")
    print_debug(f"Saving original work unit to: {orig_wu_file}")
    last_time = time.time()
    if len(glob.glob(orig_wu_file)) > 0: # handling bug where overwrite does not work for .to_fits below 6/8/2024 COC
        print_debug(f'Deleting existing {orig_wu_file}.')
        os.remove(orig_wu_file)
    orig_wu.to_fits(orig_wu_file, overwrite=True)
    elapsed = round(time.time() - last_time, 1)
    print_debug(f"{elapsed} seconds to write WorkUnit to disk: {orig_wu_file}")
    
    # gather elements needed for reproject phase
    imgs = orig_wu.im_stack
    point_on_earth = EarthLocation(1814303.74553723, -5214365.7436216, -3187340.56598756, unit='m')
    
    # Find the EBD (estimated barycentric distance) WCS for each image
    
    last_time = time.time()
    ebd_per_image_wcs, geocentric_dists = transform_wcses_to_ebd(
        orig_wu._per_image_wcs,
        imgs.get_single_image(0).get_width(),
        imgs.get_single_image(0).get_height(),
        guess_dist, # args.guess_dist, 
        [astropy.time.Time(img.get_obstime(), format='mjd') for img in imgs.get_images()],
        point_on_earth, 
        npoints=10, 
        seed=None,
    )
    elapsed = round(time.time() - last_time, 1)
    print_debug(f"{elapsed} seconds elapsed for transform WCS objects to EBD phase.")
    
    if len(orig_wu._per_image_wcs) != len(ebd_per_image_wcs):
        raise ValueError(f"Number of barycentric WCS objects ({len(ebd_per_image_wcs)}) does not match the original number of images ({len(orig_wu._per_image_wcs)}).")
    
    # Construct a WorkUnit with the EBD WCS and provenance data
    print_debug(f"Creating Barycentric WorkUnit...")
    last_time = time.time()
    ebd_wu = WorkUnit(
        im_stack=orig_wu.im_stack,
        config=orig_wu.config,
        per_image_wcs=orig_wu._per_image_wcs,
        per_image_ebd_wcs=ebd_per_image_wcs,
        heliocentric_distance=guess_dist,#args.guess_dist,
        geocentric_distances=geocentric_dists,
    )
    elapsed = round(time.time() - last_time, 1)
    print_debug(f"{elapsed} seconds elapsed to create EBD WorkUnit.")
    
    del orig_wu # 6/7/2024 COC
    
    # Reproject to a common WCS using the WCS for our patch
    print_debug(f"Reprojecting WorkUnit...")
    last_time = time.time()
    reprojected_wu = reprojection.reproject_work_unit(ebd_wu, patch_wcs, frame="ebd", max_parallel_processes=n_workers)
    elapsed = round(time.time() - last_time, 1)
    print_debug(f"{elapsed} seconds elapsed to create the reprojected WorkUnit.")
#   print_debug("Reprojected WorkUnit created")
    
    # Save the reprojected WorkUnit
    reprojected_wu_file = os.path.join(args.result_dir, "reprojected_wu.fits")
    print_debug(f"Saving reprojected work unit to: {reprojected_wu_file}")
    last_time = time.time()
    reprojected_wu.to_fits(reprojected_wu_file)
    elapsed = round(time.time() - last_time, 1)
    print_debug(f"{elapsed} seconds elapsed to create the reprojected WorkUnit: {reprojected_wu_file}")
