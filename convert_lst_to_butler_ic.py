"""
A script to convert the .lst files from the pre-TNOs KBMOD pipeline to the butler standardizer generated ImageCollections.

This allows the old results to use the current KBMOD parsl pipelines and have zero point correction and updated WCS information.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from astropy.wcs import WCS

from kbmod import ImageCollection

parser = argparse.ArgumentParser(description='Convert LST files to ImageCollection files with WCS information.')

parser.add_argument('--cached_butler_path', type=str, help='The path to the cached butler file which has both the URI and the visit/detector information.')
parser.add_argument('--full_ic_path', type=str, help='The path to the full ImageCollection representing our butler repository.')
parser.add_argument('--ic_out_dir', type=str, help='The directory to save the new ImageCollections to.')
parser.add_argument('--uri_lists_dir', type=str, help='The directory containing the LST files with the URIs to convert.')
parser.add_argumet('--guess_distance', type=float, help='A distance to reflect-correct this WorkUnit to.')

args = parser.parse_args()

def get_uris(lst_path):
    """
    Get the URIs from an .lst file.

    Parameters
    ----------
    lst_path : str
        The path to the .lst file.

    Returns
    -------
    list
        The list of URIs in the .lst file.
    """
    uris = []
    with open(lst_path) as f:
        for l in f.readlines():
            l = l.strip()  # seeing invisible trailing characters 6/12/2024 COC
            if l == "":
                continue  # skip blank lines 6/12/2024 COC
            if not l.startswith("#"):
                # Ignore commented metadata
                uris.append(l)

    # Clean up the URI strings
    for i in range(len(uris)):
        # clean up character encoding
        curr = uris[i].replace("%23", "#").strip()

        # strip off the file:// prefix if it exists
        file_prefix = "file://"
        if curr.startswith(file_prefix):
            curr = curr[len(file_prefix) :]

        uris[i] = curr
    return uris



def get_from_ic(ic, visit, detector):
    """
    Get the row from an ImageCollection with the given visit and detector.

    Parameters
    ----------
    ic : ImageCollection
        The ImageCollection to search in.
    visit : int
        The visit to search for.
    detector : int
        The detector to search for.

    Returns
    -------
    astropy.table.row.Row
        The row from the ImageCollection with the given visit and detector.
    """
    res = ic.data[(ic.data['visit'] == visit) & (ic.data['detector'] == detector)]
    if len(res) > 1:
        raise ValueError(f'Duplicate visit and detector: {visit}, {detector}')
    elif len(res) == 0:
        return None
    return res

def populate_new_ic(uris, ics):
    """
    Populate a new ImageCollection with the data from the URIs in the LST file.
    
    Parameters
    ----------
    uris : list
        The list of URIs to populate the new ImageCollection with.
        
    ics : list
        The list of ImageCollections to search for the data.
        
    Returns
    -------
    ImageCollection
        The new ImageCollection populated with the data from the URIs.
    """
    result_ic = ics[0].copy()
    result_ic.data = result_ic.data[:0]
    for i, uri in enumerate(uris):
        print(f'Looking for URI {i + 1}/{len(uris)}')
        cache_row = cached_butler[cached_butler['uri'] == uri]
        if len(cache_row) > 1:
            cache_row = cache_row[0]
        for ic in ics:
            visit = cache_row['visit'].values[0]
            detector = cache_row['detector'].values[0]
            row = get_from_ic(ic, visit, detector)
            if row is not None:
                result_ic.data.add_row(dict(row))
                break
    result_ic.data['std_idx'] = list(range(len(result_ic.data)))
    return result_ic

# TODO this was copied from other scripts in the repo.
def _create_wcs_from_corners(
    corners=None,
    image_width=None,
    image_height=None,
    pixel_scale=None,
    verbose=True,
):
    """
    Create a WCS object given the RA, Dec coordinates of the four corners of the image
    and the dimensions of the image in pixels. Optionally save as a FITS file.

    Parameters:
    corners (list of lists): [[RA1, Dec1], [RA2, Dec2], [RA3, Dec3], [RA4, Dec4]]
    image_width (int): Width of the image in pixels; if none, pixel_scale will be used to determine size.
    image_height (int): Height of the image in pixels; if none, pixel_scale will be used to determine size.
    filename (str): The name of the FITS file to save
    pixel_scale (float): The pixel scale (in units of arcseconds per pixel); used if image_width or image_height is None.
    verbose (bool): Print more messages.

    Returns:
    WCS: The World Coordinate System object for the image

    5/6/2024 COC + ChatGPT4
    """
    # TODO switch to https://docs.astropy.org/en/stable/api/astropy.wcs.utils.fit_wcs_from_points.html
    # Extract the corners
    if verbose:
        print(f"At the start, corners={corners}, type={type(corners)}")
    if type(corners) == type((0, 1)) or len(corners) == 1:
        corners = corners[0]
        print(f"After un-tuple corners are {corners}.")
    corners = np.unique(
        corners, axis=0
    )  # eliminate duplicate coords in case someone passes the repeat 1st corner used for plotting 6/5/2024 COC/DO
    if len(corners) != 4:
        raise ValueError(f"There should be four (4) corners. We saw: {corners}")
    ra = [corner[0] for corner in corners]
    dec = [corner[1] for corner in corners]
    #
    # Calculate the central position (average of the coordinates)
    center_ra = np.mean(ra)
    center_dec = np.mean(dec)
    #
    # Calculate the pixel scale in degrees per pixel
    if pixel_scale is not None:
        pixel_scale_ra = pixel_scale / 60 / 60
        pixel_scale_dec = pixel_scale / 60 / 60
    else:
        ra_range = max(ra) - min(
            ra
        )  # * np.cos(np.radians(center_dec))  # Adjust RA difference for declination; do not use cos(), results in incorrect pixel scale 6/6/2024 COC
        dec_range = max(dec) - min(dec)
        pixel_scale_ra = ra_range / image_width
        pixel_scale_dec = dec_range / image_height
    if verbose:
        print(
            f'Saw (RA,Dec) pixel scales ({pixel_scale_ra*60*60},{pixel_scale_dec*60*60})"/pixel. User-supplied: {pixel_scale}"/pixel.'
        )
    # Initialize a WCS object with 2 axes (RA and Dec)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [image_width / 2, image_height / 2]
    wcs.wcs.crval = [center_ra, center_dec]
    wcs.wcs.cdelt = [
        -pixel_scale_ra,
        pixel_scale_dec,
    ]  # RA pixel scale might need to be negative (convention)
    # Rotation matrix, assuming no rotation
    wcs.wcs.pc = [[1, 0], [0, 1]]
    # Define coordinate frame and projection
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.array_shape = (image_height, image_width)

    return wcs

def _patch_arcmin_to_pixels(patch_size, pixel_scale):
    """Operate on the patch_size array (with size (2,1)) to convert to
    pixels. Uses pixel_scale to do the conversion.

    Returns
    -------
    nd.array
        A 2d array with shape (2,1) containing the width and height of the
        patch in pixels.
    """
    patch_pixels = np.ceil(np.array(patch_size) * 60 / pixel_scale).astype(int)

    print(
        f"Derived patch_pixels (w, h) = {patch_pixels} from patch_size={patch_size}[arcmin] and pixel_scale={pixel_scale}[arcsec/pixel]."
    )

    return patch_pixels

# TODO copied from other scripts in the repo
def get_params_from_uri_file(uri_filepath):
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
    with open(uri_filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue  # deal with rogue blank lines or invisible double line endings
            if not line.startswith("#"):
                break  # comments section is done
            line = line.lstrip("#").split("=")
            lhs = line[0].strip()
            rhs = line[1].strip()
            try:
                rhs = eval(rhs)
            except ValueError:
                print(f"Unable to eval() {lhs} field with value {rhs}.")
                continue
            results[lhs] = rhs
    return results

def convert_lst_to_butler_ic(lst_filepath, full_ic, ic_out_dir, guess_distance=None):
    """
    Convert a LST file to an ImageCollection with WCS information.

    Parameters
    ----------
    lst_filepath : str
        The path to the LST file to convert.
    full_ic : ImageCollection
        The full ImageCollection representing the butler repository.
    ic_out_dir : str
        The directory to save the new ImageCollection to.
    guess_distance : float, optional
        A distance to reflect-correct this WorkUnit to, by default None

    Returns
    -------
    ImageCollection
        The new ImageCollection created from the LST file.
    """
    lst_uris = get_uris(lst_filepath)

    params = get_params_from_uri_file(lst_filepath)

    image_width, image_height = _patch_arcmin_to_pixels(params["patch_size"], params["pixel_scale"])

    ic = populate_new_ic(lst_uris, [full_ic])

    print(f"Creating WCS from patch")
    patch_wcs = _create_wcs_from_corners(
        params["patch_box"],
        image_width,
        image_height,
        pixel_scale=params["pixel_scale"],
    )
    my_header = patch_wcs.to_header()
    my_header['NAXIS2'] = image_width
    my_header['NAXIS1'] = image_height

    ic.data['global_wcs'] = np.str_(my_header) 

    ic.data['std_idx'] = list(range(len(ic)))

    if guess_distance is not None:
        print(f"Reflect-correcting to distance {guess_distance} in column 'helio_guess_distance'.")
        ic.data['helio_guess_distance'] = guess_distance

    ic_filename = '.'.join(os.path.basename(lst_filepath).split('.')[:-1]) + '.collection'
    ic_path = os.path.join(ic_out_dir, ic_filename)
    print(f"Writing to path {ic_path}")

    ic.write(ic_path, overwrite=True)

    new_ic = ImageCollection.read(ic_path)
    print(f"Converted old ic of size {len(ic)} to ic of size {len(new_ic)}")
    return new_ic

# We use cached data from the butler such as at  "/mmfs1/home/wbeebe/dirac/kbmod/kbmod_wf/kbmod_new_ic/slice3/staging/region_search_df_A0_differenceExp.csv"
cached_butler = pd.read_csv(args.cached_butler_path)

# The full butler derived info for DEEP is in the Image Collection at "/gscratch/dirac/DEEP/collab/image_collections.new/joined.collection"
full_ic = ImageCollection.read(args.full_ic_path)

# Get all the .lst files in the directory
pattern = os.path.join(args.uri_lists_dir, "*.lst")
lst_filepaths = glob.glob(pattern)

for f in lst_filepaths:
    convert_lst_to_butler_ic(f, full_ic, args.guess_distance)

