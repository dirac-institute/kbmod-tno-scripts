from astropy.io import fits
from astropy.wcs import WCS
import os

def retain_wcs_headers(input_fits: str, output_fits: str):
    print(f'Starting retain_wcs_headers(input_fits={input_fits}, output_fits={output_fits})...')
    with fits.open(input_fits) as hdul:
        # Create a new HDU list for the output file
        new_hdul = fits.HDUList()
        
        for hdu in hdul:
            header = hdu.header
            if 'NAXIS' in header and header['NAXIS'] > 0:
                # Remove image data but keep header and WCS
                new_hdu = fits.PrimaryHDU(header=header) if isinstance(hdu, fits.PrimaryHDU) else fits.ImageHDU(header=header)
            else:
                # Copy over the entire HDU if it has no image data
                new_hdu = hdu

            new_hdul.append(new_hdu)

        new_hdul.writeto(output_fits, overwrite=True)
        print(f'Wrote {output_fits} to disk.')


if __name__ == '__main__':
    import argparse # This is to enable command line arguments.
    parser = argparse.ArgumentParser(description='Strip out image data, leaving headers and WCS behind. By Colin Orion Chandler (COC) (7/16/2024)')
#	parser.add_argument('objects', metavar='O', type=str, nargs='+', help='starting row')
    parser.add_argument('files', help='fits file(s)', type=str, nargs='+') # 8/5/2021 COC: changing to not require this keyword explictly
#	parser.add_argument('--offsets', dest='offsets', help='offsets as quoted touples, like "-2,4" one per image.', type=str, nargs='+', default=[])
    parser.add_argument('--outdir', dest='outdir', help='output directory', type=str, default=None)
##	parser.add_argument('--convert-only', dest='convert_only', help='convert entire thing', type=bool, default=False)
##	parser.add_argument('--out-folder', dest='out_folder', help='output folder', type=str)
##	parser.add_argument('--do-arrows', dest='do_arrows', help='include anti-solar and anti-motion vector arrows', type=str, default='True')
#	parser.add_argument('--thumb-radius', dest='thumb_radius', help='thumbnail radius', type=int, default=None)
    args = parser.parse_args()
    
    ## Example usage
    #retain_wcs_headers('reprojected_wu.fits', 'reprojected_wu_stripped.fits')
    for fn in args.files:
        out_fn = fn.replace('.fits', '_stripped.fits')
        if args.outdir != None:
            out_fn = os.path.join(args.outdir, os.path.basename(out_fn))
        retain_wcs_headers(input_fits=fn, output_fits=out_fn)