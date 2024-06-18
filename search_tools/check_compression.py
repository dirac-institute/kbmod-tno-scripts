# Colin Orion Chandler + ChatGPT4o 20240606

import pandas as pd
from astropy.io import fits
import numpy as np

def get_uncompressed_size(hdu):
	"""Calculate the size of the uncompressed data in MB."""
	if isinstance(hdu, fits.PrimaryHDU) or isinstance(hdu, fits.ImageHDU):
		return hdu.data.nbytes / (1024 ** 2) if hdu.data is not None else 0
	elif isinstance(hdu, fits.BinTableHDU) or isinstance(hdu, fits.TableHDU):
		return hdu.data.nbytes / (1024 ** 2) if hdu.data is not None else 0
	elif isinstance(hdu, fits.CompImageHDU):
		return hdu.data.nbytes / (1024 ** 2) if hdu.data is not None else 0
	else:
		return 0
	
def examine_fits_compression(file_path):
	# Open the FITS file
	hdul = fits.open(file_path)
	
	# Prepare lists to store data
	indices = []
	names = []
	compressions = []
	uncompressed_sizes = []
	
	# Iterate over each HDU
	for i, hdu in enumerate(hdul):
		indices.append(i)
		names.append(hdu.name)
		
		# Check if the HDU is compressed
		if isinstance(hdu, fits.CompImageHDU):
			compressions.append('Compressed Image')
		elif isinstance(hdu, fits.BinTableHDU) and hasattr(hdu, 'compType') and hdu.compType:
			compressions.append('Compressed Table')
		else:
			compressions.append('None')
			
		# Get uncompressed size
		uncompressed_size = get_uncompressed_size(hdu)
		uncompressed_sizes.append(uncompressed_size)
		
	# Close the FITS file
	hdul.close()
	
	# Create a DataFrame
	df = pd.DataFrame({
		'Index': indices,
		'Name': names,
		'Compression': compressions,
		'Uncompressed Size (MB)': uncompressed_sizes
	})
	
	return df

# Example usage
file_path = 'differenceExp_DECam_VR_VR_DECam_c0007_6300_0_2600_0_855307_N10_DEEP_20190505_A0c_scienceHASHstep6_20240424T203517Z.fits'
df = examine_fits_compression(file_path)
print(df)
print(f'Total size: {np.sum(df["Uncompressed Size (MB)"])} Mb.')