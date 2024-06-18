import pandas as pd
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord# , EarthLocation, GCRS, ICRS, search_around_sky, get_body_barycentric
from astropy import units as u
import glob
from dateutil import parser
from astropy.time import Time
import datetime
from matplotlib import pyplot as plt
from region_search_functions import get_fakes_df # (desired_dates, base_csvfile='fakes_detections_simple.csv.gz', overwrite=False)
from region_search_functions import add_DEEP_ids # (df, fields_by_dt_csvfile='deep_field_by_date_coc.csv', verbose=False)
from region_search_functions import make_small_piles_go_away # (df, combined_fields_by_dt)
from region_search_functions import find_good_date_pointing_pairs # (df=df) needed for make_small_piles_go_away apparently
from region_search_functions import get_params_from_uri_file # (uri_file, verbose=False)
# TODO the above, add large pile stuff to the fakes and Skybot dataframes 6/17/2024 COC
from region_search_functions import  get_local_observing_date # (ut_datetime, utc_offset=-4, verbose=False)
from region_search_functions import make_desired_dates_str # (desired_dates, long=False)
from region_search_functions import find_closest_datetime
import os
import progressbar

from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool, cpu_count
import numpy as np



def add_visit_ids(df, pointings_df):
	visits = []
	uts = []
	ut_to_visit = pointings_df.set_index("ut")["visit"].to_dict()
	for ut in df['ut_datetime']:
		closest_ut = find_closest_datetime(datetime_list=list(set(pointings_df['ut_datetime'])), user_datetime=ut)
		visits.append(ut_to_visit(closest_ut))
	df['visit'] = visits
	return df

def get_skybot_df(skybot_csvfile, desired_dates, cache=True, verbose=True, butler_df=None, butler_csvfile=None):
	"""
	6/17/2024 COC
	"""
	desired_dates_str = make_desired_dates_str(desired_dates=desired_dates, long=False)
	mincsv = skybot_csvfile.replace('.csv.gz', f'_{desired_dates_str}.csv.gz')
	if len(glob.glob(mincsv)) == 0:
		print(f'Making {mincsv} from {skybot_csvfile}...')
		skybot_df = pd.read_csv(skybot_csvfile)
		if 'DEEP_id' not in skybot_df.columns: # 6/17/2024 COC
			skybot_df = add_DEEP_ids(skybot_df)
			skybot_df.to_csv(skybot_csvfile, index=False, compression='gzip')
			print(f'Wrote updated csvfile: {skybot_csvfile}')
		if 'visit' not in skybot_df.columns and butler_df is not None and butler_csvfile is not None: # 6/17/2024 COC
			skybot_df = add_visit_ids(df=orig_fakes_df, pointings_df=get_pointings_dataframe(df=butler_df, butler_csvfile=butler_csvfile))
			skybot_df.to_csv(skybot_csvfile, index=False, compression='gzip')
			print(f'Wrote updated csvfile: {skybot_csvfile}')
		if verbose:print(f'Available dates: {list(sorted(set(skybot_df["local_obsnight"])))}')
		rdf = skybot_df[skybot_df['local_obsnight'].isin(desired_dates)]
		rdf.to_csv(mincsv, compression='gzip', index=False)
	skybot_df = pd.read_csv(mincsv)
	return skybot_df
	

def read_table(fn):
	"""6/16/2024 COC"""
	ecsv_data = []
	with open(fn, 'r') as f:
		for line in f:
			ecsv_data.append(line)
	t = Table.read(ecsv_data, format='ascii.ecsv')
	return t

def xy_by_rate_and_mjd(x0, y0, vx, vy, t0, t):
	"""
	Figure out the (x,y) position for a given moment in time based on starting coordinates, rates.
	6/16/2024 COC
	"""
	elapsed_days = t - t0 # just use MJD I guess for simplicity
	x = x0 + vx*elapsed_days
	y = y0 + vy*elapsed_days
	return x, y


def process_row(args):
	i, t_row, mjds, headers = args
	xys = []
	rades = []
	obs_valids = []
	for j in range(len(mjds)):
		newx, newy = xy_by_rate_and_mjd(x0=t_row['x'], y0=t_row['y'], vx=t_row['vx'], vy=t_row['vy'], t0=mjds[0], t=mjds[j])
		w = WCS(headers[i])
		sky = w.pixel_to_world(newx, newy)
		rade = (sky.ra.deg, sky.dec.deg)
		rades.append(rade)
		xys.append((newx, newy))
		obs_valids.append(t_row['obs_valid'][j])
	return t_row['xy'], xys, rades, obs_valids

def main(t, mjds, hdul):
	nrows = len(t)
	headers = [hdul[f'SCI_{i}'].header for i in range(nrows)]
	t_rows = t.to_dict(orient='records')
	
	# Prepare arguments for process_map
	args = [(i, t_rows[i], mjds, headers) for i in range(nrows)]
	
	# Use process_map for parallel processing
	results = process_map(
		process_row,
		args,
		max_workers=cpu_count()
	)
	
	xy_dict = {}
	rade_dict = {}
	obs_valid_dict = {}
	
	for xy, xys, rades, obs_valids in results:
		xy_dict[xy] = xys
		rade_dict[xy] = rades
		obs_valid_dict[xy] = obs_valids
		
	return xy_dict, rade_dict, obs_valid_dict

# Example usage:
# xy_dict, rade_dict, obs_valid_dict = main(t, mjds, hdul)



def find_nearest_sbr(skybot_df, ra, dec, ut, dist_au):
	"""6/16/2024 COC"""
	our_coord = SkyCoord(ra=ra, dec=dec)
	available_uts = list(set(skybot_df["ut_datetime"]))
	closest_ut = find_closest_datetime(available_uts, ut)
	closest_ut = str(closest_ut)[0:-3] # .668000 -> .668
#	print(f'ut is {ut}, nearest was {closest_ut}')
#	print(f'len(skybot_df) = {len(skybot_df)}; available are {available_uts}')
	skybot_ut_df = skybot_df[skybot_df['ut_datetime']==closest_ut]
#	print(len(skybot_ut_df))
	coords = SkyCoord(ra=[skybot_ut_df[f'RA_{dist_au}'].iloc()[i]*u.degree for i in range(0,len(skybot_ut_df))],
										dec=[skybot_ut_df[f'Dec_{dist_au}'].iloc()[i]*u.degree for i in range(0,len(skybot_ut_df))],
#										frame='icrs'
	)
	separations = our_coord.separation(coords).arcsecond
	min_sep = min(separations)
	min_sep_idx = list(separations).index(min_sep)
	print(f'Nearest neighbor (idx={min_sep_idx}) was {min_sep}" away (ut was {ut}, closest_ut was {closest_ut}).')
	
def mjd_to_ut(mjd, offset_s=0):# thought -60 would be necessary for MJD 6/16/2024 COC
	return(parser.parse(Time(mjd, format='mjd').isot) + datetime.timedelta(seconds=offset_s))

def get_patch_mins_maxes(patch):
	min_ra = min([i[0] for i in patch])
	max_ra = max([i[0] for i in patch])
	min_dec = min([i[1] for i in patch])
	max_dec = max([i[1] for i in patch])
	d = {'min_ra':min_ra, 'max_ra':max_ra, 'min_dec':min_dec, 'max_dec':max_dec}
	return d


def assess_results(uri_file, wu_fits, results_ecsv, skybot_csvfile, fakes_csvfile='fakes_detections_simple.csv.gz', common_dir='/gscratch/dirac/coc123/common_files', multi_night=True):
	print(f'Starting assess_results(uri_file={uri_file}, wu_fits={wu_fits}, results_ecsv={results_ecsv}, skybot_csvfile={skybot_csvfile}, fakes_csvfile={fakes_csvfile}, common_dir={common_dir}, multi_night={multi_night})...')
#	uri_file = 'uris_20x20_60.0au_2019-04-02_and_2019-05-07_slice2_patch5818_lim5000.lst'
#	uri_file = 'uris_20x20_42.0au_2019-04-02_and_2019-05-07_slice3_patch5826_lim5000.lst'
	uri_params = get_params_from_uri_file(uri_file=uri_file, verbose=False)
	print(f'Available URI params: {uri_params.keys()}')
#	print(uri_params['patch_box'])
	desired_dates = uri_params['desired_dates'] # ['2019-04-02', '2019-05-07']
	desired_distances = [uri_params['dist_au']] #[60.0]
	dist_au = uri_params['dist_au']
#	skybot_csvfile = 'region_search_df_A0_differenceExp_skybot_clean_simple.csv.gz'
	# ok a bit silly to load the butler DF just for the pointing DF, but kludge for now 6/17/2024 COC
	butler_csvfile = f'{common_dir}/region_search_df_A0_differenceExp.csv.gz' # 'dataframe_A0_differenceExp.csv.gz'
	butler_df = pd.read_csv(butler_csvfile)
	skybot_df = get_skybot_df(skybot_csvfile=f'{common_dir}/{skybot_csvfile}', desired_dates=desired_dates, cache=True)
	fakes_df = get_fakes_df(desired_dates, base_csvfile=f'{common_dir}/{fakes_csvfile}', overwrite=False)
	if 'visit' not in fakes_df:
		fakes_df = get_fakes_df(desired_dates, base_csvfile=f'{common_dir}/{fakes_csvfile}', overwrite=False, butler_df=butler_df, butler_csvfile=butler_csvfile)
	print(f'skybot_df, len={len(skybot_df)}, has columns {skybot_df.columns}')
	
#	t = read_table(fn='full_results.ecsv')
	t = read_table(fn=results_ecsv)
#	t = t.sort('likelihood', reverse=True) # 6/18/2024 COC (1am oy) was going to try this but instead going to sleep
	
	n_sci = len(t['obs_valid'][0])
	print(f'There are {n_sci} science extensions.')

#	hdul = fits.open('reprojected_wu_stripped.fits')
#	hdul = fits.open('reprojected_wu_stripped_drew42auSlice3.fits')
	hdul = fits.open(wu_fits)
	print(len(hdul))
	#print(hdul.info()) # LOTS of extensions, e.g., EBD_971, WCS_971, SCI_180, EBD_109, WCS_109
	hdu_sci_test = hdul['SCI_188']
	print(list(hdu_sci_test.header.keys()))
	print(hdu_sci_test.header['MJD'])
	
	mjds = []
	obsnights = []
	sci_uts = []
	nrows = len(t)
	if len(t) > 949:
		print(f'Printing slice 949 of the table:')
		print(t[949])
	print(f'Results Table has columns {t.columns}')
	t['xy'] = [f'{t["x"][i]}_{t["y"][i]}' for i in range(0,len(t))]
	
	for i in range(0,n_sci):
		this_mjd = hdul[f'SCI_{i}'].header['MJD']
		mjds.append(this_mjd)
		this_ut = str(Time(this_mjd, format='mjd').isot)
		local_obsnight = get_local_observing_date(ut_datetime=this_ut, utc_offset=-4, verbose=False)
		sci_uts.append(this_ut)
		obsnights.append(local_obsnight) #
	#
	n_nights_dict = {}
	for i in range(0,nrows):
		obs_valid_list = t['obs_valid'][i]
		filtered_dates = [str(date) for date, flag in zip(obsnights, obs_valid_list) if flag]
#		print(filtered_dates)
		n_nights_dict[t['xy'][i]] = len(list(set(filtered_dates)))
			
	#print(f'Found {len(mjds)} MJDs.')
	#n_sci = 
	#print(t)
	#print(t['xy'])
	xy_dict = {}
	rade_dict = {}
	obs_valid_dict = {}
	# probably need to reverse (or invert later) the following so we can do search-around-sky on a timestep (layer) basis 6/16/2024 COC
#	xy_dict, rade_dict, obs_valid_dict = main(t, mjds, hdul)
	h = hdul[f'SCI_0'].header
#	h['NAXIS'] = (4563, 4563)
	w = WCS(h)
	special_mapping = {}
	special = [949] # for going after specific result slices
	for i in range(0,nrows):
#		print(f'Processing row {i} of {nrows}')
#		if i != 949: continue
		xy = t['xy'][i] # just our unique ID
		if multi_night == True and n_nights_dict[xy] < 2: continue
		if i in special:
			special_mapping[xy] = i
			print(f'encountered special {i}, xy={xy}')
		xys = []
		rades = []
		obs_valids = []
		for j in range(0,n_sci):
			newx, newy = xy_by_rate_and_mjd(x0=t['x'][i], y0=t['y'][i], vx=t['vx'][i], vy=t['vy'][i], t0=mjds[0], t=mjds[j])
			sky = w.pixel_to_world(newx, newy)
			rade = (sky.ra.deg, sky.dec.deg)
			rades.append(rade)
			#		print(f'Table row {i}, WU slice {j}: (newx, newy) = ({newx}, {newy}), (RA, Dec) = {rade[0], rade[1]}')
			xys.append((newx, newy))
			obs_valids.append(t['obs_valid'][i][j])
			#		mjds.append(hdul[f'SCI_{i}'].header['MJD']) # wait we already have MJDs
		xy_dict[xy] = xys
		rade_dict[xy] = rades
		obs_valid_dict[xy] = obs_valids
#		if i >= 100:
#			print(f'WARNING!!!!!!!!!!!! Stopping after 10th row!!!')
#			break
	print(f'After calculations, we have {len(rade_dict)} items in rade_dict.')
	#
	print(f'Slimming down skybot_df to be within patch bounds. Size before was {len(skybot_df)} rows.')
	patch_bounds = get_patch_mins_maxes(uri_params['patch_box'])
	skybot_df = skybot_df[skybot_df[f'RA_{dist_au}'] < patch_bounds['max_ra']]
	skybot_df = skybot_df[skybot_df[f'RA_{dist_au}'] > patch_bounds['min_ra']]
	skybot_df = skybot_df[skybot_df[f'Dec_{dist_au}'] < patch_bounds['max_dec']]
	skybot_df = skybot_df[skybot_df[f'Dec_{dist_au}'] > patch_bounds['min_dec']]
	print(f'After slimming, skybot_df has {len(skybot_df)} rows.')
	legended = []
	fig = plt.figure(figsize=[8,8])
	for kbmod_find in rade_dict:
		colors = []
		for i in obs_valid_dict[kbmod_find]:
			if i == False:
				colors.append('gray')
			else:
				colors.append('red')
		kbmod_ras = [rade_dict[kbmod_find][i][0] for i in range(0,len(rade_dict[kbmod_find]))]
		kbmod_decs = [rade_dict[kbmod_find][i][1] for i in range(0,len(rade_dict[kbmod_find]))]
		label = 'KBMOD'
		if label in legended:
			label = None
		else:
			legended.append(label)
#		plt.plot(kbmod_ras, kbmod_decs, color='red', label=label, linewidth=1)
		plt.plot(kbmod_ras, kbmod_decs, color='black', label=label, linewidth=0.5, alpha=0.25)
		plt.scatter(kbmod_ras, kbmod_decs, color=colors, label=label, linewidth=1, s=1)
	
#	dt_skybot_df = skybot_df[skybot_df['local_obsnight'].isin(desired_dates)]  # wait this already happened 6/17/2024 COC
#	skybot_df = skybot_df[skybot_df['DEEP_id'] == 'A0b']
#	minmax asdf
#	bound_skybot_df = skybot_df[skybot_df['RA'] > ]
	print(f'see unique DEEP_ids in skybot_df are {list(set(skybot_df["DEEP_id"]))}.')
	for objname in list(set(skybot_df['Name'])):
		obj_df = skybot_df[skybot_df['Name']==objname]
		label = 'Known SSO'
		if label in legended:
			label = None
		else:
			legended.append(label)
		plt.plot(obj_df[f'RA_{dist_au}'], obj_df[f'Dec_{dist_au}'], color='blue', label=label, alpha=0.5, linewidth=0.5)
		plt.scatter(obj_df[f'RA_{dist_au}'], obj_df[f'Dec_{dist_au}'], color='blue', label=label, s=1)
	#
	# Fakes next
	print(f'Before fakes_df diet, it had {len(fakes_df)} rows.')
	fakes_df = fakes_df[fakes_df[f'RA_{dist_au}'] < patch_bounds['max_ra']]
	fakes_df = fakes_df[fakes_df[f'RA_{dist_au}'] > patch_bounds['min_ra']]
	fakes_df = fakes_df[fakes_df[f'Dec_{dist_au}'] < patch_bounds['max_dec']]
	fakes_df = fakes_df[fakes_df[f'Dec_{dist_au}'] > patch_bounds['min_dec']]
	print(f'After fakes diet, it had {len(fakes_df)} rows.')
	print(f'see unique DEEP_ids in skybot_df are {list(set(skybot_df["DEEP_id"]))}.')
	for objname in list(set(fakes_df['ORBITID'])):
		obj_df = fakes_df[fakes_df['ORBITID']==objname]
		label = 'Fake'
		if label in legended:
			label = None
		else:
			legended.append(label)
		plt.plot(obj_df[f'RA_{dist_au}'], obj_df[f'Dec_{dist_au}'], color='purple', label=label, linewidth=0.5, alpha=0.5)
		plt.scatter(obj_df[f'RA_{dist_au}'], obj_df[f'Dec_{dist_au}'], color='purple', label=label, s=1)
	#
	plt.gca().set_aspect('equal')
	print('patch box: ', uri_params['patch_box'])
	plt.plot([uri_params['patch_box'][i][0] for i in range(0,len(uri_params['patch_box']))], [uri_params['patch_box'][i][1] for i in range(0,len(uri_params['patch_box']))], color='green', linestyle='--')
	plt.legend()
#	dist_au = 60.0 # KLUDGE 6/17/2024 COC
	plt.xlabel(f'{dist_au} au-corrected RA [°]')
	plt.ylabel(f'{dist_au} au-corrected Dec [°]')
	plot_ylims = plt.gca().get_ylim()
	plot_xlims = plt.gca().get_xlim()
	for ext in ['pdf', 'png']:
		plt.savefig(f'results_with_skybot_and_fakes_multinight{multi_night}.{ext}')
	plt.show()
	plt.close()
	#
	animate = True
	if animate:
		animdir = 'anim'
		os.makedirs(animdir, exist_ok=True)
		fakes_uts = list(set(fakes_df['ut']))
		skybot_uts = list(set(skybot_df['ut_datetime']))
		patch_ras = [uri_params['patch_box'][i][0] for i in range(0,len(uri_params['patch_box']))]
		patch_decs = [uri_params['patch_box'][i][1] for i in range(0,len(uri_params['patch_box']))]
		with progressbar.ProgressBar(max_value=n_sci) as bar:
			for timestep in range(0,n_sci):
				legended = []
				#
				# KBMOD
				for i,kbmod_find in enumerate(rade_dict):
					if obs_valid_dict[kbmod_find][timestep] == False:
							this_color = 'gray'
					else:
						this_color = 'red'
					kbmod_ra = rade_dict[kbmod_find][timestep][0]
					kbmod_dec = rade_dict[kbmod_find][timestep][1]
					label = 'KBMOD'
					if label in legended:
						label = None
					else:
						legended.append(label)
					if kbmod_find not in special_mapping:
						plt.scatter(kbmod_ra, kbmod_dec, color=this_color, label=label, s=1)
					else:
						plt.scatter(kbmod_ra, kbmod_dec, color=this_color, label=f'{special_mapping[kbmod_find]}', s=15, marker='*')
	#			plt.scatter(kbmod_ras[timestep], kbmod_decs[timestep], color=colors, label=label, linewidth=1, s=1)
				# Skybot
				nearest_skybot_ut = find_closest_datetime(datetime_list=skybot_uts, user_datetime=parser.parse(sci_uts[timestep]))
				nearest_skybot_ut = str(nearest_skybot_ut)[0:-3]
				skybot_ut_df = skybot_df[skybot_df['ut_datetime']==nearest_skybot_ut]
				label = 'Known SSO'
				if label in legended:
					label = None
				else:
					legended.append(label)
				plt.scatter(skybot_ut_df[f'RA_{dist_au}'], skybot_ut_df[f'Dec_{dist_au}'], color='blue', label=label, s=1)
				#
				# Fakes
				nearest_fakes_ut = find_closest_datetime(datetime_list=fakes_uts, user_datetime=parser.parse(sci_uts[timestep]))
				nearest_fakes_ut = str(nearest_fakes_ut)#[0:-3]
				fakes_ut_df = fakes_df[fakes_df['ut']==nearest_fakes_ut]
				if len(fakes_ut_df) == 0:
					raise KeyError(f'Empty fakes dataframe when matching {sci_uts[timestep]} sci_ut to fakes_uts {fakes_uts}.')
				label = 'Fake'
				if label in legended:
					label = None
				else:
					legended.append(label)
				plt.scatter(fakes_ut_df[f'RA_{dist_au}'], fakes_ut_df[f'Dec_{dist_au}'], color='purple', label=label, s=1)
				#
				# The Patch
				plt.plot(patch_ras, patch_decs, color='green', linestyle='--')
				# finalize
				plt.legend(loc='lower left')
			#	dist_au = 60.0 # KLUDGE 6/17/2024 COC
				plt.xlabel(f'{dist_au} au-corrected RA [°]')
				plt.ylabel(f'{dist_au} au-corrected Dec [°]')
				plt.ylim(plot_ylims)
				plt.xlim(plot_xlims)
				plt.gca().set_aspect('equal')
				plt.title(f'{sci_uts[timestep]}')
				outfile = f'{animdir}/step{timestep}.png'
				plt.savefig(outfile)
				plt.close()
				bar.update(timestep)
			
	exit()
#	for i,rade in enumerate(rade_dict[kbmod_find]):

if __name__ == '__main__':
	import argparse # This is to enable command line arguments.
	argparser = argparse.ArgumentParser(description='Strip out image data, leaving headers and WCS behind. By Colin Orion Chandler (COC) (7/16/2024)')
#	argparser.add_argument('objects', metavar='O', type=str, nargs='+', help='starting row')
#	argparser.add_argument('files', help='fits file(s)', type=str, nargs='+') # 8/5/2021 COC: changing to not require this keyword explictly
#	argparser.add_argument('--offsets', dest='offsets', help='offsets as quoted touples, like "-2,4" one per image.', type=str, nargs='+', default=[])
	argparser.add_argument('--wu-fits', dest='wufits', help='WU fitsfile (or stripped)', type=str, default=None)
	argparser.add_argument('--results-ecsv', dest='resultsecsv', help='results ecsv file', type=str, default=None)
	argparser.add_argument('--uri-file', dest='urifile', help='results ecsv file', type=str, default=None)
	argparser.add_argument('--skybot-csvfile', dest='skybotcsv', help='Skybot csvfile', type=str, default='region_search_df_A0_differenceExp_skybot_clean_simple.csv.gz')
	argparser.add_argument('--fakes-csvfile', dest='fakescsv', help='Fakes csvfile', type=str, default='fakes_detections_simple.csv.gz')
	argparser.add_argument('--common-path', dest='commonpath', help='Common path for common files like skybot csv', type=str, default='/gscratch/dirac/coc123/common_files')
	argparser.add_argument('--disable-multi-night', dest='nomultinight', help='disable requirement for multi-night finds', type=bool, default=False)
	#	parser.add_argument('--convert-only', dest='convert_only', help='convert entire thing', type=bool, default=False)
##	parser.add_argument('--out-folder', dest='out_folder', help='output folder', type=str)
##	parser.add_argument('--do-arrows', dest='do_arrows', help='include anti-solar and anti-motion vector arrows', type=str, default='True')
#	parser.add_argument('--thumb-radius', dest='thumb_radius', help='thumbnail radius', type=int, default=None)
	args = argparser.parse_args()
	assess_results(uri_file=args.urifile, 
					wu_fits=args.wufits, 
					results_ecsv=args.resultsecsv, 
					skybot_csvfile=args.skybotcsv, 
					fakes_csvfile=args.fakescsv, 
					common_dir=args.commonpath,
					multi_night=not args.nomultinight
				)
	## Example usage
	#retain_wcs_headers('reprojected_wu.fits', 'reprojected_wu_stripped.fits')
	

#if __name__ == '__main__':
#	
#	first_ra = rade_dict[list(rade_dict.keys())[0]][0][0]*u.degree
#	first_dec = rade_dict[list(rade_dict.keys())[0]][0][1]*u.degree
#	for visit in rade_dict:
#		for i,rade in enumerate(rade_dict[visit]):
#			find_nearest_sbr(skybot_df=skybot_df, ra=rade[0]*u.degree, dec=rade[1]*u.degree, ut=mjd_to_ut(mjd=mjds[i]), dist_au=60.0)