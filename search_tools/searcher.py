import glob
import os
import sys
#from os.path import dirname
if __name__ == '__main__':
	sys.path.append(f'{os.environ["HOME"]}/bin')
	sys.path.append(f'.')
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone

from functools import partial

import lsst.sphgeom as sphgeom
from lsst.sphgeom import ConvexPolygon, UnitVector3d
from dateutil import parser
import progressbar
from scipy.optimize import curve_fit
from deep_fields_coc import determine_field, fetch_deep_field_coords#, get_local_observing_date
from astropy.coordinates import SkyCoord, EarthLocation, GCRS, ICRS, search_around_sky, get_body_barycentric

from astropy import units as u
from astroquery.imcce import Skybot
from astropy.time import Time

from scipy.optimize import minimize
from tqdm.contrib.concurrent import process_map
import tqdm
from multiprocessing import Pool, cpu_count
if __name__ == '__main__':
	print(f'See cpu_count={cpu_count()}')

from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from region_search_functions import merge_dataframes # (df1, df2, key="uri")
from region_search_functions import add_DEEP_ids # (df, fields_by_dt_csvfile='deep_field_by_date_coc.csv', verbose=False)
from region_search_functions import get_available_distances
#from region_search_functions import correct_parallax # (coord, obstime, point_on_earth, heliocentric_distance)
from region_search_functions import get_local_observing_date, butler_rad_to_float_degree, dataId_to_dataIdStr
from region_search_functions import fix_fit_columns # (df, verbose=True)
from region_search_functions import purge_old_coord_cols # (df, verbose=True)
from region_search_functions import get_fakes_df # (desired_dates, base_csvfile='fakes_detections_simple.csv.gz', overwrite=False) # moved 6/17/2024 COC
from region_search_functions import make_desired_dates_str # (desired_dates, long=False) # 6/17/2024 COC
from region_search_functions import make_small_piles_go_away # (df, combined_fields_by_dt)
from region_search_functions import find_good_date_pointing_pairs # (df, min_exposures=20)
from region_search_functions import find_closest_datetime # (datetime_list, user_datetime)
from region_search_functions import get_pointings_dataframe

from linearity import assess_linearity3 # (fakes_df, desired_dates, ORBITID, verbose=True, dist_au=40.0, plot=False, show=False)
from linearity import assess_linearity2 # (fakes_df, desired_dates, ORBITID, verbose=True, dist_au=40.0, plot=False, show=False)
#def merge_dataframes(df1, df2, key="uri"):
#	# Perform the merge
#	merged_df = pd.merge(df1, df2, on=key, suffixes=('_df1', '_df2'))
#	
#	# Identify duplicate columns
#	duplicate_columns = [col for col in df1.columns if col in df2.columns and col != key]
#	
#	# Combine duplicate columns
#	for col in duplicate_columns:
#		merged_df[col] = merged_df.apply(lambda row: row[col + '_df1'] if pd.notnull(row[col + '_df1']) else row[col + '_df2'], axis=1)
#		
#		# Drop the duplicate columns
#		merged_df.drop(columns=[col + '_df1', col + '_df2'], inplace=True)
#		
#	return merged_df


# assess_coordinate_differences() moved to get_centers_and_corners 6/1/2024 COC

if __name__ == '__main__':
	butler_csvfile = 'region_search_df_A0_differenceExp.csv.gz' # 'dataframe_A0_differenceExp.csv.gz'
	print(f'butler_csvfile is {butler_csvfile}')

# merge_in_pointings_by_uri() moved to region_search_functions 6/1/2024 COC

#def process_deepid_row(args):
#	rade, dt, df, local_obsdates = args
#	deep_id_df = determine_field(rade, dt, df, local_obsdates)
#	return deep_id_df['DEEP_id']
#
#def add_DEEP_ids(df, fields_by_dt_csvfile='deep_field_by_date_coc.csv', verbose=False):
#	if verbose:
#		print(f'add_DEEP_ids(df) sees df.columns: {df.columns}')
#	what = 'butler'
#	if 'RA' in df.columns and 'DEC' in df.columns:
#		what = 'fakes'
#	print(f'Adding DEEP_id to the DataFrame. what={what}')
#	
#	# Read CSV file once and pass it to the determine_field function
#	fields_df = pd.read_csv(fields_by_dt_csvfile)
#	local_obsdates = list(set(fields_df['local_obsdate'].values))
#	
#	if what == 'butler':
#		RAs = df['pointing_ra']
#		DECs = df['pointing_dec']
#	elif what == 'fakes':
#		RAs = df['RA']
#		DECs = df['DEC']
#		
#	dts = df['local_obsnight']
#	args_list = [(rade, dt, fields_df, local_obsdates) for rade, dt in zip(zip(RAs, DECs), dts)]
#	
#	deep_ids = process_map(process_deepid_row, args_list, max_workers=cpu_count(), chunksize=10)
#	
#	df['DEEP_id'] = deep_ids
#	return df


#def determine_available_distances(df):
#	"""
#	5/22/2024 COC: making a function and adding special cases of -1 and -2
#	5/30/2024 COC: extending to use for out Butler df too
#	"""
#	available_distances = []
#	#
#	# format will be like fit_30, RA_30, Dec_30, or 
#	what = 'fakes'
#	if 'center_coord' in df.columns:
#		what = 'butler'
#	special_cols_to_use = {"RA":-1, "RA_known_r":-2, 'center_coord':-1} # uncorrected and corrected for true distance 5/22/2024 COC
#	for colname in df.columns:
#		if colname.startswith('fit_') and 'known_' not in colname:
#			if what == 'fakes':
##				print(f'Adding {colname} to desired_cols.')
#				desired_cols.append(colname)
#			available_distances.append(float(colname.split('fit_')[-1]))
#		elif colname in special_cols_to_use:
#			available_distances.append(special_cols_to_use[colname])
#	return available_distances


if __name__ == '__main__':
	df = pd.read_csv(butler_csvfile)
	print(f'See butler_csvfile df.columns are {", ".join(list(df.columns))}')
	#
	df, changed = fix_fit_columns(df=df)
	if changed:
		print(f'fit_ column fix(es) happened, saving new csvfile...')
		df.to_csv(butler_csvfile, compression='gzip', index=False)
	df, changed = purge_old_coord_cols(df=df)
	if changed:
		print(f'Old center_coord column drop(s) happened, saving new csvfile...')
		df.to_csv(butler_csvfile, compression='gzip', index=False)
	#
	available_butler_distances, some_cols = get_available_distances(df=df)
	#
	if 'center_coord_ra' not in df.columns: # this should be in incoming DF now, disabling here 6/1/2024 COC
		raise KeyError(f'Missing center_coord_ra from df.columns = {df.columns}.')
#		df['center_coord'] = df['center_coord'].apply(eval)
#		df['center_coord_ra'] = [i[0] for i in df['center_coord']]
#		df['center_coord_dec'] = [i[1] for i in df['center_coord']]
#		df.to_csv(butler_csvfile, index=False, compression='gzip')
#		print(f'Wrote updated (post center_coord_ra) csvfile: {butler_csvfile}')
	
	if 'pointing_ra' not in df.columns: # this should be in incoming DF now, disabling here 6/1/2024 COC
		raise KeyError('pointing_ra not in df.columns = {df.columns}.')
#		df = merge_in_pointings_by_uri(pointing_coords_csv='with_pointing_coords.csv.gz', main_csv=butler_csvfile)
#		df.to_csv(butler_csvfile, compression='gzip', index=False)
#		print(f'Wrote updated {butler_csvfile} to disk.')
#		df = add_DEEP_ids(df)
#		df.to_csv(butler_csvfile, index=False, compression='gzip')
#		print(f'Wrote updated csvfile: {butler_csvfile}')

if __name__ == '__main__':
	if 'DEEP_id' not in df.columns: # this should be in incoming DF now, but leaving for now 6/1/2024 COC
		df = add_DEEP_ids(df)
		df.to_csv(butler_csvfile, index=False, compression='gzip')
		print(f'Wrote updated csvfile: {butler_csvfile}')



# Function to add a new dataframe to the big dataframe
def add_dataframe_to_big_df(big_df, new_df, visit, ut_datetime):
	new_df['visit'] = visit
	new_df['ut_datetime'] = ut_datetime
#	global big_df
	big_df = pd.concat([big_df, new_df], ignore_index=True)
	return big_df

def get_skybot_dataframe(pointing_df, butler_csvfile, radius_deg=1.1, site_code=807, update=False, overwrite=False):
	"""
	Get skybot results for our pointings.
	TODO update: update results from SkyBot
	6/15/2024 COC
	"""
	skybot = Skybot()
	radius_arcsec = radius_deg * 60 * 60
	print(f'There are {len(pointing_df)} rows in the pointings dataframe.')
	skybot_csvfile = butler_csvfile.replace('.csv.gz','_skybot.csv.gz')
	if overwrite == True or len(glob.glob(skybot_csvfile)) == 0:
		print(f'SkyBot DataFrame does not exist. Creating one now...')
		# Initialize the big dataframe with the same columns as the incoming dataframes
		columns = ['visit', 'ut_datetime', 'Number', 'Name', 'RA', 'DEC', 'Type', 'V', 'posunc', 'centerdist', 'RA_rate', 
					'DEC_rate', 'geodist', 'heliodist', 'alpha', 'elong', 'x', 'y', 'z', 'vx', 
					'vy', 'vz', 'epoch']
		skybot_df = pd.DataFrame(columns=columns)
	else:
		if update == True:
			raise ValueError(f'ERROR! update feature has not been implemented yet.')
		print(f'Recycling {skybot_csvfile}...')
		skybot_df = pd.read_csv(skybot_csvfile)
	
	existing_visits = list(set(skybot_df['visit']))
	print(f'There were {len(existing_visits)} existing_visits.')
	print(f'Assembling list of tasks...')
	visits_todo = {}
	for i,visit in enumerate(pointing_df['visit']):
		if visit in existing_visits and update==False:
			print(f'Skipping existing visit {visit} as update=False.')
			continue
		obstime = pointing_df['ut_datetime'].iloc()[i]
		pointing_ra = pointing_df['pointing_ra'].iloc()[i]
		pointing_dec = pointing_df['pointing_dec'].iloc()[i]
		visits_todo[visit] = {'obstime':obstime, 'pointing_ra':pointing_ra, 'pointing_dec':pointing_dec}
	
	#
	print(f'Querying Skybot...')
	max_between_writes = 25 # started with 10, upping to 25 6/15/2024 COC
	c = 0
	changed = False
	with progressbar.ProgressBar(max_value=len(visits_todo)) as bar:	
		for i,visit in enumerate(visits_todo):
	#		field = SkyCoord(ra=pointing_ra*u.deg, dec=pointing_dec*u.deg)
			field = SkyCoord(ra=visits_todo[visit]['pointing_ra']*u.deg, dec=visits_todo[visit]['pointing_dec']*u.deg)
	#		epoch = Time(obstime)
			epoch = Time(visits_todo[visit]['obstime'])
			print(f'We have field={field}, epoch={epoch}')
			sbr = skybot.cone_search(field, radius_arcsec*u.arcsecond, epoch).to_pandas()
			print(f'Saw {len(sbr)} results. Columns are {sbr.columns}.')
			skybot_df = add_dataframe_to_big_df(big_df=skybot_df, new_df=sbr, visit=visit, ut_datetime=epoch)
			changed = True
			bar.update(i)
			c += 1
			if c >= max_between_writes:
				print(f'Writing {skybot_csvfile} to disk...')
				c = 0
				skybot_df.to_csv(skybot_csvfile, compression='gzip', index=False)
				changed = False
	if changed:
		skybot_df.to_csv(skybot_csvfile, compression='gzip', index=False)
#				exit()
	print(f'Finished processing {len(visits_todo)} visits_todo.')
#	exit() # why was this here 6/16/2024 COC
	return skybot_df
	

# adding pointing dataframe, SkyBot stuff 6/15/2024 COC
if __name__ == '__main__':
	pointing_df = get_pointings_dataframe(df=df, butler_csvfile=butler_csvfile, overwrite=False)
	print(f'pointing_df has local_obsnight {sorted(list(set(pointing_df["local_obsnight"])))}.')
	# 
	# emergency repair of Skybot dataframe here. I realized it had wrong (single too) datatime, local_obsnight
#	visit_dict = pointing_df.set_index('visit')['ut_datetime'].to_dict()
#	skydf = pd.read_csv('region_search_df_A0_differenceExp_skybot_clean_simple.csv.gz')
#	skydf['ut_datetime'] = [visit_dict[visit] for visit in skydf['visit']]
#	skydf['local_obsnight'] = [get_local_observing_date(ut_datetime=UT) for UT in skydf['ut_datetime']]
#	skydf.to_csv('region_search_df_A0_differenceExp_skybot_clean_simple.csv.gz', compression='gzip', index=None)
#	print(f'done')
	skybot_df = get_skybot_dataframe(pointing_df=pointing_df, butler_csvfile=butler_csvfile, radius_deg=1.1, update=False, overwrite=False)
	print(f'WARNING!!! Skybot Dataframe has hardcoded radius for DECam!! 6/15/2024 COC')
	print(f'WARNING!!! Skybot Dataframe has hardcoded site code for DECam!! 6/15/2024 COC')
#		skybot_df = pd.DataFrame

## Define the main coordinate and the catalog of other coordinates
#main_coord = SkyCoord(ra=[10]*u.degree, dec=[20]*u.degree, frame='icrs')
#catalog_coords = SkyCoord(ra=[10, 11, 12]*u.degree, dec=[20, 21, 22]*u.degree, frame='icrs')
#
## Find matches within 0.1 degrees
#idx1, idx2, sep2d, dist3d = main_coord.search_around_sky(catalog_coords, 0.1*u.deg)
#matches = catalog_coords[idx1]
#
#print(matches)


def find_overlapping_distances(all_distances_combined, verbose=True):
	"""
	Pass in a full list of seen distances from both the fakes and butler dataframes, return a list of the ones that overlap.
	5/30/2024 COC
	"""
	distances_combined = available_butler_distances + available_fakes_distances
	all_seen_distances = list(set(distances_combined))
	all_seen_distances.sort()
	overlapping_distances = []
	for d in all_seen_distances:
		the_count = distances_combined.count(d)
		if the_count > 2:
			raise KeyError(f'IMPOSSIBLE cannot have more than three of the same distance ({d}) spanning the two dataframes!') # just in case
		if the_count == 2:
			overlapping_distances.append(d)
	if verbose: print(f'Saw overlapping_distances = {overlapping_distances}')
	return overlapping_distances


#def analyze_plx_methods():
#	combo_csvfile = f'combined_old_new_fakes.csv'
#	if len(glob.glob(combo_csvfile)) > 0:
#		combo_df = pd.read_csv(combo_csvfile)
#	else:
#		fakes_csvfile = 'fakes_detections_simple.csv.gz'
#		print(f'Reading fakes csv file {fakes_csvfile}...')
#		fakes_df = pd.read_csv(fakes_csvfile)
#		fakes_df['orbitid_ut'] = [f'{fakes_df["ORBITID"].iloc()[i]}_{fakes_df["ut"].iloc()[i]}' for i in range(0,len(fakes_df))]
#		available_fakes_distances, some_cols = get_available_distances(df=fakes_df)
#		#
#		print(f'Reading old fakes csvfile...')
#		old_fakes_df = pd.read_csv('fakes_detections_fittingWay.gz')
#		old_fakes_df['orbitid_ut'] = [f'{old_fakes_df["ORBITID"].iloc()[i]}_{old_fakes_df["ut"].iloc()[i]}' for i in range(0,len(old_fakes_df))]
#		#
#		old_available_distances, some_cols = get_available_distances(df=old_fakes_df)
#		both_fakes_distances = find_overlapping_distances(all_distances_combined=available_fakes_distances + old_available_distances, verbose=True)
#		#
#		print(f'Merging dataframes...')
#		combo_df = pd.merge(fakes_df, old_fakes_df, on='orbitid_ut', how='left')
#		print(f'Export combined csvfile to {combo_csvfile}...')
#		combo_df.to_csv(combo_csvfile, compression='gzip', index=False)
#		print(f'combo_df.columns is {combo_df.columns}')


# Moving below to a Notebook because the full datasets are unwieldy 6/3/2024 COC
#if __name__ == '__main__':
#	analyze_plx_methods()
#	exit()

		

	
if __name__ == '__main__':
	startTime = time.time()
	fakes_csvfile = 'fakes_detections_simple.csv.gz' # this is really huge (4.31 Gb, compressed)
#	fakes_csvfile = 'fakes_2019-04-02_and_2019-05-07.csv.gz'
	print(f'WARNING: we are using a limited set fakes CSV file {fakes_csvfile}.')
#	fakes_csvfile = 'fakes_detections_20190402_20190403_20190504_20190505_20190507_simple.csv.gz'
#	fakes_df['orbitid_ut'] = [f'{fakes_df["ORBITID"].iloc()[i]}_{fakes_df["ut"].iloc()[i]}' for i in range(0,len(fakes_df))] # this was for merging different fakes DFs, maybe deprecated 6/4/2024 COC
	#df = df[df['local_obsnight'] < '2019-05-07'] # 5/3/2024 COC
	#desired_dates = ['2019-04-03', '2019-05-06'] # A0b, WE DO NOT HAVE 2019-05-06!
	#print(f'WARNING: INCORRECT DATES HERE JUST FOR TESTING WHILE KLONE IS DOWN 5/26/2024 COC')
	#desired_dates = ['2019-04-04', '2019-05-05'] # A0c, should have 104, 91 images
#	desired_dates = ['2019-04-03', '2019-05-04'] # A0c, should have 104, 91 images; now *LOCAL* dates 5/25/2024 COC
#	desired_dates = ['2019-04-02', '2019-04-03']
#	desired_dates = ['2019-05-04', '2019-05-05']
	desired_dates = ['2019-04-02', '2019-05-07'] # 5/30/2024 COC a0b
#	desired_dates = ['2019-04-03', '2019-05-05'] # 6/11/2024 COC a0c
#	desired_dates = ['2019-04-03', '2019-05-04'] # 6/11/2024 COC a0a and a0c, less overlap; have not tried yet
	desired_dates_str = make_desired_dates_str(desired_dates=desired_dates)
	#
	# TODO make something that determines a csvfile name for a cache and if it does not exist read the giant CSV then output the cache that is limited by dates 6/6/2024 COC
	print(f'Reading fakes csv file {fakes_csvfile}...')
#	fakes_df = pd.read_csv(fakes_csvfile)
	fakes_df = get_fakes_df(desired_dates=desired_dates, 
#		base_csvfile, 
		overwrite=False,
		butler_df=df,
		butler_csvfile=butler_csvfile
	)
	if 'visit' not in fakes_df:
		add_visit_ids(df=fakes_df, pointings_df=pointing_df)
	print(f'done')
	exit()
	elapsed = round(time.time() - startTime, 1)
	available_fakes_distances, some_cols = get_available_distances(df=fakes_df)
	print(f'It took {elapsed} seconds to read fakes_df from {fakes_csvfile}.')
	
#	csv_dates_str = '_'.join([i.replace('-','') for i in desired_dates]) # we were splitting CSV files out for faster reads, out for now 6/4/2024 COC
	#csvfile = 'fakes_detections_20190404_20190505_simple.csv.gz'
	#csvfile = f'fakes_detections_{csv_dates_str}_simple.csv.gz'
#	csvfile = 'fakes_detections_20190402_20190403_20190504_20190505_20190507_simple.csv.gz'
#	exit()
	print(f'There are {len(set(fakes_df["ORBITID"]))} unique fakes.')
		
	# adding local_obsnight, a local date for the start of the night 5/25/2024 COC
	if 'local_obsnight' not in fakes_df.columns:
		print(f'local_obsnight missing from fakes_df, adding now...')
		fakes_df['local_obsnight'] = [get_local_observing_date(ut_datetime=UT) for UT in fakes_df['ut_datetime']]#, utc_offset)]
		fakes_df.to_csv(fakes_csvfile, index=False, compression='gzip')
		print(f'Wrote updated csvfile: {fakes_csvfile}')
	
	# disabling for now as we do not use this field I think in fakes 5/29/2024 COC
	if 'DEEP_id' not in fakes_df:
		df = add_DEEP_ids(fakes_df)
		fakes_df.to_csv(fakes_csvfile, index=False, compression='gzip')
		print(f'After Fakes DF DEP_ids, wrote updated csvfile: {fakes_csvfile}')

	# asdf


if __name__ == '__main__':
	available_distances, some_cols = get_available_distances(df=fakes_df)
	available_distances.sort()
	print(f'Saw available_distances = {available_distances}')
	
	relabel_columns = {}
	for i in [20,30,40,50,60,70,80,90,100]:
		relabel_columns[f'RA_{i}'] = f'RA_{i}.0'
		relabel_columns[f'Dec_{i}'] = f'Dec_{i}.0'
		relabel_columns[f'fit_{i}'] = f'fit_{i}.0'
		relabel_columns[f'center_coord_{i}_ra'] = f'center_coord_{i}.0_ra'
		relabel_columns[f'center_coord_{i}_dec'] = f'center_coord_{i}.0_dec'
	for i in available_distances: # 5/23/2024 COC
		relabel_columns[f'center_coord_{int(i)}_ra'] = f'center_coord_{int(i)}.0_ra'
		relabel_columns[f'center_coord_{int(i)}_dec'] = f'center_coord_{int(i)}.0_dec'
	fakes_df = fakes_df.rename(columns=relabel_columns)
	print(f'fakes_df.columns after relabeling: {fakes_df.columns}')
	
	print(f'WARNING/NOTE: we are using a limited set of fakes that just have our dates in there. 5/13/2024 COC')
	
	pd.set_option('display.max_columns', None)

def get_fake_info(ORBITID, desired_dates, verbose=False):
	orbit_df = fakes_df[fakes_df['ORBITID']==ORBITID]
	records = []
	for dt in desired_dates:
		the_record = orbit_df[orbit_df['local_obsnight']==dt].iloc()[0]
		if verbose: # warning: very wordy!
			print(f'RA = {the_record["RA_40.0"]}, Dec = {the_record["Dec_40.0"]}, ')
			print(the_record.to_csv(None))
		records.append(the_record)
	return records

if __name__ == '__main__':
	#get_fake_info(ORBITID=3110224, desired_dates=desired_dates) # our recovered objects from two nights!!!
	fake_test = get_fake_info(ORBITID=3110224, desired_dates=desired_dates)
	print(fake_test) # shows the list of results dicts (?)
	print(fake_test[0]['MAG']) # gives just the scalar


#print(f'fakes_df.columns: {fakes_df.columns}')

# TODO move this to a fakes functions library 6/1/2024 COC
def assess_columns(fakes_df, cols=['r','d'], available_distances=[]):
	"""5/20/2024 COC"""
	desired_cols = ['r', 'd', 'aei_1', 'aei_2', 'aei_3', 'aei_4', 'aei_5', 'aei_6', 'H_VR', 'MAG', 'PERIOD', 'PHASE', 'CCDNUM', 'AMP', 'mjd_mid']
	for col in desired_cols: cols.append(col) # 6/2/2024 COC
	units = {'r':'au', 'd':'au', 'a':'au', 'i':'deg', 'PHASE':'deg'}
	for d in available_distances:
		units[f'fit_{float(d)}'] = 'au'
	print(f'assess_columns units are: {units}')
	names = {'aei_1':'a', 'aei_2':'e', 'aei_3':'i'}
	binsdict = {'CCDNUM':list(range(0,max(fakes_df['CCDNUM'])))}
	for col in cols:
		print(f'{col} range = {min(fakes_df["r"])} to {max(fakes_df["r"])}')
		bins = 100
		if col in binsdict: bins = binsdict[col]
		data = fakes_df[col]
		if col in ['PHASE']:
			data *= 180/np.pi
		plt.hist(data, bins=bins)
		plt.yscale('log')
		real_name = col
		if real_name in names:
			real_name = names[real_name]
		xlabel = real_name
		if real_name in units:
			xlabel += f' [{units[real_name]}]'
		plt.xlabel(xlabel)
		plt.title(f'Range: {min(data)} to {max(data)}')
		outfile = f'{col}_{desired_dates_str}_hist.pdf'
		plt.savefig(outfile)
		print(f'Wrote {outfile} to disk.')
		plt.close()
#	plt.show()

if __name__ == '__main__':
	#assess_columns(fakes_df=fakes_df, cols=desired_cols, available_distances=available_distances)
	print(f'WARNING: assess_columns() is turned OFF!')


if __name__ == '__main__':
	desired_orbitids = [4630883, 3110224]


#if __name__ == '__main__':
#	print(f'WARNING: discrepancy assessment is disabled!')
	#for ORBITID in desired_orbitids:
	#	discrep_mean, discrep_std = check_linearity(fakes_df=fakes_df, desired_dates=desired_dates, ORBITID=ORBITID, verbose=False)

if __name__ == '__main__':
	orbitids = {}
	for dt in desired_dates:
		orbitids[dt] = list(set(fakes_df[fakes_df['local_obsnight']==dt]['ORBITID']))
	overlapping_orbit_ids = set(orbitids[desired_dates[0]]).intersection(orbitids[desired_dates[-1]])
	print(f'There were {len(overlapping_orbit_ids)} ORBITIDs that overlapped first and last dates of {desired_dates}.')

def get_colnames_by_dist(dist_au):
	"""
	Convenience function to supply correct column names for different situations.
	Originally for fakes work (and this is still true as of 6/1/2024 COC).
	5/22/2024 COC
	"""
	dist_au = float(dist_au)
	if dist_au == float(-1.0): # this was wrong before 5/22/2024, should have been RA_known_r. -1 should be unadjusted now! 5/22/2024 COC
		ra_field = 'RA'
		dec_field = 'DEC'
		r_field = 'r'
	elif dist_au == -2.0: # 0 now here instead of above 5/22/2024 COC
		# copying from find_fakes code:
		# 		colnames = [f"RA_{dist_au}", f"Dec_{dist_au}", f"fit_{dist_au}"]
#		if dist_au == 0.0:
#			colnames = ["RA_known_r", "Dec_known_r", "fit_known_r"]
		ra_field = 'RA_known_r'
		dec_field = 'Dec_known_r'
		r_field = 'fit_known_r'
	else:
		ra_field = f'RA_{dist_au}'
		dec_field = f'Dec_{dist_au}'
		r_field = f'fit_{dist_au}'
	return ra_field, dec_field, r_field



def assess_all_linearity(overlapping_orbit_ids, method=2, dist_au=40.0, verbose=False, show=False):
	"""
	5/19-20/2021 COC
	"""
	all_discreps = []
	all_stds = []
	all_r_discreps = []
	all_maxes = []
	usable_orbit_ids = []
	misfires = []
	for ORBITID in overlapping_orbit_ids:
		discrep_max = None
		if method == 1:
			discrep, std = check_linearity(fakes_df=fakes_df, desired_dates=desired_dates, ORBITID=ORBITID, dist_au=dist_au, verbose=verbose)
		elif method == 2:
			discrep, std, r_discrep_mean = assess_linearity2(fakes_df=fakes_df, desired_dates=desired_dates, ORBITID=ORBITID, verbose=verbose, dist_au=dist_au, plot=False)
		elif method == 3:
			discrep, std, r_discrep_mean, discrep_max = assess_linearity3(fakes_df=fakes_df, desired_dates=desired_dates, ORBITID=ORBITID, verbose=verbose, dist_au=dist_au, plot=False, show=False) # plots are actually useful, just prolific 5/23/2024 COC
		else:
			raise KeyError(f'Inavlid method {method} supplied!')
		#
		if discrep == None or std == None:
			misfires.append(ORBITID)
			continue
		usable_orbit_ids.append(ORBITID)
		all_discreps.append(abs(discrep))
		all_stds.append(std)
		all_r_discreps.append(abs(r_discrep_mean)) # abs for logging probably
		all_maxes.append(abs(discrep_max))
	#
	z = np.array(all_discreps)
	max_discrep_arcsec = 2
	n_ok = len(z[z<max_discrep_arcsec])
	pct_ok = ( n_ok / len(all_discreps) ) * 100
	print(f'{dist_au} au: {n_ok} of {len(all_discreps)} ({round(pct_ok,3)}%) are ≤ {max_discrep_arcsec}".')
	#
	fig, ax1 = plt.subplots()
	ax1.errorbar(list(range(len(usable_orbit_ids))), all_discreps, all_stds, label=f'deviation', marker='.', ms=0.25, linestyle='none', linewidth=0.1)
	ax1.scatter(list(range(len(usable_orbit_ids))), all_maxes, color='red', s=0.25, label='max distance')
	for i in range(1, 5 + 1):
		ax1.plot([0, len(usable_orbit_ids)], [i * 0.263] * 2, label=f'{i} pixels', linewidth=1, alpha=0.5)
	ax1.set_yscale('log')
	ax1.set_ylabel('Deviation from Linearity (")')
	if method == 3: # 5/23/2024 COC
#		pass
		ax1.set_ylim([1e-4,1e3]) # 5/23/2024 COC # reenabled, udpated 6/13/2024 COC
	else:
		ax1.set_ylim([1e-3,1e4]) # 5/21/2024 COC
	def tick_function(t, scale_factor):
		return ["%.2f" % (val * scale_factor) for val in t]
	
	ax2 = ax1.twinx()
	mn, mx = ax1.get_ylim()
	ax2.set_ylim(mn/0.263, mx/0.263)
	ax2.set_ylabel('Deviation from Linearity (pixels)')
	ax2.set_yscale('log')
#	plt.legend()
	ax1.set_xlabel('ORBITID Index')
	if float(dist_au) == -2.0:
		plt.title(f'Corrected via Known r. Dates: {", ".join(desired_dates)}')
	elif float(dist_au) == -1.0:
		plt.title(f'No Correction. Dates: {", ".join(desired_dates)}')
	else:
		plt.title(f'{dist_au} au "guess." Dates: {", ".join(desired_dates)}')
	plt.tight_layout()
	for ext in ['pdf', 'png']:
		plt.savefig(f'overlapping_ORBITIDs_linearity_by_ORBITIDidx_{dist_au}au_{desired_dates_str}_method{method}.{ext}')
	if show: 
		plt.show()
	else:
		plt.close()
	
	fig, ax1 = plt.subplots()
	ax1.scatter(all_r_discreps, all_discreps, s=1, label='median')
	ax1.scatter(all_r_discreps, all_maxes, s=1, color='red', label='maxes')
	if method == 3: # 5/23/2024 COC
#		pass
		ax1.set_ylim([1e-4,1e3]) # 5/21/2024 COC
		ax1.set_xlim([1e-4,1e4]) # 5/21/2024 COC
	else:
		ax1.set_ylim([1e-3,1e3]) # 5/21/2024 COC
		ax1.set_xlim([1e-4,1e4]) # 5/21/2024 COC
		
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel(f'Deviation from r [au]')
	plt.ylabel(f'Deviation from Linearity ["]')
	if float(dist_au) == -2.0:
		plt.title(f'Corrected via Known r. Dates: {", ".join(desired_dates)}')
	elif float(dist_au) == -1.0:
		plt.title(f'No Correction. Dates: {", ".join(desired_dates)}')
	else:
		plt.title(f'{dist_au} au "guess." Dates: {", ".join(desired_dates)}')
	plt.tight_layout()
	for ext in ['pdf', 'png']:
		plt.savefig(f'overlapping_ORBITIDs_linearity_by_deltar_{dist_au}au_{desired_dates_str}_method{method}.{ext}')
	if show:
		plt.show()
	else:
		plt.close()
	
	#
	# let us try checking the separation of the two coordinates from eachother too 5/22/2024 COC
	ra_colname, dec_colname, r_colname = get_colnames_by_dist(dist_au=dist_au)
	coords_by_dt = {}
	for dt in desired_dates:
		coords_by_dt[dt] = []
		dt_df = fakes_df[fakes_df['local_obsnight']==dt]
		for ORBITID in usable_orbit_ids:
			orbit_df = dt_df[dt_df['ORBITID']==ORBITID]
			middle_row = len(orbit_df)//2
			coords_by_dt[dt].append([ orbit_df[ra_colname].iloc()[middle_row], orbit_df[dec_colname].iloc()[middle_row] ])
	date1_skycoord = SkyCoord(ra=[coord[0] for coord in coords_by_dt[desired_dates[0]]]*u.degree, dec=[coord[1] for coord in coords_by_dt[desired_dates[0]]]*u.degree, frame='icrs')
	date2_skycoord = SkyCoord(ra=[a[0] for a in coords_by_dt[desired_dates[-1]]]*u.degree, dec=[b[1] for b in coords_by_dt[desired_dates[-1]]]*u.degree, frame='icrs')
	print(f'date1_skycoord.shape = {date1_skycoord.shape}, date1_skycoord.shape = {date2_skycoord.shape}')
	separations = date1_skycoord.separation(date2_skycoord).arcminute
	#
	# TODO insert histogram here 5/23/2024 COC
#	plt.hist(separations, bins=100)
#	plt.xlabel(f"{desired_dates[0]} to {desired_dates[1]} Separation (')")
#	plt.ylabel(f'Number of Fakes')
	#
#	print(f'separations: {separations}')
	fig, ax1 = plt.subplots()
	ax1.errorbar(separations, all_discreps, all_stds, label=f'deviation', marker='.', ms=0.25, linestyle='none', linewidth=0.1)
	ax1.scatter(separations, all_maxes, label=f'maxes', s=0.25, color='red')
	the_xrange = [1e-2, 1e3]
	the_yrange = [1e-8, 1e-1]
	ax1.set_xlim(the_xrange)
	if method == 3:
		ax1.set_ylim([1e-4, 1e3])
	for i in range(1, 5 + 1):
#		ax1.plot([0, max(separations)], [i * 0.263] * 2, label=f'{i} pixels', linewidth=1, alpha=0.5)
		ax1.plot([the_xrange[0], the_xrange[1]], [i * 0.263] * 2, label=f'{i} pixels', linewidth=1, alpha=0.5)
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.set_ylabel('Deviation from Linearity (")')
#	ax1.set_ylim([1e-3,1e4]) # 5/21/2024 COC
	def tick_function(t, scale_factor):
		return ["%.2f" % (val * scale_factor) for val in t]
	
	ax2 = ax1.twinx()
	mn, mx = ax1.get_ylim()
	ax2.set_ylim(mn/0.263, mx/0.263)
	ax2.set_ylabel('Deviation from Linearity (pixels)')
	ax2.set_yscale('log')
#	plt.legend()
	ax1.set_xlabel("Date 1 -- Date 2 Sky Separation [']")
	if float(dist_au) == -2.0:
		plt.title(f'Corrected via Known r. Dates: {", ".join(desired_dates)}')
	elif float(dist_au) == -1.0:
		plt.title(f'No Correction. Dates: {", ".join(desired_dates)}')
	else:
		plt.title(f'{dist_au} au "guess." Dates: {", ".join(desired_dates)}')
	plt.tight_layout()
	for ext in ['pdf', 'png']:
		plt.savefig(f'overlapping_ORBITIDs_linearity_by_separation_{dist_au}au_{desired_dates_str}_method{method}.{ext}')
	if show: 
		plt.show()
	else:
		plt.close()
	# final return below
	#
	return n_ok, pct_ok



def meta_assess_deviations(distances, show=False):
	"""
	5/21/2024 COC
	"""
#	distances = available_distances + [-2, -1] # adding known distance (-2) and no offset (-1) 5/22/2024 COC
	all_n_ok = []
	all_pct_ok = []
	for dist_au in distances:#[30,40]:
		n_ok, pct_ok = assess_all_linearity(overlapping_orbit_ids=overlapping_orbit_ids, method=3, dist_au=dist_au, verbose=False)
		all_n_ok.append(n_ok)
		all_pct_ok.append(pct_ok)
	#
	plt.scatter(distances, all_n_ok)
	plt.xlabel(f'Distance "Guess" [au]')
	plt.ylabel(f'Number Objects with ≤2" Deviation from Linearity')
	plt.tight_layout()
	for ext in ['pdf', 'png']:
		plt.savefig(f'n_ok_per_distance_guess_{desired_dates_str}.{ext}')
	if show:
		plt.show()
	else:
		plt.close()
	#
	#
	plt.scatter(distances, all_pct_ok)
	plt.xlabel(f'Distance "Guess" [au]')
	plt.ylabel(f'Fraction of Objects with ≤2" Deviation from Linearity [%]')
	plt.tight_layout()
	for ext in ['pdf', 'png']:
		plt.savefig(f'pct_ok_per_distance_guess_{desired_dates_str}.{ext}')
	if show:
		plt.show()
	else:
		plt.close()
	
if __name__ == '__main__':
	print(f'WARNING! META_ASSESS is OFF!!!')
#	meta_assess_deviations(distances=available_butler_distances) # TODO get the available distances for this thing 5/29/2024 COC
	# 
	# special case
	#assess_all_linearity(overlapping_orbit_ids, method=3, dist_au=-2, verbose=False, show=False) # special case, trying uncorrected coords 5/21/2024 COC; method 2 to 3 6/13/2024 COC; dist_au from 0 to -2 6/13/2024 COC
	


#if __name__ == '__main__':
#	fakes_skycoord_40 = SkyCoord(ra=fakes_df['RA_40.0']*u.deg, dec=fakes_df['Dec_40.0']*u.deg, frame='icrs')


def export_fakes_info(dist_au):
	"""
	Probably should make dist_au be a list 5/21/2024 COC
	5/20/2024 COC
	"""
	for ORBITID in desired_orbitids:
		orbit_df = fakes_df[fakes_df['ORBITID']==ORBITID]
		print(orbit_df.iloc()[0])
		with open(f'{ORBITID}_{dist_au}au_{desired_dates_str}.csv','w') as f:
			print(f'ORBITID,RA_{dist_au},Dec_{dist_au}',file=f)
			for i in range(0,len(orbit_df)):
				print(f'{ORBITID},{orbit_df[f"RA_{dist_au}"].iloc()[i]},{orbit_df[f"Dec_{dist_au}"].iloc()[i]}', file=f)
#if __name__ == '__main__':
#	export_fakes_info(dist_au=40.0)
#	#from astropy.table import Table
#	#result_coords = Table.read('results_coords.ecsv', format='ascii.ecsv')
#	#results_coords
#	wilson_coords = []
#	fn = 'gistfile3_slower_20240520.txt'
#	#fn = 'gistfile2.txt'
#	#fn = 'wilson_matches_20240520.txt'
#	with open(fn, 'r') as f:
#		for line in f:
#			line = line.strip().replace('(','').replace(')','').split(',')
#			wilson_coords.append((float(line[0].strip().replace(',','')),float(line[1].strip().replace(',',''))))
#	#print(wilson_coords)
#	wilson_skycoord = SkyCoord(ra=[i[0] for i in wilson_coords]*u.deg, dec=[i[1] for i in wilson_coords]*u.deg, frame='icrs')
#	#wilson_skycoords = [SkyCoord(ra=i[0]*u.deg, dec=i[1]*u.deg, frame='icrs') for i in wilson_coords]
#	#print(wilson_skycoord)
#	
#	
#	idx1, idx2, sep2d, dist3d = search_around_sky(coords1=wilson_skycoord, coords2=fakes_skycoord_40, seplimit=5*u.arcsecond)
##	print(len(wilson_skycoord))
##	print(len(fakes_skycoord_40))
##	print(len(idx1))
##	print(len(idx2))
#	unique_kbmod_results = list(set(idx1))
##	print(unique_kbmod_results)
#	results_df = pd.DataFrame.from_dict({'idx1':idx1, 'idx2':idx2})
#	results_df['ORBITID'] = [fakes_df['ORBITID'].iloc()[i] for i in idx2]
#	#print(results_df)
#	want_df = results_df[results_df['ORBITID'] == desired_orbitids[0]]
##	print(want_df)
#	
##	print(list(idx1).count(67))
#	#print(fakes_df.iloc()[idx2[0]])

if __name__ == '__main__':
	#df = pd.read_csv('dataframe_A0_fakes_calexp.csv')
#	butler_csvfile = 'region_search_df_A0_differenceExp.csv.gz' #'dataframe_A0_differenceExp.csv.gz'
	print(f'butler_csvfile was already set as {butler_csvfile}')
	df = pd.read_csv(butler_csvfile)
	#
	# kludge code here
#	cols_to_drop = []
#	cols_to_rename = {}
#	for dist_au in range(0,200+1):
#		cols_to_drop.append(f'center_coord_{dist_au}')
#		cols_to_drop.append(f'center_coord_{float(dist_au)}')
#		for i in cols_to_drop:
#			try:
#				df = df.drop(i, axis=1)
#			except KeyError as msg: # column does not exist
#				pass
#		if f'fit_{int(dist_au)}' in df.columns:
#			if f'fit_{float(dist_au)}' not in df.columns:
#				cols_to_rename[f'fit_{int(dist_au)}'] = f'fit_{float(dist_au)}'
#			else:
#				cols_to_drop.append(f'fit_{int(dist_au)}')
#	df = df.rename(columns=cols_to_rename)
#	df.to_csv(csvfile, compression='gzip', index=False)
#	print(f'Wrote fixed CSV KLUDGEY')
	# end kludge code
	#df['region'] = df['region'].apply(eval)
	if 'region' in df.columns:
		df['region'] = [eval(i) for i in df['region']] # .apply did not work
	if 'center_coord' in df.columns:
		df['center_coord'] = df['center_coord'].apply(eval)
	df = df.rename(columns=relabel_columns) # KLUDGE! 5/21/2024 COC
	print(f'{butler_csvfile} df has columns {df.columns}')

if __name__ == '__main__':
	if 'local_obsnight' not in df.columns: # this should be coming with DF now, disabling 6/1/2024 COC
		raise KeyError(f'local_obsnight missing from df.column = {df.columns}')
#		print(f'local_obsnight missing from df, adding now...')
#		df['local_obsnight'] = [get_local_observing_date(ut_datetime=UT) for UT in df['ut']]#, utc_offset)]
#		df.to_csv(f'{csvfile}_{int(time.time())}', index=False, compression='gzip')
#		df.to_csv(csvfile, index=False, compression='gzip')
#		print(f'Wrote updated csvfile: {csvfile}')

# print(len(df[df['local_obsnight']=='2019-05-06']))



def compare_df_obsnights(butler_df, fakes_df):
	"""
	6/4/2024 COC
	"""
	seen = {}
	seen['butler'] = list(set(butler_df['local_obsnight']))
	seen['butler'] = list(set(butler_df['local_obsnight']))

#compare_df_obsnights(butler_df, fakes_df)


if __name__ == '__main__':
	startTime= time.time()
	fields_by_dt_butler_orig = find_good_date_pointing_pairs(df=df)
	elapsed = time.time() - startTime
	print(f'It took {round(elapsed,1)} seconds for find_good_date_pointing_pairs().')
	print(f'Saw original (unfiltered by date) fields_by_dt = {fields_by_dt_butler_orig}.')
	# 
	# Below does NOT work 6/4/2042 COC -- logic flaw, there is no visit for Fakes...
#	fields_by_dt_fakes_orig = find_good_date_pointing_pairs(df=fakes_df)
#	print(f'Saw original (unfiltered by date) fields_by_dt = {fields_by_dt_fakes_orig}.')
#	combined_fields_by_dt = {}
#	for d in [fields_by_dt_butler_orig, fields_by_dt_fakes_orig]:
#		for dt in d:
#			if dt not in combined_fields_by_dt:
#				combined_fields_by_dt[dt] = d[dt]
#	print(f'combined_fields_by_dt: {combined_fields_by_dt}')
	



if __name__ == '__main__':
	#
	if 'large_pile' not in fakes_df.columns: # 5/29/2024 COC
		df = make_small_piles_go_away(fakes_df, combined_fields_by_dt=fields_by_dt_butler_orig)
		df.to_csv(f'{fakes_csvfile}_{int(time.time())}', compression='gzip', index=False)
		df.to_csv(fakes_csvfile, compression='gzip', index=False)
		print(f'Wrote updated fakes_csv {fakes_csvfile} after large_pile.')
	#
#	fakes_df = fakes_df[fakes_df['local_obsnight'].isin(desired_dates)]
#	fakes_df.to_csv(f'fakes_{desired_dates_str}.csv.gz',compression='gzip', index=False)
#	print(f'wrote')

	if 'large_pile' not in df.columns: # 5/29/2024 COC
		df = make_small_piles_go_away(df, combined_fields_by_dt=fields_by_dt_butler_orig)
		df.to_csv(f'{butler_csvfile}_{int(time.time())}', compression='gzip', index=False)
		df.to_csv(butler_csvfile, compression='gzip', index=False)
		print(f'Finished with large_pile.')

if __name__ == '__main__':
	df_orig = df.copy() # 5/26/2024 COC gonna send this through analyze_fields a bit
	df = df[df['local_obsnight'].isin(desired_dates)] # 5/3/2024 COC
	fakes_df = fakes_df[fakes_df['local_obsnight'].isin(desired_dates)] # 6/4/2024 COC -- NOTE: gotta make sure we do not write the CSV after this point!
	# TODO caching stuff

if __name__ == '__main__':
	large_piles_only = True
	if large_piles_only == True:
		print(f'NOTE: limiting dataframe to large piles only')
		df = df[df['large_pile']==True]


if __name__ == '__main__':
	fields_by_dt = find_good_date_pointing_pairs(df=df)
	print(f'Saw filtered fields_by_dt = {fields_by_dt} for dataframe df filtered by desired_dates={desired_dates}.')
	#print(type(df['center_coord'].iloc()[0]))
#	print(f'OY CHECK THAT OUT ABOVE ... WHY THREE FIELDS ON EACH DATE??')
	true_available_distances = find_overlapping_distances(all_distances_combined = available_butler_distances + available_fakes_distances)
	print(f'NOTE: manually overriding available distances!')
#	desired_distances_au = [38,39,40,41,42,43,44] # true_available_distances#[42] # [42] # [30, 40, 50, 60, 70, 80, 90]#[80.0, 70.0, 60.0, 50.0]#[40.0, 30.0]
	desired_distances_au = true_available_distances # list(range(30,50+1)) # 6/11/2024 COC
	usable_flags = []
	#for i in range(0,len(df)):
	#	dt = str(df['local_obsnight'].iloc()[i])
	#	patch_id = df['DEEP_id'].iloc()[i]
	#	

def get_available_dates(df):
	"""5/29/2024 COC"""
	available_dates = list(set([str(df['local_obsnight'].iloc()[i]) for i in range(0,len(df))]))
	available_dates.sort()
	return available_dates


def determine_better_pointing_centers(df, verbose=False):
	"""
	The pointing centers look off on the plots. 
	Probably an effect of copying from the DEEP II paper table.
	Go for (mean(RA), mean(Dec)) instead.
	6/6/2024 COC
	"""
	if 'center_coord_ra' in df:
		what = 'butler'
		ra_field = 'center_coord_ra'
		dec_field = 'center_coord_dec'
	else:
		what = 'fakes'
		ra_field = 'RA'
		dec_field = 'DEC'
	#
	d = {}
	available_dates = list(set(df['local_obsnight']))
	available_dates.sort()
	for dt in available_dates:
		d[dt] = {}
		dt_df = df[df['local_obsnight']==dt]
		if verbose: print(f'dt_df.columns = {dt_df.columns}')
		DEEP_ids = list(set(dt_df['DEEP_id']))
		for DEEP_id in DEEP_ids:
			patch_df = dt_df[dt_df['DEEP_id']==DEEP_id]
			RA = np.mean(patch_df[ra_field])
			DEC = np.mean(patch_df[dec_field])
			d[dt][DEEP_id] = [RA, DEC]
	return d

# PAUSED: need to get fields_by_dt_butler_orig into the analyze_fields df, then redo parallelize stuff from ChatGPT
def analyze_fields(df, dist_au, desired_dates, desired_deep_patch_id=None, plot_field=True, good_pairs_only=True, better_pointing_centers=None, verbose=False):
	"""
	5/30/2024 COC: changed default desired_deep_patch_id to None. This is important, else we will not make patches for our adjacent dates that span different DEEP patches!
	5/25/2024 COC: added plot_field but the circle was very off-center. Need to come up with a center for each local obsdate TODO
	5/15/2024 COC: making a function
	"""
	dist_au = float(dist_au) # 5/23/2024 COC kludge
	print(f'Starting analyze_fields(dist_au={dist_au})...')
	if better_pointing_centers == None:
		print(f'WARNING: better_pointing_centers not supplied, so calculating every time. Expensive!')
		better_pointing_centers = determine_better_pointing_centers(df=df)
#	print(f'analyze_fields() df.columns: {df.columns}')
	fig = plt.figure(figsize=[12,8])
	arrow_colors = ['blue', 'red', 'green', 'orange', 'pink', 'purple']
	arrow_linestyles = ['--', '-']
	arrow_dts_legended = []
	fields_by_dt_butler_orig = find_good_date_pointing_pairs(df=df) # 6/6/2024 0.4s, so adding here rather than trying to pass in
	for i,dt in enumerate(desired_dates):
		df2 = df[df['local_obsnight']==dt]
		for patch_id in list(set(df2['DEEP_id'])):
			if desired_deep_patch_id != None and patch_id != desired_deep_patch_id: continue
			if good_pairs_only == True and patch_id not in fields_by_dt_butler_orig[dt]:
				if verbose: print(f'Skipping non good pair {dt} {patch_id}')
				continue
			patch_df = df2[df2['DEEP_id']==patch_id]
#			print(f'Before patch filter, df2 was len {len(df2)}') # all good; A0c pre 7260 5/25/2024 COC
#			if desired_deep_patch_id != None:
#				df2 = df2[df2['DEEP_id']==desired_deep_patch_id]
	#		print(f'After A0c filter, there are {len(df2)} rows left.') #  all good; post A0c, 4116 5/25/2024 COC
		#	print(df2['center_coord'])
#			print(f'There are {len(df2.index)} entries for date {dt}.')
	#		print(df['local_obsnight']) # debugging, getting empty dataframe for local_obsnight
	#		print(df['ut_date'])
		#	print(list(set(Xs)))
			size = 5 # 6/6/2024 COC making bigger
	#		if dt =='2019-05-07':  # disabling this clause as probably vestigial 5/26/2024 COC
	#			size = 5
	#			print(f'Saw one.')
			for j in range(0, len(patch_df.index)):
				start_coords = [patch_df['center_coord_ra'].iloc()[j], patch_df['center_coord_dec'].iloc()[j]]#[patch_df['center_coord'].iloc()[j][0], patch_df['center_coord'].iloc()[j][1]]
				end_coords = [patch_df[f'center_coord_{dist_au}_ra'].iloc()[j], patch_df[f'center_coord_{dist_au}_dec'].iloc()[j]]
				lens = [end_coords[0] - start_coords[0], end_coords[1] - start_coords[1]]
		#		plt.plot([, ], [, ], color='')
				label = None # adding labeling 6/6/2024 COC
				if dt not in arrow_dts_legended:
					label = f'{dt}'
					arrow_dts_legended.append(dt)
				plt.arrow(start_coords[0], start_coords[1], lens[0], lens[1], 
					alpha=0.75, # 0.5 to 0.75 6/6/2024 COC
					length_includes_head=True, 
					head_width=0.025, 
					color=arrow_colors[i], 
					label=label,
				)
			plt.scatter(patch_df['center_coord_ra'], patch_df['center_coord_dec'], 
#				label=dt,  # disabling; just arrows 6/6/2024 COC
				s=size, 
				alpha=0.5, 
				color='black', # 6/6/2024 COC not very legible otherwise
#				color=arrow_colors[i], # dot very visible, and edgecolor only leaves a tiny bit of color 6/6/2024 COC
#				edgecolor='black', # 6/6/2024 COC
			)
			
#			plt.scatter([ra[0] for ra in patch_df['center_coord'].values], [dec[1] for dec in patch_df['center_coord'].values], label=dt, s=size, alpha=0.5)
			# disabling below scatter as redundant with the arrows and they are cluttering6/6/2024 COC
#			plt.scatter(patch_df[f'center_coord_{dist_au}_ra'], patch_df[f'center_coord_{dist_au}_dec'], label=f'{dist_au}au {dt}', s=20, marker='+', alpha=0.5)
	if plot_field == True:
		edgecolors = ['green', 'purple', 'cyan', 'lime', 'deepskyblue', 'magenta', 'red', 'orange', 'blue']
		ax = plt.gca()
		patches_to_plot = []
		corresponding_dts = []
		corresponding_center_coords = []
		for deep_patch_id in list(set(df['DEEP_id'])):
			print(f'deep_patch_id is {deep_patch_id} here')
			print(f'first is {df["DEEP_id"].iloc()[0]}')

			deep_patch_df = df[df['DEEP_id']==deep_patch_id]
			for dt in list(set(deep_patch_df['local_obsnight'])):
				try:
					ok_fields = fields_by_dt_butler_orig[str(dt)]
				except KeyError as msg:
					raise KeyError(f'Did not find {dt} (type={type(dt)}) in fields_by_dt_butler_orig={fields_by_dt_butler_orig}')
				if good_pairs_only==True and deep_patch_id not in ok_fields: # TODO: this may be trouble using _orig here 5/29/2024 COC
					print(f'Skipping dt {dt}, field {deep_patch_id} pair not in fields_by_dt (good pairs): {ok_fields}')
					continue
				else:
					print(f'Keeping good pair {dt} with {deep_patch_id}')
#				found_deep_id = determine_field(rade=, dt [, max_separation_degrees=3 [, verbose=False ]])
				patches_to_plot.append(deep_patch_id)
				corresponding_dts.append(dt)
#				rade = fetch_deep_field_coords(deep_patch_id, dt=dt) # asdf
#				corresponding_center_coords.append(rade)
				corresponding_center_coords.append(better_pointing_centers[dt][deep_patch_id]) # 6/6/2024 COC
#			corresponding_center_coords = fetch_deep_field_coords(patches_to_plot, dt=corresponding_dts)
#		if deep_patch_id != None:
#			patches_to_plot = [deep_patch_id]
#		else:
#			patches_to_plot = list(set(df['DEEP_id']))
#			patches_to_plot.sort()
		edgecolor_idx = 0
		for i,deep_patch_to_plot in enumerate(patches_to_plot):
#			rade_original = fetch_deep_field_coords(deep_patch_to_plot) # asdf
			# 5/26/2024 COC: here we should figure out how to get the more appropriate RA, Dec for each field
#			middle_ut = df2['ut'].iloc()[len(df2)//2] # middle row UT
##			print(f'{deep_patch_to_plot} See rade_original={rade_original}, middle_ut={middle_ut}')
##			rade, fit = correct_parallax(coord=SkyCoord(ra=rade_original[0]*u.degree, dec=rade_original[1]*u.degree, frame='icrs'), obstime=], point_on_earth=EarthLocation.of_site('CTIO'), heliocentric_distance=dist_au)
#			rade_, fit = testy(radec=rade_original, obsdatetime=middle_ut, location_name='CTIO', distance_au=dist_au)
#			rade = [rade_.ra.deg, rade_.dec.deg]
#			rade = rade_original # nevermind, thought we needed parallax correction, that was incorrect
#			print(f'Got rade, fit = {rade}, {fit}')
#			this_circle = Circle(rade, 1.1, label=f'{deep_patch_to_plot} ({",".join([str(k) for k in rade])})', facecolor='none', edgecolor=edgecolors[i])
#			print(f'see corresponding_center_coords[i] = {corresponding_center_coords[i]}')
			edgecolor = edgecolors[edgecolor_idx]
			edgecolor_idx += 1
			if edgecolor_idx >= len(edgecolors): edgecolor_idx=0
			these_coords = corresponding_center_coords[i]
			this_circle = Circle(these_coords, 
				1.1, 
#				label=f'{deep_patch_to_plot} ({corresponding_dts[i]}: {",".join([str(k) for k in rade])})', 
				label=f'{deep_patch_to_plot} ({corresponding_dts[i]})',  # adding date 6/6/2024 COC
				facecolor='none', 
				edgecolor=edgecolor,
				alpha=0.75 # 6/6/2024 COC
			)
			ax.add_artist(this_circle)
			# PAUSED HERE -- plot the circle of the patch and add to legend 5/25/2024 COC
#			plt.plot()
	plt.legend(loc='lower left')
	plt.xlabel('RA')
	plt.ylabel('Dec')
	if deep_patch_id == 'A0c':
		plt.xlim([214,218])
		plt.ylim([-13,-10.5])
	elif deep_patch_id == 'A0b':
		plt.xlim([214,217.5]) # 213 if including 20au
		plt.ylim([-14.75,-12.4])
	plt.gca().set_aspect('equal')
	plt.title(f'{dist_au} au')
	plt.tight_layout()
	local_dates_str = make_desired_dates_str(desired_dates, long=True)
	available_dates = get_available_dates(df=df)
	if desired_dates == available_dates:
		local_dates_str = 'allDates'
	for ext in ['pdf', 'png']:
		outfile = f'fields_{dist_au}au_{make_desired_dates_str(desired_dates, long=True)}_DEEP_patch_{desired_deep_patch_id}.{ext}'
		if good_pairs_only == True:
			outfile = outfile.replace(f'.{ext}', f'_good_pairs_only.{ext}')
		plt.savefig(outfile)
	plt.close()
	#plt.show()
	print(f'For {dist_au}, unique dates:')
	print(set(list(df['local_obsnight'])))
	print(f'[{dist_au} au] len(df) is {len(df)}')
	print(f'[{dist_au} au] df columns are {df.columns}')
#	print(df['region'].iloc()[0])
#	print(type(df['region'].iloc[0]))
	#catalog_coords = SkyCoord(ra=df[f'center_coord_{dist_au}_ra']*u.degree, dec=df[f'center_coord_{dist_au}_dec']*u.degree, frame='icrs')
	
	min_dec = int(np.floor(min(df[f'center_coord_{dist_au}_dec'])))
	max_dec = int(np.ceil(max(df[f'center_coord_{dist_au}_dec'])))
	print(f'[{dist_au} au] Saw (ceiled and floored) min_dec = {min_dec}, max_dec = {max_dec}')
	return {'min_dec':min_dec, 'max_dec':max_dec}

def determine_available_bulter_distances(df):
	distances = []
	for colname in df.columns:
		if colname.startswith('center_coord_') and colname.endswith('_dec'):
#			print(colname)
			d = colname.replace('center_coord_','').replace('_dec','')
			distances.append(float(d))
	return(distances)


if __name__ == '__main__':
	better_pointing_centers = determine_better_pointing_centers(df=df)
	available_dates = get_available_dates(df=df_orig)
	available_dates.sort()
	print(f'saw available_dates={available_dates}')
	
	# trying some different variations 5/26/2024 COC
	#df_orig = df.copy() # 5/26/2024 COC gonna send this through analyze_fields a bit
	# 2019 dates as of 20240526 COC: 20190402, 20190403, 20190504, 20190505, 20190507, 20190601, 20190603
	date_sets_to_try = []
	date_sets_to_try.append(['2019-04-02', '2019-04-03', '2019-05-04', '2019-05-05', '2019-05-07'])
	date_sets_to_try.append(['2019-04-02', '2019-04-03'])
	date_sets_to_try.append(['2019-05-04', '2019-05-05'])
	date_sets_to_try.append(['2019-05-05', '2019-05-07'])
	date_sets_to_try.append(['2019-05-04', '2019-05-05', '2019-05-07'])
	date_sets_to_try.append(['2019-04-02', '2019-05-04'])
	date_sets_to_try.append(['2019-04-02', '2019-05-05'])
	date_sets_to_try.append(['2019-04-02', '2019-05-07'])
	date_sets_to_try.append(['2019-04-03', '2019-05-04'])
	date_sets_to_try.append(['2019-04-03', '2019-05-05'])
	date_sets_to_try.append(['2019-04-03', '2019-05-07'])
	#date_sets_to_try.append(['2019-06-01', '2019-06-03']) # empty arrays


def lots_of_analyze_fields(date_sets_to_try):
	analyze_fields(	df=df, 
					dist_au=42, 
					desired_deep_patch_id=None, 
					desired_dates=available_dates,
					better_pointing_centers=better_pointing_centers
				)
	
	# trying our plots with all fields we have instead of limiting to A0c for example 5/25/2024 COC
	distances, desired_cols = get_available_distances(df=df)
	for dist_au in distances:
		if dist_au < 1: 
			print(f'WARNING: skipping < 1 distance {dist_au} until we update some code for analyze_field ...')
			continue
		analyze_fields(df=df_orig, dist_au=dist_au, desired_deep_patch_id=None, desired_dates=[])
		analyze_fields(	df=df, 
						dist_au=dist_au, 
						desired_deep_patch_id='A0b',
						desired_dates=available_dates,
						better_pointing_centers=better_pointing_centers,
					)
	
	for date_set in date_sets_to_try:
		print(f'Trying {date_set} in analyze_fields() next...')
		analyze_fields(	df=df_orig[df_orig['local_obsnight'].isin(date_set)], 
						dist_au=42, 
						desired_deep_patch_id=None, 
						desired_dates=date_set,
						better_pointing_centers=better_pointing_centers,
					)
if __name__ == '__main__':
	print(f'NOTE: lots_of_analyze_fields is turned OFF')
#	lots_of_analyze_fields(date_sets_to_try=date_sets_to_try)


# Define the helper function
def process_distance(dist, df, desired_dates, better_pointing_centers):
	if dist == -1: 
		return None  # skip unaltered coords for this purpose
	
	results = analyze_fields(df=df, 
							dist_au=dist, 
							desired_dates=desired_dates,
							better_pointing_centers=better_pointing_centers,
							)
	
	return results

if __name__ == '__main__':
	# we could either choose a master dec range, that covers all the distances we are working with, or make a set per distance
	# TODO cache this somehow 6/6/2024 COC
	# TODO parallelize this 6/6/2024 COC
	global_min_dec = 99999
	global_max_dec = -99999
	distances, desired_cols = get_available_distances(df=df)
	# 
	# parallelizing stuff 6/6/2024 COC	
	
	# Use partial to fix additional arguments
	process_distance_partial = partial(process_distance, df=df, desired_dates=desired_dates, better_pointing_centers=better_pointing_centers)
	
	# Use process_map to apply the function in parallel
	results_list = process_map(process_distance_partial, distances, max_workers=cpu_count())
	
	# Initialize global_min_dec and global_max_dec
	global_min_dec = float('inf')
	global_max_dec = float('-inf')
	
	# Iterate over the results to update global_min_dec and global_max_dec
	for results in results_list:
		if results is not None:
			if results['min_dec'] < global_min_dec:
				global_min_dec = results['min_dec']
			if results['max_dec'] > global_max_dec:
				global_max_dec = results['max_dec']
				
	print(global_min_dec, global_max_dec)
	
#	for dist in distances:#desired_distances_au:
#		if dist == -1: continue # skip unaltered coords for this purpose
#		results = analyze_fields(df=df, 
#								dist_au=dist, 
#								desired_dates=desired_dates,
#								better_pointing_centers=better_pointing_centers,
#							)
#		if results['min_dec'] < global_min_dec:
#			global_min_dec = results['min_dec']
#		if results['max_dec'] > global_max_dec:
#			global_max_dec = results['max_dec']


def generate_patches(arcminutes, overlap_percentage, verbose=True, decRange=[-90, 90], export=True):
	"""Given a "rectangle" in (RA, Dec) touple (probably based on some chip size),
	produce a list of bounded regions on the sky,
	with user-supplied edge overlap percentage (overlap_percentage).
	The list is to be iterated over when matching against actual observations,
	i.e., for later shift-and-stack.
	v0: 12/11/2023 COC
	Note: this all assumes small angle approximation is OK. 1/9/2024 COC
	TODO: something to limit Dec range (min or max).
		E.g., at LSST (30.241° S) they cannot see anything above 59.759° N.
	TODO export
	"""
	import numpy as np
	
	def checkDec(n):
		if n > 90:
			n -= 180
		elif n < -90:
			n += 180
		return n
	
	def checkRA(n):
		if n > 360:
			n -= 360
		if n < 0:
			n += 360
		return n
	
	# Convert arcminutes to degrees (work base unit)
	arcdegrees = np.array(arcminutes) / 60.0
	
	# Calculate overlap in degrees
	overlap = arcdegrees * (overlap_percentage / 100.0)
	
	# Number of patches needed in RA, Dec space
	# TODO: consider these aren't decimal; so should be ceil for bounds 1/9/2024 COC
	num_patches_ra = int(360 / (arcdegrees[0] - overlap[0]))
	num_patches_dec = int(180 / (arcdegrees[1] - overlap[1]))
	if verbose:
		print(
			f"Number of patches in (RA, Dec): ({num_patches_ra},{num_patches_dec})."
		)  # Recall (RA, Dec) ranges are (0-360,0-180), so square inputs result in (n*2, n) ranges.
		
	# Generate patches
	patches = []
	centers = []  # 1/15/2024 COC
	skippedBecauseOfDec = 0
	skippedBecauseOfRA = 0 # 4/28/2024 COC
	for ra_index in range(num_patches_ra):
		# Calculate corner RA coordinates; moved out of dec loop 1/11/2024 COC
		ra_start = checkRA(ra_index * (arcdegrees[0] - overlap[0]))
		center_ra = checkRA(ra_start + arcdegrees[0] / 2)  # 1/15/2024 COC
		ra_end = checkRA(ra_start + arcdegrees[0])
		#
		for dec_index in range(num_patches_dec):
			# Calculate corner Dec coordinates
			dec_start = checkDec(dec_index * (arcdegrees[1] - overlap[1]) - 90)
			center_dec = checkDec(dec_start + arcdegrees[1] / 2)  # 1/15/2024 COC
			dec_end = checkDec(dec_start + arcdegrees[1])
			#
			# Make sure Dec is in allowed range; KLUDGE 1/9/2024 COC
			OK = True
			for d in [dec_start, dec_end]:
				if d < decRange[0] or d > decRange[1]:
					OK = False
					break
			if OK == False:
				skippedBecauseOfDec += 1
				#                 print(f'Something is outside of valid Dec range: dec_start={dec_start}, dec_end={dec_end}')
				continue
			#
			# Append patch coordinates to the list
			patches.append(((ra_start, dec_start), (ra_end, dec_end)))
			centers.append((center_ra, center_dec))  # 1/15/2024 COC
			
	#
	npatches = len(patches)
	info = {"npatches": npatches, "arcminutes": arcminutes, "overlap": overlap_percentage}
	if verbose:
		print(
			f"There were {npatches} produced, skipping {skippedBecauseOfDec} because Dec was outside {decRange}. Info: {info}."
		)
	#
	# produce CSV if desired
	if export == True:
		outfile = f"patches_{arcminutes[0]}x{arcminutes[1]}arcmin_{overlap_percentage}pctOverlap"
		if decRange != None:
			outfile += f"_Dec{decRange[0]}to{decRange[1]}"
		outfile += ".csv"
		with open(outfile, "w") as f:
			print(f"i,ra0,dec0,ra1,dec1", file=f)
			for i,quad in enumerate(patches):
				print(f"{i},{quad[0][0]},{quad[0][1]},{quad[1][0]},{quad[1][1]}", file=f)
			print(f"Wrote {len(patches)} patch rows to {outfile}.")
	#
	return patches, centers, info

if __name__ == '__main__':
	patch_size = [20,20]#[9,9] # making this component more mutable for this work 5/24/2024 COC
	patch_size_str = f'{patch_size[0]}x{patch_size[1]}'
	patchesNxN, centersNxN, infoNxN = generate_patches(arcminutes=(patch_size[0], patch_size[1]), overlap_percentage=0, decRange=[global_min_dec, global_max_dec])
	print(f'patch_size is {patch_size}; centersNxN[0] = {centersNxN[0]}')
	print(f'There are {len(centersNxN)} {patch_size_str} arcminute patches.')

def pythagoras(a,b):
	return np.sqrt(a**2 + b**2)


def calculate_max_separation(patch_size_arcmin, chip_size_arcmin, buffer_arcmin=0, verbose=True):
	"""
	We want the maximum separation between patches, which is diagonal-to-diagonal.
	By the Pythagorean Theorem, c^2 = a^2 + b^2.
	We take the half-lengths because we are calculating the maximum distance between the center and a corner,
	i.e., not the full diagonals of the polygons.
	N.B.: prior to the /2, 1908 patches had matches, with a total of 4,919,114 matches throughout. After: 1593, 1,236,424
	5/2/2024 COC
	"""
	
	patch_diagonal_arcmin = pythagoras(patch_size_arcmin[0]/2, patch_size_arcmin[1]/2)
	chip_diagonal_arcmin = pythagoras(chip_size_arcmin[0]/2, chip_size_arcmin[1]/2)
	max_separation_arcmin = patch_diagonal_arcmin + chip_diagonal_arcmin + buffer_arcmin
	if verbose: 
		print(f"Computed a maximum separation between a patch and a chip to be {max_separation_arcmin}', including a {buffer_arcmin}' buffer.")
	return max_separation_arcmin

if __name__ == '__main__':
	master_results_dict = {} # results dictionary 5/15/2024 COC

def is_point_in_patch(patch, rade):
	"""
	Check to see if a point is in the patch.
	NOTE: TODO probably have to deal with patches that transit RA=0° (a.k.a. 360°).
	5/24/2024 COC
	"""
	min_ra = min([i[0] for i in patch])
	max_ra = max([i[0] for i in patch])
	min_dec = min([i[1] for i in patch])
	max_dec = max([i[1] for i in patch])
	RA = rade[0]
	DEC = rade[1]
	if DEC >= min_dec and DEC <= max_dec and RA >= min_ra and RA <= max_ra:
		return True
	return False


def assess_matches(dist_au, patch_size_arcmin, chip_size_arcmin):
	"""
	Fixing to have patch_size_arcmin passed in. Caught problems 6/6/2024 COC.
	"""
	dist_au = float(dist_au) # Kludge 5/23/2024 COC
	# Define the main coordinate and the catalog of other coordinates
	#main_coord = SkyCoord(ra=[10]*u.degree, dec=[20]*u.degree, frame='icrs')
	if dist_au == -1:
		ra_field = 'center_coord_ra'
		dec_field = 'center_coord_dec'
	else:
		ra_field = f'center_coord_{dist_au}_ra'
		dec_field = f'center_coord_{dist_au}_dec'
	science_coords = SkyCoord(ra=[float(df[ra_field].iloc()[i])*u.degree for i in range(0,len(df))], dec=[float(df[dec_field].iloc()[i])*u.degree for i in range(0,len(df))], frame='icrs')
	patches_coords = SkyCoord(ra=[i[0]*u.degree for i in centersNxN], dec=[i[1]*u.degree for i in centersNxN], frame='icrs')
	
	# Find matches within 0.1 degrees
#	max_separation_arcmin = calculate_max_separation(patch_size_arcmin=[9,9], chip_size_arcmin=[9,18], buffer_arcmin=0, verbose=True)
	max_separation_arcmin = calculate_max_separation(patch_size_arcmin=patch_size_arcmin, chip_size_arcmin=chip_size_arcmin, buffer_arcmin=0, verbose=True)
	
	#max_separation_arcmin = 11 # 6.4' radius for a 9x9 patch, plus 4.5" for 1/2 of a short side of a decam chip = ~11'
	#print(f"WARNING: using a manually calculated max_separation_arcmin of {max_separation_arcmin}'.")
	print(f'See max_separation_arcmin={max_separation_arcmin}, type is {type(max_separation_arcmin)}.')
	print(f'See science (df) ra, dec fields are ({ra_field}, {dec_field}), which have types {type(float(df[ra_field].iloc()[0]))}, {type(float(df[dec_field].iloc()[0]))}.')
	print(f'See patches_coords are centersNxN[0] and [1]: {centersNxN[0][0]}, {centersNxN[0][1]}, types are {type(centersNxN[0][0])}, {type(centersNxN[0][1])}')
	
	lastTime = time.time()
	idx1, idx2, sep2d, dist3d = search_around_sky(patches_coords, science_coords, max_separation_arcmin*u.arcmin) # 0.1*u.deg
	elapsed = round(time.time() - lastTime, 1)
	print(f'Matching took {elapsed} seconds.')
	
	matches = patches_coords[idx1]
	print(f'There were {len(matches)} idx1 matches.')
	#print(matches)
	
	unique_patches_with_matches = list(set(idx1))
	print(f'There are {len(unique_patches_with_matches)} patches that have matches.')
	
	print(unique_patches_with_matches)
	
	#plt.hist(matches, bins=unique_patches_with_matches) # too slow 5/2/2024 COC
	
	print(f'idx1[0:10]: {idx1[0:10]}')
	
	counts = []
	for i in unique_patches_with_matches:
		counts.append(list(idx1).count(i))
	
	df2 = pd.DataFrame.from_dict({'patch_id':unique_patches_with_matches, 'counts':counts})
	df2 = df2.sort_values('counts', ascending=False)
	for i in range(0,20):
		print(f'{dist_au} au {i!s:3}: {df2["patch_id"].iloc()[i]} has {df2["counts"].iloc()[i]}')
		
	plt.xlabel(f'Patch ID')
	plt.ylabel(f'Number of Matches')
	plt.scatter([x for x in list(range(0,len(counts)))], counts, s=0.5)
	for ext in ['pdf', 'png']:
		plt.savefig(f'matches_{patch_size_str}_{dist_au}au_{desired_dates_str}_largeOnly_{large_piles_only}.{ext}')
	#plt.show()
	plt.close()
	#
	# NEXT: plot the matches from our top hit
	match_df = pd.DataFrame.from_dict({'idx1':idx1, 'idx2':idx2, 'sep2d':sep2d, 'dist3d':dist3d})
	return {'match_df':match_df, 'df2':df2}

if __name__ == '__main__':
	for dist_au in desired_distances_au:
		master_results_dict[dist_au] = assess_matches(dist_au=dist_au, patch_size_arcmin=patch_size, chip_size_arcmin=[9,18])

def getRegionCorners(region):
	"""
	Using the 2D boundingBox() from an input region (convexPolygon), we
	extract the (RA, Dec) coordinates of each vertex.
	As there are four vertices, the input object is a quadrilateral.
	2/2/2024 COC
	"""
	corners = []
	bbox = region.getBoundingBox()
#	print(bbox) # note: these are in radians
	corners.append((bbox.getLon().getA().asDegrees(), bbox.getLat().getA().asDegrees()))
	corners.append((bbox.getLon().getA().asDegrees(), bbox.getLat().getB().asDegrees()))
	corners.append((bbox.getLon().getB().asDegrees(), bbox.getLat().getB().asDegrees()))
	corners.append((bbox.getLon().getB().asDegrees(), bbox.getLat().getA().asDegrees()))
	print(f'corners: {corners}')
#	exit()
	return corners

def make_radebox(patch):
	coords = []
	ras = [i[0] for i in patch]
	decs = [i[1] for i in patch]
	coords.append([min(ras), min(decs)])
	coords.append([min(ras), max(decs)])
	coords.append([max(ras), max(decs)])
	coords.append([max(ras), min(decs)])
	coords.append([min(ras), min(decs)]) # close the shap]e
	return coords

def kludge_decam_box_deg(coord):
	"""
	Make a box using our knoweldge of DECam chips, assuming normal orientation on-sky.
	TODO: deal with spanning 0° RA 6/6/2024 COC
	5/5/2024 COC
	"""
	ra = coord[0]
	dec = coord[1]
	decam_ra_radius_arcmin = 9
	decam_dec_radius_arcmin = 4.5
	coords = []
	BL = [(ra - decam_ra_radius_arcmin/60), (dec - decam_dec_radius_arcmin/60)]
	TL = [(ra - decam_ra_radius_arcmin/60), (dec + decam_dec_radius_arcmin/60)]
	TR = [(ra + decam_ra_radius_arcmin/60), (dec + decam_dec_radius_arcmin/60)]
	BR = [(ra + decam_ra_radius_arcmin/60), (dec - decam_dec_radius_arcmin/60)]
	coords.extend([BL, TL, TR, BR])
	return coords


def latLonFromRaDecDeg(ra, dec, verbose=False):
	"""
	Return a sphgeom LonLat object given an input RA, Dec (in degrees).
	We correct Dec values outside of ±90° (e.g., subtract 180 from Dec=90.1).
	"""
	if dec > 90:
		print(f"WARNING: Dec > 90° ({dec}°) so subtracting 180°.")
		dec -= 180
	elif dec < -90:
		print(f"WARNING: Dec < -90° ({dec}°) so adding 180°.")
		dec += 180
	if ra > 360:
		print(f"WARNING: RA > 360° ({ra}°) so subtracting 360°.")
		ra -= 360
	elif ra < 0:  # just in case 1/9/2024 COC
		print(f"WARNING: RA < 0° ({ra}°) so adding 360°.")
		dec += 180
	t = lsst.sphgeom.LonLat(
		lsst.sphgeom._sphgeom.NormalizedAngle(np.deg2rad(ra)), lsst.sphgeom.Angle(np.deg2rad(dec))
	)
	return t


def patches_to_sphgeom(input_data):
	"""
	Convert a list of patches or FITS files to sphgeom regions.

	Parameters:
	- input_data: List of patches or FITS file paths.

	Returns:
	- List of sphgeom.SphericalPolygon objects.
	"""
	
	def read_patch_fits(fits_path):
		# Read FITS file and extract WCS information
		hdul = fits.open(fits_path)
		wcs = astropy.wcs.WCS(hdul[0].header)
		
		# Extract coordinates from the WCS
		ra, dec = wcs.all_pix2world(np.array([0, 0]), np.array([0, 0]), 0)
		ra_start, dec_start = ra[0], dec[0]
		
		ra, dec = wcs.all_pix2world(
			np.array([hdul[0].data.shape[1], hdul[0].data.shape[0]]), np.array([1, 1]), 0
		)
		ra_end, dec_end = ra[0], dec[0]
		
	#         return sphgeom.ConvexPolygon.from_radec_sequence(
	#             [(ra_start, dec_start), (ra_start, dec_end), (ra_end, dec_end), (ra_end, dec_start)]
	#         )
		
	sphgeom_regions = []
	
	for item in input_data:
		if isinstance(item, tuple):
			# If it's a patch tuple, convert to sphgeom.SphericalPolygon
			ra_start, dec_start = item[0]
			ra_end, dec_end = item[1]
			box = lsst.sphgeom.Box(
				latLonFromRaDecDeg(ra_start, dec_start), latLonFromRaDecDeg(ra_end, dec_end)
			)
			sphgeom_regions.append(box)
		elif isinstance(item, str):
			# NOTE: untested 2/15/2024 COC TODO
			# If it's a FITS file path, read the file and convert to sphgeom.SphericalPolygon
			sphgeom_regions.append(read_patch_fits(item))
		else:
			raise ValueError(
				"Unsupported input type. Supported types are tuple (patch) or str (FITS file path)."
			)
			
	return sphgeom_regions

# Example usage:
# Assuming you have a list of patches or FITS file paths
# patches_or_fits = generate_patches(arcminutes_input, overlap_percentage_input)
# sphgeom_regions_result = patches_to_sphgeom(patches_or_fits)

def get_patch_radius():
	r = pythagoras(a=patch_size[0],b=patch_size[1])/2
	print(f'Calculated a patch size of {r} for patch_size {patch_size}.')
	return r
	


def plot_by_match_id(df, fakes_df, dist_au, master_results_dict, match_id=0, max_uris=None, show=False, fakes_plot_method=1, min_overlapping=0, large_piles_only=True):
	"""
	TODO: implement min_overlapping 5/22/2024 COC/WSB
	5/7/2024 COC
	"""
	if large_piles_only == True:
		print(f'NOTE: limiting fakes to large_pile==True in plot_by_match_id().')
		fakes_df = fakes_df[fakes_df['large_pile']==True]
#	match_id = 6
	fig = plt.figure(figsize=[8,8])
	df2 = master_results_dict[dist_au]['df2']
	match_df = master_results_dict[dist_au]['match_df']
	top_patch_id = df2['patch_id'].iloc()[match_id]
	match_df2 = match_df[match_df['idx1']==top_patch_id]
	sep_sorted_df2 = match_df2.sort_values('sep2d', ascending=True)
#	print(sep_sorted_df2)
#	top_N_idx = np.argsort(match_df2['sep2d'])[:max_uris]
#	print(f'top N via np: {top_N_idx[0:10]}')
#	exit()
#	top_N_uri = [df['uri'].iloc()[i] for i in top_N_idx]
	# Moving this up here so we can output to the URI file 5/24/2024 COC
	patch_box = make_radebox(patchesNxN[top_patch_id])
	
	# moving URI export to later so we skip it for min_overlapping use 5/30/2024 COC
	def plot_coords_stuff(reflex='uncorrected', combo=False, extra_plots=False):
		"""
		Plot the centers and chip bounds
		Combo: also add fakes arrows
		TODO: add real (known) objects?
		5/5/2024 COC
		"""
		fig = plt.figure(figsize=[20,10])
		coords2 = []
		coords2_byDate = {}
		uts_byObsnight = {}
		idx2_overlapping_patch = [] # 6/7/2024 COC
		#coords2_boxes = []
		############
		### 5/13/2024 COC/WSB want to see what fakes we have
	#	patch_center = centersNxN[top_patch_id]
		patch_center_skycoord = SkyCoord(centersNxN[top_patch_id][0]*u.deg, centersNxN[top_patch_id][1]*u.deg, frame='icrs')
		fakes_by_date = {}
#		max_sep_arcmin = get_patch_radius() #12.8/2 # center to corner of square, ceiling, via Pythagorean theorem 5/15/2024 COC # disabled, not using any more; removed from print statements too 6/6/2024 COC
		all_fake_ids = []
		for dt in desired_dates:
			fakes_for_dt = fakes_df[fakes_df['local_obsnight'] == dt]
			rade_this_dist = SkyCoord([i*u.deg for i in fakes_for_dt[f'RA_{dist_au}']], [i*u.deg for i in fakes_for_dt[f'Dec_{dist_au}']], frame='icrs')
			fakes_for_dt['in_the_patch'] = [is_point_in_patch(patch=patchesNxN[top_patch_id], rade=[fakes_for_dt[f'RA_{dist_au}'].iloc()[i], fakes_for_dt[f'Dec_{dist_au}'].iloc()[i]]) for i in range(0,len(fakes_for_dt))]
#			fakes_for_dt['separation_arcmin'] = patch_center_skycoord.separation(rade_this_dist).arcmin
#			fakes_for_dt = fakes_for_dt[fakes_for_dt['separation_arcmin'] < max_sep_arcmin]
			fakes_for_dt = fakes_for_dt[fakes_for_dt['in_the_patch']==True]
			unique_fakes_for_dt = list(set(fakes_for_dt['ORBITID']))
			fakes_by_date[dt] = unique_fakes_for_dt
			all_fake_ids.extend(unique_fakes_for_dt)
			print(f'[{dist_au} au, slice {match_id}] {dt} had {len(unique_fakes_for_dt)} unique fake ORBITIDs in the patch center.')
		all_fake_ids = list(set(all_fake_ids))
		intersected = set(fakes_by_date[desired_dates[0]]).intersection(set(fakes_by_date[desired_dates[1]]))
		if len(list(intersected)) < min_overlapping:
			print(f'{len(list(intersected))} overlapping patches from both dates, < {min_overlapping}, for top_patch_id (slice?) = {top_patch_id} so skipping.')
			return None
		
		############
		print(f'[{reflex} reflex, {top_patch_id} patch_id] Here is an example FITS file from our pile: (TODO: get one that is very close to center of patch)')
		print(df['uri'].iloc()[match_df2['idx2'].iloc()[0]])
		colors = ['blue', 'orange']
		dts_legended = []
		boxes_to_plot = [] # 5/23/2024 COC: kludge so we can get number of UTs in the legend
		labels_for_plots = []
		colors_to_plot = []
		chip_linestyles = []
		for idx in match_df2['idx2']:
			if reflex == 'uncorrected':
				RA = df['center_coord_ra'].iloc()[idx] # df['center_coord'].iloc()[idx][0]
				DEC = df['center_coord_dec'].iloc()[idx] #df['center_coord'].iloc()[idx][1]
				the_coords = [RA, DEC]
			else:
				the_coords = [df[f'center_coord_{reflex}_ra'].iloc()[idx], df[f'center_coord_{reflex}_dec'].iloc()[idx]]
			coords2.append(the_coords)
#			dt = parser.parse(df['local_obsnight'].iloc()[idx]).date()
			dt = df['local_obsnight'].iloc()[idx] # trying without parser now that we are on local_obsnight 5/25/2024 COC
			if dt not in coords2_byDate: coords2_byDate[dt] = []
			coords2_byDate[dt].append(the_coords)
			if dt not in uts_byObsnight: uts_byObsnight[dt] = []
			uts_byObsnight[dt].append(df['ut_datetime'].iloc()[idx]) # visit not available I think, so we must use ut_datetime
	#		the_box_test = getRegionCorners(df['region'].iloc()[idx]) # uncomment to see coords of original region corners (wrong tho!)
			the_box = kludge_decam_box_deg(the_coords)
			#
			# The idea below is that we check each corner of the chip and if any corner falls within the patch then it is overlapping. 6/7/2024 COC
			# WARNING: this could backfire somehow? Actually I this this is pretty reasonable... 6/7/2024 COC
			# TODO: consider adding a buffer to the function that checks if a point is in the patch 6/7/2024 COC
			in_the_patch = False
			for corner_coord in the_box: # note the box does not yet have a repeat of coord 1 so we check all
				in_the_patch = is_point_in_patch(patch=patchesNxN[top_patch_id], rade=corner_coord)
#				print(f'Checking corner_coord {corner_coord} for being in patch {top_patch_id} {patchesNxN[top_patch_id]} resulted in in_the_patch={in_the_patch}')
				if in_the_patch == True:
#					print(f'Saw {idx} was in the patch')
#					idx2_overlapping_patch.append(df['uri'].iloc()[idx])
					idx2_overlapping_patch.append(idx) # instead of URI, use idx2 which gives us flexibility to, for example, find visits, not just URIs
					break # just need one to be in the patch, and must stop checking!
#			print(f'in_the_patch = {in_the_patch}')
			#
			# if the chip is not overlapping the patch, then skip it.
			if in_the_patch == False:
				chip_linestyle = ':'
				# TODO REENABLE CONTINUE 6/7/2024 COC
#				continue # do not bother plotting if it does not overlap with the patch
			else:
				chip_linestyle = '-'
			chip_linestyles.append(chip_linestyle)
			the_box.append(the_box[0]) # for plotting we must return to the original point to complete the polygon
		#	coords2_boxes.append(the_box)
			if str(dt) == str(desired_dates[0]):
				color = colors[0]
				colors_to_plot.append(color)
			else:
				color = colors[1]
				colors_to_plot.append(color)
			if dt not in dts_legended: # just send the first to the legend
				dts_legended.append(dt)
#				label = f'{dt} chip'
				labels_for_plots.append(dt)
			else:
#				label = None
				labels_for_plots.append(None)
			boxes_to_plot.append(the_box)
#			plt.plot(list(zip(*the_box))[0], list(zip(*the_box))[1], color=color, label=label)
		for i, the_box in enumerate(boxes_to_plot):
			label = labels_for_plots[i]
			color = colors_to_plot[i]
			if label != None:
				dt = label # we were saving dt for these
				label = f'{dt} chip ({len(list(set(uts_byObsnight[dt])))} UTs)' # asdf
			plt.plot(list(zip(*the_box))[0], list(zip(*the_box))[1], color=color, label=label, linestyle=chip_linestyles[i])
		print(f'[{reflex} reflex, {top_patch_id} patch_id] patchesNxN[top_patch_id={top_patch_id}] = {patchesNxN[top_patch_id]}')	
		
		#plt.scatter(centersNxN[top_patch_id][0], centersNxN[top_patch_id][1], label='patch center')
		patch_box = make_radebox(patchesNxN[top_patch_id]) # moving up (and out of function) so we can output to URI file 4/24/2024 COC; uncommenting 5/24/2024 COC
		print(f'[{reflex} reflex, {top_patch_id} patch_id] patch_box = {patch_box}')
		
		print(f'{reflex} reflex, patch center coords (slice {match_id}): {centersNxN[top_patch_id]}')
		
		if combo == False:
			plt.plot(list(zip(*patch_box))[0], list(zip(*patch_box))[1], label='shift-and-stack region', linestyle='--', color='green')
		patch_coords = []
		
		print(f'coords2[0] = {coords2[0]}')
		
		for coord in coords2:
			if coord[1] > 0:
				print(f'Dec > 0: {coord}')
		
		#plt.plot(patches_coords)
		#plt.scatter([x[0] for x in coords2], [y[1] for y in coords2], s=0.5, label='matches')
#		Xs = []
#		Ys = []
#		for coord in coords2:
#			Xs.append(coord[0])
#			Ys.append(coord[1])
#		print(f'min(Xs): {min(Xs)}')
		# DISABLED 5/16/2024 COC
#		for dt in coords2_byDate: # TODO disable this, but move the label somewhere else too 5/16/2024 COC
#			plt.scatter([coord[0] for coord in coords2_byDate[dt]], [coord[1] for coord in coords2_byDate[dt]], s=0.5, label=f'{dt} ({len(set(uts_byDate[dt]))})')#uts_byDate
#			plt.scatter([coord[0] for coord in coords2_byDate[dt]], [coord[1] for coord in coords2_byDate[dt]], s=0.5, label=f'{dt} ({len(coords2_byDate[dt])})')#uts_byDate
			
		#plt.scatter(Xs, Ys, s=0.5, label='matches')
		# disabling titles as we specify now in the xlabel, ylabel 5/16/2024 COC
#		if reflex == 'uncorrected':
#			plt.title(f'{reflex}')
#		else:
#			plt.title(f'{reflex} au')
		plt.gca().set_aspect('equal')
		#
		# Deal with axis labels
		xlabel = 'RA'
		ylabel = 'Dec'
		if reflex != 'uncorrected':
			xlabel += f' ({reflex} au SSB)'
			ylabel += f' ({reflex} au SSB)'
		#
		reflex_str = reflex
		if reflex != 'uncorrected':
			reflex_str += 'au'
		if combo == False:
			plt.legend()
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.tight_layout()
			for ext in ['pdf', 'png']:
				plt.savefig(f'matches_on_sky_{patch_size_str}_{reflex_str}_{desired_dates_str}_slice{match_id}_patch{top_patch_id}_largeOnly_{large_piles_only}.{ext}')
			if show: plt.show()
			plt.close()
		
		# moving intersected calculation to top so we can check for overlapping fakes first 5/30/2024 COC
		intersected_plotted = [] # 5/24/2024 COC
		intersected_colors = ['red', 'purple', 'cyan', 'magenta', 'deepskyblue', 'lime']
		intersected_linestyles = ['-', ':', '--', '-.']
		intersected_color_counter = 0
		intersected_linestyle_counter = 0
		print(f'[{dist_au} au, slice {match_id}] fakes_by_date = {fakes_by_date}')
		print(f'[{dist_au} au, slice {match_id}] Of the {len(all_fake_ids)} unique orbit IDs within the patch, there were {len(list(intersected))} ORBITIDs from both dates: {intersected}')
		#
		coords_by_dt = {} # should be a dict of dates that each have a list of end RA, Dec coords
		colors = ['blue', 'orange', 'red']
		linestyles = ['--', ':', '-']
		labels = []
		for k in [desired_dates[0], desired_dates[1], 'both dates']:
			labels.append(f'fake ({k})')
		if reflex == 'uncorrected':
			fakes_ra_column = 'RA' # RA, Dec
			fakes_dec_column = 'DEC'
		else:
			fakes_ra_column = f'RA_{dist_au}'
			fakes_dec_column = f'Dec_{dist_au}'
		for dt in desired_dates:
			coords_by_dt[dt] = []
			fakes_for_dt = fakes_df[fakes_df['local_obsnight'] == dt]
			for ORBITID in all_fake_ids:# unique_fakes_for_dt[dt]:
				orbitid_df = fakes_for_dt[fakes_for_dt['ORBITID']==ORBITID]
				print(f'[{dist_au} au, slice {match_id}] See len(orbitid_df) = {len(orbitid_df)}')
				center_row = len(orbitid_df) // 2 # integer division
				if dt == desired_dates[0]:
					the_row = 0
				else:
					the_row = -1
				if center_row == 0: # KLUDGE 5/15/2024 COC
					coord = [0,0]
				else:
					coord = [orbitid_df[fakes_ra_column].iloc()[the_row], orbitid_df[fakes_dec_column].iloc()[the_row]]
				coords_by_dt[dt].append(coord)
		
		labels_done_list = []
#		fakes_plot_method = 1
		if fakes_plot_method == 0:
			for i,ORBITID in enumerate(all_fake_ids):
				RAs = []
				DECs = []
				for dt in desired_dates:
					RAs.append(coords_by_dt[dt][i][0])
					DECs.append(coords_by_dt[dt][i][1])
				if 0 in RAs or 0 in DECs: # only had one date for this ORBITID, see KLUDGE above
					for j in range(0,1+1):
						if RAs[j] != 0:
							plt.scatter(RAs[j], DECs[j], 
	#							label=labels[j], 
								color=colors[j], 
								marker='x', 
								s=2,
							)
					print(f'[{dist_au} au, slice {match_id}] Saw a 0 in (RAs, DECs)')
					continue
				arrowXlen = RAs[-1] - RAs[0]
				arrowYlen = DECs[-1] - DECs[0]
				if ORBITID not in intersected:
					if ORBITID in fakes_by_date[desired_dates[0]]:
						plot_id = 0
					elif ORBITID in fakes_by_date[desired_dates[1]]:
						plot_id = 1
					else:
						raise KeyError(f'[{dist_au} au, slice {match_id}] ERROR: did not find ORBITID in the fakes_by_date dict!')
				else: # overlapping id
					plot_id = 2
				# TODO: try arrows instead 5/15/2024 COC
		#		plt.plot(RAs, DECs, color=colors[plot_id], linestyle=linestyles[plot_id], label=labels[plot_id])
				head_width = 0.025
				if plot_id == 2: # overlapping one, better legend for these
					# asdf
					fake_info = get_fake_info(ORBITID=ORBITID, desired_dates=desired_dates)[0]
					MAG = round(fake_info['MAG'].iloc()[0],1)
					a = round(fake_info['aei_1'].iloc()[0],1)
					e = round(fake_info['aei_2'].iloc()[0],1)
					incl = round(fake_info['aei_3'].iloc()[0],1)
					plt.arrow(RAs[0], DECs[0], arrowXlen, arrowYlen, 
						color=colors[plot_id], 
			#			linestyle=linestyles[plot_id], 
						length_includes_head=True, 
						head_width=head_width, 
						label=f'ID: {ORBITID} (a={a} au, e={e}, i={incl}°, m={MAG})'
					)
				elif plot_id not in labels_done_list:
					labels_done_list.append(plot_id)
					plt.arrow(RAs[0], DECs[0], arrowXlen, arrowYlen, 
						color=colors[plot_id], 
			#			linestyle=linestyles[plot_id], 
						length_includes_head=True, 
						head_width=head_width, 
						label=labels[plot_id]
					)
				else:
					plt.arrow(RAs[0], DECs[0], arrowXlen, arrowYlen, 
						color=colors[plot_id], 
			#			linestyle=linestyles[plot_id], 
						length_includes_head=True, 
						head_width=head_width, 
	#					label=labels[plot_id]
					)
		#
		# plotting the entire fakes trajectories instead 5/16/2024 COC
		elif fakes_plot_method == 1:
			dts_legended = []
#			orbitids_legended = [] # 5/22/2024 COC
			for ORBITID in all_fake_ids:
#				if ORBITID not in intersected:
#					if ORBITID in fakes_by_date[desired_dates[0]]:
#						plot_id = 0
#					elif ORBITID in fakes_by_date[desired_dates[1]]:
#						plot_id = 1
#					else:
#						raise KeyError(f'[{dist_au} au, slice {match_id}] ERROR: did not find ORBITID in the fakes_by_date dict!')
#				else: # overlapping id
#					plot_id = 2
				orbitid_df = fakes_df[fakes_df['ORBITID']==ORBITID]#fakes_for_dt[fakes_for_dt['ORBITID']==ORBITID]
				connecting_ras = []
				connecting_decs = []
				for k,dt in enumerate(desired_dates):
					orbitid_dt_df = orbitid_df[orbitid_df['local_obsnight']==dt]
					if connecting_ras == []:
						if len(orbitid_dt_df) == 0:
							connecting_ras.append(0)
							connecting_decs.append(0)
						else:
							connecting_ras.append(orbitid_dt_df[fakes_ra_column].iloc()[0])
							connecting_decs.append(orbitid_dt_df[fakes_dec_column].iloc()[0])
					else:
						if len(orbitid_dt_df) == 0:
							connecting_ras.append(0)
							connecting_decs.append(0)
						else:
							connecting_ras.append(orbitid_dt_df[fakes_ra_column].iloc()[-1])
							connecting_decs.append(orbitid_dt_df[fakes_dec_column].iloc()[-1])
					if dt not in dts_legended:
						dts_legended.append(dt)
						label = f'{dt} fake' # asdf put number of UTs here?
					else:
						label = None
					if ORBITID in intersected: # 5/20/2024 COC
						fake_info = get_fake_info(ORBITID=ORBITID, desired_dates=desired_dates)[k]
#						print(fake_info)
						MAG = round(fake_info['MAG'],1)
						a = round(fake_info['aei_1'],1)
						e = round(fake_info['aei_2'],1)
						incl = round(fake_info['aei_3'],1)
						r = round(fake_info['r'],1)
#						label = f'ID: {ORBITID} (m={MAG}, r={r} au, a={a} au, e={e}, i={incl}°)'
						label = None
					plt.plot(orbitid_dt_df[fakes_ra_column], orbitid_dt_df[fakes_dec_column], color=colors[k], label=label) # asdf
				if 0 not in connecting_ras:
					if ORBITID in intersected and ORBITID not in intersected_plotted:
						intersected_plotted.append(ORBITID)
						label = f'ID: {ORBITID} (m={MAG}, r={r} au, a={a} au, e={e}, i={incl}°)'
						line_color = intersected_colors[intersected_color_counter] #'red' # asdf
						linestyle = intersected_linestyles[intersected_linestyle_counter]
						intersected_color_counter += 1
						if intersected_color_counter >= len(intersected_colors):
							intersected_color_counter = 0
							intersected_linestyle_counter += 1
							if intersected_linestyle_counter >= len(intersected_linestyles):
								raise KeyError(f'ERROR: WE ARE OUT OF LINESTYLE COLOR COMBOS!')
					else:
						label = None
						line_color = 'black'
						linestyle = '-'
					plt.plot(connecting_ras, connecting_decs, color=line_color, alpha=0.25, linewidth=1, label=label, linestyle=linestyle)
			
				
		patch_box = make_radebox(patchesNxN[top_patch_id])
		print(f'[{dist_au} au, slice {match_id}] patch_box = {patch_box}')
		print(f'[{dist_au} au, slice {match_id}] Patch center coords: {centersNxN[top_patch_id]}')
		plt.plot(list(zip(*patch_box))[0], list(zip(*patch_box))[1], label=f"{patch_size_str}' patch", linestyle='--', color='green')
		#
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.gca().set_aspect('equal')
#		plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
		plt.legend(loc='lower left') # added lower left 6/7/2024 COC
		plt.tight_layout()
		if combo == False:
			for ext in ['pdf','png']:
				outfile = f'fakes_{patch_size_str}_{reflex_str}_{desired_dates_str}_slice{match_id}_patch{top_patch_id}_m{fakes_plot_method}_largeOnly_{large_piles_only}.{ext}'
				plt.savefig(outfile, bbox_inches = "tight")
				print(f'Wrote {outfile} to disk.')
			plt.close()
		
		if combo == True:
			for ext in ['pdf', 'png']:
				outfile = f'matches_with_fakes_{patch_size_str}_{reflex_str}_{desired_dates_str}_slice{match_id}_patch{top_patch_id}_m{fakes_plot_method}_largeOnly_{large_piles_only}.{ext}'
				# new 6/11/2024 COC: limit RA, Dec to keep the plots looking similar for comparisons
				plot_radius = ( max(infoNxN["arcminutes"]) / 60) * 2
				center_coord = centersNxN[top_patch_id]
				plt.xlim([center_coord[0]-plot_radius, center_coord[0]+plot_radius])
				plt.ylim([center_coord[1]-plot_radius, center_coord[1]+plot_radius])
				plt.savefig(outfile, bbox_inches = "tight")
				print(f'Wrote {outfile} to disk.')
			plt.close()
		return coords2, idx2_overlapping_patch
	
	for combo_option in [True, False]: # disabling false too many plots!!! 5/22/2024 COC; WAIT no we need it sadly (the uncorrected version is just empty otherwise)
		coords2, idx2_overlapping_patch = plot_coords_stuff(reflex=f'{dist_au}', combo=combo_option)
		if coords2 == None:
			print(f'Saw coords2==None, so skipping rest of plotting here...')  # 5/30/2024 COC
			return None
	plot_coords_stuff(reflex='uncorrected', combo=False) # moved this out of True/Falso combo bit as it looks useless with combo True 5/22/2024 COC
	
	coords2_freq = {}
	for coord in coords2:
		coord_str = str(round(coord[0],2)) + str(round(coord[1],2))
		if coord_str not in coords2_freq:
			coords2_freq[coord_str] = 0
		coords2_freq[coord_str] += 1
	print(f'[{dist_au} au, slice {match_id}] len(coords2_freq) = {len(coords2_freq)}')
	print(f'[{dist_au} au, slice {match_id}] coords2_freq = {coords2_freq}')

	clown_color_plot = False # disabling the famed "clown color plot" per WSB 6/6/2024 COC
	if clown_color_plot == True:
		for i, key in enumerate(list(coords2_freq.keys())):
			plt.scatter(i, coords2_freq[key])
		plt.xlabel('Index Number')
		plt.ylabel('Frequency of rounded (RA, Dec) coordinate')
		for ext in ['pdf']: # ,'png'
			outfile = f'freq_{patch_size_str}_{dist_au}au_{desired_dates_str}_slice{match_id}_patch{top_patch_id}_m{fakes_plot_method}_largeOnly_{large_piles_only}.{ext}'
			plt.savefig(outfile)
			print(f'[{dist_au} au, slice {match_id}] Wrote {outfile} to disk.')
		if show: plt.show()
		plt.close()
	#
	# MOVING the fakes part into the earlier function so we can have them all on the same plot 5/16/2024 COC
	#
	# moved uri export down here 5/30/2024 COC
	good_file_uris = [df['uri'].iloc()[i] for i in idx2_overlapping_patch] # sep_sorted_df2['idx2']]
	n_available_uris = len(good_file_uris)
	uri_file_visits = list(set(df['visit'].iloc()[i] for i in sep_sorted_df2['idx2']))
	if max_uris != None: 
		print(f'Limiting URI list to {max_uris} URIs. There were {n_available_uris} available.')
		good_file_uris = good_file_uris[0:max_uris]
		uri_file_visits = uri_file_visits[0:max_uris]
	unique_visits = len(list(set(uri_file_visits))) # this should be the number of unique timestamps in the final reprojected workunit
	#
	# we output the URI file helpful information that is also needed for work unit scripts 6/6/2024 COC
	with open(f'uris_{patch_size_str}_{dist_au}au_{desired_dates_str}_slice{match_id}_patch{top_patch_id}_lim{max_uris}.lst', 'w') as f:
		print(f'#list_generated_datetime={datetime.now()}') # note datetime.datetime!
		print(f'#desired_dates={desired_dates}', file=f)
		print(f'#dist_au={dist_au}', file=f)
		print(f'#patch_size={patch_size}', file=f)
		print(f'#patch_id={top_patch_id}', file=f)
		print(f'#patch_center_coords={centersNxN[top_patch_id]}', file=f)
#		print(f'#patch_info={infoNxN[top_patch_id]}', file=f) # KeyError 47438
		print(f'#patch_box={patch_box}', file=f)
		print(f'#unique_visits={unique_visits}', file=f)
		print(f'#pixel_scale=0.263', file=f) # TODO make this not hardwired 6/6/2024 COC
		print(f'#n_available_uris={n_available_uris}', file=f)
		print(f'#n_included_uris={len(good_file_uris)}', file=f)
		print('\n'.join(good_file_uris), file=f)


if __name__ == '__main__':
	max_slice = 10
	for i in range(0,max_slice+1): # to ,50 while we look for *any* overlaps in the nearer date sets 5/30/2024 COC; to 10 6/11/2024 COC
		for dist_au in desired_distances_au: # NTS: do not try to override these here, change desired_distances_au instead!
			if dist_au < 1:
				print(f'NOTE: skipping dist_au = {dist_au} for plot_by_match for now.')
				continue
	# PAUSED 5/30/2024 COC: thought about upping the number of match_ids we iterate through (i.e., look at all of them) but need to figure out the maximum number of matches? probably with a minimum number of matches (i.e., not 0); also it really is convenient to go by match_id first and not distance, so we get the top hits coming out first
	#			n_matches = master_results_dict[dist_au]['df2']
	#			match_df = master_results_dict[dist_au]['match_df']
	#			top_patch_id = df2['patch_id'].iloc()[match_id]
			plot_by_match_id(df=df, 
				fakes_df=fakes_df, 
				dist_au=float(dist_au), 
				master_results_dict=master_results_dict, 
				match_id=i, 
				show=False, 
				max_uris=5000, 
				fakes_plot_method=1, 
				min_overlapping=0, # probably should be 1 for chasing our best fakes, and if we set the range above to more than 20 say 5/30/2024 COC
			)

#
#def process_match(i, dist_au, df, fakes_df, master_results_dict):
#	if dist_au < 1:
#		print(f'NOTE: skipping dist_au = {dist_au} for plot_by_match for now.')
#		return
#	plot_by_match_id(
#		df=df,
#		fakes_df=fakes_df,
#		dist_au=float(dist_au),
#		master_results_dict=master_results_dict,
#		match_id=i,
#		show=False,
#		max_uris=5000,
#		fakes_plot_method=1,
#		min_overlapping=0,
#	)
#	
## Define the function outside of match_plotting
#def process_match_wrapper(args):
#	x, df, fakes_df, master_results_dict = args
#	return process_match(x[0], x[1], df, fakes_df, master_results_dict)
#
#def match_plotting(df, fakes_df, master_results_dict):
#	max_slice = 10
#	desired_distances_au = [1, 2, 3, 4, 5]  # Example distances
#	
#	# Prepare the list of inputs for process_map
#	inputs = []
#	for i in range(0, max_slice + 1):
#		for dist_au in desired_distances_au:
#			inputs.append(((i, dist_au), df, fakes_df, master_results_dict))
#			
#	# Use process_map to parallelize
#	process_map(process_match_wrapper, inputs, max_workers=cpu_count())
#	
#if __name__ == '__main__':
#	# Define or import your variables df, fakes_df, master_results_dict here
#	df = ...  # replace with actual data
#	fakes_df = ...  # replace with actual data
#	master_results_dict = ...  # replace with actual data
#	
#	match_plotting(df=df, fakes_df=fakes_df, master_results_dict=master_results_dict)
#	