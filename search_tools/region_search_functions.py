# 6/1/2024 COC breaking out some functions for sanity
import glob
import os
import sys
#from os.path import dirname
if __name__ == '__main__':
	sys.path.append(f'{os.environ["HOME"]}/bin')

COCOMMON = None
try:
	COCOMMON = os.environ['COCOMMON']
except KeyError:
	pass

import time

from datetime import datetime, timedelta, timezone
from dateutil import parser

#import lsst
#import lsst.daf.butler as dafButler
#import lsst.sphgeom as sphgeom

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map
import tqdm

from dateutil.parser import parse

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors

from scipy.optimize import minimize

import progressbar

from astropy.time import Time  # for converting Butler visitInfo.date (TAI) to UTC strings
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, GCRS, ICRS, get_body_barycentric
import astropy.io.fits as fits

from deep_fields_coc import determine_field


def butler_rad_to_float_degree(n, verbose=False):
	"""
	Take a butler rad, and convert to degrees, through kludgy string manipulation.
	Note that this is never as ideal as getting degrees from the original object.
	5/31/2024 COC
	"""
	n_deg = np.degrees(float(str(n).replace(' rad','')))
	if verbose: print(f'butler_rad_to_float_degree(): n={n} --> {n_deg} (degrees, but saved as a unitles float).')
	return n_deg

def clean_uris(uri_list):
	"""
	Standardize this to be consistent and useful for our purposes.
	Replace HTML code %23 with #, drop file://.
	5/31/2024 COC
	"""
	new_uris = []
	for uri in uri_list:
		new_uris.append(uri.replace('file://','').replace('%23','#'))
	return new_uris

def dataId_to_dataIdStr(dataId, force_clean=False, verbose=False):
	"""
	We will store a shortened, concatenated version of this unique triple.
	This is needed because dict values in a column are not hashable.
	Example: "{instrument: 'DECam', detector: 1, visit: 898287}" becomes "DECam_1_898287"
	4/23/2024 COC
	"""
	if force_clean:
		dataId = clean_dataId(dataId=dataId, verbose=verbose)
	#
	dataId_str = f"{dataId['instrument']}_{dataId['detector']}_{dataId['visit']}"
	#
	if verbose: 
		print(f'dataId {dataId} became: {dataId_str}')
	#
	return dataId_str

def get_available_distances(df):
	"""
	5/22/2024 COC: making a function and adding special cases of -1 and -2
	5/30/2024 COC: extending to use for out Butler df too
	6/1/2024 COC: renaming this to get_ from determine_, consolidating for both dataframes
	"""
	available_distances = []
	#
	# format will be like fit_30, RA_30, Dec_30, or 
	desired_cols = []
	what = 'fakes'
	if 'center_coord' in df.columns or 'center_coord_ra' in df.columns:
		what = 'butler'
	special_cols_to_use = {"RA":-1, "RA_known_r":-2, 'center_coord':-1} # uncorrected and corrected for true distance 5/22/2024 COC
	skip_colnames = ['center_coord_ra', 'center_coord_dec']
	for colname in df.columns:
		if colname in skip_colnames: continue
		if colname.startswith('fit_') and 'known_' not in colname:
			if what == 'fakes':
#				print(f'Adding {colname} to desired_cols.')
				desired_cols.append(colname)
			available_distances.append(float(colname.split('fit_')[-1]))
		elif colname in special_cols_to_use:
			available_distances.append(special_cols_to_use[colname])
	return available_distances, desired_cols


#def get_available_distances(df, verbose=True):
#	"""
#	Figure out what distances we have in the DataFrame already.
#	TODO: unify for fakes and butler DFs
#	5/21/2024 COC
#	"""
#	distances = []
#	skip_colnames = ['center_coord_ra', 'center_coord_dec']
#	for colname in df.columns:
#		if colname.startswith('center_coord_') and colname not in skip_colnames:
#			this_dist = colname.split('center_coord_')[-1].split('_')[0]
#			if this_dist not in distances:
#				distances.append(this_dist)
#	distances.sort()
#	if verbose: print(f'Saw distances: {", ".join(distances)}.')
#	return distances

def get_local_observing_date(ut_datetime, utc_offset=-4, verbose=False):
	"""
	Function to determine local observing night given a UTC offset.
	Default is UTC-4, for DECam.
	TODO: check how daylight savings time works for this.
	"""
	if verbose: print(f'get_local_observing_date() sees ut_datetime is {ut_datetime} (type: {type(ut_datetime)}) with utc_offset={utc_offset}.')
	if type(ut_datetime == type('')):
		ut_datetime = parser.parse(ut_datetime)
	# Convert the UTC offset to a timedelta
	offset = timedelta(hours=utc_offset)
	
	# Convert the UT datetime to local time
	local_datetime = ut_datetime + offset
	
	# Check if the local time is after midnight but before noon
	if 0 <= local_datetime.hour < 12:
		dt0 = local_datetime.date() - timedelta(days=1)
	else:
		dt0 = local_datetime.date()
		
	return dt0


def look_for_duplicates(df, verbose=True, fix=True):
	"""
	As a reality check, we make sure there are no duplicate pairs of identical dataId + UT.
	3/11/2024 COC
	"""
	if verbose:
		print(f"Starting look_for_duplicates()...")
	changed = False
	# make sure no duplicate URIs first (easy + fast)
	if 'uri' in df.columns:
		if len(df["uri"]) != len(set(df["uri"])):
			duplicated = df[df['uri'].duplicated()]
			for i in range(0, len(duplicated.index)):
				URI = duplicated["uri"].iloc()[i]
				print(f'ERROR: duplicate URI: {URI}. DataIDs:')
				these_dupes = df[df['uri']==URI]
				for j in range(0, len(these_dupes.index)):
					print(df['data_id'].iloc()[i])
#			raise AttributeError(
			print(
				f'WARNING: There were {len(set(df["uri"]))} unique URIs. But there are {len(df["uri"])} URIs in the DF. Keeping firsts.'
			)
			df = df[df['uri'].duplicated()==False]
			changed = True
#			export_both(df=df, df_filename=df_filename, df_csvname=df_csvname, verbose=verbose)
		else:
			print(f'No duplicate URIs found, so no need to write out dataframe.')
	#
	if (
		not df["data_id"].apply(str).is_unique
	):  # len(df["data_id"]) != len(set([str(df["data_id"].iloc()[i]) for i in range(0,len(df.index))])):
		if fix == True:
			print(f'Found non-unique dataIds, keeping only the most recent one.')
			df = df.mask(df["data_id"].apply(str).duplicated(keep='last')==True) # 5/4/2024 COC/WSB
		else:
			raise AttributeError(f"Encountered non-unique set of dataIds.")
	#
	if 'ut' in df.columns:
		dataId_uts = df["ut"].astype(str) + df["data_id"].astype(
			str
		)  # e.g., "2019-09-27T00:20:22.932{instrument: 'DECam', detector: 1, visit: 898286}"
		if len(dataId_uts) != len(set(dataId_uts)):
			raise AttributeError(f"dataID + UT combinations were not unique")
	if verbose:
		print(f"No duplication found amongst the {len(df.index)} dataframe rows.")
	return df, changed


def merge_dataframes(df1, df2, key="uri"):
	# Perform the merge
	merged_df = pd.merge(df1, df2, on=key, suffixes=('_df1', '_df2'))
	
	# Identify duplicate columns
	duplicate_columns = [col for col in df1.columns if col in df2.columns and col != key]
	
	# Combine duplicate columns
	for col in duplicate_columns:
		merged_df[col] = merged_df.apply(lambda row: row[col + '_df1'] if pd.notnull(row[col + '_df1']) else row[col + '_df2'], axis=1)
		
		# Drop the duplicate columns
		merged_df.drop(columns=[col + '_df1', col + '_df2'], inplace=True)
		
	return merged_df

def purge_old_coord_cols(df, verbose=True):
	"""
	Only keeping _ra and _dec columns, not the base coord column.
	E.g., center_coord_20.0 goes away, keeping only center_coord_20.0_ra and center_coord_20.0_dec
	6/2/2024 COC
	"""
	changed = False
	to_drop = []
	for colname in df.columns:
		if 'center_coord' not in colname: continue # only look for column names with colname...
		if '_ra' in colname or '_dec' in colname: continue # ... but skip columns that already have been split into ra and dec
		altcol = colname # look for the float version as well, if possible; if not, then altcol and colname are the same anyway
		if 'center_coord_' in altcol:
			dist_au = altcol.replace('center_coord_', '')
			dist_au = float(dist_au)
			altcol = f'center_coord_{dist_au}_ra'
		if f'{colname}_ra' in df.columns or altcol in df.columns: # good enough; if this is there, we assume dec and fit are there too
			if verbose: print(f'Adding {colname} to the drop list as we saw a corresponding _ra column is in the dataframe.')
			to_drop.append(colname)
		else:
			raise KeyError(f'Saw a basic center_coord {colname} without a corresponding {colname}_ra in the dataframe.')
	if len(to_drop) > 0:
		if verbose: print(f'Dropping df columns: {to_drop}')
		df = df.drop(columns=to_drop)
		changed = True
	return df, changed


def fix_fit_columns(df, verbose=True):
	"""
	Seeing fit_20.0.1 in addition to fit_20.0, for example.
	I have not seen where this is coming from; I suspect an old df merge call, but for now we need to drop offending columns.
	6/2/2024 COC
	"""
	changed = False
	#
	to_drop = []
	renames = {}
	# first we deal with duplicates based on extra .1 (e.g., fit_20.0 and fit_20.0.1)
	for colname in df.columns:
		if 'fit_' not in colname: continue
		if 'known' in colname: continue # special case, no float expected
		if colname.replace('fit_','').count('.') > 1: # trouble
			okname = colname.rstrip('.1')
			if okname in df.columns:
				if verbose: print(f'Saw {colname} and {okname}, so dropping {colname}. First of each: {df[colname].iloc()[0]}, {df[okname].iloc()[0]}.')
				to_drop.append(colname)
			else:
				if verbose: print(f'Saw {colname} but not {okname}, so renaming {colname} to {okname}.')
				renames[colname] = okname
	if len(to_drop) > 0:
		if verbose: print(f'Dropping columns: {to_drop}')
		df = df.drop(columns=to_drop)
		changed = True
	if len(renames) > 0:
		if verbose: print(f'Renaming columns: {renames}')
		df = df.rename(columns=renames)
		changed = True
	#
	to_drop = []
	renames = {}
	for colname in df.columns:
		if 'fit_' not in colname: continue
		if 'known' in colname: continue # special case, no float expected
		n = colname.replace('fit_','')
		if n.count('.') == 0: # integer name
			floatname = f'fit_{float(n)}'
			if floatname in df.columns:
				if verbose: print(f'Saw {floatname} so dropping {colname}. First of each: {df[floatname].iloc()[0]} and {df[colname].iloc()[0]}, respectively.')
				to_drop.append(colname)
			else:
				if verbose: print(f'No {floatname} so renaming {colname} to {floatname}.')
				renames[colname] = floatname
	if len(to_drop) > 0:
		if verbose: print(f'Dropping columns: {to_drop}')
		df = df.drop(columns=to_drop)
		changed = True
	if len(renames) > 0:
		if verbose: print(f'Renaming columns: {renames}')
		df = df.rename(columns=renames)
		changed = True
	return df, changed
			

def check_df_colname_integrity(df):
	"""
	Check that column names are OK, without double-decimals (e.g., fit_20.0.1), and that all values are floats (no ints).
	TODO
	6/2/2024 COC
	"""
	pass
		
			
			
#def df_cleanup(df):
#	"""
#	Seeing:
#		center_coord_dist (with dist being float or integer) surviving
#		fit_20.0.1
#	6/2/2024 COC
#	"""
#	to_drop = []
#	#
#	ignore_cols = ['fit_known_r']
#	colnames = df.columns
#	# first pass: deal with weird extra fit_ columns (e.g., fit_20.0.1)
#	for col in colnames:
#		if 'fit_' in col and '.0.1' in col:
#			okname = col.rstrip('.1')
#			if okname in colnames:
#				
#		
#	
#	
#	# first convert integer-ending colnames (which is deprecated) to float-ending colnames
#	for i in range(0,200+1):
#		pairs = []
#		coord_intcol = f'center_coord_{i}'
#		coord_floatcol = f'center_coord_{float(i)}'
#		fit_intcol = f'fit_{i}'
#		fit_floatcol = f'fit_{float(i)}'
#		ra_intcol = f'center_coord_{i}_ra'
#		ra_floatcol = f'center_coord_{float(i)}_ra'
#		dec_intcol = f'center_coord_{i}_dec'
#		dec_floatcol = f'center_coord_{float(i)}_dec'
#		#
#		if coord_intcol in df.columns:
#			if coord_floatcol in df.columns:
#				df = df.drop(columns=[coord_intcol])
#				print(f'Dropped {coord_intcol} from df.')
#			else:
#				df = df.rename(columns={coord_intcol:coord_floatcol})
#				print(f'Renamed {coord_intcol} to {coord_floatcol}.')
#		if fit_intcol in df.columns:
#			if fit_floatcol in df.columns:
#				df = df.drop(columns=[fit_intcol])
#				print(f'Dropped {fit_intcol} in favor of {fit_floatcol}.')
#			else:
#				df = df.rename(columns={fit_intcol:fit_floatcol})
		
				

def post_reflex_correction(df, dist_au, verbose=True):
	"""
	Drop SkyCoord containing columns in favor of float columns, one each for RA, Dec.
	This will hopefully help with the bloated size of the .pickles coming out.
	At some point we hope to drop the pickling bit altogether now that regions are being left behind as well.
	6/1/2024 COC
	"""
	dist_au = float(dist_au)
	anything_changed = False
	base_colname = f'center_coord_{float(dist_au)}'
	#
	if base_colname not in df.columns:
		if verbose:
			print(f'{base_colname} was not in df, so nothing to do.')
		return df, anything_changed
	#
	ra_col = f'{base_colname}_ra'
	dec_col = f'{base_colname}_dec'
	if ra_col in df.columns:
		if verbose:
			print(f'{base_colname} already in df, no need to add these.')
	else: # need the separate columns now
		try:
			df[ra_col] = [i.ra.deg for i in df[f'center_coord_{dist_au}']]
			df[dec_col] = [i.dec.deg for i in df[f'center_coord_{dist_au}']]
		except AttributeError as msg: # no .ra or .deg
			print(f'post_reflex_correction() AttributeError bypass for dist_au = {dist_au}.')
			df[ra_col] = [i[0] for i in df[f'center_coord_{dist_au}']]
			df[dec_col] = [i[1] for i in df[f'center_coord_{dist_au}']]
		anything_changed = True
	if ra_col not in df.columns:
		raise KeyError(f'IMPOSSIBLE: added a column ({ra_col}) to the dataframe but it vanished.') # just in case 6/1/2024 COC
	return df, anything_changed

def purge_bad_uts(df):
	bad_uts = df[df['ut'].isna()]
	if len(bad_uts) > 0:
		print(f'There were {len(bad_uts)} bad UTs in the DataFrame:')
		print(bad_uts)
		df = df[df['ut'].notna()]
	return df



def export_df(df, fn, verbose=False):
	"""
	Updated to do new safe write approach 5/31/2024 COC
	Functionizing 5/21/2024 COC
	"""
	startTime = time.time()
	distances, desired_cols = get_available_distances(df=df) # 5/21/2024 COC
	df_export = df.copy()
	for dist_au in distances:
		df, changed = post_reflex_correction(df=df, dist_au=dist_au, verbose=verbose)
	cols_to_drop = ['data_id', 'ut', 'ut_date']
	for dist_au in distances:
		cols_to_drop.append(f'center_coord_{dist_au}')
		cols_to_drop.append(f'center_coord_{float(dist_au)}')
	for i in cols_to_drop:
		try:
			df_export = df_export.drop(i, axis=1)
		except KeyError as msg: # column does not exist
			pass
	df_export_fn = fn
	time_int = f'{int(startTime)}'
	old_fn = f'{fn}_{time_int}_old'
	tmp_fn = f'{fn}_{time_int}'
#	df_export.to_csv(f'{df_export_fn}_{int(time.time())}', index=False, compression='gzip') # 5/27/2024 COC
	df_export.to_csv(tmp_fn, index=False, compression='gzip') # 5/27/2024 COC
	if len(glob.glob(tmp_fn)) > 0: # success
		if len(glob.glob(fn)) > 0:
			os.renames(fn, old_fn)
		os.renames(tmp_fn, fn)
	elapsed = round( (time.time() - startTime)/60 )
	print(f'In {elapsed} minutes we Wrote {fn} to disk and preserved the old one as {old_fn}.')
	return fn


def safe_export_pickle(df, fn, verbose=False):
	"""
	Because Klone ckpt partition gives as little as 12 seconds of notice to kill a job, we need to safely write out our pickles (and probably CSVs too).
	We had been saving the pickles twice, once with an int(time.time()) at the end, so as not to trash the original file if a kill happens.
	5/31/2024 COC
	"""
	startTime = time.time()
	now_int = f'{int(startTime)}' # keep int the same for both shuffled files to make it easier to identify which are related post-kill
	tmpfn = f'{fn}_{now_int}'
	oldfn = f'{fn}_old_{now_int}'
	df.to_pickle(tmpfn)
	if len(glob.glob(tmpfn)) > 0: # we got to this point, so we should be safe
		if len(glob.glob(fn)) > 0:
			os.renames(fn, oldfn)
		os.renames(tmpfn, fn)
	elapsed = round( (time.time() - startTime) / 60, 1)
	print(f'It took {elapsed} minutes to write out {fn}. The old one is now {oldfn}.')
	
	
def export_both(df, df_filename, df_csvname, verbose=True):
	"""
	6/2/2024 COC: added the df integrity checks fix_fit_columns and purge_old_coords
	6/1/2024 COC
	"""
	if verbose: print(f'Exporting pickle and CSV now...')
	df, changed = fix_fit_columns(df=df, verbose=verbose)
	df, changed = purge_old_coord_cols(df=df, verbose=verbose)
	safe_export_pickle(df=df, fn=df_filename)
	export_df(df=df, fn=df_csvname)
	return df, changed


# Not sure if we used this recently but it is useful 6/1/2024 COC
def analyze_detector(df, detector_number=26, basedir='.'):
	"""
	26 is near the center of the DECam field.
	Functionized 5/2/2024 COC
	"""
	prop_cycle = plt.rcParams['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']
	
	df4 = df[df['detector']==detector_number]
	done_dates = []
	color_counter = 0
	linestyle_counter = 0
	linestyles = ['-','--',':', '-.']
	
	for i in range(0,len(df4.index)):
		dt = df4['local_obsnight'].iloc[i] # ut to local_obsnight 5/25/2024 COC
		if dt in done_dates: continue
		done_dates.append(dt)
		start = df4['center_coord'].iloc[i]
		detector = df4['detector'].iloc[i]
		tmp = df4[f'center_coord_{dist_au}'].iloc[i]
		end = [tmp.ra.deg, tmp.dec.deg]
	#	plt.scatter(start[0], start[1], marker='$' + str(i) + '$')
		plt.arrow(x=start[0], y=start[1], dx=(end[0] - start[0])*arrow_scaling, dy=(end[1] - start[1])*arrow_scaling, 
			color=colors[color_counter], 
			linestyle = linestyles[linestyle_counter],
			head_width=0.025, 
			head_length=0.1, 
			alpha=0.5,
			label = str(dt)
		)
		color_counter += 1
		if color_counter >= len(colors):
			color_counter = 0
			linestyle_counter += 1
			if linestyle_counter >= len(linestyles): # wrap, basically 4/27/2024 COC
				linestyle_counter = 0
	plt.xlabel('RA')
	plt.ylabel('Dec')
	plt.legend()
	for ext in ['pdf', 'png']:
		plt.savefig(f'{basedir}/{label}_chip{detector_number}_reflex_correction.{ext}')



# again dunno when we use this but looks useful 6/1/2024 COC
## Dataframe is all ready. Now on to the meat of the work...
#if __name__ == '__main__':
#	# Create lookup tables for data_id_str to center_coord and ut_datetime for vast speed-ups.
#	id_to_coord = df.set_index("data_id_str")["center_coord"].to_dict()
			
			
def analyze_stuff(date_str='2019-04-03', visit_id=845583, dist_au=30, basedir='.'):
	"""
	UNTESTED dist_au 5/15/2024 COC
	Functionized 5/2/2024 COC.
	"""
	id_to_date = df.set_index("data_id_str")["ut_datetime"].dt.date.to_dict() # moved here, noting df is not passed in though so to fix 6/1/2024 COC
	unique_dates = sorted(set(id_to_date.values()))
	print(f'See the following unique dates in the dataframe: ')
	print("\n".join([str(dt) for dt in unique_dates]))
	
	df2 = df[df['ut_date']==parse(f'{date_str} 00:00:00').date()]
	df3 = df2[df2['visit']==visit_id]
	
	starts = []
	ends = []
	for i in range(0,len(df3.index)):
		start = df3['center_coord'].iloc[i]
		starts.append(start)
		tmp = df3[f'center_coord_{dist_au}'].iloc[i]
		end = [tmp.ra.deg, tmp.dec.deg]
		ends.append(end)
		arrow_scaling = 0.8
		plt.arrow(x=start[0], y=start[1], dx=(end[0] - start[0])*arrow_scaling, dy=(end[1] - start[1])*arrow_scaling, color='blue', head_width=0.025, head_length=0.1, alpha=0.5)
	#	plt.scatter(start[0], start[1], color='blue', zorder=1, alpha=0.5, marker='s', s=15, label='original')
	#	plt.scatter(end[0], end[1], color='red', zorder=1, label='corrected')
	plt.scatter([j[0] for j in starts], [j[1] for j in starts], color='blue', zorder=1, alpha=0.5, marker='s', s=15, label='original')
	plt.scatter([j[0] for j in ends], [j[1] for j in ends], color='red', zorder=1, label='corrected')
	#	print(f'start -> end is {start} -> {end}')
	#	break
	plt.xlabel('RA')
	plt.ylabel('Dec')
	plt.legend()
	plt.tight_layout()
	#if label != None: plt.title(label.replace('_', ' '))
	for ext in ['pdf','png']:
		plt.savefig(f'{basedir}/before_and_after_{label}_{date_str}_visit{visit_id}.{ext}')
		
	plt.close()
	
	for i in range(0,len(df3.index)):
		start = df3['center_coord'].iloc[i]
		detector = df3['detector'].iloc[i]
		plt.scatter(start[0], start[1], marker='$' + str(i) + '$')
	plt.xlabel('RA')
	plt.ylabel('Dec')
	for ext in ['pdf','png']:
		plt.savefig(f'{basedir}/detectors_example.{ext}')
	plt.close()



def find_good_date_pointing_pairs(df, min_exposures=20):
	"""
	Find pairs of dates/DEEP patch IDs (e.g., A0c) that have high numbers of exposures.
	5/26/2024 COC
	"""
	print(f'Starting find_good_date_pointing_pairs(df, min_exposures={min_exposures})')
	what = 'butler'
	if 'RA' in df.columns:
		what = 'fakes'
	if 'visit' not in df.columns:
		raise KeyError(f'Missing! visit column is not here. We see columns: {df.columns}')
	local_nights = list(set(df['local_obsnight']))
	local_nights.sort()
	pairs = {} # key: date, values: list of deep patch IDs (e.g., A0c) # asdf
	for dt in local_nights:
		if dt not in pairs: pairs[dt] = [] # important we have empty lists
		dt_df = df[df['local_obsnight']==dt]
		dt_deep_ids = list(set(dt_df['DEEP_id']))
		for deep_id in dt_deep_ids:
			deep_id_df = dt_df[dt_df['DEEP_id']==deep_id]
			these_uts = list(set(deep_id_df['visit'].values))
			if len(these_uts) >= min_exposures:
#				print(f'{dt} ID {deep_id} had {len(these_uts)}: {these_uts}') # did not see similar UTs by-eye; warning: wordy! 5/26/2024 COC
				print(f'{dt} ID {deep_id} had {len(these_uts)} UTs.')
				pairs[dt].append(deep_id)
	return pairs

def make_small_piles_go_away(df, combined_fields_by_dt):
	ok_flags = []
	for i in range(0,len(df)):
		dt = df['local_obsnight'].iloc()[i]
		deep_id = df['DEEP_id'].iloc()[i]
		if dt not in combined_fields_by_dt or deep_id not in combined_fields_by_dt[dt]: # added dt not in catch 6/4/2024 COC
			ok_flags.append(False)
		else:
			ok_flags.append(True)
	df['large_pile'] = ok_flags
	return df


### DEEP ID stuff

def process_deepid_row(args):
	rade, dt, df, local_obsdates = args
	deep_id_df = determine_field(rade, dt, df, local_obsdates)
	return deep_id_df['DEEP_id']

def add_DEEP_ids(df, fields_by_dt_csvfile='deep_field_by_date_coc.csv', verbose=False):
	if len(glob.glob(fields_by_dt_csvfile)) == 0 and COCOMMON != None:
		fields_by_dt_csvfile = os.path.join(COCOMMON,fields_by_dt_csvfile)
	if verbose:
		print(f'add_DEEP_ids(df) sees df.columns: {df.columns}')
	if 'RA' in df.columns and 'DEC' in df.columns:
		what = 'fakes'
		ra_col = 'RA'
		dec_col = 'DEC'
	else:
		what = 'butler'
		ra_col = 'center_coord_ra'
		dec_col = 'center_coord_dec'
	#
	print(f'Adding DEEP_id to the DataFrame. what={what}')
	#
	# Read CSV file once and pass it to the determine_field function
	fields_df = pd.read_csv(fields_by_dt_csvfile)
	local_obsdates = list(set(fields_df['local_obsdate'].values))
	#
	RAs = df[ra_col]
	DECs = df[dec_col]
#	if what == 'butler':
#		RAs = df['pointing_ra']
#		DECs = df['pointing_dec']
#	elif what == 'fakes':
#		RAs = df['RA']
#		DECs = df['DEC']
	print(f'df[{ra_col}][0] is {df[ra_col].iloc()[0]}, type {type(df[ra_col].iloc()[0])}')
	dts = df['local_obsnight']
	args_list = [(rade, dt, fields_df, local_obsdates) for rade, dt in zip(zip(RAs, DECs), dts)]
	
	deep_ids = process_map(process_deepid_row, args_list, max_workers=cpu_count(), chunksize=10)
	
	df['DEEP_id'] = deep_ids
	return df

def make_desired_dates_str(desired_dates, long=False):
	"""
	Returns a clean string that has _and_ joined the various dates.
	TODO: consider removing the dashes from the dates 5/26/2024 COC
	5/26/2024 COC
	"""
	if long == False:
		desired_dates_str = '_and_'.join(desired_dates)
	else:
		desired_dates_str = '_'.join([str(i).replace('-','') for i in desired_dates])
	return(desired_dates_str)

def add_visit_ids(df, pointings_df):
	"""
	Adding visit, which is a unique integer from the Butler.
	6/17/2024 COC
	"""
	print(f'Starting add_visit_ids(df, pointings_df)...')
	visits = []
	uts = []
#	print(f'See pointings_df columns are {pointings_df.columns}')
#	print(f'See df.columns are {df.columns}')
	ut_to_visit = pointings_df.set_index("ut_datetime")["visit"].to_dict()
	pointing_uts = list(set(pointings_df['ut_datetime']))
	if 'ut_datetime' in df.columns:
		utcol = 'ut_datetime'
	elif 'ut' in df.columns:
		utcol = 'ut'
	else:
		raise KeyError(f'Do not know what column to use for ut_datetime in df. Available columns were {df.columns}.')
	#
	unique_ut_to_visit = {}
	unique_uts = list(set(df[utcol]))
	with progressbar.ProgressBar(max_value=len(unique_uts)) as bar:	# len(df)
		for i,ut in enumerate(unique_uts): # enumerate(df[utcol]):
			closest_ut = find_closest_datetime(datetime_list=pointing_uts, user_datetime=parser.parse(ut))
			visit = ut_to_visit[str(closest_ut)[0:-3]]
#			print(f'found {visit} visit for closest_ut {closest_ut}')
#			visits.append(visit)
			unique_ut_to_visit[ut] = visit
			bar.update(i)
	#
	for i,ut in enumerate(df[utcol]):
		visits.append(unique_ut_to_visit[ut])
	df['visit'] = visits
	return df


def get_pointings_dataframe(df, butler_csvfile, overwrite=False):
	"""
	Make a dataframe of just the pointings, useful for iterating over for SkyBot queries.
	6/15/2024 COC
	"""
	pointing_csvfile = butler_csvfile.replace('.csv.gz', '_pointings.csv.gz')
	skybot_csvfile = butler_csvfile.replace('.csv.gz', '_skybot.csv.gz')
	if overwrite == True or len(glob.glob(pointing_csvfile)) == 0: # need to make it
		pointing_df = df.drop_duplicates(subset=['visit']).sort_values(by='visit')
		keep_columns = ['visit', 'pointing_ra', 'pointing_dec', 'DEEP_id', 'local_obsnight', 'ut_datetime']
		columns_to_drop = []
		for colname in pointing_df.columns:
			if colname not in keep_columns:
				columns_to_drop.append(colname)
		pointing_df = pointing_df.drop(columns=columns_to_drop)
		print(f'Writing pointing_csvfile to disk...')
		pointing_df.to_csv(pointing_csvfile, compression='gzip', index=False)
	else:
		print(f'Recycling {pointing_csvfile}...')
		pointing_df = pd.read_csv(pointing_csvfile)
	print(f'After all is said and done, pointing_df has columns {pointing_df.columns}.')
	return pointing_df


def get_fakes_df(desired_dates, base_csvfile='fakes_detections_simple.csv.gz', overwrite=False, butler_df=None, butler_csvfile=None):
	"""
	Too slow to pull in the big fakes CSV file so we will start caching desired_dates_str-based subsets.
	6/6/2024 COC
	"""
	desired_dates_str = make_desired_dates_str(desired_dates=desired_dates, long=True)
#	cache_csv = f'fakes_cache_{desired_dates_str}.csv.gz'
	cache_csv = base_csvfile.replace('.csv.gz', f'_{desired_dates_str}.csv.gz') # 6/17/2024 COC
	if overwrite == True or len(glob.glob(cache_csv)) == 0:
		print(f'No cache_csv {cache_csv} found, so creating now. First, reading large fakes csv...')
		orig_fakes_df = pd.read_csv(base_csvfile)
		#
		# we have to have this column. We technically do not need anything else (e.g., DEEP_ids) so just requiring this column here. WARNING: copy/pasted from main body. 6/6/2024 COC
		if 'local_obsnight' not in orig_fakes_df.columns:
			print(f'local_obsnight missing from orig_fakes_df, adding now...')
			orig_fakes_df['local_obsnight'] = [get_local_observing_date(ut_datetime=UT) for UT in orig_fakes_df['ut_datetime']]#, utc_offset)]
			orig_fakes_df.to_csv(base_csvfile, index=False, compression='gzip')
			print(f'Post local_obsnight, wrote updated csvfile: {base_csvfile}')
		if 'visit' not in orig_fakes_df.columns and butler_df is not None and butler_csvfile is not None:
			orig_fakes_df = add_visit_ids(df=orig_fakes_df, pointings_df=get_pointings_dataframe(df=butler_df, butler_csvfile=butler_csvfile))
			orig_fakes_df.to_csv(base_csvfile, index=False, compression='gzip')
			print(f'Post visit IDs, wrote updated csvfile: {base_csvfile}')
		#
		print(f'Slicing to only include desired_dates = {desired_dates}...')
		df = orig_fakes_df[orig_fakes_df['local_obsnight'].isin(desired_dates)]
		print(f'Writing cache_csv: {cache_csv}')
		df.to_csv(cache_csv, compression='gzip', index=False)
		print(f'Wrote {cache_csv}.')
	else:
		print(f'Recycling cache_csv {cache_csv}...')
		df = pd.read_csv(cache_csv)
		if 'visit' not in df.columns and butler_df is not None and butler_csvfile is not None:
			df = add_visit_ids(df=df, pointings_df=get_pointings_dataframe(df=butler_df, butler_csvfile=butler_csvfile))
			df.to_csv(cache_csv, index=False, compression='gzip')
			print(f'Post visit IDs, wrote updated csvfile: {cache_csv}')
	return df

#######

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


def find_closest_datetime(datetime_list, user_datetime):
	"""
	Find the closest datetime to the user-supplied datetime.

	Parameters:
	datetime_list (list of datetime): List of datetime objects to search through.
	user_datetime (datetime): The user-supplied datetime object.

	Returns:
	datetime: The closest datetime object from the list.
	"""
	if type(datetime_list[0]) == type(''):
		datetime_list = [parser.parse(i) for i in datetime_list]
	closest_datetime = min(datetime_list, key=lambda x: abs(x - user_datetime))
	return closest_datetime


########

def remove_nans(df):
	"""
	Saw this 5/2/2024 COC:
		data_id region  detector data_id_str instrument  visit   ut ut_datetime ut_date center_coord
		34131     NaN    NaN       NaN         NaN        NaN    NaN  NaN         NaT     NaN          NaN

	Adding change checking to avoid unnecessary writing of dataframe 5/26/2024 COC
	5/2/2024 COC
	"""
	start_len = len(df.index)
	print(f'Starting remove_nans(df)...')
	if len(df[df['ut'].isna()]) > 0:
		print(f'WARNING: encountered one or more rows with NaNs: {df[df["ut"].isna()]}')
		print(f'Dropping NaN/NaT rows...')
		df = df.dropna()
	changed = False
	if len(df.index) != start_len:
		changed = True
	return df, changed


#### correct parallax

#def correct_parallax(coord, obstime, point_on_earth, guess_distance, method=None, verbose=False):
#	"""Calculate the parallax corrected postions for a given object at a given time and distance from Earth.
#
#	Attributes
#	----------
#	coord : `astropy.coordinate.SkyCoord`
#		The coordinate to be corrected for.
#	obstime : `astropy.time.Time` or `string`
#		The observation time.
#	point_on_earth : `astropy.coordinate.EarthLocation`
#		The location on Earth of the observation.
#	guess_distance : `float`
#		The guess distance to the object from Earth.
#
#	Returns
#	----------
#	An `astropy.coordinate.SkyCoord` containing the ra and dec of the point in ICRS.
#
#	References
#	----------
#	.. [1] `Jupyter Notebook <https://github.com/DinoBektesevic/region_search_example/blob/main/02_accounting_parallax.ipynb>`_
#	"""
#	
#	loc = (
#		point_on_earth.x.to(u.m).value,
#		point_on_earth.y.to(u.m).value,
#		point_on_earth.z.to(u.m).value,
#	) * u.m
#	
#	# line of sight from earth to the object,
#	# the object has an unknown distance from earth
#	los_earth_obj = coord.transform_to(GCRS(obstime=obstime, obsgeoloc=loc))
##    los_earth_obj = GCRS(ra=coord.ra, dec=coord.dec, obstime=obstime, obsgeoloc=loc) # this was much worse 4/1/2024 COC
#	
#	cost = lambda d: np.abs(
#		guess_distance
#		- GCRS(ra=los_earth_obj.ra, dec=los_earth_obj.dec, distance=d * u.AU, obstime=obstime, obsgeoloc=loc)
#		.transform_to(ICRS())
#		.distance.to(u.AU)
#		.value
#	)
#	
#	fit = minimize(
#		fun=cost,
#		x0=(guess_distance,),
#		method=method,
#		options={'disp':verbose} # , 'maxiter':100}
#	)
#	answer = SkyCoord(ra=los_earth_obj.ra, dec=los_earth_obj.dec, distance=fit.x[0]*u.AU, obstime=obstime, obsgeoloc=loc, frame=GCRS).transform_to(ICRS)
##     answer = GCRS(
##         ra=los_earth_obj.ra, dec=los_earth_obj.dec, distance=fit.x[0] * u.AU, obstime=obstime, obsgeoloc=loc
##     ).transform_to(ICRS())
##     print(f'The type() of answer is {type(answer)}.')
#	return answer, fit.x[0]

# now using new drew geometric one 5/31/2024 COC
def correct_parallax(coord, obstime, point_on_earth, heliocentric_distance):
	"""Calculate the parallax corrected postions for a given object at a given time and distance from Earth.

	Attributes
	----------
	coord : `astropy.coordinate.SkyCoord`
		The coordinate to be corrected for.
	obstime : `astropy.time.Time` or `string`
		The observation time.
	point_on_earth : `astropy.coordinate.EarthLocation`
		The location on Earth of the observation.
	heliocentric_distance : `float`
		The guess distance to the object from the Sun.

	Returns
	----------
	An `astropy.coordinate.SkyCoord` containing the ra and dec of the point in ICRS, and the best fit geocentric distance (float).

	References
	----------
	.. [1] `Jupyter Notebook <https://github.com/DinoBektesevic/region_search_example/blob/main/02_accounting_parallax.ipynb>`_
	"""
	# Compute the Earth location relative to the barycenter.
	# times = Time(obstime, format="mjd")
	
	# Compute the Earth's location in to cartesian space centered the barycenter.
	# This is an approximate position. Is it good enough?
	earth_pos_cart = get_body_barycentric("earth", obstime)
	ex = earth_pos_cart.x.value + point_on_earth.x.to(u.au).value
	ey = earth_pos_cart.y.value + point_on_earth.y.to(u.au).value
	ez = earth_pos_cart.z.value + point_on_earth.z.to(u.au).value
	
	# Compute the unit vector of the pointing.
	loc = (point_on_earth.to_geocentric()) * u.m
	los_earth_obj = coord.transform_to(GCRS(obstime=obstime, obsgeoloc=loc))
	
	pointings_cart = los_earth_obj.cartesian
	vx = pointings_cart.x.value
	vy = pointings_cart.y.value
	vz = pointings_cart.z.value
	
	# Solve the quadratic equation for the ray leaving the earth and intersecting
	# a sphere around the sun (0, 0, 0) with radius = heliocentric_distance
	a = vx * vx + vy * vy + vz * vz
	b = 2 * vx * ex + 2 * vy * ey + + 2 * vz * ez
	c = ex * ex + ey * ey + ez * ez - heliocentric_distance * heliocentric_distance
	disc = b * b - 4 * a * c
	
	if (disc < 0):
		return None, -1.0
	
	# Since the ray will be starting from within the sphere (we assume the 
	# heliocentric_distance is at least 1 AU), one of the solutions should be positive
	# and the other negative. We only use the positive one.
	dist = (-b + np.sqrt(disc))/(2 * a)
	
	answer = SkyCoord(
		ra=los_earth_obj.ra, # this was coord.ra
		dec=los_earth_obj.dec, # this was coord.dec
		distance=dist * u.AU,
		obstime=obstime,
		obsgeoloc=loc,
		frame="gcrs",
	).transform_to(ICRS())
	
	return answer, dist


# updating testy stuff to pass df along 6/1/2024 COC
def testy(radec, obsdatetime, location_name, distance_au, method=None, verbose=False, dist_offset_au=0):
	"""
	Wrapper for correct_parallax.
	method is currently deprecatred
	verbose is currently deprecated
	"""
	obstime = parse(str(obsdatetime))
	obsloc = EarthLocation.of_site(location_name)
	pointing = SkyCoord(str(radec[0]), str(radec[1]), unit='deg')
	results, fit = correct_parallax(coord=pointing, 
									obstime=Time(obstime), 
									point_on_earth=obsloc, 
									heliocentric_distance=distance_au + dist_offset_au,
								)
	return results, fit

def process_testy_row(args):
	"""
	Updating to be generic for butler and fakes 6/3/2024 COC
	"""
	df, i, guess_distance_au = args
	if 'center_coord_ra' in df.columns:
		what = 'butler'
		ra_col = 'center_coord_ra'
		dec_col = 'center_coord_dec'
	else:
		what = 'fakes'
		ra_col = 'RA'
		dec_col = 'DEC'
	radec = [df[ra_col].iloc()[i], df[dec_col].iloc()[i]] # df['center_coord'].iloc[i]
	obsdatetime = df['ut_datetime'].iloc[i]
	return testy(radec=radec, obsdatetime=obsdatetime, location_name='CTIO', distance_au=guess_distance_au)

def add_reflex_correction(df, guess_distance_au, overwrite=False, verbose=True):
	guess_distance_au = float(guess_distance_au)
	if verbose: print(f'Starting add_reflex_correction(df, guess_distance_au={guess_distance_au}, overwrite={overwrite})...')
	startTime = time.time()
	colname = f"center_coord_{guess_distance_au}"
	done_colname = f'{colname}_ra'
	fitcolname = f"fit_{float(guess_distance_au)}"
	changed = False
	if (fitcolname in df.columns or done_colname in df.columns) and not overwrite:
		print(f'{colname} already exists in the dataframe, and overwrite={overwrite} so nothing to do.')
		return df, changed
	
	args = [(df, i, guess_distance_au) for i in range(len(df))]
	results_tmp, fittings = zip(*process_map(process_testy_row, args, max_workers=cpu_count(), chunksize=100)) # 10 to 100 as it is so much faster now thanks to Drew 6/2/2024 COC
	
	df[colname] = results_tmp
	df[fitcolname] = fittings
	df = df.copy() # fragmentation warning handling 6/1/2024 COC
	changed = True
	elapsed = round( (time.time() - startTime)/60, 2) # minutes
	if verbose: print(f'Finished reflex-correction for {guess_distance_au} au in {elapsed} minutes; there were {len(df)} rows.')
	return df, changed

# This was mostly temporary code but may still be of use if we separate this task out 6/1/2024 COC
def merge_in_pointings_by_uri(pointing_coords_csv, main_csv):	
	# Load the dataframes from CSV files
	pointing_coords_df = pd.read_csv('with_pointing_coords.csv.gz')
	dataframe_A0_differenceExp_df = pd.read_csv(main_csv, compression='gzip')
	
	# Select the required columns from pointing_coords_df
	pointing_coords_subset = pointing_coords_df[['uri', 'pointing_ra', 'pointing_dec']]
	
	# Merge the dataframes on the 'uri' column
	merged_df = pd.merge(dataframe_A0_differenceExp_df, pointing_coords_subset, on='uri', how='left')
	
	if type(merged_df['pointing_ra']) != type(0.):
		merged_df['pointing_ra'] = [np.degrees(float(i.split(' ')[0])) for i in merged_df['pointing_ra']]
		merged_df['pointing_dec'] = [np.degrees(float(i.split(' ')[0])) for i in merged_df['pointing_dec']]
		
		# Save the merged dataframe to a new CSV file
		merged_df.to_csv('merged_dataframe.csv.gz', index=False, compression='gzip')
		
		print("The dataframes have been merged and saved to 'merged_dataframe.csv.gz'.")
		
		return merged_df


#def testy(radec, obsdatetime, location_name, distance_au, method=None, verbose=False, dist_offset_au=0):
#	# Assuming definitions of `parse`, `correct_parallax` exist elsewhere in the code
#	obstime = parse(str(obsdatetime))
#	obsloc = EarthLocation.of_site(location_name)
#	pointing = SkyCoord(str(radec[0]), str(radec[1]), unit='deg')
#	results, fit = correct_parallax(coord=pointing, obstime=Time(obstime), point_on_earth=obsloc, heliocentric_distance=distance_au + dist_offset_au)
##	results, fit = correct_parallax(coord=pointing, obstime=obstime, point_on_earth=obsloc, guess_distance=distance_au + dist_offset_au, method=method, verbose=verbose)
#	return results, fit
#
#def process_testy_row(args):
#	i, guess_distance_au = args
#	radec = df['center_coord'].iloc[i]
#	obsdatetime = df['ut_datetime'].iloc[i]
#	return testy(radec=radec, obsdatetime=obsdatetime, location_name='CTIO', distance_au=guess_distance_au)
#
#def add_reflex_correction(df, guess_distance_au, overwrite=False):
#	guess_distance_au = float(guess_distance_au) # 5/31/2024 COC
#	print(f'Starting add_reflex_correction(df, guess_distance_au={guess_distance_au}, overwrite={overwrite})...')
#	colname = f"center_coord_{guess_distance_au}"
#	fitcolname = f"fit_{float(guess_distance_au)}"
#	changed = False
#	if fitcolname in df.columns and not overwrite: # to fitcolname from colname 6/1/2024 COC (we will start dropping the SkyCoord cols soon)
#		print(f'{colname} already exists in the dataframe, and overwrite={overwrite} so nothing to do.')
#		return df, changed
#
#	args = [(i, guess_distance_au) for i in range(len(df))]
#	results_tmp, fittings = zip(*process_map(process_testy_row, args, max_workers=cpu_count(), chunksize=10))
#	
#	df[colname] = results_tmp
#	df[fitcolname] = fittings
#	changed = True
#	return df, changed
	