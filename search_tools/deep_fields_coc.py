import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from dateutil import parser
import glob
from datetime import datetime, timedelta, timezone
import os

COCOMMON = None
try:
	COCOMMON = os.environ["COCOMMON"]
except KeyError as msg:
	pass

# Function to find the closest date
def find_closest_date(dates_list, target_date):
	"""
	5/26/2024 COC
	"""
	if type(dates_list[0]) == type(''): # string
		dates_list = [parser.parse(i).date() for i in dates_list]
	if type(target_date) == type(''):
		target_date = parser.parse(target_date).date()
	closest = min(dates_list, key=lambda date: abs(date - target_date))
	days_difference = abs((closest - target_date).days)
	return closest, days_difference


def dashless_date_fixer(dt, verbose=False):
	new_dt = dt[0:4] + '-' + dt[4:6] + '-' + dt[6:]
	if verbose: print(f'dt --> {new_dt}')
	return new_dt

# if __name__ == '__main__': print(dashless_date_fixer(dt='20190406')) # test

def clean_csv_dates(csvfile='deep_field_by_date_coc_toClean.csv', verbose=False):
	"""
	Fix date column of format 20190402 to be of format 2019-04-02.
	5/26/2024 COC
	"""
	print(f'Starting clean_csv_dates(csvfile={csvfile}, verbose={verbose})...')
	if '_toClean' not in csvfile: # just in case
		raise ValueError(f'Missing "_toclean" in filename, probably wrong file!')
	newfile = csvfile.replace('_toClean','')
	#
	df = pd.read_csv(csvfile)
	cleaned = [dashless_date_fixer(str(dt)) for dt in df['local_obsdate']]
	if verbose: print(f'cleaned: {cleaned}')
	df['local_obsdate'] = cleaned
	df.to_csv(newfile, index=False)
	print(f'Wrote "{newfile}" to disk.')

fields_by_dt_csvfile = 'deep_field_by_date_coc.csv'
if glob.glob(fields_by_dt_csvfile) == 0 and COCOMMON != None:
	fields_by_dt_csvfile = os.path.join(COCOMMON, fields_by_dt_csvfile)

if __name__ == '__main__':
	if len(glob.glob(fields_by_dt_csvfile)) == 0: 
		clean_csv_dates()


def fetch_deep_field_coords(deep_patch_id, dt, verbose=True):
	"""
	New function that takes the datetime into account.
	5/28/2024 COC
	"""
	if len(deep_patch_id) != len(dt):
		raise ValueError(f'deep_patch_id and dt must be the same length arrays!')
	df = pd.read_csv(fields_by_dt_csvfile)
	print(f'columns: {df.columns}')
	all_coords = []
	for i in range(0, len(deep_patch_id)):
		this_dt = dt[i]
		this_patch_id = deep_patch_id[i]
		closest_date, days_difference = find_closest_date(dates_list=list(set(df['local_obsdate'])), target_date=this_dt)
		deep_df = df[df['local_obsdate']==str(closest_date)]
		if len(deep_df) == 0:
			raise KeyError(f'deep_df was empty after using closest_date {closest_date} (closest to {this_dt}). In df, there are {list(set(df["local_obsdate"]))}.')
		deep_df = deep_df[deep_df['DEEP_id']==this_patch_id]
		if len(deep_df) > 1:
			raise KeyError(f'More than one match for deep_patch_id={this_patch_id} on closest_date={closest_date} (dt={this_dt}) from fetch_deep_field_coords() somehow. This should not happen!')
		if len(deep_df) == 0:
			raise KeyError(f'Did not find a match at all for deep_patch_id={this_patch_id} on dt={closest_date} (closest to {this_dt}) with fetch_deep_field_coords(). Available were {list(set(df[df["local_obsdate"]==str(closest_date)]["DEEP_id"]))}')
		coords = [deep_df['RA'].iloc()[0], deep_df['Dec'].iloc()[0]]
		if verbose: print(f'Found {coords} for {this_patch_id} on {this_dt}.')
		all_coords.append(coords)
	return all_coords


#def fetch_deep_field_coords(deep_patch_id):#, local_obsdate):
#	"""
#	Give the nominal pointing coordinate given 
#		(1) a DEEP patch ID (e.g., A0c), and 
#		[actually local_obsdate yet 5/25/2024 COC] (2) a local obsdate (local date at start of observing night).
#	5/25/2024 COC
#	"""
#	df = pd.read_csv('deep_fields_clean_coc.csv')
#	patch_df = df[df['Field'] == deep_patch_id]
#	if len(patch_df) == 0:
#		raise KeyError(f'Unable to find {deep_patch_id} in df. Available: {df["Field"].values}')
#	print(f'patch_df = {patch_df}') # 5/26/2024 COC debugging
#	coords = [ patch_df['RA'].iloc()[0], patch_df['Dec'].iloc()[0] ]
#	return coords
##print(fetch_coords('A0c'))



## Given date and list of dates
#target_date_str = '2019-04-02'
#dates_list_str = ['2019-03-29', '2019-04-01', '2019-04-03', '2019-04-05']
#
## Convert strings to datetime objects
#target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
#dates_list = [datetime.strptime(date, '%Y-%m-%d') for date in dates_list_str]

## Check if target date is in the list
#if target_date in dates_list:
#	print(f"The date {target_date_str} is in the list.")
#else:
#	closest = closest_date(dates_list, target_date)
#	days_difference = abs((closest - target_date).days)
#	print(f"The date {target_date_str} is not in the list.")
#	print(f"The closest date is {closest.strftime('%Y-%m-%d')}.")
#	print(f"The number of days between {target_date_str} and {closest.strftime('%Y-%m-%d')} is {days_difference} days.")
#############


#def determine_field(rade, dt, max_separation_degrees=3, verbose=False):
#	"""
#	Figure out which DEEP field (a.k.a. "target") a given (RA, Dec) tuple belongs to.
#	Derived from Chad Trujillo's DEEP II paper Table 3; please cite accordingly.
#	Overhauled version 5/26/2024 COC: 
#		- Incorporating the local observing date into the calculation.
#		- Adding a maximum separation.
#	5/6/2024 COC
#	"""
#	df = pd.read_csv(fields_by_dt_csvfile)
#	local_obsdates = list(set(df['local_obsdate'].values))
#	
#	if dt in local_obsdates:
#		best_dt = dt
#		days_off = 0
#	else:
#		best_dt, days_off = find_closest_date(dates_list=local_obsdates, target_date=dt)
#
#	df = df[df['local_obsdate'] == str(best_dt)] # limiting to our best date; str() is critical 5/29/2024 COC
#	
#	if len(df) == 0:
#		raise KeyError(f'ERROR: for best_dt {best_dt}, closest to {dt}, days_off={days_off}, we got an empty DataFrame. Should be impossble!')
#
#	field_coords = SkyCoord(ra=df['RA']*u.degree, dec=df['Dec']*u.degree, frame='icrs')
#	
#	our_coord = SkyCoord(ra=rade[0]*u.degree, dec=rade[1]*u.degree, frame='icrs')
#	
#	seps = field_coords.separation(our_coord)
#	
#	min_sep = min(seps)
#	min_sep_idx = list(seps).index(min_sep)
##	print(f'min_sep_idx = {min_sep_idx}')
#	min_sep_df = df.iloc()[min_sep_idx]
#	results_dict = {}
#	results_dict['days_off'] = days_off
#	the_sep_degrees = min_sep.degree
#	results_dict['separation_degrees'] = the_sep_degrees
##	min_sep_df['days_off'] = days_off
##	min_sep_df['min_sep'] = the_sep_degrees
#	results_dict['nearest_DEEP_id'] = min_sep_df['DEEP_id']
#	if the_sep_degrees > max_separation_degrees:
##		min_sep_df['DEEP_id'] = 'Unknown'
#		results_dict['DEEP_id'] = 'unknown'
#	else:
#		results_dict['DEEP_id'] = min_sep_df['DEEP_id']
#	if verbose: print(f'dt={dt}, rade={rade}, days_off={days_off}, min_sep.degree={the_sep_degrees}, min_sep_df: {min_sep_df}')
#	return results_dict#['Field']

import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

def determine_field(rade, dt, df='', local_obsdates='', max_separation_degrees=3, verbose=False):
	"""
	Figure out which DEEP field (a.k.a. "target") a given (RA, Dec) tuple belongs to.
	Derived from Chad Trujillo's DEEP II paper Table 3; please cite accordingly.
	Overhauled version 5/26/2024 COC: 
		- Incorporating the local observing date into the calculation.
		- Adding a maximum separation.
	5/6/2024 COC
	"""
	
	if type(df) == type('') or type(local_obsdates) == type(''):
		print(f'WARNING: reading csvfile and determining local_obsdates (probably for every excution). Inefficient!')
		df = pd.read_csv(fields_by_dt_csvfile)
		local_obsdates = list(set(df['local_obsdate'].values))
	
	if dt in local_obsdates:
		best_dt = dt
		days_off = 0
	else:
		best_dt, days_off = find_closest_date(dates_list=local_obsdates, target_date=dt)
		
	df_best_dt = df[df['local_obsdate'] == str(best_dt)]  # limiting to our best date; str() is critical 5/29/2024 COC
	
	if len(df_best_dt) == 0:
		raise KeyError(f'ERROR: for best_dt {best_dt}, closest to {dt}, days_off={days_off}, we got an empty DataFrame. Should be impossible!')
		
	field_coords = SkyCoord(ra=[i*u.degree for i in df_best_dt['RA']], dec=[i*u.degree for i in df_best_dt['Dec']], frame='icrs')
	our_coord = SkyCoord(ra=rade[0]*u.degree, dec=rade[1]*u.degree, frame='icrs')
	
	seps = field_coords.separation(our_coord)
	min_sep = min(seps)
	min_sep_idx = seps.argmin()
	min_sep_df = df_best_dt.iloc[min_sep_idx]
	
	results_dict = {
		'days_off': days_off,
		'separation_degrees': min_sep.degree,
		'nearest_DEEP_id': min_sep_df['DEEP_id'],
		'DEEP_id': min_sep_df['DEEP_id'] if min_sep.degree <= max_separation_degrees else 'unknown'
	}
	
	if verbose:
		print(f'dt={dt}, rade={rade}, days_off={days_off}, min_sep.degree={min_sep.degree}, min_sep_df: {min_sep_df}')
		
	return results_dict

# Example usage:
# df = pd.read_csv(fields_by_dt_csvfile)
# local_obsdates = list(set(df['local_obsdate'].values))
# result = determine_field(rade, dt, df, local_obsdates)


#def determine_field(rade):
#	"""
#	Figure out which DEEP field (a.k.a. "target") a given (RA, Dec) tuple belongs to.
#	Derived from Chad Trujillo's DEEP II paper Table 3; please cite accordingly.
#	5/6/2024 COC
#	"""
#	df = pd.read_csv('deep_fields_clean_coc.csv')
#	
#	field_coords = SkyCoord(ra=df['RA']*u.degree, dec=df['Dec']*u.degree, frame='icrs')
#	
#	our_coord = SkyCoord(ra=rade[0]*u.degree, dec=rade[1]*u.degree, frame='icrs')
#	
#	seps = field_coords.separation(our_coord)
#	
#	min_sep = min(seps)
#	min_sep_idx = list(seps).index(min_sep)
##	print(f'min_sep_idx = {min_sep_idx}')
#	min_sep_df = df.iloc()[min_sep_idx]
#	print(f'min_sep_df: {min_sep_df}')
#	return min_sep_df#['Field']


if __name__ == '__main__':
# TODO fix by adding dt 5/26/2024 COC
#	A0c_patch_from_patches = [215.62499999999997, -12.375000000000004]
#	found_field_df = determine_field(rade=A0c_patch_from_patches)
#	found_field = found_field_df['Field']
#	print(f'Found {found_field} for A0c ({A0c_patch_from_patches}).')


#astroarchive_df = pd.read_csv('DEEP_instcals.csv')
#field_names = []
#for i in range(0, len(astroarchive_df.index)):
#	ra = astroarchive_df['ra'].iloc()[i]
#	dec = astroarchive_df['dec'].iloc()[i]
#	field_name = determine_field(rade=[ra, dec])
#	field_names.append(field_name)
#astroarchive_df['DEEP_Field'] = field_names
#astroarchive_df.to_csv('DEEP_instcals_with_DEEP_fields.csv')
#
#

	df = pd.read_csv('DEEP_instcals_with_DEEP_fields.csv')
	df2 = df[df['caldat']=='2019-05-06']
	print(len(df2.index))
	
	print(f'len(df)={len(df)}')
	print(df.columns)
	nancount = 0
	for i in range(0, len(df.index)):
		OBJECT = str(df['object'].iloc()[i]).strip()
		DEEP_ID = str(df['DEEP_Field'].iloc()[i]).strip()
		if str(OBJECT) in ['nan', '']: 
			nancount += 1
			continue
		if OBJECT.lower() != DEEP_ID.lower():
			print(f'{OBJECT} != {DEEP_ID} for {df["archive_filename"].iloc()[i]}')
	print(f'there were {nancount} NaNs (missing OBJECT info)')
