import progressbar
import pandas as pd
import os
import glob
import sys
if __name__ == '__main__':
    sys.path.append(f'{os.environ["HOME"]}/bin')
    sys.path.append(f'.')
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u
from astropy.time import Time
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count
from dateutil.parser import parse
from dateutil import parser
from astropy.coordinates import SkyCoord
import numpy as np
import time
if __name__ == '__main__': lastTime = time.time()

from region_search_functions import get_local_observing_date
from region_search_functions import correct_parallax # (coord, obstime, point_on_earth, heliocentric_distance)
from region_search_functions import testy # (radec, obsdatetime, location_name, distance_au, method=None, verbose=False, dist_offset_au=0)
from region_search_functions import purge_bad_uts # (df)
import gzip

def check_csvfile_integrity(csvfile):
    """
    Astroquery Skybot problems, some rows with incorrect number of fields.
    6/15/2024 COC
    """
#   comma_counts = []
    good_comma_count = None
    with open(csvfile, 'r') as f:
        for line in f:
            if 'Number' in line: # header_row
                good_comma_count = line.count(',')
                break
    print(f'good_comma_count = {good_comma_count}')
    goodlines = []
    badlines = []
    with open(csvfile, 'r') as f:
        for line in f:
            if 'Number' in line: # header_row
                goodlines.append(line)
                badlines.append(line)
                continue
            if line.count(',') != good_comma_count:
                badlines.append(line)
            else:
                goodlines.append(line)
    #
    print(f'Saw {len(goodlines)} good lines and {len(badlines)} bad lines.')
    good_name = csvfile.replace('.csv', '_clean.csv')
    bad_name = csvfile.replace('.csv', '_badrows.csv')
    with open(good_name, 'w') as f:
        for line in goodlines:
            print(line, file=f)
        print(f'Wrote good lines to {good_name}.')
    if len(badlines) > 0:
        with open(bad_name, 'w') as f:
            for line in badlines:
                print(line, file=f)
        print(f'Wrote badlines to {bad_name}.')
#       for line in f:
#           line = line.split(',')
#           comma_counts.append(len(line))
#   unique_counts = list(set(comma_counts))
#   r = {}
#   maxcount = 0
#   corresponding_comma_count = 0
#   for comma_count in unique_counts:
#       the_count = comma_counts.count(comma_count)
#       if the_count > maxcount:
#           maxcount = the_count
#           corresponding_comma_count = comma_count
#       print(f'There were {the_count} {comma_count} comma rows.')
#   
    newcsvfile = csvfile
    if len(badlines) > 0:
        newcsvfile = good_name
    return newcsvfile
            
    
if __name__ == '__main__':
    print(f'See cpu_count={cpu_count()}.')

def mjd_to_ut(mjd):
    return Time(mjd, format='mjd').to_datetime()


def get_file_size_in_kb(file_path):
    file_size = os.path.getsize(file_path)  # Get the size in bytes
    file_size_kb = file_size / 1024  # Convert bytes to kilobytes
    return file_size_kb

def find_csvfile(csvfile, verbose=True):
    """
    Utility to 
        (1) find the gz version of a requested csvfile, or 
        (2) the non-gz if that does not exist, and to
        (3) remove csvfiles that are basically empty.
    """
    basename = csvfile.split('.gz')[0]
    gzipname = f'{basename}.gz'
    final_fn = None
    for fn in [gzipname, basename]:
        if len(glob.glob(fn)) > 0:
            if get_file_size_in_kb(file_path=fn) < 2: # smaller than 2 kilobytes
                print(f'Overly small file (<2 kb) so deleting {fn} now.')
                os.remove(fn)
            else:
                if verbose: print(f'Found {fn} so returning that...')
                return(fn)
    return final_fn



#def get_local_observing_date(ut_datetime, utc_offset=-4):
#   """
#   5/26/2024 COC: consider if this should be string or datetime.
#   """
##	print(f'see ut_datetime is: {ut_datetime} (type: {type(ut_datetime)})')
#   if type(ut_datetime == type('')):
#       ut_datetime = parser.parse(ut_datetime)
#   # Convert the UTC offset to a timedelta
#   offset = timedelta(hours=utc_offset)
#   
#   # Convert the UT datetime to local time
#   local_datetime = ut_datetime + offset
#   
#   # Check if the local time is after midnight but before noon
#   if 0 <= local_datetime.hour < 12:
#       dt0 = local_datetime.date() - timedelta(days=1)
#   else:
#       dt0 = local_datetime.date()
#       
#   return dt0

## Example usage
#ut_datetime = datetime(2024, 5, 25, 3, 0, 0, tzinfo=timezone.utc)  # Example UT datetime
#utc_offset = -4  # UTC offset for La Serena, Chile
#dt0 = get_local_observing_date(ut_datetime, utc_offset)
#print("Local observing date at the start of the night:", dt0)


def purge_small_distances(df, max_distance_au=1): # 5 au to 0.01 with Drew's new correct_parallax; nope; to 1 au (ok; 5/31/2024 COC)
    """
    This is not an ideal solution but for now we will purge fakes that have distances under 3.1 au.
    to 4 au as we are still getting failures for some situations (TODO investigate someday) 5/28/2024 COC
    5/27/2024 COC
    """
    changed = False
    start_len = len(df)
    dist_col = 'r'
    if 'heliodist' in df.columns:
        dist_col = 'heliodist'
    df = df[df[dist_col] >= max_distance_au]
    if len(df) != start_len:
        changed = True
        print(f'NOTE: df went from {start_len} to {len(df)} after purging rows with r above {max_distance_au} au.')
    return df, changed


def safe_export_csv(df, fn, verbose=True):
    """
    Keeping the old CSV files as we go, and safely write out in case of kill on a cluster.
    """
    if verbose: print(f'Starting safe_export_csv(df, fn={fn})...')
    if not fn.endswith('.gz'): 
        if verbose:
            print(f'Adding missing .gz to csv name {fn} now.')
        fn += '.gz'
    time_int = f'{int(time.time())}'
    old_fn = f'{fn}_{time_int}_old'
    tmp_fn = f'{fn}_{time_int}'
    df.to_csv(tmp_fn, compression='gzip', index=False)
    if len(glob.glob(tmp_fn)) > 0: # success
        if len(glob.glob(fn)) > 0:
            os.renames(fn, old_fn)
        os.renames(tmp_fn, fn)
    if verbose: print(f'Saved {fn}, leaving {old_fn} behind as backup.')
    

def import_csv(csvfile='fakes_detections.csv', desired_dates=None):
    """
    Note: the whole _simple concept has been abandoned. TODO remove _simple from naming structure?
    TODO deal with gzip or not gzip of skybot csvfiles (currently manually gunzipping for cleaning phase)
    5/13/2024 COC (approximate)
    """
    print(f'Running import_csv(csvfile={csvfile}, desired_dates={desired_dates})...')
    lastTime = time.time()
    #
    # adding Skybot special case of bad rows 6/15/2024 COC
    if 'skybot' in csvfile and 'skybot_clean' not in csvfile: # KLUDGE
        csvfile = check_csvfile_integrity(csvfile=csvfile)
    #
#   csvfile = 'fakes_detections.csv'
    #csvfile = 'fakes_detections_first100.csv'
    csvfile_simple = csvfile.replace('.csv', '_simple.csv')
    original_simple_csvfile = csvfile_simple
    if desired_dates != None and desired_dates != []:
        datestr = '_'.join(desired_dates).replace('-','')
        print(f'datestr is {datestr}')
        csvfile_simple = csvfile_simple.replace('_simple.csv', f'_{datestr}_simple.csv')
    
    # 5/13/2024 COC: columns were not dropping anyway, so just keeping all for now
    
    if find_csvfile(csvfile_simple) == None:  # need to make simplified CSV
        if find_csvfile(original_simple_csvfile) != None:
            print(f'NOTE: Recycling full simple csv {original_simple_csvfile} (this can take a while)...')
            df = pd.read_csv(find_csvfile(original_simple_csvfile), low_memory=False)
        else:
            print(f'No simple_csv around, generating...')
            df = pd.read_csv(find_csvfile(csvfile), low_memory=False)
#       print(df.columns)
#       keeping_columns = ['RA', 'DEC', 'EXPNUM', 'CCDNUM', 'ORBITID', 'mjd_mid', 'MAG']
#       for colname in df.columns:
#           if colname not in keeping_columns:
#               df = df.drop(colname, axis=1)
        # disabling next two lines as we haven't done anything with anything up to this point 5/26/2024 COC
#       if not csvfile_simple.endswith('.gz'): csvfile_simple += '.gz'
#       df.to_csv(csvfile_simple, index=False, compression='gzip')
    else:
        print(f'NOTE: recycling {csvfile_simple}')
        df = pd.read_csv(find_csvfile(csvfile_simple))
    
    print(f'There are {len(df)} rows in the DataFrame.')
    if len(df) == 0:
        raise AttributeError(f'ERROR: empty df encountered!')
    elapsed = round(time.time() - lastTime, 1)
    lastTime = time.time()
    print(f'{elapsed} seconds elapsed (startup + reading dataframe). Next: making SkyCoords for fakes...')
    
    if 'ut_datetime' in df.columns and 'ut' not in df.columns: # for skybot 6/15/2024 COC
        df = df.rename(columns={'ut_datetime':'ut'})
    
    if 'ut' not in df.columns:
        # TIMING NOTE: ut took 133 s for 4726326 rows, COCMBP, 4 workers, chunksize=100; 5/9/2024 COC
        print(f'Adding missing ut to DataFrame...')
        lastTime = time.time()
        
        # Apply conversion in parallel
        df['ut'] = process_map(mjd_to_ut, df['mjd_mid'], max_workers=cpu_count(), desc="Converting MJD to UT", chunksize=100)  # Adjust max_workers as needed
        elapsed = round(time.time() - lastTime, 1)
        lastTime = time.time()
        print(f'{elapsed} seconds elapsed for ut. Next: ut_date...')
        # TIMING NOTE: ut_date took < 10s for 4726326 rows, COCMBP; 5/9/2024 COC

#   6/15/2024 COC: disabling this bit as we no longer use
#   if 'ut_date' not in df.columns: # splitting this out 6/15/2024 COC
#       df['ut_date'] = [dt.date() for dt in df['ut']]
#       elapsed = round(time.time() - lastTime, 1)
#       lastTime = time.time()
#       print(f'{elapsed} seconds elapsed for ut_date. Writing CSV next...')
#       safe_export_csv(df=df, fn=csvfile_simple)
#       elapsed = round(time.time() - lastTime, 1)
#       print(f'Writing {csvfile_simple} took {elapsed} seconds.')
    
    # bad UT check added 5/10/2024 COC
    bad_uts = df[df['ut'].isna()]
    if len(bad_uts) > 0:
        print(f'Found {len(bad_uts)} bad UTs. Removing...')
        df = purge_bad_uts(df=df)
        safe_export_csv(df=df, fn=csvfile_simple)
    
    # adding local_obsnight, a local date for the start of the night 5/25/2024 COC
    if 'local_obsnight' not in df.columns:
#       print(f'See df["ut"] is {df["ut"]}')
        df['local_obsnight'] = [get_local_observing_date(ut_datetime=UT) for UT in df['ut']]#, utc_offset)]
        safe_export_csv(df=df, fn=csvfile_simple)
    
    if desired_dates != None and desired_dates != []:
        print(f'Limiting dates to ***local dates***: {desired_dates}...')
        available_dates = list(set(df['local_obsnight']))
        available_dates.sort()
        print(f'Saw available_dates in df: {available_dates}')
        before_len = len(df)
#       df = df[df['ut_date'].isin(desired_dates)]
        if type(df['local_obsnight'].iloc()[0]) == type(''):
            df = df[df['local_obsnight'].isin(desired_dates)]
        else:
            df = df[df['local_obsnight'].isin([parser.parse(dt).date() for dt in desired_dates])]
        if len(df) != before_len:
            print(f'Writing updated CSV...')
            safe_export_csv(df=df, fn=csvfile_simple)
    
    df, changed = purge_small_distances(df=df) # , max_distance_au)
    if changed:
        print(f'Writing new CSV as there were small distance rows purged.')
        safe_export_csv(df=df, fn=csvfile_simple)
        print(f'Finished writing CSV files to disk post-purge_small_distances.')
    return df, csvfile_simple


# Define a function to create a SkyCoord object
def create_skycoord(row):
    return SkyCoord(ra=row['RA']*u.deg, dec=row['DEC']*u.deg, frame='icrs')

    
#def correct_parallax(coord, obstime, point_on_earth, heliocentric_distance):
#   """Calculate the parallax corrected postions for a given object at a given time and distance from Earth.
#
#   Attributes
#   ----------
#   coord : `astropy.coordinate.SkyCoord`
#       The coordinate to be corrected for.
#   obstime : `astropy.time.Time` or `string`
#       The observation time.
#   point_on_earth : `astropy.coordinate.EarthLocation`
#       The location on Earth of the observation.
#   heliocentric_distance : `float`
#       The guess distance to the object from the Sun.
#
#   Returns
#   ----------
#   An `astropy.coordinate.SkyCoord` containing the ra and dec of the point in ICRS, and the best fit geocentric distance (float).
#
#   References
#   ----------
#   .. [1] `Jupyter Notebook <https://github.com/DinoBektesevic/region_search_example/blob/main/02_accounting_parallax.ipynb>`_
#   """
#   loc = (
#       point_on_earth.x.to(u.m).value,
#       point_on_earth.y.to(u.m).value,
#       point_on_earth.z.to(u.m).value,
#   ) * u.m
#   
#   # line of sight from earth to the object,
#   # the object has an unknown distance from earth
#   los_earth_obj = coord.transform_to(GCRS(obstime=obstime, obsgeoloc=loc))
#   
#   cost = lambda geocentric_distance: np.abs(
#       heliocentric_distance
#       - GCRS(
#           ra=los_earth_obj.ra,
#           dec=los_earth_obj.dec,
#           distance=geocentric_distance * u.AU,
#           obstime=obstime,
#           obsgeoloc=loc,
#       )
#       .transform_to(ICRS())
#       .distance.to(u.AU)
#       .value
#   )
#   
#   try:
#       fit = minimize(
#       cost,
#       (heliocentric_distance,),
#       bounds = [(max(0., heliocentric_distance-1.02), heliocentric_distance+1.02)],
#   )
#   except ValueError as msg:
#       raise ValueError(f'ValueError! coord={coord}, obstime={obstime}, heliocentric_distance={heliocentric_distance}')
#   
#   answer = SkyCoord(
#       ra=los_earth_obj.ra,
#       dec=los_earth_obj.dec,
#       distance=fit.x[0] * u.AU,
#       obstime=obstime,
#       obsgeoloc=loc,
#       frame="gcrs",
#   ).transform_to(ICRS())
#   
#   return answer, fit.x[0]


#def testy(radec, obsdatetime, location_name, distance_au, method=None, verbose=False, dist_offset_au=0):
#   # Assuming definitions of `parse`, `correct_parallax` exist elsewhere in the code
#   obstime = parse(str(obsdatetime))
#   obsloc = EarthLocation.of_site(location_name)
#   pointing = SkyCoord(str(radec[0]), str(radec[1]), unit='deg')
#   results, fit = correct_parallax(coord=pointing, obstime=Time(obstime), point_on_earth=obsloc, heliocentric_distance=distance_au + dist_offset_au)
#   return results, fit


#def assemble_rad_de_ut(df, dist_au_list):
#   """
#   Cache the hard bit of utrade, leaving the dist_au and i for later
#   5/31/2024 COC
#   """
#   results_dict = {}
#   with progressbar.ProgressBar(max_value=len(df.index)) as bar:
#       c = 0
#       c_lim = 10000 # to 10000 5/27/2024 COC
#       indices = list(range(0,len(df)))
#       for dist_au in dist_au_list:
#           if float(dist_au) == float(-2):
##                   dist_au = df['r'].iloc()[i]
#               distances = df['r'].values()
#           else:
#               distances = [dist_au]*len(df)
#           args = list(zip(indices, distances, df['RA'].values(), df['DEC'].values(), df['ut'].values()))
#           results_dict[dist_au] = args
#           bar.update(i)
#   return results_dict


def process_reflex_row(args):
    i, guess_distance_au, RA, DEC, obsdatetime = args
#   radec = df['center_coord'].iloc[i]
#   radec = [ df['RA'].iloc[i], df['DEC'].iloc[i] ]
    radec = [RA, DEC]
#   obsdatetime = df['ut_datetime'].iloc[i]
#   obsdatetime = df['ut'].iloc[i]
    return testy(radec=radec, obsdatetime=obsdatetime, location_name='CTIO', distance_au=guess_distance_au)

def add_reflex_correction(df, guess_distance_au, overwrite=False, verbose=True):
    """
    If a distance of 0 is entered, then use the known distance. 5/21/2024 COC
    Docs added 5/21/2024 COC
    """
    guess_distance_au = float(guess_distance_au) # 6/1/2024 COC
    #
    if 'center_coord_ra' in df or 'center_coord' in df:
        what = 'butler'
        ra_col = 'center_coord_ra'
        dec_col = 'center_coord_dec'
    elif 'posunc' in df.columns:
        what = 'skybot'
        ra_col = 'RA'
        dec_col = 'DEC'
        dist_col = 'heliodist'
    else:
        what = 'fakes'
        ra_col = 'RA'
        dec_col = 'DEC'
        dist_col = 'r'
    #
    use_real_distance = False
    if guess_distance_au == -2:#0.0:
        if what == 'butler':
            raise ValueError(f'Cannot do real distance for butler DF.')
        use_real_distance = True
        print(f'True r range: {min(df[dist_col])} to {max(df[dist_col])} au.')
    #
    print(f'Starting add_reflex_correction(df (len(df)={len(df)}, guess_distance_au={guess_distance_au}, overwrite={overwrite}, use_real_distance={use_real_distance})...')
    if len(df) == 0:
        raise AttributeError(f'ERROR: df was empty!')
    #
    if what in ['fakes', 'skybot']:
        colnames = [f"RA_{guess_distance_au}", f"Dec_{guess_distance_au}", f"fit_{guess_distance_au}"]
    elif what == 'butler':
        colnames = [f"center_coord_{guess_distance_au}_ra", f"center_coord_{guess_distance_au}_dec", f"fit_{guess_distance_au}"]
    if use_real_distance == True:
        colnames = ["RA_known_r", "Dec_known_r", "fit_known_r"]
    for colname in colnames:
        if colname in df.columns and not overwrite:
            print(f'{colname} already exists in the dataframe, and overwrite={overwrite} so nothing to do.')
            return df, False
    #
    if verbose: print(f'Assembling {len(df.index)} args for multiprocessing...')
#   args = []
#   with progressbar.ProgressBar(max_value=len(df.index)) as bar:
#       c = 0
#       c_lim = 10000 # to 10000 5/27/2024 COC
#       for i in range(len(df)):
#           if use_real_distance == True:
#               dist_au = df['r'].iloc()[i]
#           else:
#               dist_au = guess_distance_au
#           args.append([i, dist_au, df['RA'].iloc()[i], df['DEC'].iloc()[i], df['ut'].iloc[i]])
#           c += 1
#           if c >= c_lim:
#               c = 0
#               bar.update(i)
    # faster approach 5/31/2024 COC
    indices = list(range(0,len(df)))
    if float(guess_distance_au) == float(-2):
        distances = df[dist_col]
        try:
            distances = distances.values()
        except TypeError as msg:
            pass
    else:
        distances = [dist_au]*len(df)
    try:
        args = list(zip(indices, distances, df[ra_col].values(), df[dec_col].values(), df['ut'].values()))
    except TypeError as msg:
        print(f'TypeError bypass (message was {msg}).')
        args = list(zip(indices, distances, df[ra_col], df[dec_col], df['ut']))
    #
    startTime = time.time()
    if verbose: print(f'Starting actual reflex-correction work...')
    results_tmp = process_map(process_reflex_row, args, max_workers=cpu_count(), chunksize=10)
    elapsed_minutes = round( (time.time() - startTime) / 60 )
    if verbose: print(f'{elapsed_minutes} minutes elapsed for {guess_distance_au} au reflex-correction.')
    RAs = []
    Decs = []
    fits = []
    for i in results_tmp:
        RAs.append(i[0].ra.deg)
        Decs.append(i[0].dec.deg)
        fits.append(i[1])
    df[colnames[0]] = RAs
    df[colnames[1]] = Decs
    df[colnames[2]] = fits
    return df, True



if __name__ == '__main__':
    import argparse # This is to enable command line arguments.
    argparser = argparse.ArgumentParser(description='Process KBMOD fakes catalog, adding reflex-correction. By Colin Orion Chandler (COC), 2024-05.')
    default_distances = [-2,20,30,40,42,50,60,70,80,90,100]
    default_distances += list(range(30,50+1))
    default_distances += [-2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 150.0, 200.0]
    default_distances = list(set(default_distances))
    default_distances.sort()
    argparser.add_argument('--distances', nargs='+', type=float, dest='distances', default = default_distances, help="distances to calculate parameters for. Default is {default_distances}. Use -2 for real (stored) r values.")
    argparser.add_argument('--csvfile', dest='csvfile', type=str, default='fakes_detections.csv', help='CSV file to work on.')
    argparser.add_argument('--dates', nargs='+', dest='dates', type=str, default=[], help=f'dates of format YYYY-MM-DD (default is everything)')
#   argparser.add_argument('--timedelta', dest='time_delta', type=int, default=30, help=f'How many seconds after the UT to calculate. Used to make ephemeris more current (e.g., it takes time to query JPL, or you are planning for some time in the near future).')
    argparser.add_argument('--verbose', dest='verbose', type=bool, default=False, help=f'say "--verbose True" to see more messages.')
    args = argparser.parse_args()
    desired_distances_au = args.distances
    desired_dates = args.dates
    desired_dates.sort() # important! 5/20/2024 COC
#   csvfile = 'fakes_detections.csv'
    csvfile = args.csvfile
#   csvfile = 'fakes_detections_first100.csv'
#   csvfile_simple = csvfile.replace('.csv', '_simple.csv')
#   df, newcsvfile = import_csv(csvfile=csvfile, desired_dates = ['2019-04-04', '2019-05-05'])
    print(f'About to do import_csv(csvfile={csvfile}, desired_dates = {args.dates}). desired_distances_au = {desired_distances_au}')
    df, newcsvfile = import_csv(csvfile=csvfile, desired_dates = args.dates)
    print(f'df.columns = {df.columns}')
    if len(df.index) == 0:
        raise AttributeError(f'Empty DataFrame df for csvfile={csvfile}, desired_dates={desired_dates}')
    #
    reflexes_done_without_saving = 0 # 6/1/2024 COC compromising b/w csv writeout times and risk of job running out of time/being killed
    max_reflexes_without_saving = 3
    for dist_au in desired_distances_au:
        if float(dist_au) == float(-1):
            print(f'NOTE: skipping dist_au=-1 (uncorrected values) as this does not apply to reflex-correction.') # safety check, just in case 5/26/2024 COC
            continue
#       colnames = []
#       for k in [dist_au, float(dist_au)]: # in case it is stored with or without the float version (we changed this at some point) 5/27/2024 COC
#           colnames.append(f"RA_{k}")
#           colnames.append(f"Dec_{k}")
#           colnames.append(f"fit_{k}")
#       colnames = list(set(colnames))
#       if float(dist_au) == float(-2):
#           colnames = ["RA_known_r", "Dec_known_r", "fit_known_r"]
        df, changed = add_reflex_correction(df=df, guess_distance_au=dist_au, overwrite=False)
        if changed:
            reflexes_done_without_saving += 1
        if reflexes_done_without_saving >= max_reflexes_without_saving:
            reflexes_done_without_saving = 0
            safe_export_csv(df=df, fn=newcsvfile)
    if reflexes_done_without_saving > 0:
        safe_export_csv(df=df, fn=newcsvfile)
    exit()
    
    
if __name__ == '__main__':
#   df = import_csv(csvfile='fakes_detections.csv')
    df, newcsvfile = import_csv(csvfile='fakes_detections_first100.csv', desired_dates = ['2019-04-03', '2019-05-04'])
    print(f'df columns: {df.columns}')
    lastTime = time.time()
    desired_dates = ['2019-04-03', '2019-05-04'] # to local obsnights (minus one for each date here)
#   print(f'We see the following UT dates: {list(set(df["ut_date"]))}')
    print(f'We see the following local obsnights: {list(set(df["local_obsnight"]))}') # 6/15/2024 COC dropping ut_date
    if desired_dates != []:
        df = df[df['local_obsnight'].isin(desired_dates)]
    if len(df.index) == 0:
        print(f'Empty DataFrame. Exiting...')
        exit()
    
    print(f'After desired_dates filter, {len(df)} rows remain.')
    elapsed = round(time.time() - lastTime, 1)
    lastTime = time.time()
    print(f'{elapsed} seconds elapsed for date filtering. Next: making SkyCoords for fakes...')
    
    
    # Apply the function in parallel using process_map with a proper chunksize
    # TIMING NOTE: about 30 seconds for the 178014 surviving rows, 4 workers, COCMBP, chunksize=1000; 5/9/2024 COC
    df['skycoord'] = process_map(create_skycoord, [row for _, row in df.iterrows()], chunksize=1000, max_workers=cpu_count(), desc="Generating SkyCoords",)
    
    elapsed = round(time.time() - lastTime, 1)
    lastTime = time.time()
    print(f'{elapsed} seconds elapsed for making SkyCoords. Next: separation calculation for our desired (RA, Dec)...')
    
    max_sep_arcmin = 4.5
    
    rade = (215.77499999999998, -12.52500000000001)
    search_coord = SkyCoord(ra=rade[0]*u.deg, dec=rade[1]*u.deg, frame='icrs')
    
    print(df['skycoord'].iloc[0])
#   print(list(df['skycoord']))
    
    # Calculate separation
    fake_coords = SkyCoord(ra=[ra*u.deg for ra in df['RA']], dec=[dec*u.deg for dec in df['DEC']], frame='icrs')
    
    # TIMING NOTE: 52s on COC MBP for the 
    df['separation'] = search_coord.separation(fake_coords).arcmin
    
    elapsed = round(time.time() - lastTime, 1)
    lastTime = time.time()
    print(f'{elapsed} seconds elapsed for making separation calculation. Next: limit results to those with max separation...')
    print(type(df['separation'].iloc[0]))
    print(df['separation'].iloc[0])
    
    df2 = df[df['separation'] < max_sep_arcmin]
    
    print(f'There were {len(df2)} matches:')
    print(df2)
    print()
    all_orbitids = list(df2["ORBITID"])
    unique_orbitids = list(set(all_orbitids))
    orbit_df = pd.DataFrame.from_dict({'ORBITID':unique_orbitids})
    print(f'{len(unique_orbitids)}: Unique ORBITIDs: {unique_orbitids}')
    print()
    print(f"There were {list(df2['ORBITID'].duplicated()).count(True)} duplicated items.")
    
    counts = {}
    for dt in desired_dates:
        if dt not in counts: counts[dt] = []
        dtdf = df2[df2['local_obsnight'] == dt]
        for orbitid in unique_orbitids:
            one_orbitid_df = df2[df2['ORBITID']==orbitid]
            
#           print(f'{dt} "dtdf": {len(dtdf)}')
#           orbit_df['{dt}_count'] = [all_orbitids.count(orbit_id) for orbit_id in unique_orbitids]
    
    sums = []
    for i in range(0, len(orbit_df)):
        n = 0
        for dt in desired_dates:
            n += orbit_df[f'{dt}_count'].iloc()[i]
        sums.append(n)
    orbit_df['sum'] = sums
    
    orbit_df = orbit_df[orbit_df['sum']>0]
    orbit_df = orbit_df.sort_values(f'{desired_dates[0]}_count', ascending=False)
    print(f'There were {len(orbit_df)} fakes spanning both dates.')
    print(orbit_df)
    