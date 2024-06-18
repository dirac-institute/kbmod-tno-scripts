# 6/18/2024 COC
import pandas as pd
import os
import glob
import numpy as np

COCOMMON = None
try:
	COCOMMON = os.environ["COCOMMON"]
except KeyError as msg:
	pass


def fits_and_dist_from_urifile(urifile, butler_df):
	uris = []
	with open(urifile, 'r') as f:
		for line in f:
			line = line.strip()
			if line == '': continue
			if 'dist_au' in line: # example line: #dist_au=60.0
				dist_au = float(line.split('=')[-1])
			if line.startswith('#'): continue
			uris.append(line)
	uris_by_visit = {}
	for uri in uris:
		visit = int(uri.split('/VR/')[-1].split('/')[1])
		if visit not in uris_by_visit: uris_by_visit[visit] = []
		uris_by_visit[visit].append(uri)
	#
	#
	uri_to_dist = df.set_index("uri")[f"fit_{dist_au}"].to_dict()
	uri_to_ut = df.set_index("uri")[f"ut_datetime"].to_dict()
	all_dists = []
	all_visits = []
	all_uts = []
	for visit in uris_by_visit:
		all_visits.append(visit)
		dists = []
		for uri in uris_by_visit[visit]:
			dists.append(uri_to_dist[uri])
		fit_mean = np.mean(dists)
		fit_std = np.std(dists)
		print(f'visit {visit} had fit {fit_mean} ± {fit_std} au.')
		all_dists.append(fit_mean)
		all_uts.append(uri_to_ut[uri])
	newdf = pd.DataFrame.from_dict({'visit':all_visits, 'fit_mean_au':all_dists, 'ut_datetime':all_uts})
	outfile = urifile.replace('uris_', 'dists_').replace('.lst','.csv')
	newdf.to_csv(outfile, index=False)
	print(f'Wrote {outfile} to disk.')
	return newdf
	


if __name__ == '__main__':
	import argparse # This is to enable command line arguments.
	argparser = argparse.ArgumentParser(description='Strip out image data, leaving headers and WCS behind. By Colin Orion Chandler (COC) (7/16/2024)')
#	argparser.add_argument('objects', metavar='O', type=str, nargs='+', help='starting row')
	argparser.add_argument('files', help='URI file(s)', type=str, nargs='+') # 8/5/2021 COC: changing to not require this keyword explictly
	args = argparser.parse_args()
	#
	butler_csvfile='region_search_df_A0_differenceExp.csv.gz'
	if len(glob.glob(butler_csvfile)) == 0 and COCOMMON != None:
		butler_csvfile = os.path.join(COCOMMON, butler_csvfile)
	print(f'butler_csvfile is {butler_csvfile}')
	df = pd.read_csv(butler_csvfile)
	
	for fn in args.files:
#		dist_au, visits = visits_and_dist_from_urifile(urifile=fn)
#		get_geodists(butler_csvfile='region_search_df_A0_differenceExp.csv.gz', dist_au=dist_au, visits=visits)
		fits_and_dist_from_urifile(urifile=fn, butler_df=df)