# butler-specific functionality, split off 6/1/2024 COC
import glob
import os
import sys
#from os.path import dirname
if __name__ == '__main__':
	sys.path.append(f'{os.environ["HOME"]}/bin')
import time

from datetime import datetime, timedelta, timezone
from dateutil import parser

import lsst
import lsst.daf.butler as dafButler
import lsst.sphgeom as sphgeom

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map
import tqdm
from functools import partial

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

def get_collection_names(butler, basedir='.', verbose=False, export=True, target=None):
	"""
	Making this a function 2/6/2024 COC.
	adding target option 5/1/2024 COC
	"""
	print(f'Starting get_collection_names(butler, basedir={basedir}, verbose={verbose}, export={export})')
	all_collection_names = []
	
	if target == None:
		q = sorted(butler.registry.queryCollections("*"))
	else:
		q = sorted(butler.registry.queryCollections(f"*"))
	
	for c in q:
		all_collection_names.append(c)
		
	if export == True:
		outfile = f"{basedir}/all_collection_names.lst"
		with open(outfile, "w") as f:
			for c in all_collection_names:
				print(c, file=f)
				
	if verbose:
		message = f"Found {len(all_collection_names)} collections in the Butler."
		if export == True:
			message += f' Wrote to "{outfile}".'
		print(message)
	return all_collection_names

def getDatasetTypeStats(butler, overwrite=False, label=None, basedir='.'):
	"""
	Get information on all datasetTypes found in a Butler.
	TODO implement caching if desired. If not, get rid of overwrite option.
	Added label option for handling different parameter selection and their caching 4/23/2024 COC.
	2/1/2024 COC
	"""
	datasetTypes = {}
	
	cache_file = f"{basedir}/dataset_types.csv"
	if label != None:
		cache_file = cache_file.replace('.csv', f'_{label}.csv')
	cache_exists = False
	if len(glob.glob(cache_file)) > 0:
		cache_exists = True
		
	if overwrite == False and cache_exists == True:
		print(f"Recycling {cache_file} as overwrite was False...")
		with open(cache_file, "r") as f:
			for line in f:
				print(line)
				line = line.strip().split(",")
				datasetTypes[line[0]] = int(line[1])
		print(f"Read {len(datasetTypes)} datasetTypes from disk.")
		return datasetTypes
	
	q = sorted(butler.registry.queryDatasetTypes())
	
	print(f'Fetching datasetTypes...')
	with progressbar.ProgressBar(max_value=len(q)) as bar:
		for j, dt in enumerate(q):
			n = 0
			for i, ref in enumerate(
				butler.registry.queryDatasets(datasetType=dt, collections=desired_collections)
			):
				n += 1
			if n > 0:
				if dt.name not in datasetTypes:
					datasetTypes[dt.name] = 0
				datasetTypes[dt.name] += n
			bar.update(j)
			
	if cache_exists == False or overwrite == True:
		print(f"Saving {len(datasetTypes)} datasetTypes to {cache_file} now...")
		with open(cache_file, "w") as f:
			for key in datasetTypes:
				print(f"{key},{datasetTypes[key]}", file=f)
	else:
		print(f"Saw {len(datasetTypes)} datasetTypes.")
	return datasetTypes

# see a pattern, like this:
# DEEP/20190402/A0a
# DEEP/20190402/A0a/raw
# DEEP/20190402/A0a/science#step0/20240307T214206Z
# DEEP/20190402/A0a/science#step0/20240418T000511Z
# DEEP/20190402/A0a/science#step0/20240418T002734Z
# DEEP/20190402/A0a/science#step1/20240307T214547Z
# DEEP/20190402/A0a/science#step1/20240309T104915Z
# DEEP/20190402/A0a/science#step1/20240312T234713Z
# DEEP/20190402/A0a/science#step2/20240404T191753Z
# DEEP/20190402/A0a/science#step2/20240404T222951Z
# DEEP/20190402/A0a/science#step3/20240404T224002Z
# DEEP/20190402/A0a/science#step4/20240404T224213Z
# DEEP/20190402/A0a/science#step5/20240404T225201Z
# DEEP/20190402/A0a/science#step6/20240404T235221Z

# We looked through the collections already.
# We will string manipulate to get to what we need.
# Previously, we used a list file on disk. (This could be a better option for some users.)



def clean_dataId(dataId, verbose=False):
	"""
	We need a way to extract a portable (non-LSST object) version of the DataId.
	A dataId normally looks like a dictionary with keys of instrument, detector, visit.
	4/22/2024 COC
	"""
	fields = ['instrument', 'detector', 'visit']
	clean_id = {}
	#
	if type(dataId) == type({}) and list(dataId.keys()) == fields:
		if verbose:
			print(f'dataId was already a properly formatted dict: {dataId}.')
		return dataId
	#
	try:
		dataId = dataId.values_tuple()
	except AttributeError as msg:
		if verbose:
			print(f'dataId did not have a values_tuple() attribute. Message was {msg}.')
		pass
	#
	if type(dataId) == type((1,1,1)):
		for i, label in enumerate(fields):
			clean_id[label] = dataId[i]
		if verbose:
			print(f'Derived {clean_id} from tuple {dataId}.')
		return clean_id
	#
	for i, label in enumerate(fields):
		try:
			clean_id[label] = dataId[label]
		except KeyError as msg:
			print(f'KeyError for {label} with dataId= {dataId}. Message was: {msg}.')
			break
	#
	if clean_id == {}:
		raise ValueError(f'Failed to create clean_id for dataId={dataId}.')
	#
	return clean_id


def get_vdr_data(butler, desired_collections, desired_datasetTypes, verbose=False):
	"""

	Made as function 2/6/2024 COC.
	"""
	# VDR === Visit Detector Region
	# VDRs hold what we need in terms of region hashes and unique dataIds.
	# NOTE: this typically takes < 5s to run 2/1/2024 COC
	# NOTE: tried iterating over desired_collections vs supplying desired_collections; same output 2/1/2024 COC
	
	vdr_dict = {"data_id": [], "region": [], "detector": []}
	#     vdr_ids = []
	#     vdr_regions = []
	#     vdr_detectors = []
	
	for dt in desired_datasetTypes:
		datasetRefs = butler.registry.queryDimensionRecords(
			"visit_detector_region", datasets=dt, collections=desired_collections
		)
		for ref in datasetRefs:
			vdr_dict["data_id"].append(clean_dataId(ref.dataId, verbose=verbose))
			vdr_dict["region"].append(
				ref.region
			)  # keeping as objects for now; should .encode() for caching/export
			vdr_dict["detector"].append(ref.detector)  # 2/2/2024 COC
			# BUT if we decided to export this or cache this, we should write the encode() version to disk
			#
			example_vdr_ref = ref  # this leaves a VDR Python object we can play with
			# other data available:
			#    id = ref.id# id -- e.g., 1592350 (for DEEP dataset, I think UUIDs for newer Butlers)
			#    visit = ref.dataId.full['visit'] # e.g., 946725
			# vdr_filters.append(ref.dataId.full['band']) # e.g., VR
			# vdr_detectors.append(ref.dataId.full['detector']) # e.g., 1
	df = pd.DataFrame.from_dict(vdr_dict)
	return df, example_vdr_ref

def getInstruments(butler, vdr_ids, desired_collections, datasetType='calexp', first_instrument_only=True):
	"""Iterate through our records to determine which instrument(s) are involved.
	Return a list of the identified instruments.
	If first_instrument_only is True, stop as soon as we found an instrument.
	"""
	# KLUDGE: snag the instrument name of the first record we find in a visitInfo query.
	print(f'Starting getInstruments(butler, vdr_ids, datasetType={datasetType}, first_instrument_only={first_instrument_only})...') # 4/20/2024 COC
	instrument_names = []
	for i, dataId in enumerate(vdr_ids):
		visitInfo = butler.get(f"{datasetType}.visitInfo", dataId=dataId, collections=desired_collections)
		instrument_name = visitInfo.instrumentLabel
		if instrument_name not in instrument_names:
			print(f'Found {instrument_name}. Adding to "desired_instruments" now.')
			instrument_names.append(instrument_name)
		if first_instrument_only == True and len(instrument_names) > 0:
			print(
				f"WARNING: we are not iterating over all rows to find instruments, just taking the first one."
			)
			break
	return instrument_names

# The single-thead approach (below) requires some 2 hours to execute. So instead we will multiprocess.
# paths = [butler.getURI(desired_datasetTypes[0], dataId=dataId, collections=desired_collections) for dataId in vdr_ids]

###########
# 6/1/2024 COC: we do not use these two I believe
def chunked_dataIds(dataIds, chunk_size=200):
	"""Yield successive chunk_size chunks from dataIds."""
	for i in range(0, len(dataIds), chunk_size):
		yield dataIds[i : i + chunk_size]		
		
def get_uris(dataIds_chunk, repo_path, desired_datasetTypes, desired_collections):
	"""Fetch URIs for a list of dataIds."""
	chunk_uris = []
	butler = dafButler.Butler(repo_path)
	for dataId in dataIds_chunk:
		try:
			uri = butler.getURI(desired_datasetTypes[0], dataId=dataId, collections=desired_collections)
			uri = uri.geturl()  # Convert to URL string
			chunk_uris.append(uri)
		except Exception as e:
			print(f"Failed to retrieve path for dataId {dataId}: {e}")
	return chunk_uris

############
# 6/1/2024 COC: we use this one
def getURIs(butler, dataIds, repo_path, desired_datasetTypes, desired_collections, overwrite=False, label=None, basedir='.'):
	"""
	Get URIs from a Butler for a set of dataIDs.
	Cache results to disk for future runs.
	TODO: consider exporting as CSV so we can validate URIs against dataIds. 2/6/2024 COC
	TODO: consider the safety of this cache system. 5/26/2024 COC
	Updated 2/5/2024 COC
	"""
	print(f"Starting getURIs(butler, dataIds, repo_path={repo_path}, desired_datasetTypes, desired_collections, overwrite={overwrite}, label={label})...")
	
	paths = []
	
	cache_file = f"{basedir}/uri_cache.lst"
	if label != None: cache_file = cache_file.replace(".lst", f"_{label}.lst")
	cached_exists = False
	if len(glob.glob(cache_file)) > 0:
		cached_exists = True
		
	if cached_exists == True and overwrite == False:
		with open(cache_file, "r") as f:
			for line in f:
				paths.append(line.strip())
		print(f"Recycled {len(paths)} paths from {cache_file} as overwrite was {overwrite}.")
		return paths
	
	# Prepare dataId chunks
	dataId_chunks = list(chunked_dataIds(dataIds))
	
	# Execute get_uris in parallel and preserve order
	with ProcessPoolExecutor() as executor:
		# Initialize progress bar
		with progressbar.ProgressBar(max_value=len(dataId_chunks)) as bar:
			# Use map to execute get_uris on each chunk and maintain order
			result_chunks = list(
				executor.map(
					get_uris,
					dataId_chunks,
					[repo_path] * len(dataId_chunks),
					[desired_datasetTypes] * len(dataId_chunks),
					[desired_collections] * len(dataId_chunks),
				)
			)
			
			for i, chunk_uris in enumerate(result_chunks):
				paths.extend(chunk_uris)  # Add the retrieved URIs to the main list
				bar.update(i)
				
	with open(cache_file, "w") as f:
		for path in paths:
			print(path, file=f)
		print(f"Wrote {len(paths)} paths to disk for caching purposes.")
		
	return paths


def get_timestamps(repo_path, desired_collections, dataset_type, dataIds_chunk):
	"""Get timestamps for a chunk of dataIds"""
	chunked_data = []
	butler = dafButler.Butler(repo_path)
	for dataId in dataIds_chunk:
		try:
			visitInfo = butler.get(f"{dataset_type}.visitInfo", dataId=dataId, collections=desired_collections)
			t = Time(str(visitInfo.date).split('"')[1], format="isot", scale="tai")
			tutc = str(t.utc)
			chunked_data.append(tutc)
		except Exception as e:
			print(f"Failed to retrieve timestamp for dataId {dataId}: {e}")
	return chunked_data

def getTimestamps(dataIds, repo_path, desired_collections, dataset_type, overwrite=False, label=None, basedir='.'):
	"""
	Get timestamps for the given dataIds
	"""
	timestamps = []
	cache_file = f"{basedir}/vdr_timestamps.lst"
	if label is not None:
		cache_file = cache_file.replace('.lst', f'_{label}.lst')
		
	if not overwrite and glob.glob(cache_file):
		with open(cache_file, "r") as f:
			timestamps = [line.strip() for line in f]
		print(f"Recycled {len(timestamps)} from {cache_file}.")
		return timestamps
	
	def chunked_dataIds(dataIds, chunk_size=200):
		for i in range(0, len(dataIds), chunk_size):
			yield dataIds[i:i + chunk_size]
			
	dataId_chunks = list(chunked_dataIds(dataIds))
	
	# Create a partial function with repo_path, desired_collections, and dataset_type
	partial_get_timestamps = partial(get_timestamps, repo_path, desired_collections, dataset_type)
	
	# Use tqdm's process_map for parallel processing with progress bar
	result_chunks = process_map(partial_get_timestamps, dataId_chunks, max_workers=None, chunksize=1, desc="Processing dataId chunks")
	
	timestamps = [timestamp for chunk in result_chunks for timestamp in chunk]
	
	with open(cache_file, "w") as f:
		for ts in timestamps:
			print(ts, file=f)
	print(f"Wrote {len(timestamps)} lines to {cache_file} for future use.")
	
	print(f"Obtained {len(timestamps)} timestamps.")
	return timestamps


#def get_timestamps(dataIds_chunk):
#	"""Needs repo_path, desired_collections, dataset_type too somehow"""
#	chunked_data = []
#	butler = dafButler.Butler(repo_path)
#	for dataId in dataIds_chunk:
#		try:
#			visitInfo = butler.get(f"{dataset_type}.visitInfo", dataId=dataId, collections=desired_collections)
#			t = Time(str(visitInfo.date).split('"')[1], format="isot", scale="tai")
#			tutc = str(t.utc)
#			chunked_data.append(tutc)
#		except Exception as e:
#			print(f"Failed to retrieve timestamp for dataId {dataId}: {e}")
#	return chunked_data
#
#def getTimestamps(dataIds, repo_path, desired_collections, overwrite=False, label=None, basedir='.'):
#	"""
#	5/26/2024 COC note: questioning the safety of this; TODO consult WSB.
#	"""
#	timestamps = []
#	cache_file = f"{basedir}/vdr_timestamps.lst"
#	if label is not None:
#		cache_file = cache_file.replace('.lst', f'_{label}.lst')
#		
#	if not overwrite and glob.glob(cache_file):
#		with open(cache_file, "r") as f:
#			timestamps = [line.strip() for line in f]
#		print(f"Recycled {len(timestamps)} from {cache_file}.")
#		return timestamps
#	
#	def chunked_dataIds(dataIds, chunk_size=200):
#		for i in range(0, len(dataIds), chunk_size):
#			yield dataIds[i:i + chunk_size]
#			
#	dataId_chunks = list(chunked_dataIds(dataIds))
#	
#	# Use process_map instead of the executor
#	result_chunks = process_map(
#		get_timestamps,
#		dataId_chunks,
#		[repo_path] * len(dataId_chunks),
#		[desired_collections] * len(dataId_chunks),
#		max_workers=None,  # Auto-selects the number of workers, you can specify if needed
#		chunksize=1,  # How many items each worker should take at once, adjust based on need
#		desc="Processing dataId chunks"
#	)
#	
#	timestamps = [timestamp for chunk in result_chunks for timestamp in chunk]
#	
#	with open(cache_file, "w") as f:
#		for ts in timestamps:
#			print(ts, file=f)
#	print(f"Wrote {len(timestamps)} lines to {cache_file} for future use.")
#	
#	print(f"Obtained {len(timestamps)} timestamps.")
#	return timestamps


#def get_dates_with_targets(collection_list, specific_step=None):
#	dates_targets = []
#	for collection in collection_list:
#		if '#step' not in collection: continue
#		if 'DEEP/2' not in collection: continue
#		dt = collection.split('/')[1]
#		target = collection.split('/')[2]
#		dt_target = f'{dt}_{target}'
#		if dt_target not in dates_targets: dates_targets.append(dt_target)

		
# commenting out 6/1/2024 COC
#def get_dates_and_targets(collection_list):
#	"""
#	From a list of collection names, extract unique dates and targets (pointing groups, e.g., A0a).
#	4/20/2024 COC
#	"""
#	dates = []
#	targets = []
#	dates_targets = []
#	for collection in collection_list:
#		if 'science#step' not in collection: continue
#		if not collection.startswith('DEEP/20'): continue
#		if collection.endswith('flat'): continue
#		
#		dt = collection.split('/')[1]
#		target = collection.split('/')[2]
#		if dt not in dates: dates.append(dt)
#		if target not in targets: targets.append(target)
#		dt_target = f'{dt}/{target}'
#		if dt_target not in dates_targets: dates_targets.append(dt_target)
#	return {'dates':dates, 'targets':targets, 'dt_targets':dates_targets}


# commenting out 6/1/2024 COC
#def get_latest_version(collection_list, dt_target, step=None):
#	"""
#	For a collection list, return the latest collection name for a given step.
#	If no step is supplied, then he latest is returned.
#	5/1/2024 COC update: this apparently is bad according to Steven, as they are each incomplete, so we need all, not the latest
#	4/20/2024 COC
#	"""
#	desired_collection = None
#	for collection in collection_list:
#		if '#step' not in collection: continue
#		if dt_target not in collection: continue
#		if step != None and f'#step{step}' not in collection: continue
#		desired_collection = collection
#	print(f'Got latest for dt_target={dt_target} was {desired_collection}')
#	return desired_collection

	
# commenting out 6/1/2024 COC
#def get_desired_collections(all_collections_list, desired_collection_list=None, mode=None, step=None, target=None):
#	"""
#	Produce a list of collections that will be used for querying the Butler.
#
#	If desired_collection_list is None, then a hard-wired "default" approach
#	(for Haden/DEEP) is carried out, requiring:
#		1. "Pointing" must be in the collection name.
#		2. "/imdiff_r/" must be in the collection name.
#		3. "/2021" may not be in the collection name.
#
#	Otherwise, desired_collection_list can be either
#		1. a Python list of desired collection names, or
#		2. a filename (ending in .lst) that specifies the desired collections.
#	Either way, the collection names are verified against the (required) collections_list.
#
#	Made this into a function 2/6/2024 COC.
#
#	NOTE/TODO: untested are the supplied list and list file approaches.
#
#	Adding a mode option so we can do a Steven-style Butler 4/20/2024 COC.
#	"""
#	
#	desired_collections = []
#	
#	if desired_collection_list == None:
#		if mode == None:
#			for collection_name in all_collection_names:
#				if (
#					"Pointing" in collection_name
#					and "/imdiff_r" in collection_name
#					and "/2021" not in collection_name
#				):
#					desired_collections.append(collection_name)
#		elif mode == 'Steven2':
#			c = 0#
#		elif mode == 'Steven':
#			c = get_dates_and_targets(collection_list=all_collections_list)
#			for dt_target in c['dt_targets']:
#				if target != None and target not in dt_target:
#					continue
#				desired_collection = get_latest_version(collection_list=all_collections_list, dt_target=dt_target, step=step)
#				if desired_collection == None:
#					print(f'Got None for dt_target={dt_target}, step={step}.')
#					continue
#				desired_collections.append(desired_collection)
#	else:
#		if type(desired_collection_list) == type(""):
#			with open(desired_collection_list, "r") as f:
#				for line in f:
#					desired_collections.append(line.strip())
#		else:
#			for entry in desired_collection_list:
#				desired_collections.append(entry)
#	#
#	# Validate entries
#	for entry in desired_collections:
#		if entry not in all_collections_list:
#			raise KeyError(f'"{entry}" is not in the master list of collections supplied.')
#	#
#	return desired_collections

# commenting out 6/1/2024 COC
#if __name__ == '__main__':
#	# need to keep track of our parameters for the purpose of caching, etc. 4/23/2024 COC
#	#param_dict = {'step':2, 'mode':'Steven', 'target':'A0b'}
#	param_dict = {'step':2, 'mode':'Steven', 'target':None}
#	label = f"{param_dict['mode']}_step{param_dict['step']}_{param_dict['target']}"
#	
#	#desired_collections = get_desired_collections(	all_collections_list=all_collection_names, 
#	#												desired_collection_list=None, 
#	#												mode=param_dict['mode'], 
#	#												step=param_dict['step'],
#	#												target=param_dict['target'],
#	#											)
#

# commenting out 6/1/2024 COC
#def get_desired_collections_string(target):
#	"""
#	5/1/2024 COC
#	"""
#	s = f"*/{target}*"
#	return s
