import matplotlib.pyplot as plt
import numpy as np
import os

import kbmod
from kbmod.analysis.plotting import *
from kbmod.search import StampCreator
from kbmod.results import Results
from kbmod.work_unit import WorkUnit

from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u
from astropy.table import Table
import astropy.time 

import argparse

"""
A script for loading a KBMOD work unit and results table and saving plots for each of the multi-night results.

Note that light curves are only plotted for results that have a "psi_curve" and "phi_curve" column in the results table, and are saved to a separate file.
"""

def mjd_to_day(mjd):
    """
    Converts an MJD to a day string in the format "YYYY-MM-DD"
    """
    # TODO Should be consistent on using local obstime or UTC obstime (different plots use different ones)
    return str(astropy.time.Time(mjd, format='mjd').to_value('datetime')).split()[0]


def generate_num_days(res, wu, verbose=False):
    """
    Generates the number of unique days observed for each result in the results table
    
    Parameters
    ----------
    res : Results
        The results object containing the results table
    wu : WorkUnit
        The work unit object containing the image stack    
    
    Returns
    -------
    num_days : list
        A list of the number of unique days observed for each result in the results table
    """
    # For each result find the number of unique days observed.
    num_days = []
    for idx in range(len(res)):
        # Whether an observation was "valid" and included in the result
        is_valid = res[idx]["obs_valid"]
        
        # Get all of the observation times that were valid and included in the result
        valid_obstimes = [] 
        for i in range(len(is_valid)):
            if is_valid[i]:
                valid_obstimes.append(wu.im_stack.get_obstime(i))
        
        # Convert the obstimes to days and generate the number of days.
        num_days.append(len(set([mjd_to_day(t) for t in valid_obstimes])))

    if verbose:
        print(f"len(res): f{len(res)}")
        print(f"len(num_days): f{len(num_days)}")
        print(num_days)
    return num_days


def generate_daily_coadds(wu, stamps, res, idx):

    """
    Generates the daily coadds for the given result index
    
    Parameters
    ----------
    wu: WorkUnit
        The work unit object containing the image stack
    stamps:
        A list of stamps for each observation in the work unit
    res : Results
        The results object containing the results table
    idx : int
        The index of the result to generate the daily coadds for

    Returns
    -------
    daily_coadds : dict
        A dictionary mapping each day to a coadded stamp of observations on that day
    valid_obstimes : list
        A list of the observation times that were included in the result
    """
    # Map each day for a result to its coadded stamp
    daily_coadds = {}
    result_row = res[idx]
    for i in range(wu.im_stack.img_count()):
        if result_row["obs_valid"][i]:
            day = mjd_to_day( wu.im_stack.get_obstime(i))
            curr_stamp = stamps[i]
            # Depending on where "stamps" were generated may be a RawImage
            if not isinstance(curr_stamp, np.ndarray):
                curr_stamp = curr_stamp.image

            if day not in daily_coadds:
                # Create the initial coadd
                daily_coadds[day] = curr_stamp.copy()
            else:
                # Add the stamps together
                daily_coadds[day] += curr_stamp
    return daily_coadds

def plot_lc_from_result_row(res, idx, figure, ax):
    """Plot a lightcurve for a single row of the results table.

    Parameters
    ----------
    row : `astropy.table.row.Row`
        The information from the results to plot.
    figure : `matplotlib.pyplot.Figure` or `None`
        Figure, `None` by default.
    """
    row = res[idx]
    if figure is None:
        figure = plt.figure(layout="constrained")

    if "psi_curve" in row.colnames and "psi_curve" in row.colnames:
        print("plotting lightcurve")
        psi = row["psi_curve"]
        phi = row["phi_curve"]
        lc = np.full(psi.shape, 0.0)

        valid = (phi != 0) & np.isfinite(psi) & np.isfinite(phi)
        if "obs_valid" in row.colnames:
            valid = valid & row["obs_valid"]

        lc[valid] = psi[valid] / phi[valid]
        plot_time_series(lc, None, indices=valid, figure=figure, ax=ax, title=f"Light curve for result {idx}")

def plot_multi_night_results(res, wu, save_path, verbose=True):
    # Generate the number of days column
    if "num_days" not in res.table.columns:
        res.table["num_days"] = generate_num_days(res, wu)
        if verbose:
            print("Generated number of days")

    # Generate the stamps for all results
    trajectories = res.make_trajectory_list()
    all_stamps = [StampCreator.get_stamps(wu.im_stack, trj, 10) for trj in trajectories]
    if "all_stamps" in res.table.columns:
        # plot_result_row is significantly slower if "all_stamps" is included in the table
        # It will try to plot each individual stamp and is both slow and unreadable
        res.table.remove_column("all_stamps")

    for idx in range(len(res)):
        # Look at multi-night results
        if res[idx]["num_days"] > 1:
            if verbose:
                print(f"Generating plot for result {idx}")
            daily_coadds = generate_daily_coadds(wu, all_stamps[idx], res, idx)
            # We want to plot a row of the images for this results the first being the full coadd stamp
            imgs = [res["stamp"][idx]]
            labels = [f'Coadd for result {idx}']
            for day, coadd in daily_coadds.items():
                imgs.append(coadd)
                labels.append(str(day))
            fig = plt.figure(layout="constrained")
            plot_multiple_images(imgs, labels=labels, figure=fig, norm=True)
            plt.savefig(os.path.join(save_path, f"result_{idx}_coadds.png"))
            lc_fig, lc_ax = plt.subplots()
            plot_lc_from_result_row(res, idx, figure=lc_fig, ax=lc_ax)
            plt.savefig(os.path.join(save_path, f"result_{idx}_lc.png"))
    
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate a plot for each of the multi-night results")
    parser.add_argument("--work_unit", help="The work unit file to generate the results load")
    parser.add_argument("--results", help="The KBMOD results directory to load")
    parser.add_argument("--save_path", help="The path to save the plots to")
    args = parser.parse_args()

    if not os.path.exists(args.work_unit):
        raise ValueError("The work unit file does not exist")
    
    if not os.path.exists(args.results):
        raise ValueError("The results directory does not exist")
    elif not os.path.isdir(args.results):
        # The results path exists but is not a directory
        raise ValueError("The results path must be a directory")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    elif not os.path.isdir(args.save_path):
        # The safe path exists but is not a directory
        raise ValueError("The save path must be a directory")

    # Load the results
    res = Results.read_table(os.path.join(args.results, "results.ecsv"))
    
    # Load the work unit
    wu = WorkUnit.from_fits(args.work_unit)

    # Plot the multi-night results
    plot_multi_night_results(res, wu, args.save_path)
