#!/bin/sh
#SBATCH --job-name="ReprojWU"
#SBATCH --output="RepojectWU-%A_%a.out"
#SBATCH --mem=512G  # 410 used for 189 UTs with 6 images each UT layer
#SBATCH -c8       #x cores - should be 8 for reprojecting step (more will require much more RAM which we do not have)
#SBATCH --array=1%250# START WITH ONE FOR SED!!
#SBATCH --account=escience
#SBATCH --partition=ckpt-all
#SBATCH --signal=SIGALRM
#SBATCH --time=239 # one minute under the limit
#SBATCH --export=ALL

# Below is to deal with astropy race conditions; hopefully we will not need 6/10/2024 COC
#export XDG_CONFIG_HOME=/gscratch/dirac/`whoami`/.astropy/config/$SLURM_ARRAY_TASK_ID/astropy/
#export XDG_CACHE_HOME=/gscratch/dirac/`whoami`/.astropy/cache/$SLURM_ARRAY_TASK_ID/astropy/
#mkdir -p $XDG_CONFIG_HOME
#mkdir -p $XDG_CACHE_HOME

echo "hostname is $(hostname)"

scriptdir="/gscratch/dirac/coc123/kbmod-tno-scripts"

# We want bindir set in case the correct environment was accidentally not loaded prior to sbatch.
bindir="/mmfs1/gscratch/dirac/coc123/conda_envs/kbmod_coc/bin"

MULT=1

rundir=$(pwd)
echo "$(date) -- rundir is $rundir"


check_uris() {
	# Check if uris.lst exists in the current directory
	if [ -f "uris.lst" ]; then
		# Set the environment variable to uris.lst
		export URIS_FILE="uris.lst"
		echo "Found uris.lst, setting URIS_FILE to uris.lst"
	else
		# Find the first file in the current directory
		alternate_file=$(find . -maxdepth 1 -type f -name \*uri\*.lst | head -n 1)
		if [ -n "$alternate_file" ]; then
			# Set the environment variable to the alternate file
			export URIS_FILE="$alternate_file"
			echo "uris.lst not found. Using alternate file: $alternate_file"
		else
			# Warn the user if no file is found
			echo "uris.lst not found and no alternate file found in the current directory."
		fi
	fi
}

echo "$(date) -- setting URI file variable."

# Call the function to check for URI file and set the environment variable.
check_uris

echo "$(date) -- URIS_FILE was $URIS_FILE"

# UNTESTED; TODO test
# If no Image Collection ecsv, make it now, though we are in the reproject phase, this could prevent a job failure. 6/10/2024 COC
if [  $(find . -maxdepth 1 -name "ic.ecsv" | wc -l) -lt 1 ];then
	echo "$(date) -- Could not find ic.ecsv. Generating now, but this will only work if file paths do not require augmentation."
	srun python $bindir/python "$scriptdir/create_ic.py" --target_uris_file "$URIS_FILE" --ic_output_file "ic.ecsv"
fi

result_dir="output"

# UNTESTED; TODO test
## If no original wu, complain, but proceed 6/10/2024 COC
if [  $(find $result_dir -maxdepth 1 -name "orig_wu.fits" | wc -l) -lt 1 ];then
	echo "$(date) -- WARNING we could not find orig_wu.fits which means our mode just became Original WorkUnit instead of Reproject."
fi

echo "$(date) -- Using $SLURM_CPUS_PER_TASK as n_workers" # 6/17/2024 COC

srun $bindir/python "$scriptdir/ic_to_wu.py" --ic_input_file="ic.ecsv" --result_dir="$result_dir" --search_config="$scriptdir/search_config.yaml" --uri_file "$URIS_FILE" --n_workers "$SLURM_CPUS_PER_TASK"

echo "$(date) -- Finished reproject phase."
