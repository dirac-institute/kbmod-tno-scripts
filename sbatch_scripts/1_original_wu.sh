#!/bin/sh
#SBATCH --job-name="OrigWU"
#SBATCH --output="OriginalWU-%A_%a.out"
#SBATCH --mem=512G  # 410 used for 189 UTs with 6 images each "layer"
#SBATCH -c1                                    #x cores - should be 1 for Original WU
#SBATCH --array=1%250# START WITH ONE FOR SED!!
#SBATCH --account=escience
#SBATCH --partition=gpu-a40
#SBATCH --signal=SIGALRM
#SBATCH --time=120
#SBATCH --export=ALL

# Below is to deal with astropy race conditions; hopefully we will not need 6/10/2024 COC
#export XDG_CONFIG_HOME=/gscratch/dirac/`whoami`/.astropy/config/$SLURM_ARRAY_TASK_ID/astropy/
#export XDG_CACHE_HOME=/gscratch/dirac/`whoami`/.astropy/cache/$SLURM_ARRAY_TASK_ID/astropy/
#mkdir -p $XDG_CONFIG_HOME
#mkdir -p $XDG_CACHE_HOME


scriptdir="/gscratch/dirac/coc123/kbmod-tno-scripts"

# We want bindir set in case the correct environment was accidentally not loaded prior to sbatch.
bindir="/mmfs1/gscratch/dirac/coc123/conda_envs/kbmod_coc/bin"

#rundir=/projects/asteroids/SAFARI2/bin
rundir=$(pwd)
echo "$(date) -- rundir is $rundir"

startdir=$(pwd)

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

# If no Image Collection ecsv, make it now 6/10/2024 COC
if [  $(find . -maxdepth 1 -name "ic.ecsv" | wc -l) -lt 1 ];then
	echo "$(date) -- Could not find ic.ecsv. Generating now, but this will only work if file paths do not require augmentation."
	srun python $bindir/python "$scriptdir/create_ic.py" --target_uris_file "$URIS_FILE" --ic_output_file "ic.ecsv"
fi

result_dir="output"
## If already an original wu, complain, but proceed 6/10/2024 COC
if [  $(find $result_dir -maxdepth 1 -name "orig_wu.fits" | wc -l) -gt 0 ];then
	echo "$(date) -- WARNING we found an existing orig_wu.fits but we are in Original WorkUnit mode. This job will likely never finish."
fi

srun $bindir/python "$scriptdir/ic_to_wu.py" --ic_input_file="ic.ecsv" --result_dir="output" --search_config="$scriptdir/search_config.yaml" --uri_file "$URIS_FILE"

echo "$(date) -- Finished original WorkUnit phase."
