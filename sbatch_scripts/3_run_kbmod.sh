#!/bin/sh
#SBATCH --job-name="KBMOD"
#SBATCH --output="KBMOD-%A_%a.out"
#SBATCH --mem=300G  # per WSB 6/13/2024
#SBATCH -c1       #x cores - should be 8 for reprojecting step (more will require much more RAM which we do not have)
#SBATCH --gpus=1
#SBATCH --array=0-1%25
#SBATCH --account=escience
#SBATCH --partition=ckpt-g2
#SBATCH --signal=SIGALRM
#SBATCH --time=479 # one minute under the GPU limit
#SBATCH --export=ALL

# Below is to deal with astropy race conditions; hopefully we will not need 6/10/2024 COC
#export XDG_CONFIG_HOME=/gscratch/dirac/`whoami`/.astropy/config/$SLURM_ARRAY_TASK_ID/astropy/
#export XDG_CACHE_HOME=/gscratch/dirac/`whoami`/.astropy/cache/$SLURM_ARRAY_TASK_ID/astropy/
#mkdir -p $XDG_CONFIG_HOME
#mkdir -p $XDG_CACHE_HOME

echo "hostname is $(hostname)"

echo ""
nvidia-smi
echo ""

scriptdir="/gscratch/dirac/coc123/kbmod-tno-scripts"

# We want bindir set in case the correct environment was accidentally not loaded prior to sbatch.
bindir="/mmfs1/gscratch/dirac/coc123/conda_envs/kbmod_coc/bin"

MULT=1

rundir=$(pwd)
echo "$(date) -- rundir is $rundir"

# e.g., "exhaustive_search_config_fast_vel.yaml"
config_start="exhaustive_search_config_"
config_end="_vel.yaml"

result_dir="output_""$SLURM_JOB_ID"_"$SLURM_ARRAY_TASK_ID"
mkdir -p "$result_dir"

# TODO make this prettier
s="slow"
if [ $SLURM_ARRAY_TASK_ID -gt 0 ];then
	s="fast"
fi
configfile="$config_start""$s""$config_end"
echo "$(date) -- -configfile was $configfile"

wu_file=$(find . -maxdepth 2 -name "reprojected_wu.fits" | head -n1)
srun $bindir/python "$scriptdir/kbmod_run_wu.py" --wu_input_file "$wu_file" --result_dir="$result_dir" --search_config="$scriptdir/$configfile"

echo "$(date) -- Finished KBMOD phase with configfile: $configfile"
