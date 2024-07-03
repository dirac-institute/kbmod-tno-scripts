from parsl import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider

walltimes = {
    "compute-bigmem": "01:00:00", # change this to be appropriate
}

def klone_config():
    return Config(
        executors=[
            HighThroughputExecutor(
                label="small_cpu",
                provider=SlurmProvider(
                    partition="compute-bigmem",
                    account="astro",
                    min_blocks=0,
                    max_blocks=4,
                    init_blocks=1,
                    parallelism=1,
                    nodes_per_block=1,
                    cores_per_node=1, # perhaps should be 8???
                    mem_per_node=64, # In GB
                    exclusive=False,
                    walltime=walltimes["compute-bigmem"],
                ),
            ),
            ThreadPoolExecutor(
                label="local_dev_testing",
            )
        ]
    )