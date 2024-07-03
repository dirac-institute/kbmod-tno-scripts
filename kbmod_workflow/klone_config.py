from parsl import Config
from parsl.executors import WorkQueueExecutor, ThreadPoolExecutor
from parsl.providers import SlurmProvider

def klone_config():
    return Config(
        executors=[
            WorkQueueExecutor(
                label="small_cpu",
                use_cache=True,
                max_retries=1,
                provider=SlurmProvider(
                    partition="compute-bigmem",
                    account="astro",
                ),

            ),
            ThreadPoolExecutor(
                label="local_dev_testing",
            )
        ]
    )