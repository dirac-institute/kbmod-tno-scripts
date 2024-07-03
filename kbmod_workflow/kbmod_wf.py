import os
import parsl
from datetime import datetime
from parsl import python_app, File
import parsl.executors

from klone_config import klone_config


@python_app(executors=['local_dev_testing', 'small_cpu'])
def uri_to_ic(inputs=[], outputs=[], logger=None):
    logger.info('Starting uri_to_ic')
    return 42


with parsl.load(klone_config()):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging_file_path = os.path.join(os.getcwd(), f'kbmod_wf_{timestamp}.log')
    parsl_logger = parsl.set_file_logger(filename=logging_file_path)
    uri_list = File(os.path.join(os.getcwd(), 'uri_list.txt'))

    uri_to_ic_future = uri_to_ic(
        inputs=[uri_list],
        outputs=[File(os.path.join(os.getcwd(), 'ic.ecsv'))],
        logger=parsl_logger
    )

    print(uri_to_ic_future.result())


parsl.clear()