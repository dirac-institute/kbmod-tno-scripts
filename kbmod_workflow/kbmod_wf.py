import os
import parsl
from datetime import datetime
from parsl import python_app, File

from klone_config import klone_config
from utilities import configure_logger


@python_app(executors=['local_dev_testing'])
def uri_to_ic(inputs=[], outputs=[]):
    logger = configure_logger(
        name='python_app:uri_to_ic',
        file_path=inputs[-1].filepath
    )

    logger.info('Starting uri_to_ic')


with parsl.load(klone_config()):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging_file = File(os.path.join(os.getcwd(), f'kbmod_wf_{timestamp}.log'))
    uri_list = File(os.path.join(os.getcwd(), 'uri_list.txt'))

    uri_to_ic_future = uri_to_ic(
        inputs=[uri_list, logging_file],
        outputs=[
            File(os.path.join(os.getcwd(), 'ic.ecsv')),
            logging_file,
        ],
    )

parsl.clear()