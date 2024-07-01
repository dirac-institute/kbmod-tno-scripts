import os
import parsl
from parsl import bash_app, python_app, File

from klone_config import klone_config


@python_app
def uri_to_ic(input=[], ouput=[]):
    pass


with parsl.load(klone_config()):
    uri_list = File(os.path.join(os.getcwd(), 'uri_list.txt'))
    uri_to_ic_future = uri_to_ic(input=[uri_list], output=[File(os.path.join(os.getcwd(), 'ic.txt'))])


parsl.clear()