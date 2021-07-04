import sys
import socket
import os
import glob
import os
import logging
from importlib import reload
import shutil as sh

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

from intan2kwik.core.intan.util.read_header import read_header
from intan2kwik.core.intan import load_intan as li
from intan2kwik.core import intan

from intan2kwik import kwd as ikwd
logger.info('All modules loaded')

rhd_folder = '/Users/bioel/Desktop/intan2kwik-master/raw_rhs'
simp_path = 'demo/which_path.docx'
abs_path = os.path.abspath(simp_path)

print(abs_path)

kwd_file_path = os.path.join(rhd_folder, 'experiment.raw.kwd')
kwe_file_path = os.path.join(rhd_folder, 'experiment.raw.kwe')

ihdr = ikwd.intan_to_kwd(rhd_folder, kwd_file_path, rec=0, include_channels=['amplifier', 'board_adc'], board='rhs')