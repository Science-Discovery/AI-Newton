import os
import sys
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(work_dir)
from aiphy.experiment.core import *
