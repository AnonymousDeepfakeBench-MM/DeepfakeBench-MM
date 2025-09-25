import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.registry import DETECTOR

from detectors.mds_detector import MDSDetector
from detectors.frade_detector import FRADEDetector
from detectors.avh_sup_detector import AVHSupDetector