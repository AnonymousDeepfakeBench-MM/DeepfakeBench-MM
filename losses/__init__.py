import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.registry import LOSSFUNC

from losses.standard_loss import CrossEntropyLoss, CrossEntropyLossWeightedFakeAVCeleb, CrossEntropyLossWeightedMegaMMDF
from losses.mds_loss import MDSL1, MDSL1WeightedFakeAVCeleb, MDSL1WeightedMegaMMDF
from losses.frade_loss import FRADEContrastiveLoss, FRADEContrastiveLossWeightedFakeAVCeleb, \
    FRADEContrastiveLossWeightedMegaMMDF, FRADECenterClusterLoss, FRADECenterClusterLossWeightedFakeAVCeleb, \
    FRADECenterClusterLossWeightedMegaMMDF
