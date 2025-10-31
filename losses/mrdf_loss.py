import torch
from torch import nn

from losses import LOSSFUNC
from losses.abstract_loss import AbstractLoss

@LOSSFUNC.register_module(module_name="mrdf_cmr")
class MRDFLcmr(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.margin = 0.0
        self.cos_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, v_feat, a_feat, labels):
        """
        Computes the cross-modality regulartion loss.
        Args:
            v_feat: (tensor in [B, C]) video feature tensor
            a_feat: (tensor in [B, C]) audio feature tensor
            labels: (tensor in [B]) multimodal labels tensor
        Returns:
            A scalar tensor representing the cross-modality regularization loss.
        """
        distance = self.cos_similarity(v_feat, a_feat)
        loss = labels * (1 - distance) + \
               (1 - labels) * torch.clip(distance - self.margin, min=0.0)

        return loss.mean()

@LOSSFUNC.register_module(module_name="mrdf_cmr_fakeavceleb_weighted")
class MRDFLcmrFakeAVCeleb(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.margin = 0.0
        self.cos_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, v_feat, a_feat, labels):
        """
        Computes the cross-modality regulartion loss.
        Args:
            v_feat: (tensor in [B, C]) video feature tensor
            a_feat: (tensor in [B, C]) audio feature tensor
            labels: (tensor in [B]) multimodal labels tensor
        Returns:
            A scalar tensor representing the cross-modality regularization loss.
        """
        distance = self.cos_similarity(v_feat, a_feat)
        loss = labels * (1 - distance) + \
               12595 / 302 * (1 - labels) * torch.clip(distance - self.margin, min=0.0)

        return loss.mean()

@LOSSFUNC.register_module(module_name="mrdf_cmr_megammdf_weighted")
class MRDFLcmrdMegaMMDF(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.margin = 0.0
        self.cos_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, v_feat, a_feat, labels):
        """
        Computes the cross-modality regulartion loss.
        Args:
            v_feat: (tensor in [B, C]) video feature tensor
            a_feat: (tensor in [B, C]) audio feature tensor
            labels: (tensor in [B]) multimodal labels tensor
        Returns:
            A scalar tensor representing the cross-modality regularization loss.
        """
        distance = self.cos_similarity(v_feat, a_feat)
        loss = labels * (1 - distance) + \
               613048 / 59636 * (1 - labels) * torch.clip(distance - self.margin, min=0.0)

        return loss.mean()
