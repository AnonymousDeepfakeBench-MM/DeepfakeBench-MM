import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat

from losses.abstract_loss import AbstractLoss
from utils.registry import LOSSFUNC

@LOSSFUNC.register_module(module_name="frade_center_cluster")
class FRADECenterClusterLoss(AbstractLoss):
    def __init__(self, input_dim=768):
        super().__init__()
        self.center = nn.Parameter(torch.randn(input_dim, device='cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, fusion_feature, labels, gamma2=0.25):
        """
        Computes the cross-entropy loss.

        Args:
            mds_score: A PyTorch tensor of size (batch_size, num_classes) containing MDS scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the cross-entropy loss.
        """
        real_feats = fusion_feature[labels == 0]
        if real_feats.shape[0] == 0:
            real_loss =  torch.tensor(0.)
        else:
            real_loss = F.mse_loss(real_feats, repeat(self.center, 'd -> b d', b=real_feats.shape[0]))

        fake_feats = fusion_feature[labels == 1]
        if fake_feats.shape[0] == 0:
            fake_loss = torch.tensor(0.)
        else:
            fake_loss = F.mse_loss(fake_feats, repeat(self.center, 'd -> b d', b=fake_feats.shape[0]))

        return real_loss + fake_loss


@LOSSFUNC.register_module(module_name="frade_center_cluster_fakeavceleb_weighted")
class FRADECenterClusterLossWeightedFakeAVCeleb(AbstractLoss):
    def __init__(self, input_dim=768):
        super().__init__()
        self.center = nn.Parameter(torch.randn(input_dim, device='cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, fusion_feature, labels, gamma2=0.25):
        """
        Computes the cross-entropy loss.

        Args:
            mds_score: A PyTorch tensor of size (batch_size, num_classes) containing MDS scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the cross-entropy loss.
        """
        real_feats = fusion_feature[labels == 0]
        if real_feats.shape[0] == 0:
            real_loss =  torch.tensor(0.)
        else:
            real_loss = F.mse_loss(real_feats, repeat(self.center, 'd -> b d', b=real_feats.shape[0]))

        fake_feats = fusion_feature[labels == 1]
        if fake_feats.shape[0] == 0:
            fake_loss = torch.tensor(0.)
        else:
            fake_loss = F.mse_loss(fake_feats, repeat(self.center, 'd -> b d', b=fake_feats.shape[0]))

        return real_loss * 12595 / 302 + fake_loss


@LOSSFUNC.register_module(module_name="frade_center_cluster_megammdf_weighted")
class FRADECenterClusterLossWeightedMegaMMDF(AbstractLoss):
    def __init__(self, input_dim=768):
        super().__init__()
        self.center = nn.Parameter(torch.randn(input_dim, device='cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, fusion_feature, labels, gamma2=0.25):
        """
        Computes the cross-entropy loss.

        Args:
            mds_score: A PyTorch tensor of size (batch_size, num_classes) containing MDS scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the cross-entropy loss.
        """
        real_feats = fusion_feature[labels == 0]
        if real_feats.shape[0] == 0:
            real_loss =  torch.tensor(0.)
        else:
            real_loss = F.mse_loss(real_feats, repeat(self.center, 'd -> b d', b=real_feats.shape[0]))

        fake_feats = fusion_feature[labels == 1]
        if fake_feats.shape[0] == 0:
            fake_loss = torch.tensor(0.)
        else:
            fake_loss = F.mse_loss(fake_feats, repeat(self.center, 'd -> b d', b=fake_feats.shape[0]))

        return real_loss * 613048 / 59636 + fake_loss


@LOSSFUNC.register_module(module_name="frade_contrastive_loss")
class FRADEContrastiveLoss(AbstractLoss):
    def __init__(self):
        super().__init__()

    def forward(self, video_feature, audio_feature, labels, gamma1=0.2):
        """
        Computes the cross-entropy loss.

        Args:
            mds_score: A PyTorch tensor of size (batch_size, num_classes) containing MDS scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the cross-entropy loss.
        """
        cos_similarities = F.cosine_similarity(video_feature, audio_feature)
        loss = labels.float() * cos_similarities + (1 - labels.float()) * torch.clamp((1 - cos_similarities), min=gamma1)

        return loss.mean()


@LOSSFUNC.register_module(module_name="frade_contrastive_loss_fakeavceleb_weighted")
class FRADEContrastiveLossWeightedFakeAVCeleb(AbstractLoss):
    def __init__(self):
        super().__init__()

    def forward(self, video_feature, audio_feature, labels, gamma1=0.2):
        """
        Computes the cross-entropy loss.

        Args:
            mds_score: A PyTorch tensor of size (batch_size, num_classes) containing MDS scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the cross-entropy loss.
        """
        cos_similarities = F.cosine_similarity(video_feature, audio_feature)
        loss = labels.float() * cos_similarities + \
               (1 - labels.float()) * torch.clamp((1 - cos_similarities), min=gamma1) * 12595 / 302

        return loss.mean()


@LOSSFUNC.register_module(module_name="frade_contrastive_loss_megammdf_weighted")
class FRADEContrastiveLossWeightedMegaMMDF(AbstractLoss):
    def __init__(self):
        super().__init__()

    def forward(self, video_feature, audio_feature, labels, gamma1=0.2):
        """
        Computes the cross-entropy loss.

        Args:
            mds_score: A PyTorch tensor of size (batch_size, num_classes) containing MDS scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the cross-entropy loss.
        """
        cos_similarities = F.cosine_similarity(video_feature, audio_feature)
        loss = labels.float() * cos_similarities + \
               (1 - labels.float()) * torch.clamp((1 - cos_similarities), min=gamma1) * 613048 / 59636

        return loss.mean()