import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backbones/av_hubert/avhubert"))
import hubert_pretraining, hubert, hubert_asr
from fairseq import checkpoint_utils


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.abstract_detector import AbstractDetector
from detectors import DETECTOR
from losses import LOSSFUNC


@DETECTOR.register_module(module_name='AVH-Sup')
class AVHSupDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if not self.config["ddp"]:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['WORLD_SIZE'] = '1'
            os.environ['RANK'] = '0'

        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.mlp = AVH_Sup_MLP()
    def build_backbone(self, config):
        models, _, task = checkpoint_utils.load_model_ensemble_and_task([config["pretrained"]])
        backbone = models[0].encoder.w2v_model.cuda().eval()
        return backbone

    def build_loss(self, config):
        ce_loss = LOSSFUNC[config['loss_func'][0]]
        return {'CE': ce_loss()}

    @torch.no_grad()
    def features(self, data_dict: dict) -> torch.tensor:
        # convert from RGB to gray, cut mouth region
        device = data_dict["video"].device
        to_gray_weight = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1, 1)
        video_input = torch.sum(data_dict["video"] * to_gray_weight, dim=1, keepdim=True)  # [B, 1, T, H, W]
        video_input = video_input[:, :, :, 52:140, 52:140]    # in AVH-Align, it uses 96x96 mouth crop and 88x88 center crop
        video_input = (video_input - 0.421) / 0.165           # normalize of gray scale frames, mean and std are from self_large_vox_433h.pt

        # rearrange Log-Mel, 4 slice as 1 audio frame
        b, t, f = data_dict["audio"].shape
        if t % 4 != 0:
            pad = 4 - t % 4
            data_dict["audio"] = torch.cat([data_dict["audio"], torch.zeros((b, pad, f), device=device)], dim=1)
        audio_input = data_dict["audio"].reshape(b, -1, 4 * f)
        with torch.no_grad():
            audio_input = F.layer_norm(audio_input, audio_input.shape[-1:])
        audio_input = audio_input.transpose(1, 2)
        # print(video_input.shape, audio_input.shape)
        video_feature, _ = self.backbone.extract_finetune({"video": video_input, "audio": None}, None, None)
        # print(video_feature.shape)
        audio_feature, _ = self.backbone.extract_finetune({"video": None, "audio": audio_input}, None, None)
        # print(audio_feature.shape)

        return {'video': video_feature, 'audio': audio_feature}

    def classifier(self, features: dict) -> torch.tensor:
        return self.mlp(features)


    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:     # Todo
        return {'overall': self.loss_func['CE'](pred_dict['cls'], data_dict['label'])}


    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        pred = torch.cat((-pred.unsqueeze(1), pred.unsqueeze(1)), dim=1)

        # we use this in most cases!!!
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {
            'cls': pred,
            'prob': prob,
            # 'feat': features
            }
        return pred_dict


"""
The following code is from https://github.com/bit-ml/AVH-Align/blob/main/avh_sup/mlp.py
"""
class AVH_Sup_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 1024
        self.visual_proj = nn.Linear(1024, hidden_dim // 2)
        self.audio_proj = nn.Linear(1024, hidden_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_feats):
        video_feats, audio_feats = input_feats["video"], input_feats["audio"]
        visual_proj = self.visual_proj(video_feats)     # [B, T, 512]
        audio_proj = self.audio_proj(audio_feats)       # [B, T, 512]
        fused_features = torch.cat((visual_proj, audio_proj), dim=-1)   # [B, T, 1024]
        output = self.mlp(fused_features)[:, :, 0]      # [B, T]
        return torch.logsumexp(output, dim=-1)          # [B]