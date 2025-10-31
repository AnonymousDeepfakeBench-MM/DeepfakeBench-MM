import os
import sys
import torch
import torch.nn as nn

from fairseq.modules import LayerNorm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backbones/av_hubert"))
from avhubert import hubert, hubert_pretraining

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from detectors.abstract_detector import AbstractDetector
from detectors import DETECTOR
from losses import LOSSFUNC


@DETECTOR.register_module(module_name='MRDF')
class MRDFDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.loss_func = self.build_loss(config)
        self.backbone = self.build_backbone(config)

        # module after AV-HuBERT backbone
        self.embed = 768
        self.dropout = 0.1

        self.project_video = nn.Sequential(LayerNorm(self.embed),
                                           nn.Linear(self.embed, self.embed),
                                           nn.Dropout(self.dropout))
        self.project_audio = nn.Sequential(LayerNorm(self.embed),
                                           nn.Linear(self.embed, self.embed),
                                           nn.Dropout(self.dropout))

        self.video_classifier = nn.Sequential(nn.Linear(self.embed, 2))
        self.audio_classifier = nn.Sequential(nn.Linear(self.embed, 2))
        self.mm_classifier = nn.Sequential(nn.Linear(self.embed, self.embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(self.embed, 2))

    def build_backbone(self, config):
        backbone_cfg = hubert.AVHubertConfig
        backbone_cfg.audio_feat_dim = 104
        backbone_cfg.label_rate = 25 # Not used in this model. But to avoid unnecessary "str" / "int", we set an integer.
        backbone = hubert.AVHubertModel(cfg=backbone_cfg,
                                        task_cfg=hubert_pretraining.AVHubertPretrainingConfig,
                                        # dictionaries=hubert_pretraining.AVHubertPretrainingTask)
                                        dictionaries = [])
        backbone.feature_extractor_video.resnet.frontend3D[0] = \
            nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        return backbone

    def build_loss(self, config):
        ce_loss = LOSSFUNC[config['loss_func'][0]]
        cmr_loss = LOSSFUNC[config['loss_func'][1]]
        return {"CE": ce_loss(), "CMR": cmr_loss()}

    def features(self, data_dict: dict) -> torch.tensor:
        # rearrange Log-Mel, 4 slice as 1 audio frame
        device = data_dict["video"].device
        b, t, f = data_dict["audio"].shape
        if t % 4 != 0:
            pad = 4 - t % 4
            data_dict["audio"] = torch.cat([data_dict["audio"], torch.zeros((b, pad, f), device=device)], dim=1)
        data_dict["audio"] = data_dict["audio"].reshape(b, -1, 4 * f)
        data_dict["audio"] = data_dict["audio"].transpose(1, 2)

        # unimodal feature
        video_feature = self.backbone.feature_extractor_video(data_dict["video"]).transpose(1, 2)   # [B, T, 768]
        audio_feature = self.backbone.feature_extractor_audio(data_dict["audio"]).transpose(1, 2)   # [B, T, 768]

        # multimodal feature
        av_feature = torch.cat([audio_feature, video_feature], dim=2)                      # [B, T, 768*2]
        av_feature = self.backbone.layer_norm(av_feature)
        av_feature = self.backbone.post_extract_proj(av_feature)
        av_feature = self.backbone.dropout_input(av_feature)
        av_feature, _ = self.backbone.encoder(av_feature)
        return {"video": video_feature, "audio": audio_feature, "av": av_feature}

    def classifier(self, features: dict) -> torch.tensor:
        video_feature = self.project_video(features["video"]).mean(1)
        audio_feature = self.project_audio(features["audio"]).mean(1)
        video_logit = self.video_classifier(video_feature)
        audio_logit = self.audio_classifier(audio_feature)
        av_logit = self.mm_classifier(features["av"][:, 0, :])
        return {"video": video_logit, "audio": audio_logit, "av": av_logit}

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        ce_loss = self.loss_func["CE"](pred_dict["av_logits"], data_dict["label"])
        a_ce_loss = self.loss_func["CE"](pred_dict["a_logits"], data_dict["audio_label"])
        v_ce_loss = self.loss_func["CE"](pred_dict["v_logits"], data_dict["video_label"])
        cmr_loss = self.loss_func["CMR"](pred_dict["v_feature"], pred_dict["a_feature"], data_dict["label"])

        return {"overall": ce_loss + a_ce_loss + v_ce_loss + cmr_loss,
                "ce_loss": ce_loss, "cmr_loss": cmr_loss, "a_ce_loss": a_ce_loss, "v_ce_loss": v_ce_loss}

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        # we use this in most cases!!!
        # get the probability of the pred
        prob = torch.softmax(pred["av"], dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {
            "cls": pred["av"],
            "prob": prob,
            "a_logits": pred["audio"],
            "v_logits": pred["video"],
            "av_logits": pred["av"],
            "v_feature": features["video"].mean(1),
            "a_feature": features["audio"].mean(1),
        }
        return pred_dict