"""Lightning module that fuses TAMER and ICAL ideas."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn

try:
    import lightning as L
except ImportError:  # pragma: no cover
    import pytorch_lightning as L  # type: ignore

from datamodule import vocab
from TAMER.tamer.model.tamer import TAMER
from TAMER.tamer.utils.utils import (
    ExpRateRecorder,
    Hypothesis,
    ce_loss,
    to_bi_tgt_out,
    to_struct_output,
)

from .cam import CharacterAidedModule


class LitTAMERICAL(L.LightningModule):
    """Hybrid Lightning model with auxiliary character prediction."""

    def __init__(
        self,
        d_model: int = 256,
        growth_rate: int = 24,
        num_layers: int = 16,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        dc: int = 32,
        cross_coverage: bool = True,
        self_coverage: bool = True,
        beam_size: int = 10,
        max_len: int = 200,
        alpha: float = 1.0,
        early_stopping: bool = False,
        temperature: float = 1.0,
        learning_rate: float = 8e-2,
        patience: int = 10,
        milestones: List[int] | None = None,
        cam_lambda: float = 0.2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        milestones = milestones or [40, 55]

        self.tamer = TAMER(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            vocab_size=len(vocab),
        )
        self.cam_head = CharacterAidedModule(d_model, len(vocab))
        self.cam_lambda = cam_lambda
        self.learning_rate = learning_rate
        self.patience = patience
        self.milestones = milestones
        self.beam_size = beam_size
        self.max_len = max_len
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.temperature = temperature
        self.exprate_recorder = ExpRateRecorder()

    def forward(self, img: torch.Tensor, img_mask: torch.Tensor, tgt: torch.Tensor):
        feature, mask = self.tamer.encoder(img, img_mask)

        #추가된 파트
        feat_4d = feature.permute(0, 3, 1, 2)
        char_logits = self.cam_head(feat_4d)
        
        feature = torch.cat((feature, feature), dim=0)
        mask = torch.cat((mask, mask), dim=0)
        out, sim = self.tamer.decoder(feature, mask, tgt)
        return out, sim, char_logits

    def training_step(self, batch: Dict[str, torch.Tensor], *_) -> torch.Tensor:
        imgs = batch["image"].to(self.device)
        img_mask = batch["mask"].to(self.device)
        indices = [seq.tolist() for seq in batch["latex_ids"]]
        char_target = batch["char_target"].to(self.device)

        tgt, out = to_bi_tgt_out(indices, self.device)
        struct_out, illegal = to_struct_output(indices, self.device)

        out_hat, sim, char_logits = self.forward(imgs, img_mask, tgt)

        seq_loss = ce_loss(out_hat, out)
        struct_loss = ce_loss(sim, struct_out, ignore_idx=-1)
        char_loss = nn.BCEWithLogitsLoss()(char_logits, char_target)

        loss = seq_loss + struct_loss + self.cam_lambda * char_loss

        self.log_dict(
            {
                "train/seq_loss": seq_loss,
                "train/struct_loss": struct_loss,
                "train/char_loss": char_loss,
                "train/loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(indices),
        )
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], *_) -> None:
        imgs = batch["image"].to(self.device)
        img_mask = batch["mask"].to(self.device)
        indices = [seq.tolist() for seq in batch["latex_ids"]]
        tgt, out = to_bi_tgt_out(indices, self.device)

        out_hat, _, _ = self.forward(imgs, img_mask, tgt)
        val_loss = ce_loss(out_hat, out)
        self.log("val/loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(
            self.parameters(), lr=self.learning_rate, eps=1e-6, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def approximate_joint_search(self, img: torch.Tensor, mask: torch.Tensor) -> List[Hypothesis]:
        return self.tamer.beam_search(
            img,
            mask,
            beam_size=self.beam_size,
            max_len=self.max_len,
            alpha=self.alpha,
            early_stopping=self.early_stopping,
            temperature=self.temperature,
        )
