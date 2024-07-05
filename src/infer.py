from typing import List
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import InferDataset
from .model import TimesFM
from .utils import compile_outputs
from .constants import *


class Infer:
    def __init__(
        self,
        window_size: int | None = None,
        context_len: int | None = None,
        horizon_len: int | None = None,
        batch_size: int = 128,
    ):
        self.window_size = window_size
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.batch_size = batch_size

        self.model = TimesFM.from_pretrained()

        if DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(DEVICE)
        self.model.to(self.device)
        self._log(f"Using device: {self.device}")

    def _log(self, msg):
        if VERBOSE:
            print(msg)

    def _predict_on_batch(
        self,
        data: DataLoader,
    ):
        pred = []
        gids = []

        for batch in data:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = self.model(**batch)
            pred.append(output)
            gids.append(batch["gids"])

        return pred, gids

    @torch.no_grad()
    def predict(
        self,
        data: List[np.array],
        freq: List[int] = None,
    ):
        dataset = InferDataset(
            data=data,
            freq=freq,
            window_size=self.window_size,
            context_len=self.context_len,
            horizon_len=self.horizon_len,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        pred, gids = self._predict_on_batch(dataloader)

        return pred, gids

    @torch.no_grad()
    def predict_on_df(
        self,
        df: pd.DataFrame,
        freq: str,
        group_id: str,
        target: str,
        time_idx: str,
        num_jobs: int = 1,
    ):
        dataset = InferDataset.from_dataframe(
            data=df,
            freq=freq,
            group_id=group_id,
            target=target,
            time_idx=time_idx,
            window_size=self.window_size,
            context_len=self.context_len,
            horizon_len=self.horizon_len,
            num_jobs=num_jobs,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        pred, gids = self._predict_on_batch(dataloader)

        for ix in range(len(gids)):
            if gids[ix] is not None:
                gids[ix] = dataset.inverse_group_id_map[gids[ix].item()]

        pred_df = compile_outputs(pred, gids)

        return pred_df
