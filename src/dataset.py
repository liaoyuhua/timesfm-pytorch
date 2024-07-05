from typing import List
import warnings
import multiprocessing
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import freq_map, process_group, moving_average


class InferDataset(Dataset):
    def __init__(
        self,
        data: List[np.array],
        freq: List[int] = None,
        gids: List[str] = None,
        window_size: int = None,
        context_len: int = None,
        horizon_len: int = None,
        group_id_map: dict = None,
    ):
        self.freq = freq
        self.gids = gids
        self.window_size = window_size
        self.context_len = context_len
        self.horizon_len = horizon_len

        self.group_id_map = group_id_map

        if self.group_id_map is not None:
            self.inverse_group_id_map = {v: k for k, v in self.group_id_map.items()}
        else:
            self.inverse_group_id_map = None

        self.input_ts, self.input_padding, self.input_freq = self._preprocess(
            data, freq
        )

    def _padding(self, data: List[np.ndarray], freq: List[int]):
        input_ts, input_padding, input_freq = [], [], []
        for i, ts in enumerate(data):
            input_len = ts.shape[0]
            padding = np.zeros(shape=(input_len + self.horizon_len,), dtype=float)
            if input_len < self.context_len:
                num_front_pad = self.context_len - input_len
                ts = np.concatenate(
                    [np.zeros(shape=(num_front_pad,), dtype=float), ts], axis=0
                )
                padding = np.concatenate(
                    [np.ones(shape=(num_front_pad,), dtype=float), padding], axis=0
                )
            elif input_len > self.context_len:
                ts = ts[-self.context_len :]
                padding = padding[-(self.context_len + self.horizon_len) :]

            input_ts.append(ts)
            input_padding.append(padding)
            input_freq.append(freq[i])

        return (
            np.stack(input_ts, axis=0),
            np.stack(input_padding, axis=0),
            np.stack(input_freq, axis=0).astype(np.int32).reshape(-1, 1),
        )

    def _preprocess(self, data: List[np.ndarray], freq: List[int]):
        data = [np.array(ts)[-self.context_len :] for ts in data]

        if self.window_size is not None:
            new_inputs = []
            for ts in data:
                new_inputs.extend(moving_average(ts, self.window_size))
            data = new_inputs

        if freq is None:
            warnings.warn("No frequency provided via `freq`. Default to high (0).")
            freq = [0] * len(data)

        return self._padding(data, freq)

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        freq: str,
        context_len: int,
        horizon_len: int,
        group_id: str = "unique_id",
        target: str = None,
        time_idx: str = "ds",
        window_size: int = None,
        num_jobs: int = 1,
    ):
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            print("Multiprocessing context has already been set.")

        df_sorted = data.sort_values([group_id, time_idx], ignore_index=True)

        # if group_id column is string type, build a dictionary to map group_id to integer
        if df_sorted[group_id].dtype == "object":
            group_id_map = {
                gid: i for i, gid in enumerate(df_sorted[group_id].unique())
            }
            df_sorted["__gid__"] = df_sorted[group_id].map(group_id_map)
        else:
            group_id_map = None
            df_sorted["__gid__"] = df_sorted[group_id]

        new_inputs = []
        gids = []

        if num_jobs == 1:
            for key, group in df_sorted.groupby("__gid__"):
                inp, gid = process_group(
                    key,
                    group,
                    target,
                    context_len,
                )
                new_inputs.append(inp)
                gids.append(gid)
        else:
            if num_jobs == -1:
                num_jobs = multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=num_jobs) as pool:
                results = pool.starmap(
                    process_group,
                    [
                        (key, group, target, context_len)
                        for key, group in df_sorted.groupby("__gid__")
                    ],
                )
            new_inputs, gids = zip(*results)

        freq_inps = [freq_map(freq)] * len(new_inputs)

        return cls(
            new_inputs,
            freq_inps,
            gids,
            window_size,
            context_len,
            horizon_len,
            group_id_map,
        )

    def __len__(self):
        return len(self.input_ts)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.input_ts[idx], dtype=torch.float32),
            "paddings": torch.tensor(self.input_padding[idx], dtype=torch.float32),
            "freq": torch.tensor(self.input_freq[idx], dtype=torch.int32),
            "gid": (
                torch.tensor([self.gids[idx]], dtype=torch.int32) if self.gids else None
            ),
        }


if __name__ == "__main__":
    import os

    print(os.getcwd())

    data = pd.read_csv("./data/ETTm1.csv")

    data["ds"] = pd.to_datetime(data["date"])
    data["unique_id"] = "AAA"

    dataset = InferDataset.from_dataframe(
        data=data,
        freq="MIN",
        context_len=168,
        horizon_len=24,
        group_id="unique_id",
        target="OT",
        time_idx="ds",
    )

    print(dataset[0])

    print(dataset[0]["x"].shape)

    print(dataset[0]["paddings"].shape)

    print(dataset[0]["freq"].shape)

    print(dataset[0]["gid"])

    print(dataset[0]["gid"].shape)

    print(dataset.inverse_group_id_map[dataset[0]["gid"].item()])
