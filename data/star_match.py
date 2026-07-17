import os
from pathlib import Path

import torch as th
import torch.utils.data as th_data
from torch_geometric.data import Batch

from omegaconf import DictConfig

from nagsn_runtime.star_match import (
    PreprocessStarData,
    init_match_pairs,
    load_star_csv,
    reset_indices,
    validate_bidirectional_matches,
)


class StarGraphData:
    def __init__(self, data_dir, index, config, mode):
        self.data_dir = data_dir
        self.sample_name = Path(data_dir).name
        self.index = index
        self.config = config
        self.mode = mode

        self.match_pair = None
        self.img_graph = None
        self.ast_graph = None

        self.pre = PreprocessStarData(config, self.data_dir, mode)
        self.camera_params = self.pre.camera_params
        self.coordinate_tolerance = float(
            self.config.validation.coordinate_tolerance
        )

        self._load_data()

    def _load_data(self):
        img_path = str(Path(self.data_dir) / "img.csv")
        ast_path = str(Path(self.data_dir) / "ast.csv")

        # output: coord1, coord2, gray/mag, star_index, match_index
        img_array = load_star_csv(
            img_path,
            "img",
            self.camera_params.image_size,
            self.coordinate_tolerance,
        )
        ast_array = load_star_csv(
            ast_path,
            "ast",
            self.camera_params.image_size,
            self.coordinate_tolerance,
        )
        validate_bidirectional_matches(
            img_array,
            ast_array,
            img_path,
            ast_path,
        )
        img_data = th.from_numpy(img_array)
        ast_data = th.from_numpy(ast_array)

        # match pairs
        img_indices, ast_indices = reset_indices(img_data, ast_data)
        self.match_pair = init_match_pairs(img_indices, ast_indices )

        # preprocess
        img_data = self.pre.preprocess_img(img_data[:, :3])
        ast_data = self.pre.preprocess_ast(ast_data[:, :3])

        # construct graphs
        self.img_graph = self.pre.construct_graph(img_data)
        self.ast_graph = self.pre.construct_graph(ast_data)


class StarInfoData(th_data.Dataset):
    def __init__(self,
                 data_dir: str,
                 load_type: str,
                 config: DictConfig):
        super().__init__()

        if load_type not in {"train", "val", "test"}:
            raise ValueError(f"load_type must be 'train', 'val', or 'test', got {load_type!r}.")

        self.data_dir = Path(data_dir) / load_type
        self.load_type = load_type
        self.config = config

        if not self.data_dir.is_dir():
            raise NotADirectoryError(
                f"Star data split directory does not exist: {self.data_dir}"
            )

        sample_names = sorted(
            entry.name for entry in os.scandir(self.data_dir) if entry.is_dir()
        )
        if not sample_names:
            raise ValueError(f"No valid star samples found in {self.data_dir}.")

        self.files_name = tuple(sample_names)
        self.sample_paths = tuple(
            self.data_dir / name for name in self.files_name
        )

    def __len__(self):
        L = len(self.sample_paths)
        return L

    def __getitem__(self, item):
        data_path = self.sample_paths[item]
        graph_data = StarGraphData(
            str(data_path),
            item,
            self.config,
            self.load_type,
        )
        return (
            graph_data.img_graph,
            graph_data.ast_graph,
            graph_data.match_pair,
            graph_data.sample_name,
        )


class MatchPairBatch:
    def __init__(self):
        self.match_pairs_list = []
        self.num_pairs = []
        self.img_offsets = []
        self.ast_offsets = []
        self.sample_names = []

        self.is_finish = False

    def update(self,
               pair,
               img_offset,
               ast_offset,
               sample_name=None):
        self.match_pairs_list.append(pair)
        self.num_pairs.append(pair.shape[0])
        self.img_offsets.append(img_offset)
        self.ast_offsets.append(ast_offset)
        self.sample_names.append(sample_name)

    def finish(self):
        non_empty_pairs = [pair for pair in self.match_pairs_list if pair.numel()]
        if not non_empty_pairs:
            self.match_pairs = th.empty((0, 2), dtype=th.long)
        else:
            self.match_pairs = th.cat(non_empty_pairs, dim=0)

        pair_counts = th.tensor(self.num_pairs, dtype=th.long)
        self.pair_ptr = th.cat((
            th.zeros(1, dtype=th.long),
            pair_counts.cumsum(dim=0),
        ))
        self.img_offsets = th.tensor(self.img_offsets, dtype=th.long)
        self.ast_offsets = th.tensor(self.ast_offsets, dtype=th.long)
        self.is_finish = True

    def pin_memory(self):
        if not self.is_finish:
            raise RuntimeError("finish() must be called before pinning match pairs.")
        self.match_pairs = self.match_pairs.pin_memory()
        self.pair_ptr = self.pair_ptr.pin_memory()
        self.img_offsets = self.img_offsets.pin_memory()
        self.ast_offsets = self.ast_offsets.pin_memory()
        return self

    def to(self, device, *args, **kwargs):
        if not self.is_finish:
            raise RuntimeError("finish() must be called before moving match pairs.")
        self.match_pairs = self.match_pairs.to(device, *args, **kwargs)
        self.pair_ptr = self.pair_ptr.to(device, *args, **kwargs)
        self.img_offsets = self.img_offsets.to(device, *args, **kwargs)
        self.ast_offsets = self.ast_offsets.to(device, *args, **kwargs)
        return self

    def to_list(self):
        if not self.is_finish:
            raise RuntimeError(
                "finish() must be called before converting match pairs."
            )

        match_pairs = self.match_pairs.detach().cpu()
        pair_ptr = self.pair_ptr.detach().cpu()
        img_offsets = self.img_offsets.detach().cpu()
        ast_offsets = self.ast_offsets.detach().cpu()
        match_pairs_list = []
        for start_index, end_index, img_offset, ast_offset in zip(
            pair_ptr[:-1],
            pair_ptr[1:],
            img_offsets,
            ast_offsets,
        ):
            offsets = th.stack((img_offset, ast_offset))
            match_pair = match_pairs[start_index:end_index] - offsets
            match_pairs_list.append(match_pair.numpy())

        return match_pairs_list


def collate_fn(batch):
    if not batch:
        raise ValueError("Cannot collate an empty star-match batch.")
    img_graphs = []
    ast_graphs = []
    match_batch = MatchPairBatch()

    img_offset = 0
    ast_offset = 0

    for sample in batch:
        if len(sample) == 4:
            img_graph, ast_graph, match_pair, sample_name = sample
        elif len(sample) == 3:
            img_graph, ast_graph, match_pair = sample
            sample_name = None
        else:
            raise ValueError(
                f"Expected a 3- or 4-item star sample, got {len(sample)} items."
            )

        img_graphs.append(img_graph)
        ast_graphs.append(ast_graph)

        if not th.is_tensor(match_pair):
            raise TypeError(
                f"Sample {sample_name!r} match pairs must be a torch.long "
                f"tensor, got {type(match_pair).__name__}."
            )
        if match_pair.ndim != 2 or match_pair.shape[1] != 2:
            raise ValueError(
                f"Sample {sample_name!r} match pairs must have shape [N, 2], "
                f"got {match_pair.shape}."
            )
        num_pair = match_pair.shape[0]
        if match_pair.dtype != th.long:
            raise TypeError(
                f"Sample {sample_name!r} match pairs must be a torch.long "
                f"tensor, got {type(match_pair).__name__} with "
                f"dtype {getattr(match_pair, 'dtype', None)}."
            )
        if num_pair > 0:
            if th.any(match_pair < 0):
                raise ValueError(
                    f"Sample {sample_name!r} contains negative match indices."
                )
            if th.any(match_pair[:, 0] >= img_graph.num_nodes):
                raise ValueError(
                    f"Sample {sample_name!r} contains an image match index "
                    f"outside [0, {img_graph.num_nodes})."
                )
            if th.any(match_pair[:, 1] >= ast_graph.num_nodes):
                raise ValueError(
                    f"Sample {sample_name!r} contains a catalog match index "
                    f"outside [0, {ast_graph.num_nodes})."
                )

        offsets = th.tensor([img_offset, ast_offset], dtype=th.long)
        match_pairs_b = match_pair + offsets

        match_batch.update(match_pairs_b,
                           img_offset,
                           ast_offset,
                           sample_name)
        img_offset += img_graph.num_nodes
        ast_offset += ast_graph.num_nodes

    img_batch = Batch.from_data_list(img_graphs)
    ast_batch = Batch.from_data_list(ast_graphs)

    match_batch.finish()
    if match_batch.match_pairs.shape[0] == 0:
        raise ValueError(
            "A star-match batch must contain at least one matched pair; "
            f"samples: {match_batch.sample_names}."
        )
    return img_batch, ast_batch, match_batch
