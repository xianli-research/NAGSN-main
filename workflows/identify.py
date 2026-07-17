import os
import warnings
import numpy as np
from datetime import datetime

from torch.utils.data import DataLoader

from nagsn_runtime.matching import (
    predict_metric,
    metric_calculate,
)
from nagsn_runtime.identification import (
    build_affine_pairs,
    collate_graph_blocks,
    identify_graph_batch,
    IdentifyGraphData,
    IdentifyTestData,
    InsufficientBrightStarsError,
    load_correct_model,
    load_match_model,
    match_initial_graphs,
    merge_block_predictions,
    ResumeState,
    print_resume_record,
)


def _print_identify_result(data_path, test_data, eval_result_sample):
    print(f"___ {data_path} ___")
    print("\tMetric   \t\tValue")
    print("_________________________________________________________")
    print(f"\tf1       \t\t{eval_result_sample[0]:.6f}")
    print(f"\tprecision\t\t{eval_result_sample[1]:.6f}")
    print(f"\trecall   \t\t{eval_result_sample[2]:.6f}")
    print("_________________________________________________________")
    print(f"\tnum_img  \t\t{test_data.num_img:d}")
    print(f"\tnum_ast  \t\t{test_data.num_ast:d}")
    print("_________________________________________________________")


def identify_sample(data_path, device, config, match_model, resume_state=None):
    data_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3]
    print(f"Start processing `{data_path}`({data_now}).")

    if resume_state is not None:
        cached = resume_state.get_any(data_path)
        if cached is not None:
            print(f"[resume] Skip completed sample `{data_path}`.")
            return cached

    # === Load the data === #
    test_data = IdentifyTestData(data_path,
                                 device,
                                 config.match.dataset)

    # === correct the distort coordinates === #
    image_size = test_data.camera_params.image_size
    distort_level = test_data.camera_params.level
    if distort_level > 0:
        correct_model = load_correct_model(config.correct.model.correct,
                                           device,
                                           image_size,
                                           distort_level,
                                           config.correct.checkpoint_dir,
                                           config.correct.checkpoint_template)
        test_data.update_correct_model(correct_model)
        test_data.correct_coords()

    # === Select the bright stars to calculate affine parameters === #
    num_graph = config.match.dataset.divide.star_num_blk
    data_load = test_data.select_bright(num_graph)

    score_threshold = float(config.match.model.match.graph.score_thres)
    max_exact_nodes = int(config.match.model.match.graph.max_exact_nodes)
    pred_set = match_initial_graphs(
        data_load["img_graph"],
        data_load["ast_graph"],
        match_model,
        device,
        score_threshold,
        max_exact_nodes,
    )
    pred_pairs = build_affine_pairs(
        pred_set,
        data_load["img_coords"],
        data_load["ast_coords"],
    )

    if pred_pairs is not None:
        # Calculate the affine parameters and get the inliers
        ransac_rng = (
            int(config.reproduce.seed)
            if bool(config.reproduce.is_open)
            else None
        )
        test_data.pre.get_affine(
            pred_pairs[:, 1],
            pred_pairs[:, 0],
            rng=ransac_rng,
        )

        # === Load the graph data === #
        graph_data = IdentifyGraphData(data=test_data, config=config)
        data_loader = DataLoader(
            dataset=graph_data,
            batch_size=config.batch_size,
            collate_fn=collate_graph_blocks,
            shuffle=False,
            num_workers=config.num_workers_data,
            pin_memory=False,
            persistent_workers=False,
        )
        # === Process graph blocks === #
        batch_results = [
            identify_graph_batch(
                batch,
                device,
                match_model,
                score_threshold,
                max_exact_nodes,
            )
            for batch in data_loader
        ]
        preds = merge_block_predictions(batch_results)
    else:
        preds = np.empty([0, 3])

    # === Calculate the metrics === #
    gts = test_data.match_pairs_gt
    statistic_results = predict_metric(preds[:, :2], gts)
    eval_result_sample = metric_calculate(
        statistic_results[0],
        statistic_results[1],
        statistic_results[2]
    )
    _print_identify_result(data_path, test_data, eval_result_sample)
    print_resume_record(data_path, statistic_results, eval_result_sample)
    return statistic_results, eval_result_sample


def identify_group(group_path, config, device, match_model, resume_state=None):
    if resume_state is not None:
        cached = resume_state.get_any(group_path)
        if cached is not None:
            print(f"[resume] Skip completed group `{group_path}`.")
            count_result, eval_result = cached
            return np.expand_dims(count_result, axis=0), np.expand_dims(eval_result, axis=0)

    data_names = sorted(
        entry.name for entry in os.scandir(group_path) if entry.is_dir()
    )
    results = []
    for sample_name in data_names:
        sample_path = os.path.join(group_path, sample_name)
        try:
            result = identify_sample(
                data_path=sample_path,
                device=device,
                config=config,
                match_model=match_model,
                resume_state=None,
            )
        except InsufficientBrightStarsError as error:
            warnings.warn(
                f"Skipping identification sample {sample_path}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        results.append(result)
    if not results:
        warnings.warn(
            f"Skipping identification group with no processable samples: {group_path}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    count_results, eval_result_samples = zip(*results)

    count_results = np.stack(count_results, axis=0)
    eval_result_samples = np.stack(eval_result_samples, axis=0)
    f1, pre, rec = metric_calculate(
        np.sum(count_results[:, 0]),
        np.sum(count_results[:, 1]),
        np.sum(count_results[:, 2])
    )
    stds = np.std(eval_result_samples, axis=0)
    print(f"___ {group_path} ___")
    print("\tMetric   \t\tValue")
    print("_________________________________________________________")
    print(f"\tf1       \t\t{f1:.6f}±{stds[0]:.6f}")
    print(f"\tprecision\t\t{pre:.6f}±{stds[1]:.6f}")
    print(f"\trecall   \t\t{rec:.6f}±{stds[2]:.6f}")
    print("_________________________________________________________")
    print_resume_record(
        group_path,
        np.sum(count_results, axis=0),
        (f1, pre, rec),
    )
    return count_results, eval_result_samples


def identify_process(data_dir, config, device):
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Identification data directory does not exist: {data_dir}")
    data_groups = sorted(
        entry.name for entry in os.scandir(data_dir) if entry.is_dir()
    )
    resume_state = ResumeState.from_config(config)

    match_model = load_match_model(cfg_mdl=config.match.model.match,
                                   device=device,
                                   ckpt_path=config.match.test.ckpt_path)

    group_results = []
    for group_name in data_groups:
        result = identify_group(
            group_path=os.path.join(data_dir, group_name),
            config=config,
            device=device,
            match_model=match_model,
            resume_state=resume_state,
        )
        if result is not None:
            group_results.append(result)

    if not group_results:
        raise RuntimeError(
            f"No processable sample groups found in identification directory: {data_dir}"
        )
    count_results, eval_result_groups = zip(*group_results)
    count_results = np.concatenate(count_results, axis=0)
    eval_result_groups = np.concatenate(eval_result_groups, axis=0)

    f1, pre, rec = metric_calculate(
        np.sum(count_results[:, 0]),
        np.sum(count_results[:, 1]),
        np.sum(count_results[:, 2])
    )
    stds = np.std(eval_result_groups, axis=0)

    print(f"\n===={data_dir}====")
    print("\tMetric   \t\tValue")
    print("=========================================================")
    print(f"\tf1        \t\t{f1:.6f}±{stds[0]:.6f}")
    print(f"\tprecision \t\t{pre:.6f}±{stds[1]:.6f}")
    print(f"\trecall    \t\t{rec:.6f}±{stds[2]:.6f}")
    print("=========================================================\n")
