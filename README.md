# NAGSN

NAGSN is a graph-neural-network-based framework for star identification. It
provides workflows for distortion correction and adaptation, star matching,
and end-to-end star identification.

This repository includes the model architectures, primary data-loading
pipelines, PyTorch Lightning modules, and executable entry points. To comply
with institutional release restrictions, selected low-level components are
distributed separately as the precompiled `nagsn-runtime` wheel.

## Environment

The reference environment uses Python 3.9, PyTorch 2.0.0, and CUDA 11.8.

```bash
conda create -n nags python=3.9 pip -y
conda activate nags
python -m pip install --upgrade pip
python -m pip install torch==2.0.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

Install the runtime wheel included with the release. The supplied wheel targets
CPython 3.9 on Linux x86-64.

```bash
python -m pip install dist/runtime/nagsn_runtime-0.1.0-cp39-cp39-linux_x86_64.whl
```

## Data

The datasets, directory layouts, and field descriptions are available on
[Zenodo](https://zenodo.org/records/21403901).

## Usage

Run the following commands from the project root. Command-line arguments
override the corresponding Hydra settings under `config/`.

Set `DATA_PATH` to the root directory of the extracted dataset before running
the examples:

```bash
export DATA_PATH=/path/to/dataset
```

### Train the distortion-correction model

```bash
python scripts/train_correct.py \
  data_path=${DATA_PATH}/distortion_data/Sim6144_1 \
  hardware.gpus=0
```

Replace `Sim6144_1` with another dataset directory under `distortion_data` as
needed.

### Evaluate the distortion-correction model

```bash
python scripts/eval_correct.py \
  data_path=${DATA_PATH}/distortion_data/Sim6144_1 \
  test.ckpt_path=ckpts/test/correct/6144_1.ckpt \
  hardware.gpus=0
```

Pretrained distortion-correction checkpoints are provided in
`ckpts/test/correct/`.

### Train the star-matching model

```bash
python scripts/train_match.py \
  data_path=${DATA_PATH}/match_subsets \
  hardware.gpus=0
```

### Evaluate the star-matching model

```bash
python scripts/eval_match.py \
  data_path=${DATA_PATH}/match_subsets \
  test.ckpt_path=ckpts/test/nags.ckpt \
  hardware.gpus=0
```

The pretrained star-matching checkpoint is provided in `ckpts/test/`.

### Run the star-identification pipeline

```bash
python scripts/identify.py \
  data_path=${DATA_PATH} \
  match.test.ckpt_path=ckpts/test/nags.ckpt \
  hardware.gpus=0
```

For samples that require distortion correction, the appropriate checkpoint is
selected automatically according to the image size and distortion level:

```text
ckpts/test/correct/{image_size}_{level}.ckpt
```

### Adapt the distortion-correction model

```bash
python scripts/adapt_correct.py \
  data_path=${DATA_PATH}/distortion_data/AdCapture01 \
  match.test.ckpt_path=ckpts/test/nags.ckpt \
  hardware.gpus=0 \
  logger.name=adaptation
```

Additional training and evaluation options can be configured in
`config/correct.yaml`, `config/match.yaml`, and `config/identify.yaml`, or
overridden directly through Hydra command-line arguments.
