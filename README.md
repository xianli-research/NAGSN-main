# NAGS

NAGS is a graph neural network project for star identification. It provides
workflows for star matching, distortion correction, adaptive correction, and
end-to-end star identification.

This repository releases the model architectures, primary data-loading
pipelines, PyTorch Lightning modules, and executable entry points. Low-level
implementations are provided separately through the binary `nagsn-runtime`
wheel.

## Environment

The reference environment uses Python 3.9, PyTorch 2.0.0, and CUDA 11.8.
Conda is used only to create the minimal Python environment; project
dependencies are installed with pip.

```bash
conda create -n nags python=3.9 pip -y
conda activate nags
python -m pip install --upgrade pip
python -m pip install torch==2.0.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

Install the private runtime wheel supplied with the release:

```bash
python -m pip install dist/runtime/nagsn_runtime-0.1.0-cp39-cp39-linux_x86_64.whl
```

Verify the installation:

```bash
python -c "import nagsn_runtime; print(nagsn_runtime.API_VERSION)"
```

## Data

The datasets, directory layouts, and field descriptions are available at:

<https://zenodo.org/records/21403901>

## Usage

Run the following commands from the project root. Command-line arguments
override the corresponding Hydra settings under `config/`.

### Train the distortion-correction model

```bash
python scripts/train_correct.py \
  data_path=${DATA_PATH}/distortiodistortion_datan_dataset/sub_data \
  hardware.gpus=0
```
example:
```bash
python scripts/train_correct.py \
  data_path=${DATA_PATH}/distortion_data/Sim6144_1 \
  hardware.gpus=0
```


### Evaluate the distortion-correction model

```bash
python scripts/eval_correct.py \
  data_path=${DATA_PATH}/distortion_data/sub_data \
  test.ckpt_path=ckpts/test/correct/{size}_{level}.ckpt \
  hardware.gpus=0
```
example:
```bash
python scripts/eval_correct.py \
  data_path=${DATA_PATH}/distortion_data/Sim6144_1 \
  test.ckpt_path=ckpts/test/correct/6144_1.ckpt \
  hardware.gpus=0
```

### Train the star-matching model

```bash
python scripts/train_match.py \
  data_path=${DATA_PATH}/subgraphs \
  hardware.gpus=0
```

### Evaluate the star-matching model

```bash
python scripts/eval_match.py \
  data_path=${DATA_PATH}/subgraphs \
  test.ckpt_path=ckpts/test/nags.ckpt \
  hardware.gpus=0
```

### Run star identification process

```bash
python scripts/identify.py \
  data_path=${DATA_PATH} \
  match.test.ckpt_path=ckpts/test/nags.ckpt \
  hardware.gpus=0
```

For samples requiring distortion correction, the correction checkpoint is
selected automatically using the following convention:

```text
ckpts/test/correct/{image_size}_{level}.ckpt
```

### Run adaptive distortion correction

```bash
python scripts/adapt_correct.py \
  data_path=${DATA_PATH}/distortion_data/AdCapture01 \
  match.test.ckpt_path=ckpts/test/nags.ckpt \
  hardware.gpus=0 \
  logger.name=adaption
```

Additional training and evaluation options can be configured in
`config/correct.yaml`, `config/match.yaml`, and `config/identify.yaml`, or
overridden directly through Hydra command-line arguments.
