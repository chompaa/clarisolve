# Clarisolve

> A DL-based super-resolution and colorization tool.

## Requirements

You must have Python with minimum version `3.10.0` installed
your system.

To install the required packages, you can run:

```shell
pip install -r requirements.txt
```

## Usage

> ⚠️ **Warning**: This tool does **not** catch errors. As a result, running it incorrectly may produce incoherent traces.

### Evaluation

A GUI tool is provided to super-resolve and colorize images. To run, use:

```shell
python main.py
```

CLI options are also present, see the `super-resolve.py` and `colorize.py` files for
details.

### Training

You can train any model yourself using `train.py` as follows:

```shell
python train.py --model { "srcnn", "srcnnc", "srres", "iccnn", "icres" } \
                --train-data TRAIN_DATA \
                --eval-data EVAL_DATA \
                --output-dir OUTPUT_DIR \
                [--checkpoint-path CHECKPOINT_PATH] \
                [--learn-rate LEARN_RATE] \
                [--end-epoch END_EPOCH] \
                [--num-workers NUM_WORKERS] \
                [--seed SEED]
```

Note that for SR models, a `.h5` file is required for both datasets, and for IC, a
directory is required.

### Datasets

A utility script `util/make.py` is provided for `.h5` file creation.
