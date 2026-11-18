# Iris Flower Image Classification with CNN + SAM

A refactored image-classification project based on the uploaded notebook's two experiments:
1. Grayscale images with a custom cross-entropy implementation and a SAM optimizer attempt.
2. RGB images with class balancing and a CNN.

This version consolidates the experiments into one reproducible pipeline and corrects key training/evaluation issues.

## What changed

- Uses RGB images consistently.
- Uses a compact CNN with adaptive pooling rather than a large fixed-size fully connected layer.
- Implements SAM as the intended two-pass optimization procedure (gradient ascent perturbation, second forward/backward pass, restore weights, then base optimizer update).
- Uses a stratified train/validation/test split.
- Applies augmentation only to the training set.
- Handles class imbalance with a weighted sampler.
- Selects the best checkpoint using validation accuracy.
- Reports final metrics on a separate held-out test set.
- Saves the model, metrics, training curves, and confusion matrix.

## Dataset layout

Place the dataset in a directory with one folder per class:

```text
dataset/
├── iris-setosa/
├── iris-versicolor/
└── iris-virginica/
```

Folder names determine the class labels. The code supports any number of classes (at least two), provided each class has enough images for a stratified split.

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Training

```bash
python train.py --data_dir ./dataset --epochs 30
```

Optional settings:

```bash
python train.py --data_dir ./dataset --epochs 40 --batch_size 32 --image_size 128 --learning_rate 0.001
```

## Output files

The `outputs/` directory contains:
- `best_model.pth` — model checkpoint selected by validation accuracy
- `metrics.json` — test metrics and training history
- `training_curves.png` — training and validation loss/accuracy
- `confusion_matrix.png` — confusion matrix from the held-out test set

## Important notes

- The uploaded notebook pointed to a Kaggle dataset path (`/kaggle/input/iris-computer-vision`). This project expects you to download the dataset and pass its local folder path using `--data_dir`.
- The code has been rewritten and reviewed structurally, but it has not been executed against your dataset in this environment. Do not publish performance numbers until you run it and verify the test results.
- The test set is held out from training and model selection; it provides a more meaningful estimate than evaluating on the training loader.
- If you build on an external dataset or earlier implementation, retain the dataset license and give appropriate attribution in your repository.
