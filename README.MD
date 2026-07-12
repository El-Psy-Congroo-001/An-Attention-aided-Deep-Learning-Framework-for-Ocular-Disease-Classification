# ODIR-5K Retinal Disease Classification Experiments

This repository contains Jupyter notebooks for training and evaluating deep-learning models that classify retinal fundus images. The experiments are organised into three tasks:

- **5-class classification** — a five-category retinal disease dataset.
- **7-class classification** — a seven-category retinal disease dataset.
- **Cataract classification** — a cataract-focused classification dataset.

Each task compares three transfer-learning backbones—**DenseNet121**, **EfficientNet-B3**, and **ResNet50**—under four experiment settings: baseline, baseline with data augmentation, baseline with CBAM attention, and CBAM with data augmentation.

> **Important:** This is research code and is not a medical device. Predictions must not be used for diagnosis, treatment, or other clinical decisions without appropriate validation, regulatory review, and qualified clinical oversight.

## Contents

- [Project scope](#project-scope)
- [Repository layout](#repository-layout)
- [Models and experiment variants](#models-and-experiment-variants)
- [Data](#data)
- [Methodology](#methodology)
- [Requirements](#requirements)
- [Getting started](#getting-started)
- [Running an experiment](#running-an-experiment)
- [Outputs and evaluation](#outputs-and-evaluation)
- [Reproducibility notes](#reproducibility-notes)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

## Project scope

The notebooks implement supervised image classification using PyTorch and `torchvision.datasets.ImageFolder`. They load a folder-per-class dataset, split it into training, validation, and test sets, fine-tune an ImageNet-pretrained convolutional network, and report standard classification metrics.

The class labels are read directly from the dataset directory names through `ImageFolder`; therefore, the authoritative label order is the alphabetical order of the folders in the dataset copy used for a run. Keep the same directory names and structure when reproducing an experiment.

## Repository layout

```text
.
├── 5 Class Classification/
│   ├── Densenet121/
│   ├── EfficientnetB3/
│   └── Resnet50/
├── 7 Class Classification/
│   ├── Densenet121/
│   ├── EfficientnetB3/
│   └── Resnet50/
├── Cataract/
│   ├── Densenet121/
│   ├── EfficientNetB3/
│   └── Resnet50/
└── README.MD
```

Every model directory contains four notebook variants:

| Suffix | Meaning |
| --- | --- |
| `Base` | Transfer-learning baseline. |
| `Base+Augmentation` | Baseline with training-time image augmentation. |
| `Base+CBAM` | Baseline augmented with Convolutional Block Attention Module (CBAM). |
| `Base+CBAM+Augmentation` | CBAM model with training-time augmentation. |

## Models and experiment variants

| Backbone | Implementation approach |
| --- | --- |
| DenseNet121 | ImageNet-pretrained DenseNet121 with a task-specific classification head. |
| EfficientNet-B3 | EfficientNet-B3 fine-tuning using the `efficientnet-pytorch` package. |
| ResNet50 | ImageNet-pretrained ResNet50 with a task-specific classification head. |
| CBAM | Channel and spatial attention modules inserted into the model variant to emphasise informative features. |

The final classifier output dimension is determined at runtime from `len(full_dataset.classes)`, so it matches the selected dataset task.

## Data

The datasets themselves are **not included** in this repository. The notebooks currently point to Kaggle input locations, including:

- 7-class: `odir-5k-multi-class-7class-full-dataset`
- 5-class: `oidr-5k-org-multi-class-5-gcahm-class-full-dataset` (some notebooks use a similarly named `odir` dataset path)
- Cataract: `odir-5k-org-full-cataract-classification`

Download or attach the appropriate dataset, then update the `dataset_path` cell near the beginning of the notebook. The expected `ImageFolder` format is:

```text
dataset_root/
├── class_1/
│   ├── image_001.jpg
│   └── ...
├── class_2/
│   ├── image_002.jpg
│   └── ...
└── ...
```

Use only data you are authorised to access and redistribute. Review the source dataset's licence, consent conditions, and usage restrictions before training or publishing results.

## Methodology

The implemented workflow is broadly as follows:

1. Load images and labels with `torchvision.datasets.ImageFolder`.
2. Create reproducible stratified splits using `StratifiedShuffleSplit` and `random_state=42`:
   - 70% training
   - 15% validation
   - 15% testing
3. Apply ImageNet normalisation (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
4. In augmented variants, apply transformations such as random resized crops, horizontal/vertical flips, rotation, colour jitter, affine translation, grayscale conversion, and random erasing.
5. Fine-tune the selected pretrained backbone.
6. Handle class imbalance with inverse-frequency class weights in cross-entropy loss.
7. Optimise with AdamW, learning-rate scheduling, gradient clipping, and early-stopping logic where present.
8. Save the best validation-accuracy checkpoint as `best_model_acc.pth` in the notebook's working directory.
9. Evaluate on the held-out test split using classification reports, confusion matrices, per-class statistics, and prediction visualisations.

Notebook hyperparameters vary by task and variant. For example, the notebooks use a batch size of 32, while training duration is commonly 35 or 70 epochs. Treat the values in the selected notebook as the source of truth for a particular run.

## Requirements

Use Python 3.10+ and install a PyTorch build appropriate for your operating system and CUDA version. A GPU is strongly recommended for practical training times.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision
pip install efficientnet-pytorch numpy matplotlib seaborn scikit-learn jupyter
```

Main Python dependencies:

- `torch`, `torchvision`
- `efficientnet-pytorch` (EfficientNet-B3 notebooks)
- `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `jupyter`

For CUDA-enabled PyTorch installation, follow the platform-specific command from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Getting started

1. Clone or download this repository.
2. Create and activate a Python environment, then install the dependencies above.
3. Obtain the dataset for the experiment you want to run.
4. Open a notebook in JupyterLab, Jupyter Notebook, Kaggle, or Google Colab.
5. Change `dataset_path` to the absolute path of the local or attached dataset.
6. Select a GPU-backed runtime if available.
7. Run the notebook cells in order.

```bash
jupyter lab
```

For a Kaggle run, add the corresponding dataset as a notebook input and retain or adjust the existing `/kaggle/input/...` path.

## Running an experiment

Choose one notebook according to the target task, architecture, and variant. For example:

```text
7 Class Classification/Densenet121/
└── odir-5k-org-full-7-class-densenet-121 Base+CBAM+Augmentation.ipynb
```

Before starting training, verify:

- `dataset_path` resolves to the dataset root—not a parent folder containing multiple datasets.
- The class folders correspond to the intended task.
- The discovered `full_dataset.classes` are correct.
- The selected device is CUDA when a GPU is available.
- The output/checkpoint location will not overwrite results you need to keep.

To compare architectures or variants fairly, use the same dataset revision, class-folder layout, split seed, evaluation protocol, and hardware/runtime settings. Record package versions and the exact notebook filename with every result.

## Outputs and evaluation

Depending on the notebook, execution produces:

- Training and validation loss/accuracy history.
- A best-validation checkpoint named `best_model_acc.pth`.
- Test-set accuracy and per-class accuracy.
- `classification_report` metrics (precision, recall, F1-score, and support).
- Confusion-matrix visualisations.
- Sample image predictions showing true and predicted labels.

Checkpoints and generated figures are not versioned in this repository. Store experimental artefacts in a separate, clearly named results directory and record the task, model, variant, date, and dataset revision.

## Reproducibility notes

- Data splits use stratified sampling with `random_state=42` in the provided notebooks.
- Augmentations introduce stochasticity during training; exact results can still vary across GPUs, CUDA/cuDNN versions, PyTorch versions, and worker settings.
- Set Python, NumPy, and PyTorch seeds and enable deterministic settings if exact repeatability is required. This may reduce performance.
- Some notebook cells retain earlier commented-out experiments. Run the active cells and inspect them before interpreting results.
- Dataset paths are environment-specific and must be changed outside Kaggle or when using a different Kaggle dataset attachment.

## Limitations

- The notebooks are experimental and do not constitute a deployable inference application.
- Reported performance is meaningful only for the precise dataset split and preprocessing used in a run.
- External validation, subgroup analysis, calibration, bias assessment, and clinical review are necessary before any real-world use.
- Retinal images and labels can be sensitive health data; protect them appropriately and follow applicable privacy and ethics requirements.

## Contributing

Contributions are welcome. Please keep changes reproducible:

1. Explain the task, dataset revision, and experiment variant.
2. Avoid committing raw patient or source images, large checkpoints, or credentials.
3. Record dependency versions and random seeds for new results.
4. Preserve the train/validation/test separation and avoid test-set-driven model selection.
5. Add a concise description of methodology and evaluation changes.

## License

No licence file is currently included in this repository. All rights are reserved by default unless the repository owner adds a licence. Do not assume permission to reuse, modify, or redistribute the code or associated datasets; contact the project owner for licensing terms.
