# 🌸 FlowerFlow
### Robust Computer Vision Data Pipelines with PyTorch

**FlowerFlow** is a machine learning engineering project focused on building reliable, reusable, and observable data pipelines for image-based computer vision tasks using PyTorch.

Rather than focusing solely on model architecture, this project explores the engineering foundations that make computer vision workflows dependable: image validation, preprocessing, data loading, error handling, dataset splitting, and pipeline monitoring.

The goal is to build a data pipeline that can handle imperfect image data gracefully, provide useful diagnostics, and support reproducible downstream model development.

---

## 📌 Project Overview

Computer vision models depend on the quality and reliability of their input pipelines. Missing files, corrupted images, inconsistent image formats, and inefficient data loading can interrupt training or silently compromise an experiment.

FlowerFlow addresses these challenges by implementing a monitored PyTorch data pipeline that can detect loading problems, apply image transformations, organize datasets, and measure data retrieval performance.

### Key Objectives

- Build a custom PyTorch `Dataset` for image loading and label management.
- Validate images and handle corrupted or missing files.
- Apply preprocessing and image transformations using `torchvision`.
- Organize samples using dataset subsets and train/validation/test splits.
- Use PyTorch `DataLoader` objects for batching and shuffling.
- Track image access frequency, loading times, and data-loading errors.
- Develop a workflow that can be inspected, tested, and extended for future computer vision experiments.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| PyTorch | Dataset abstractions, data loading, and pipeline integration |
| Torchvision | Image transformations and preprocessing |
| Pillow (PIL) | Image decoding, validation, and conversion |
| NumPy | Numerical operations and label handling |
| SciPy | Loading MATLAB `.mat` label files |
| Matplotlib | Data visualization and image inspection |
| tqdm | Progress monitoring |
| Fastai | Supporting image-display utilities |
| Google Colab | Interactive development and notebook execution |

## 🏗️ Pipeline Architecture

The project follows a modular data-processing workflow:

```text
Flower Image Dataset
        |
        v
Label Loading and Alignment
        |
        v
Image Discovery and Validation
        |
        v
Custom PyTorch Dataset
        |
        v
Error Handling and Diagnostics
        |
        v
Image Preprocessing and Transforms
        |
        v
Dataset Splitting
        |
        v
PyTorch DataLoaders
        |
        v
Pipeline Monitoring and Evaluation
        |
        v
Downstream Computer Vision Training
```

Each stage addresses a different aspect of data reliability, from verifying the source files to measuring how efficiently samples are retrieved.

## ✨ Key Features

### 1. Custom Dataset Implementation

A custom `Dataset` class connects image files with their corresponding labels and integrates with the PyTorch data-loading API.

The implementation supports indexed sample retrieval and provides a foundation for reusable computer vision data pipelines.

### 2. Image Validation and Error Handling

Image files are not always valid or accessible. The pipeline incorporates checks to identify common loading failures, including:

- Missing image files.
- Corrupted or unreadable images.
- Images that do not meet the configured minimum size.
- Image-decoding and preprocessing errors.

The monitoring workflow records diagnostic information to help identify problematic samples.

### 3. Image Preprocessing

The pipeline uses image-processing utilities and `torchvision` transformations to prepare images for downstream machine learning workflows.

This provides a place to introduce operations such as resizing, tensor conversion, normalization, and data augmentation when required by a future model.

### 4. Dataset Organization

PyTorch dataset utilities support organizing samples into subsets and training, validation, and test partitions.

This separation provides a foundation for evaluating downstream models without using the same samples indiscriminately across all stages.

### 5. DataLoader Integration

PyTorch `DataLoader` objects provide batching and optional shuffling, making the pipeline compatible with standard model-training loops.

### 6. Pipeline Monitoring

The monitored dataset extends the loading workflow with operational diagnostics, including:

- Sample access counts.
- Image retrieval times.
- Average and maximum observed loading times.
- Loading-error summaries.
- Samples that have not been accessed during a monitoring pass.

These measurements help reveal data-quality problems and potential performance bottlenecks before model training begins.

## 📂 Project Structure

The following is the intended project layout; adjust filenames to match the committed repository.

```text
ML-engineering-portfolio/
└── projects/
    └── 02-FlowerFlow/
        ├── README.md
        ├── requirements.txt
        ├── notebooks/
        │   └── FlowerFlow.ipynb
        └── helper_utils.py
```

**Note:** The dataset directory is shown for illustration. The complete image archive may need to be downloaded separately rather than committed to Git.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or another Python version compatible with the project's pinned dependencies.
- Git.
- Access to the flower-image dataset.
- Google Colab or a compatible local Python environment.

### 1. Clone the repository

```bash
git clone https://github.com/SethSekyere/ML-engineering-portfolio.git
cd ML-engineering-portfolio
```

### 2. Install dependencies

From the repository root, install the project's requirements:

```bash
python -m pip install -r projects/02-FlowerFlow/requirements.txt
```

If your requirements file is stored elsewhere, adjust the path accordingly.

### 3. Obtain the dataset

FlowerFlow uses image files and a MATLAB label file.

The expected dataset layout is:

```text
corrupted_flower_data/
├── jpg/
└── imagelabels.mat
```

Place the extracted dataset in the project directory, or update the notebook's dataset path to match your local setup.

**Dataset download:** Add the verified dataset source or archive URL here before publishing the project. Do not assume the full dataset is included in the Git repository.

### 4. Open the notebook

Launch Jupyter from the repository root:

```bash
jupyter notebook
```

Open:

`projects/02-FlowerFlow/notebooks/FlowerFlow.ipynb`

Alternatively, open the notebook in Google Colab and follow the setup instructions provided in the notebook.

### 5. Run the pipeline

Execute the notebook cells in order to:

1. Configure the environment and import dependencies.
2. Inspect the dataset and its file structure.
3. Load and align the labels.
4. Initialize the custom dataset.
5. Validate image retrieval and error handling.
6. Configure image transformations and data loaders.
7. Run the monitoring workflow and inspect its statistics.

Before running a full-dataset validation pass, confirm that the number of available image files matches the expected dataset and that the labels are aligned with those images.

## 🧪 Validation and Testing

The project uses dataset-level checks and monitored sample retrieval to inspect pipeline behavior.

Relevant validation checks include:

- Confirming that expected image files exist.
- Verifying that images can be decoded and converted to RGB.
- Checking image dimensions.
- Ensuring that requested indices are valid.
- Tracking loading failures and retrieval times.
- Confirming that the dataset and label counts are consistent.

A successful run over a small test subset demonstrates that the pipeline can execute on those samples; it does not, by itself, establish that the entire dataset is valid.

## 📊 Current Development Status

FlowerFlow is being developed as a practical data-engineering foundation for computer vision.

The initial implementation covers custom dataset loading, image validation, error diagnostics, and access-time monitoring.

**Dataset limitation:** During development, the local image directory contained 10 JPG files while the MATLAB label file contained 8,189 label entries. A temporary 10-sample subset was used to validate the loading and monitoring workflow.

A complete end-to-end dataset run requires restoring or downloading the full image collection and verifying its alignment with the labels.

The project should not be interpreted as demonstrating a trained flower-classification model or benchmark accuracy unless those results are added separately.

## 🔭 Future Improvements

Potential next steps include:

- [ ] Add a reproducible dataset-download and integrity-checking workflow.
- [ ] Automate dataset and label consistency checks.
- [ ] Add unit tests for missing, corrupted, and undersized images.
- [ ] Make error logging configurable instead of printing every failure.
- [ ] Benchmark loading performance under different `DataLoader` configurations.
- [ ] Evaluate parallel loading with `num_workers` and pinned memory where appropriate.
- [ ] Add reproducible dataset splits and random seeds.
- [ ] Integrate the pipeline with a downstream computer vision training experiment.
- [ ] Track experiments and data-pipeline metrics using MLflow.
- [ ] Add continuous integration to test the pipeline automatically.

## 🎯 Engineering Takeaways

This project explores several practical machine learning engineering principles:

- Reliable model training begins with reliable data ingestion.
- Dataset abstractions make preprocessing and loading reusable.
- Validation and diagnostics help surface data-quality problems early.
- Monitoring provides evidence about pipeline behavior and performance.
- Reproducible setup instructions make machine learning projects easier to review and maintain.



Machine Learning Engineering | PyTorch | Computer Vision | Model Deployment

Explore the broader portfolio:

[ML Engineering Portfolio on GitHub](https://github.com/SethSekyere/ML-engineering-portfolio)

---

*FlowerFlow — Building dependable data pipelines for computer vision, one sample at a time.*
