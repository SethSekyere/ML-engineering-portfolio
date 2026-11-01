# PixelForge
### Image Color Quantization Using K-Means Clustering

PixelForge is an unsupervised machine learning project that implements the **K-Means clustering algorithm from scratch** and applies it to image color quantization.

The project explores how K-Means can group similar RGB pixel values into a smaller set of representative colors, reducing an image's color palette while attempting to preserve its visual appearance.

By replacing individual pixel colors with their corresponding cluster centroids, PixelForge demonstrates a practical application of unsupervised learning in image processing and image compression.

---

## Project Overview

Digital images contain pixels represented by Red, Green, and Blue (RGB) color channels. Each channel can take values between 0 and 255, allowing for over 16 million possible RGB color combinations.

Storing all the original color information can require considerable space.

PixelForge addresses this challenge through **color quantization**.

Rather than retaining every original color, the project applies K-Means clustering to group similar pixel colors into a predefined number of clusters. Each cluster is represented by a centroid, which acts as its representative RGB color.

The original pixel colors are then replaced with their corresponding cluster centroid colors.

This reduces the number of distinct colors required to represent the image.

## Project Objectives

- Implement the K-Means clustering algorithm from scratch using Python and NumPy.
- Understand centroid initialization, cluster assignment, and centroid optimization.
- Apply K-Means to RGB pixel data from a digital image.
- Perform image color quantization using a reduced color palette.
- Visualize RGB color distributions and their corresponding cluster centroids.
- Explore the relationship between the number of clusters, image quality, and color reduction.

## How It Works

PixelForge applies K-Means clustering to image pixel values through the following workflow:

### 1. Image Loading and Preprocessing

The original image is loaded and represented as a NumPy array.

Each pixel contains three color channels:

- Red (R)
- Green (G)
- Blue (B)

The image is reshaped into a two-dimensional dataset where each row represents one pixel and its three RGB features.

### 2. K-Means Clustering

The K-Means algorithm groups similar RGB pixel values into K clusters.

The algorithm follows these steps:

1. Initialize K centroids.
2. Assign each pixel to its nearest centroid using a distance metric.
3. Recalculate each centroid using the mean of the assigned pixels.
4. Repeat the assignment and centroid update steps until convergence or the maximum number of iterations is reached.

### 3. Color Quantization

Once clustering is complete, each pixel is assigned the RGB value of its corresponding centroid.

For example:

Original pixel:

`[250, 100, 80]`

Representative centroid:

`[245, 105, 85]`

The original pixel color is replaced with the representative centroid color.

This process is repeated across the entire image, resulting in an image represented by a smaller color palette.

### 4. Visualization

The project visualizes the original RGB color distribution and the resulting cluster centroids in three-dimensional RGB space.

This helps illustrate how K-Means groups similar colors and identifies representative colors within the image.

---

## Technical Implementation

| Component | Implementation |
|---|---|
| Machine Learning Approach | Unsupervised Learning |
| Clustering Algorithm | K-Means |
| Algorithm Implementation | From Scratch |
| Data Representation | RGB Pixel Vectors |
| Distance Measurement | Euclidean Distance |
| Centroid Optimization | Mean of Assigned Data Points |
| Image Processing | Color Quantization |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |

## Technology Stack

- **Python** – Core programming language.
- **NumPy** – Numerical computing, array manipulation, and vectorized operations.
- **Matplotlib** – Visualization of RGB color distributions, centroids, and clustering results.
- **Pillow** – Image processing support, where required.
- **Jupyter Notebook** – Interactive development and experimentation.

---

## Repository Structure

```text
03-PixelForge/
│
├── PixelForge.ipynb
│
├── utils.py
│
├── requirements.txt
│
├── README.md
│
├── bird_small.png
│
└── data/
    └── ex7_X.npy
```

*Note: Update the notebook filename or structure above if your final repository uses different names or additional files.*

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/SethSekyere/ML-engineering-portfolio.git
```

### 2. Navigate to the Project Directory

```bash
cd ML-engineering-portfolio/projects/03-PixelForge
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `PixelForge.ipynb` and execute the cells sequentially.

---

## Experimentation

The number of clusters, K, determines how many representative colors are used to reconstruct the image.

A smaller K produces a more restricted color palette, while a larger K generally preserves more of the original image's color variation.

For example:

| Number of Clusters (K) | Expected Result |
|---|---|
| 4 | Very limited color palette |
| 8 | Strong color reduction |
| 16 | Reduced palette with more color variation |
| 32 | Greater preservation of original colors |
| 64 | More detailed color representation |

These values provide opportunities to investigate the trade-off between color reduction and visual quality.

## Key Machine Learning Concepts

This project provides practical experience with:

- Unsupervised machine learning.
- K-Means clustering.
- Feature representation and numerical arrays.
- Euclidean distance calculations.
- Centroid initialization and optimization.
- Iterative cluster assignment.
- Vectorized numerical operations.
- Image processing and color quantization.
- Visualization of high-dimensional data in RGB space.

## Potential Improvements

Future development could extend PixelForge with:

- **Automated K Selection:** Investigate methods such as the Elbow Method to identify suitable cluster counts.
- **Image Quality Evaluation:** Compare reconstructed images using metrics such as Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR).
- **Compression Analysis:** Compare original and quantized image file sizes using an indexed-color representation.
- **Performance Optimization:** Explore vectorization and more efficient implementations for large image datasets.
- **Interactive Experiments:** Allow users to select different values of K and compare the resulting images.

---

## Important Note on Compression

PixelForge demonstrates image color quantization, which is a technique that can support image compression.

The K-Means implementation reduces the number of distinct colors used to represent an image. However, actual file-size reduction depends on how the quantized image is subsequently encoded and saved.

The project therefore focuses on the machine learning and color quantization process rather than implementing a complete image compression format.

---

## Author

**Seth Sekyere**

Machine Learning | Deep Learning | ML Engineering

GitHub: [@SethSekyere](https://github.com/SethSekyere)

---

*PixelForge — Exploring how unsupervised learning can reduce image color complexity through K-Means clustering.*
