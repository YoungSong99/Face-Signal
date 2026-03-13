
# Synthetic Face Detection & Bias Analysis

A research project investigating whether traditional image-signal
features used for synthetic image detection introduce demographic bias
across gender and skin tone groups.

This project analyzes artifact, color, frequency, and
reconstruction-error features extracted from face images to understand
how detection signals behave across different AI generation models and
demographic attributes.

------------------------------------------------------------------------

# Project Motivation

Recent advances in generative models such as GANs and diffusion models
allow synthetic faces to appear extremely realistic.

Detecting synthetic images has therefore become an important problem in
AI forensics and digital media integrity.

Previous research (e.g., AI-Face) shows that many detection models
exhibit demographic bias across gender and skin tone groups.

This project investigates whether traditional signal-based features used
in synthetic image detection also show demographic bias.

------------------------------------------------------------------------

# Project Overview

Dataset\
↓\
Face / Skin Region Extraction\
↓\
Feature Extraction\
↓\
Bias Analysis\
↓\
Visualization & Interpretation

------------------------------------------------------------------------

# Dataset

The dataset contains both real and AI-generated face images.

Real images: - FFHQ - IMDB-Wiki

Synthetic images: - StyleGAN - StyleGAN2 - AttGAN - StarGAN - Latent
Diffusion - Stable Diffusion - Commercial AI tools

Each image includes metadata:

-   Model type
-   Predicted gender
-   Predicted age group
-   Skin tone (Monk scale)
-   Skin tone group

Three regions are analyzed:

  Region   Description
  -------- ----------------------------------------------
  Full     Original full image
  Face     Cropped face region
  Skin     Skin-only region extracted from segmentation

------------------------------------------------------------------------

# Feature Categories

## 1. Color Features

Used to detect color inconsistencies.

Examples: - RGB statistics - Channel correlation - Chroma-luma
inconsistency - Colorfulness metric - Entropy - HSV / Lab statistics

------------------------------------------------------------------------

## 2. Frequency Features

AI-generated images often produce abnormal frequency patterns.

Examples: - FFT radial energy - Frequency peak statistics -
High-frequency energy ratio

------------------------------------------------------------------------

## 3. Artifact & Residual Features

Generative models sometimes leave high-frequency artifacts.

Examples: - Residual variance - Residual skewness - Residual kurtosis -
Laplacian variance - High-pass energy ratio

------------------------------------------------------------------------

## 4. VAE Reconstruction Features

A pretrained Variational Autoencoder (VAE) reconstructs images.

Synthetic images often produce larger reconstruction errors.

Examples: - LPIPS distance - MSE reconstruction error - Residual
statistics

------------------------------------------------------------------------

# Bias Analysis

Experiments investigate feature behavior across:

-   Gender
-   Skin tone
-   Generation model

Methods include:

-   Feature distribution comparison
-   Random Forest feature importance
-   PCA visualization
-   Balanced sampling
-   Demographic performance comparison