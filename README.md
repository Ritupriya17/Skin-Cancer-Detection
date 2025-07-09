# Skin Cancer Detection

A machine learning project to detect skin cancer from image data. This repository contains code and documentation for building, training, and evaluating models that classify skin lesions as benign or malignant.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Skin cancer is among the most common types of cancer worldwide. Early detection can significantly increase the chances of successful treatment. This project leverages deep learning techniques to classify skin lesions using dermatoscopic images.

## Features

- Image preprocessing and augmentation
- Multiple model architectures (e.g., CNN, transfer learning with pre-trained models)
- Model evaluation and visualization
- Support for new data and easy retraining

## Dataset

This project uses the [ISIC](https://www.isic-archive.com/) dataset or similar public datasets of dermatoscopic images. You must download the dataset separately and place it in the appropriate directory (see [Usage](#usage)).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ritupriya17/Skin-Cancer-Detection.git
   cd Skin-Cancer-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Dependencies include: TensorFlow/Keras, NumPy, Pandas, Matplotlib, scikit-learn, and others.*

## Usage

1. **Prepare the dataset:**
   - Download and extract the dataset into the `data/` directory.
   - Update dataset paths in the code if necessary.

2. **Train the model:**
   ```bash
   python train.py
   ```

3. **Evaluate the model:**
   ```bash
   python evaluate.py
   ```

4. **Predict on new images:**
   ```bash
   python predict.py --image path_to_image.jpg
   ```

## Project Structure

```
Skin-Cancer-Detection/
│
├── data/                 # Dataset directory (not included)
├── models/               # Saved models
├── notebooks/            # Jupyter notebooks for experiments
├── src/                  # Source code
│   ├── preprocess.py     # Data preprocessing scripts
│   ├── model.py          # Model architectures
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── predict.py        # Inference script
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── LICENSE
```

## Model Training

- Hyperparameters can be adjusted in `train.py`.
- Example of transfer learning using a pre-trained model (e.g., ResNet, VGG).
- Checkpoints and logs are saved in `models/` and `logs/`.

## Results

| Model       | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|-----------|--------|----------|
| CNN         | 0.85     | 0.83      | 0.86   | 0.84     |
| ResNet50    | 0.90     | 0.89      | 0.91   | 0.90     |


## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
*Developed by [Ritupriya17](https://github.com/Ritupriya17)*
