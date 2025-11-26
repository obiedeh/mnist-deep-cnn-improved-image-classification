mnist-deep-cnn-improved-image-classification
<p align="center"> <img src="https://img.shields.io/badge/MNIST-Deep%20CNN-blue?style=for-the-badge"> <img src="https://img.shields.io/badge/BatchNorm-Enabled-green?style=for-the-badge"> <img src="https://img.shields.io/badge/Dropout-0.25%2F0.25%2F0.5-orange?style=for-the-badge"> <img src="https://img.shields.io/badge/Augmentation-Active-purple?style=for-the-badge"> </p>

A research study comparing a baseline CNN to an improved CNN using Batch Normalization, Dropout, and Data Augmentation on the MNIST handwritten digit dataset.

📘 Project Overview

This repository contains a single Jupyter Notebook that implements a complete research workflow for MNIST handwritten digit classification. The notebook introduces a baseline CNN and an improved CNN, then compares their learning behavior, accuracy, robustness, and convergence stability.

The improved model integrates modern deep learning techniques such as Batch Normalization, Dropout regularization, and Data Augmentation to significantly enhance generalization and robustness to variations in handwritten digits.

This project is ideal for demonstrating foundational deep learning concepts, experimentation, and research-oriented performance analysis.

🎯 Research Objectives

This study answers several important research questions:

How does a simple CNN perform on MNIST?

How do BatchNorm and Dropout affect convergence?

How much does data augmentation improve generalization?

How do baseline and improved architectures differ in robustness?

Can a small CNN approach state-of-the-art results on MNIST?

🧠 What’s Inside the Notebook

Your single .ipynb file includes:

✔ 1. Dataset Loading & Preprocessing

MNIST import

Normalization to [0, 1]

Reshaping to (28, 28, 1)

✔ 2. Baseline CNN Model

Architecture:

Conv(32) → MaxPool →
Conv(64) → MaxPool →
Flatten → Dense(128) → Dense(10)

✔ 3. Improved CNN Model

Enhancements applied:

Data Augmentation (rotation, translation, zoom)

Batch Normalization (Conv + Dense layers)

Dropout (0.25, 0.25, 0.5)

Architecture:

Data Augmentation →
Conv(32) + BN + ReLU + MaxPool + Dropout →
Conv(64) + BN + ReLU + MaxPool + Dropout →
Flatten → Dense(128) + BN + ReLU + Dropout →
Dense(10)

✔ 4. Model Training

Adam optimizer

Sparse categorical crossentropy

Validation split = 0.10

5 training epochs

✔ 5. Evaluation

Training & validation accuracy curves

Training & validation loss curves

Test set evaluation

Sample predictions with confidence

Baseline vs Improved comparison

📊 Key Results
Baseline CNN

~99% test accuracy

Minimal overfitting

Smooth, stable learning curves

Excellent performance on clean MNIST digits

Improved CNN

Higher accuracy than baseline

Better robustness to rotated/shifted digits

More stable training

Better generalization due to augmentation, BN, and dropout

📈 Why the Improvements Work
Batch Normalization

Stabilizes training

Enables higher learning rates

Reduces internal covariate shift

Dropout

Prevents co-adaptation

Reduces overfitting

Encourages redundant robust features

Data Augmentation

Expands training distribution

Improves invariance to small transformations

Enhances real-world performance

🚀 How to Run This Project
1. Clone the repository
git clone https://github.com/obiedeh/mnist-deep-cnn-improved-image-classification.git
cd mnist-deep-cnn-improved-image-classification

2. Install dependencies
pip install -r requirements.txt

3. Launch Jupyter notebook
jupyter notebook

4. Open the notebook file

Your single notebook file (e.g., mnist_cnn.ipynb) contains the full workflow.

📚 Technologies Used

TensorFlow / Keras

NumPy

Matplotlib

Python 3.x

Jupyter Notebook

🧪 Research Questions Explored

This notebook investigates:

How do CNNs learn representations of handwritten digits?

What causes validation–training performance gaps?

How does augmentation improve robustness?

What architectural choices yield stability vs performance?

How small can a CNN be and still achieve ~99% accuracy?

🤝 Contributions

Since this is a research-oriented notebook, contributions are welcome in the form of:

New architectures (e.g., ResNet-style MNIST)

Additional augmentation strategies

Hyperparameter sweeps

Improved visualization tools

Comparison with other models (MLP, SVM, Transformers)

📄 License

MIT License.
