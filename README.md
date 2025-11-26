# mnist-deep-cnn-improved-image-classification

📘 Project Overview

This repository presents a research-driven exploration into improving the performance and generalization of Convolutional Neural Networks (CNNs) on the MNIST handwritten digit dataset.
We begin with a baseline 2-block CNN and progressively enhance it using:

Batch Normalization

Dropout Regularization

Data Augmentation

Improved training dynamics

The result is a deeper, more stable, and more robust CNN architecture capable of achieving state-of-the-art MNIST performance with strong generalization characteristics.

This repository provides both the baseline and improved models for comparison, enabling reproducible benchmarking and architecture experimentation.

🎯 Research Objectives

This project investigates:

1. How classic CNNs behave on MNIST using minimal architecture.

Baseline model uses only Conv → Pool → Conv → Pool → Dense.

2. How to systematically improve model robustness and generalization using:

Batch normalization

Dropout at multiple depths

Data augmentation (rotation, translation, zoom)

Improved regularization strategies

3. Convergence stability & learning dynamics

Training vs validation curves

Loss behaviors

Train–validation accuracy gap

Sensitivity to augmentation

4. Generalization to shifted, rotated, or deformed digits

Evaluating how augmentation affects robustness.

📊 Key Results
Baseline CNN

Accuracy: ~99% test accuracy

Behavior: Fast convergence, no overfitting

Weakness: Less robust to shifts & rotations

Improved CNN

Accuracy: Higher-than-baseline

Techniques used:
✓ Batch Normalization
✓ Dropout (0.25/0.25/0.5)
✓ Data Augmentation

Behavior:

Smoother optimization

Higher generalization

More stable validation loss

Stronger performance on perturbed digits

🧠 Architecture Summary
Baseline Architecture
Input → Conv(32) → MaxPool →
        Conv(64) → MaxPool →
        Flatten → Dense(128) → Dense(10)

Improved Architecture
Input → Data Augmentation →
    Conv(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    Conv(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    Flatten →
    Dense(128) → BatchNorm → ReLU → Dropout(0.5) →
    Dense(10, softmax)

📦 Repository Structure

A clean, research-style folder layout:

mnist-deep-cnn-improved-image-classification/
│
├── notebooks/
│   ├── 01_baseline_cnn.ipynb
│   ├── 02_improved_cnn.ipynb
│   └── 03_results_analysis.ipynb
│
├── src/
│   ├── baseline_model.py
│   ├── improved_model.py
│   ├── train.py
│   └── utils.py
│
├── results/
│   ├── plots/
│   │   ├── baseline_accuracy_curve.png
│   │   ├── improved_accuracy_curve.png
│   │   └── comparison.png
│   └── metrics/
│       ├── baseline_metrics.json
│       └── improved_metrics.json
│
├── README.md
└── requirements.txt

🚀 How to Run
Clone repo
git clone https://github.com/obiedeh/mnist-deep-cnn-improved-image-classification.git
cd mnist-deep-cnn-improved-image-classification

Install dependencies
pip install -r requirements.txt

Train baseline model
python src/train.py --model baseline

Train improved model
python src/train.py --model improved

📈 Training Visualizations

The repository includes comparison plots:

Baseline vs Improved accuracy curves

Baseline vs Improved loss curves

Prediction samples with confidence

Generalization performance improvements

These are generated automatically inside the results/plots/ folder.

📝 Research Notes & Insights
Findings

Batch Normalization improves optimization stability.

Dropout prevents co-adaptation of layers.

Augmentation significantly increases robustness to digit variability.

Improved CNN consistently outperforms baseline on rotated/shifted digits.

Baseline CNN already performs well, but improvement techniques make the model deploy-ready.

Why This Matters

MNIST is often considered “too easy,” but it provides the perfect controlled environment for studying:

Learning dynamics

Regularization effectiveness

Convergence patterns

Model robustness

This project demonstrates how classic CNNs can be turned from high-performing into highly robust.

📚 Technologies Used

TensorFlow / Keras

NumPy

Matplotlib

Python 3.10+

🧪 Sample Research Questions the Repo Addresses

How much does data augmentation influence CNN generalization?

Does dropout at early layers or late layers matter more?

What role does batch normalization play in convergence?

How close can a small CNN get to “state-of-the-art”?

How do baseline and improved models differ in learning dynamics?

🤝 Contributions

Pull requests for additional architectures (e.g., ResNet-style MNIST, depthwise CNNs, transformer baselines) are welcome.

📄 License

MIT License.
