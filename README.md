# README: Entropy-Gradient Even/Odd MNIST Classifier

## Project Overview
This project implements a **custom single-layer perceptron** for classifying MNIST digits as **even (0)** or **odd (1)**. Instead of using conventional **loss-based optimization**, it updates parameters based on the **entropy gradient**, with a constraint on knowledge updates:

\[ z_{i+1} - z_i < \delta \]

This constraint stabilizes training by limiting knowledge changes between iterations.

## Objective
- Extend the classical MNIST perceptron to support **entropy-gradient updates**.
- Implement a **dual-weight structure** with weights \( w_{1,j} \) and \( G_{1,j} \).
- Enforce the **knowledge constraint** \( \delta \) during parameter updates.

## Model Architecture
- **Input Layer:** 784 features (flattened MNIST images)
- **Output Layer:** Single neuron with **sigmoid activation** for binary classification

## Key Equations
1. **Knowledge Mapping:**
\[ z_k = \sum_{j=1}^{n}(w_{1,j} + G_{1,j}) x_j^{(0)} + b_1^{(0)} \]

2. **Activation:**
\[ D_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}} \]

3. **Entropy Gradient:**
\[ \frac{\partial H(z)}{\partial z} \Bigg|_k = -\frac{1}{\ln 2} z_k D_k (1 - D_k) \]

4. **Parameter Updates:**
\[ w_{1,j} \leftarrow w_{1,j} - \eta \left( \frac{\partial H(z)}{\partial z} \cdot x_j^{(0)} \right) \]
\[ G_{1,j} \leftarrow G_{1,j} - \eta \left( \frac{\partial H(z)}{\partial z} \cdot x_j^{(0)} \right) \]
\[ b_1 \leftarrow b_1 - \eta \left( \frac{\partial H(z)}{\partial z} \right) \]

5. **Constraint on Knowledge Updates:**
If \( z_{i+1} - z_i \geq \delta \), scale down updates:
\[ \text{Update} \times= \frac{\delta}{|z_{i+1} - z_i|} \]

## Training Details
- **Dataset:** MNIST (even/odd labels)
- **Optimizer:** Custom entropy-gradient updates
- **Batch Size:** 32
- **Epochs:** 5 (or more, adjustable)
- **Learning Rate:** Dynamic (initial \( \eta = 0.01 \))
- **Knowledge Constraint:** \( \delta = 0.05 \)

## Performance Monitoring
- **Test Accuracy** printed after each epoch
- **Accuracy Curve** plotted over epochs
- **Visualization:** 10 random test images with true/predicted labels

## Challenges & Insights
- **Slow learning:** When \( \delta \) is too small, updates get scaled down excessively.
- **Bias dominance:** If initialized poorly, bias can force all outputs to 1 (odd).
- **Dynamic learning rate:** Helps escape plateaus when accuracy stagnates.

## How to Run the Code
1. **Install dependencies:**
```bash
pip install numpy tensorflow matplotlib
```

2. **Download MNIST manually (if SSL issues occur):**
[MNIST Dataset](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)

3. **Run the script:**
```bash
python entropy_classifier.py
```

4. **Visualize results:**
The script will save and show test images with predictions.

---
This README summarizes the implementation, training process, and key considerations for the entropy-gradient even/odd classifier. Let me know if you’d like any adjustments or enhancements! 🚀

