# Evaluating Batch Normalization and L2 Regularization in an MLP for MNIST Classification

## Introduction

This project explores the effects of two popular regularization techniques—Batch Normalization and L2 Regularization—on the performance of a fully connected feedforward neural network trained on the MNIST handwritten digit dataset. The goal is to understand how these techniques influence training stability and generalization.

## Network Architecture

- Input layer: 784 units (flattened 28x28 pixel images)
- Hidden layers: [20, 7, 5] units
- Output layer: 10 units (one per digit class)
- Activation functions: ReLU for hidden layers, Softmax for output
- Initialization: He initialization
- Optimization: Gradient Descent with Backpropagation
- Regularization:
  - L2 norm (weight decay)
  - Batch Normalization (optional)

## Training Setup

- Batch size: 256
- Learning rate: 0.009
- Epochs: until convergence (up to 100,000 steps)
- Evaluation: Accuracy and cost metrics tracked during training and validation

## Experiment Configurations and Results

| Configuration                         | Test Accuracy |
|--------------------------------------|---------------|
| No L2, No BatchNorm                  | 94.31%        |
| With BatchNorm                       | 89.86%        |
| With L2 Regularization Only          | 91.92%        |
| With BatchNorm & L2 Regularization   | 93.46%        |

## Training and Validation Cost Plots

Each figure below shows the training and validation cost per iteration under a different experimental setup:

- `With BatchNorm.png`
- `With L2_Norm without BatchNorm.png`
- `Without BatchNorm & L2_Norm.png`
- `With BatchNorm & L2_Norm.png`

These plots illustrate convergence speed and overfitting behavior under different configurations.

## Discussion

- **BatchNorm alone** degraded test accuracy in this small architecture, suggesting that its stabilizing effects may not always translate to better generalization.
- **L2 Regularization** consistently improved performance, reducing overfitting while maintaining stable convergence.
- The combination of both techniques did not yield additive benefits in this case.

## Conclusion

This study highlights that regularization strategies should be selected based on model architecture and data. While L2 norm helped mitigate overfitting, BatchNorm was less effective on its own in this shallow network.