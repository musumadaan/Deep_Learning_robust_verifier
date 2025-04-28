# Deep_Learning_robust_verifier
This project implements a robustness verification tool for deep neural networks, specifically designed to check adversarial robustness against small perturbations.

The verifier targets:
- 7 Fully Connected Networks (fc1 to fc7)
- 3 Convolutional Neural Networks (conv1 to conv3)

The goal is to certify whether small adversarial perturbations (within an L-infinity norm ball) can change the classification result of the models on the MNIST dataset.

# Project Features
- Supports both Fully Connected (FC) and Convolutional Neural Networks (CNNs).
- Symbolic bound propagation using DeepPoly-style relaxations.
- ℓ∞-norm perturbation verification.
- Verifies whether adversarial examples can exist within a specified epsilon.
- Optimized for performance using numpy, PyTorch, and scipy.

