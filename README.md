# Hybrid Federated Deep Learning for Privacy-Preserving Systems

## Overview

This project explores the integration of Hybrid Deep Learning architectures within a Federated Learning environment for privacy-preserving collaborative image classification. The system simulates decentralized client-server training using the Flower (FLWR) framework while evaluating the performance of hybrid neural architectures across multiple benchmark datasets.

## Key Features

- Hybrid Deep Learning architectures (CNN+RNN, CNN+FNN, VGG16+VGG19)
- Federated Learning using Flower (FLWR)
- Distributed multi-client training simulation
- Privacy-preserving collaborative learning
- FedAvg aggregation strategy
- Transfer Learning with pretrained VGG models
- Benchmark evaluation on MNIST, FMNIST, and CIFAR-10

## Federated Learning Workflow

The project simulates decentralized training across multiple clients where:

1. Clients train models locally on partitioned datasets
2. Model parameters are aggregated using FedAvg
3. Global weights are synchronized across communication rounds
4. Privacy is preserved by avoiding raw data sharing

## Tech Stack

- Python
- PyTorch
- TensorFlow
- Flower (FLWR)
- TorchVision
- NumPy
- Matplotlib
- Federated Learning
- Transfer Learning

## Results

Hybrid architectures demonstrated improved classification performance compared to standalone models in multiple experimental setups. The VGG16+VGG19 hybrid configuration achieved the strongest performance on CIFAR-10 within the federated setting.

## Future Improvements

- Homomorphic Encryption integration
- Differential Privacy
- Non-IID client optimization
- Personalized Federated Learning
- Scalable multi-node deployment
