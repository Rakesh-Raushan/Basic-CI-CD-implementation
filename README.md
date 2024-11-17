# MNIST Classification with PyTorch

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification using PyTorch, with automated testing through GitHub Actions.

## Model Architecture

The model is a lightweight CNN with:
- 2 convolutional layers
- Max pooling layers
- 2 fully connected layers
- Less than 25,000 parameters
- Designed to achieve >95% accuracy in one epoch

## Project Structure

- `network.py`: Contains the CNN architecture definition
- `train.py`: Training script for the MNIST dataset
- `test_model.py`: Test cases for model parameters and accuracy
- `.github/workflows/model_tests.yml`: GitHub Actions workflow configuration

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pytest
