# MNIST Classification with PyTorch

[![Model Tests](https://github.com/Rakesh-Raushan/Basic-CI-CD-implementation/actions/workflows/model_tests.yml/badge.svg)](https://github.com/Rakesh-Raushan/Basic-CI-CD-implementation/actions/workflows/model_tests.yml)

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification using PyTorch, with automated testing through GitHub Actions.

## Model Architecture

The model is a lightweight CNN with:
- 2 convolutional layers
- Max pooling layer
- Batch normalization layer
- 1 fully connected layer
- Less than 25,000 parameters, actually 21162 parameters
- Designed to achieve >95% accuracy in one epoch

## Project Structure

- `network.py`: Contains the CNN architecture definition
- `train.py`: Training script for the MNIST dataset
- `test_model.py`: Test cases for model parameters and accuracy
- `.github/workflows/model_tests.yml`: GitHub Actions workflow configuration
- `README.md`: This file
- `requirements.txt`: List of dependencies

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pytest
- tqdm

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

### Workflow Steps
1. Triggered on every push to main branch and pull requests
2. Sets up Python 3.8 environment
3. Installs CPU-only version of PyTorch and other dependencies
4. Trains the model from scratch
5. Runs automated tests to verify:
   - Model parameter count (< 25,000)
   - Model accuracy (> 95% in one epoch)
6. Saves trained model and metrics as artifacts

### Artifacts
After each successful workflow run:
- Trained model weights (`mnist_model.pth`)
- Training metrics (`training_metrics.txt`)
are saved and available for download from the Actions tab

### Monitoring
- View workflow runs in the GitHub Actions tab
- Each run shows training progress and test results

