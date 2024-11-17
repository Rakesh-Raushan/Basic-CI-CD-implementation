import torch
from network import MNISTNet
import pytest

def test_model_parameters():
    model = MNISTNet()
    param_count = model.count_parameters()
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"

def test_model_accuracy():
    # Load the trained model
    model = MNISTNet()
    model.load_state_dict(torch.load('mnist_model.pth'))
    
    # Get accuracy from the saved metrics
    with open('training_metrics.txt', 'r') as f:
        accuracy = float(f.read().strip())
    
    assert accuracy > 95.0, f"Model accuracy is {accuracy}%, should be greater than 95%" 