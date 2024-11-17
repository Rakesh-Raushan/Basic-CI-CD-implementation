import torch
from network import MNISTNet
import pytest

def test_model_parameters():
    model = MNISTNet()
    param_count = model.count_parameters()
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"

def test_model_accuracy():
    from train import main
    accuracy = main()
    assert accuracy > 95.0, f"Model accuracy is {accuracy}%, should be greater than 95%" 