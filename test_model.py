import torch
from network import MNISTNet
import pytest
from torchvision import datasets, transforms
import torch.nn.functional as F

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

def test_model_output_shape():
    model = MNISTNet()
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    assert output.shape == (batch_size, 10), f"Expected output shape (32, 10), got {output.shape}"

@pytest.mark.parametrize("batch_size", [1, 16, 32, 64, 128])
def test_different_batch_sizes(batch_size):
    model = MNISTNet()
    x = torch.randn(batch_size, 1, 28, 28)
    try:
        output = model(x)
        assert output.shape == (batch_size, 10), f"Failed for batch size {batch_size}"
    except Exception as e:
        pytest.fail(f"Failed to process batch size {batch_size}: {str(e)}")

def test_model_deterministic_inference():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create model and fixed input
    model = MNISTNet()
    x = torch.randn(10, 1, 28, 28)
    
    # First forward pass
    model.eval()
    with torch.no_grad():
        output1 = model(x)
    
    # Second forward pass with same input
    with torch.no_grad():
        output2 = model(x)
    
    # Check if outputs are identical
    assert torch.allclose(output1, output2, rtol=1e-5), "Model outputs are not deterministic"


