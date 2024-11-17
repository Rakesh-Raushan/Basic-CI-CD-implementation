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

def test_model_output_probabilities():
    model = MNISTNet()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    probabilities = torch.exp(output)  # Convert log_softmax to probabilities
    
    # Check if probabilities sum to 1 (with small tolerance for numerical errors)
    assert torch.abs(probabilities.sum() - 1.0) < 1e-6, "Output probabilities don't sum to 1"
    # Check if all probabilities are between 0 and 1
    assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1), "Invalid probability values"

@pytest.mark.parametrize("batch_size", [1, 16, 32, 64, 128])
def test_different_batch_sizes(batch_size):
    model = MNISTNet()
    x = torch.randn(batch_size, 1, 28, 28)
    try:
        output = model(x)
        assert output.shape == (batch_size, 10), f"Failed for batch size {batch_size}"
    except Exception as e:
        pytest.fail(f"Failed to process batch size {batch_size}: {str(e)}")

def test_model_on_real_data():
    # Load a small subset of real MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    
    model = MNISTNet()
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()
    
    # Test on first batch
    data, targets = next(iter(dataloader))
    with torch.no_grad():
        outputs = model(data)
        predictions = outputs.max(1)[1]
        accuracy = predictions.eq(targets).float().mean().item() * 100
    
    assert accuracy > 90.0, f"Model accuracy on real test data: {accuracy}% (should be > 90%)"

def test_model_gradients():
    model = MNISTNet()
    x = torch.randn(1, 1, 28, 28, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    # Check if gradients are computed
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradients computed for {name}"
        assert not torch.all(param.grad == 0), f"Zero gradients for {name}"

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

def test_model_deterministic_training():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create two identical models
    model1 = MNISTNet()
    model2 = MNISTNet()
    
    # Ensure models have identical weights
    state_dict = model1.state_dict()
    model2.load_state_dict(state_dict)
    
    # Create fixed input and target
    x = torch.randn(10, 1, 28, 28)
    target = torch.randint(0, 10, (10,))
    
    # Train both models with identical input
    criterion = torch.nn.CrossEntropyLoss()
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    
    # First model forward and backward pass
    output1 = model1(x)
    loss1 = criterion(output1, target)
    optimizer1.zero_grad()
    loss1.backward()
    optimizer1.step()
    
    # Second model forward and backward pass
    output2 = model2(x)
    loss2 = criterion(output2, target)
    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()
    
    # Check if models remain identical after training step
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert torch.allclose(param1, param2, rtol=1e-5), f"Parameters diverged for {name1}"

def test_model_seed_reproducibility():
    # Test with different seeds
    seeds = [42, 123, 7]
    results = {}
    
    for seed in seeds:
        # Set seed
        torch.manual_seed(seed)
        
        # Create model and input
        model = MNISTNet()
        x = torch.randn(5, 1, 28, 28)
        
        # Get output
        with torch.no_grad():
            output = model(x)
        
        results[seed] = output.clone()
        
        # Test reproducibility with same seed
        torch.manual_seed(seed)
        model_repeat = MNISTNet()
        with torch.no_grad():
            output_repeat = model_repeat(x)
        
        assert torch.allclose(output, output_repeat, rtol=1e-5), f"Model not reproducible with seed {seed}"