# ClearView Test Suite

This directory contains comprehensive unit tests and integration tests for the ClearView image deraining framework.

## Test Structure

```bash
tests/
├── conftest.py              # Shared pytest fixtures
├── test_models.py           # Tests for neural network models and building blocks
├── test_losses.py           # Tests for loss functions
├── test_data.py            # Tests for data loading and datasets
├── test_utils.py           # Tests for utility functions
├── test_integration.py     # End-to-end integration tests
└── README.md              # This file
```

## Running Tests

### Prerequisites

Install the project with development dependencies:

```bash
pip install -e ".[dev]"
```

Or install just the test dependencies:

```bash
pip install pytest pytest-cov
pip install -r requirements.txt
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Files

```bash
# Test models
pytest tests/test_models.py

# Test losses
pytest tests/test_losses.py

# Test data pipeline
pytest tests/test_data.py

# Test utilities
pytest tests/test_utils.py

# Test integration
pytest tests/test_integration.py
```

### Run with Coverage

```bash
pytest tests/ --cov=clearview --cov-report=term-missing
```

### Run Specific Test Classes or Methods

```bash
# Run a specific test class
pytest tests/test_models.py::TestUNet

# Run a specific test method
pytest tests/test_models.py::TestUNet::test_forward_pass
```

### Verbose Output

```bash
pytest tests/ -v
```

## Test Coverage

The test suite covers:

### Unit Tests

- **Model Building Blocks** (`test_models.py`)

  - ConvBlock with different activations and configurations
  - DoubleConv blocks
  - DownBlock and UpBlock for encoder/decoder
  - AttentionGate mechanism
  - UNet, UNetSmall, UNetLarge architectures

- **Loss Functions** (`test_losses.py`)

  - Pixel-wise losses: L1, L2, MAE, MSE, Charbonnier
  - Structural losses: SSIM, Multi-Scale SSIM
  - Edge-based losses
  - Perceptual losses
  - Combined loss functions

- **Data Pipeline** (`test_data.py`)

  - ImagePairDataset for paired rainy/clean images
  - SingleFolderDataset for inference
  - Rain100Dataset for benchmark datasets
  - DataLoader integration
  - Different image formats and extensions

- **Utility Functions** (`test_utils.py`)
  - Image normalization and denormalization
  - RGB to grayscale conversion
  - Tensor-NumPy conversion
  - Metrics: PSNR, SSIM, MAE, MSE
  - Checkpoint save/load
  - Image processing utilities

### Integration Tests

- **End-to-End Workflows** (`test_integration.py`)
  - Full training pipeline
  - Model save and load
  - Batch inference
  - Metrics evaluation during training
  - Training with different loss functions
  - Gradient flow verification
  - CPU and GPU testing

## Continuous Integration

These tests are designed to run in CI/CD pipelines. The test suite includes:

- Fast unit tests for individual components
- Integration tests for end-to-end workflows
- GPU tests (skipped if CUDA is not available)
- Comprehensive coverage reports

## Writing New Tests

When adding new features, please add corresponding tests:

1. **Unit tests**: Test individual functions/classes in isolation
2. **Integration tests**: Test how components work together
3. **Fixtures**: Use shared fixtures from `conftest.py`
4. **Type hints**: All test functions should have proper type hints
5. **Docstrings**: Each test should have a descriptive docstring

Example:

```python
def test_new_feature(self, sample_dataset_dir: Path) -> None:
    """Test the new feature with sample data."""
    # Test implementation
    assert result == expected
```

## Test Fixtures

Common fixtures available in `conftest.py`:

- `device`: PyTorch device (CPU or CUDA)
- `sample_image_tensor`: Sample 3-channel image tensor
- `sample_image_pair`: Pair of rainy/clean image tensors
- `sample_batch`: Batch of image pairs
- `sample_numpy_image`: NumPy image array
- `sample_pil_image`: PIL Image
- `temp_image_file`: Temporary image file
- `temp_dir`: Temporary directory
- `sample_dataset_dir`: Sample dataset with image pairs
- `model_config`: Default model configuration
- `training_config`: Default training configuration

## Troubleshooting

### Import Errors

If you get import errors, make sure the package is installed in editable mode:

```bash
pip install -e .
```

### Missing Dependencies

Install all dependencies:

```bash
pip install -r requirements.txt
```

### CUDA Tests Failing

GPU tests are automatically skipped if CUDA is not available. To run GPU tests, ensure you have:

- NVIDIA GPU
- CUDA toolkit installed
- PyTorch with CUDA support

## Contributing

When contributing tests:

1. Follow the existing test structure
2. Use descriptive test names
3. Add docstrings to all tests
4. Use type hints
5. Keep tests focused and independent
6. Use fixtures for common setup
7. Ensure tests are deterministic
