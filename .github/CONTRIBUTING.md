# Contributing to Image Deraining

First off, thanks for taking the time to contribute! 🎉

The following is a set of guidelines for contributing to this project. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, images, etc.)
- **Describe the behavior you observed** and what you expected
- **Include your environment details** (OS, Python version, PyTorch version, GPU)

**Bug Report Template:**

```markdown
**Description:**
A clear description of the bug.

**To Reproduce:**
Steps to reproduce the behavior:

1. Run command '...'
2. Use config '...'
3. See error

**Expected behavior:**
What you expected to happen.

**Environment:**

- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10]
- PyTorch version: [e.g., 2.0.1]
- CUDA version: [e.g., 11.8]
- GPU: [e.g., RTX 3090]

**Additional context:**
Any other relevant information.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the proposed functionality
- **Explain why this enhancement would be useful**
- **Include examples** of how it would be used

### Contributing Code

#### Good First Issues

Look for issues labeled `good first issue` if you're new to the project. These are typically:

- Documentation improvements
- Simple bug fixes
- Adding tests for existing functionality
- Code cleanup and refactoring

#### Areas That Need Help

- **New architectures**: Implement Transformer-based or diffusion models
- **Optimization**: Improve inference speed, reduce memory usage
- **Documentation**: Tutorials, examples, API docs
- **Testing**: Increase test coverage
- **Datasets**: Add support for new datasets
- **Deployment**: Docker, TensorRT, ONNX optimizations

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/clearview.git
cd clearview
```

### 2. Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n deraining python=3.10
conda activate deraining
```

### 3. Install Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Or manually:
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Verify Installation

```bash
# Run tests
pytest tests/

# Check code formatting
black --check src/ scripts/ tests/

# Type checking
mypy src/
```

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# Or for bug fixes:
git checkout -b fix/bug-description
```

Branch naming conventions:

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Adding or updating tests

### 2. Make Your Changes

- Write clear, commented code
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Check code coverage
pytest --cov=src/deraining tests/

# Test formatting
black --check .
isort --check .

# Type checking
mypy src/
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add attention mechanism to U-Net"
# See commit message guidelines below
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

- **Clear title** describing the change
- **Description** of what changed and why
- **Link to related issues** (e.g., "Closes #123")
- **Screenshots/examples** if relevant
- **Checklist** confirming you've tested and documented

**PR Template:**

```markdown
## Description

Brief description of changes.

## Related Issues

Closes #(issue number)

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

- [ ] Tests pass locally
- [ ] Added new tests for new features
- [ ] Updated documentation

## Screenshots (if applicable)

Add before/after images if relevant.
```

### 6. Code Review

- Be open to feedback and discussion
- Make requested changes promptly
- Keep the PR focused on a single change
- Update your branch if main has changed: `git rebase main`

## Coding Standards

### Python Style

We follow PEP 8 with some modifications:

```python
# Use black for formatting (line length: 88)
black src/ scripts/ tests/

# Use isort for import sorting
isort src/ scripts/ tests/

# Type hints for all functions
def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    epochs: int = 100
) -> Dict[str, float]:
    """Train the deraining model.

    Args:
        model: PyTorch model to train
        data_loader: DataLoader for training data
        epochs: Number of training epochs

    Returns:
        Dictionary containing training metrics
    """
    pass
```

### Code Organization

```python
# Order of imports:
# 1. Standard library
import os
from pathlib import Path
from typing import Dict, Optional

# 2. Third-party
import torch
import torch.nn as nn
import numpy as np

# 3. Local
from clearview.models import UNet
from clearview.losses import CombinedLoss
```

### Documentation

- **Docstrings**: Use Google style docstrings
- **Comments**: Explain _why_, not _what_
- **Type hints**: Required for all public functions
- **README updates**: Update if adding features

Example docstring:

```python
def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (512, 384),
    normalize: bool = True
) -> torch.Tensor:
    """Preprocess image for model input.

    Resizes, normalizes, and converts image to PyTorch tensor.

    Args:
        image: Input image as numpy array (H, W, C)
        target_size: Target (height, width) for resizing
        normalize: Whether to normalize to [0, 1] range

    Returns:
        Preprocessed image tensor (C, H, W)

    Raises:
        ValueError: If image shape is invalid

    Example:
        >>> img = cv2.imread('rainy.jpg')
        >>> tensor = preprocess_image(img, target_size=(384, 512))
        >>> print(tensor.shape)
        torch.Size([3, 384, 512])
    """
    pass
```

### Testing

- Write tests for all new functionality
- Aim for >80% code coverage
- Use pytest fixtures for common setup
- Test edge cases and error conditions

```python
# tests/test_models.py
import pytest
import torch
from clearview.models import UNet

def test_unet_forward_pass():
    """Test U-Net forward pass with various input sizes."""
    model = UNet(in_channels=3, out_channels=3)

    # Test multiple resolutions
    for size in [(256, 256), (512, 384), (1024, 768)]:
        x = torch.randn(2, 3, *size)
        y = model(x)
        assert y.shape == x.shape, f"Output shape mismatch for {size}"

def test_unet_invalid_input():
    """Test U-Net handles invalid inputs gracefully."""
    model = UNet(in_channels=3, out_channels=3)

    with pytest.raises(ValueError):
        # Should fail with wrong number of channels
        x = torch.randn(2, 4, 256, 256)
        model(x)
```

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```html
<type
  >[optional scope]:
  <description> [optional body] [optional footer(s)]</description></type
>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks (dependencies, configs)
- `perf`: Performance improvements

### Examples

```bash
feat: add Transformer-based architecture option

feat(models): implement attention U-Net with configurable gates

fix: resolve CUDA out of memory during training

fix(data): handle grayscale images in dataset loader

docs: add training tutorial with examples

docs(api): update docstrings for loss functions

test: add unit tests for perceptual loss

refactor: simplify data augmentation pipeline

perf: optimize inference speed using torch.jit

chore: update PyTorch to 2.0.1
```

## Questions?

- **General questions**: Open a [GitHub Discussion](https://github.com/dronefreak/clearview/discussions)
- **Bug reports**: Open an [Issue](https://github.com/dronefreak/clearview/issues)
- **Feature requests**: Open an [Issue](https://github.com/dronefreak/clearview/issues) with `enhancement` label

## Recognition

Contributors will be recognized in:

- The main README.md
- Release notes for features they contribute
- GitHub's contributor graph

Thank you for contributing! 🙏
