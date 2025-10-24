# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

### 1. **Do Not** Open a Public Issue

Please **do not** report security vulnerabilities through public GitHub issues. This helps prevent malicious actors from exploiting the vulnerability before a fix is available.

### 2. Report Privately

Report the vulnerability privately through one of these methods:

- **GitHub Security Advisories** (Preferred):

  - Go to the [Security tab](https://github.com/dronefreak/clearview/security)
  - Click "Report a vulnerability"
  - Fill out the form with details

- **Direct Email**:
  - Contact the maintainer through GitHub
  - Use the subject line: `[SECURITY] Brief description of vulnerability`

### 3. Include Detailed Information

Please include:

- **Description** of the vulnerability
- **Steps to reproduce** the issue
- **Potential impact** (e.g., data exposure, code execution)
- **Affected versions**
- **Suggested fix** (if you have one)
- **Your contact information** for follow-up

**Example Report:**

```markdown
**Vulnerability Type:** Arbitrary Code Execution

**Description:**
The model loading function uses `torch.load()` without checking file
integrity, allowing malicious actors to embed arbitrary code in
pretrained weight files.

**Steps to Reproduce:**

1. Create malicious .pth file with embedded code
2. Load using `model.load_weights('malicious.pth')`
3. Arbitrary code executes during unpickling

**Impact:**
High - Could lead to remote code execution on user machines

**Affected Versions:**
All versions <= 1.0.0

**Suggested Fix:**
Use `torch.load(..., weights_only=True)` in PyTorch 2.0+
or verify checksums before loading
```

### 4. Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Varies by severity
  - **Critical**: Emergency patch within 7 days
  - **High**: Patch within 30 days
  - **Medium**: Patch within 90 days
  - **Low**: Next regular release

### 5. Disclosure Policy

- We follow **coordinated disclosure**
- Security advisories will be published after a fix is available
- You will be credited (unless you prefer anonymity)
- We aim for a 90-day disclosure window

## Security Best Practices for Users

### Safe Model Loading

```python
# ✅ SAFE: Use weights_only for PyTorch 2.0+
model.load_state_dict(
    torch.load('model.pth', weights_only=True)
)

# ❌ UNSAFE: Default loading can execute arbitrary code
model.load_state_dict(
    torch.load('untrusted_model.pth')  # Dangerous!
)
```

### Verify Model Checksums

```bash
# Always verify checksums for downloaded models
sha256sum pretrained_model.pth
# Compare with published checksum
```

### Sandboxed Environments

Run untrusted models in isolated environments:

```bash
# Use Docker for isolation
docker run --rm -it \
    --gpus all \
    -v $(pwd):/workspace \
    deraining:latest \
    python scripts/inference.py --weights untrusted.pth
```

### Input Validation

When processing user-uploaded images:

```python
# Validate image before processing
def safe_load_image(path: str, max_size: int = 10 * 1024 * 1024):
    """Safely load and validate image.

    Args:
        path: Path to image file
        max_size: Maximum file size in bytes (default: 10MB)
    """
    # Check file size
    if os.path.getsize(path) > max_size:
        raise ValueError("Image too large")

    # Use safe image loading
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Invalid image file")

    # Validate dimensions
    h, w = img.shape[:2]
    if h > 4096 or w > 4096:
        raise ValueError("Image dimensions too large")

    return img
```

### Dependency Security

Keep dependencies updated:

```bash
# Check for known vulnerabilities
pip install safety
safety check

# Update packages
pip install --upgrade -r requirements.txt
```

## Known Security Considerations

### Model Weights

- **Pickle Vulnerability**: PyTorch model files (.pth) use pickle, which can execute arbitrary code
- **Mitigation**: Only load models from trusted sources, use `weights_only=True` in PyTorch 2.0+

### Input Data

- **Large Images**: Can cause memory exhaustion
- **Mitigation**: Validate image dimensions before processing

### Dependencies

- **Third-party libraries**: May contain vulnerabilities
- **Mitigation**: Regular dependency updates, security scanning

### GPU Memory

- **OOM Attacks**: Maliciously large inputs can crash GPU
- **Mitigation**: Input size validation, memory monitoring

## Security Updates

Security patches will be announced through:

- GitHub Security Advisories
- Release notes with `[SECURITY]` tag
- Updated CHANGELOG.md

## Attribution

Responsible disclosure researchers will be credited in:

- Security advisory
- Release notes
- CONTRIBUTORS.md (if desired)

## Questions?

For non-security questions about safety best practices:

- Open a [GitHub Discussion](https://github.com/dronefreak/clearview/discussions)
- Check existing security documentation

Thank you for helping keep this project secure! 🔒
