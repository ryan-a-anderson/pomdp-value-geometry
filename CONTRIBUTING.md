# Contributing to POMDP Value Geometry

Thank you for your interest in contributing to this project! This document provides guidelines for contributing code, reporting issues, and suggesting improvements.

## Code of Conduct

Be respectful and constructive in all interactions. This is an academic research project, and we welcome contributions from researchers and practitioners in reinforcement learning, optimization, and related fields.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (Python version, OS, etc.)
- Minimal code example demonstrating the issue

### Suggesting Enhancements

For feature requests or improvements:
- Check if the suggestion already exists in issues
- Clearly describe the enhancement and its motivation
- Provide examples of how it would be used
- Consider implementation implications

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** following the style guidelines below
4. **Add tests** if applicable
5. **Update documentation** as needed
6. **Commit with clear messages**: `git commit -m "Add feature X"`
7. **Push to your fork**: `git push origin feature/your-feature-name`
8. **Open a Pull Request** with a clear description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/pomdp-value-geometry.git
cd pomdp-value-geometry

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (if added)
pip install -r requirements-dev.txt
```

## Code Style Guidelines

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters (relaxed from PEP 8's 79 for readability)
- Use descriptive variable names

```python
# Good
def optimize_policy(initial_distribution, learning_rate=0.1):
    """Optimize policy using projected gradient ascent."""
    ...

# Avoid
def opt_pol(rho, lr=0.1):
    ...
```

### Documentation

All public functions should have docstrings:

```python
def solve_v(self, p):
    """
    Solve for value function given policy parameters.

    Args:
        p (np.ndarray): Policy parameters of shape (n_obs,) where
            p[k] = π(a=1|o=k) for binary actions.

    Returns:
        np.ndarray: Value function v of shape (n_states,).

    Raises:
        np.linalg.LinAlgError: If the Bellman equation is singular.

    Example:
        >>> pomdp = POMDPAnalyzer(P, R, Beta, gamma=0.9)
        >>> policy = np.array([0.5, 0.8])
        >>> value = pomdp.solve_v(policy)
    """
    ...
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `POMDPAnalyzer`)
- **Functions/methods**: `snake_case` (e.g., `optimize_projected_gradient`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_ITERATIONS`)
- **Private methods**: `_leading_underscore` (e.g., `_compute_A`)

### Mathematical Notation

Maintain consistency with paper notation:
- `rho` or `ρ` for initial state distribution
- `p` for policy parameters (π in paper)
- `v` for value function (V in paper)
- `alpha`, `beta` for transition/observation kernels (α, β in paper)
- `gamma` or `γ` for discount factor

## Testing Guidelines

### Adding Tests

If adding new functionality, include tests:

```python
# tests/test_pomdp_analyzer.py
import numpy as np
from pomdp_optim_dynamics import POMDPAnalyzer

def test_solve_v():
    """Test value function solution."""
    P = np.array([[[0.9, 0.1], [0.2, 0.8]],
                  [[0.7, 0.3], [0.3, 0.7]]])
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    Beta = np.eye(2)

    pomdp = POMDPAnalyzer(P, R, Beta, gamma=0.9)
    policy = np.array([0.5, 0.5])
    value = pomdp.solve_v(policy)

    assert value.shape == (2,)
    assert np.all(np.isfinite(value))
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_pomdp_analyzer.py

# Run with coverage
pytest --cov=. --cov-report=html
```

## Commit Message Guidelines

Write clear, concise commit messages:

```
Short summary (50 chars or less)

More detailed explanation if necessary. Wrap at 72 characters.
Explain what changed and why, not how (code shows how).

- Bullet points are okay
- Use present tense: "Add feature" not "Added feature"
- Reference issues: "Fixes #123"
```

### Examples

```
Add gradient clipping to optimization

Prevents exploding gradients in high-noise POMDPs by clipping
gradient norm to a maximum value.

Fixes #42
```

```
Optimize feasibility checking computation

Replace nested loops with vectorized operations, reducing
runtime from O(n³) to O(n²). Achieves 10x speedup on
typical problems.
```

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass (`pytest`)
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed? What problem does it solve?

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
How was this tested? Include test cases or examples.

## Related Issues
Fixes #123
Related to #456

## Screenshots (if applicable)
Add screenshots for visualization changes
```

## Documentation Contributions

### Improving Documentation

Documentation improvements are highly valued:
- Fix typos or unclear explanations
- Add usage examples
- Improve docstrings
- Create tutorials or guides

### Documentation Style

- Use clear, concise language
- Provide concrete examples
- Link to relevant sections of the paper
- Include mathematical notation where helpful

## Questions?

If you have questions about contributing:
- Open an issue with the "question" label
- Email the maintainer
- Check existing issues and discussions

## Recognition

Contributors will be acknowledged in:
- README.md Contributors section
- Paper acknowledgments (for substantial contributions)
- Git commit history

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
