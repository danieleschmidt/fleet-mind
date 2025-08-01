# Contributing to Fleet-Mind

We love your input! We want to make contributing to Fleet-Mind as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/fleet-mind
cd fleet-mind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
black .
isort .
flake8 .
mypy fleet_mind/
```

### Code Style

- We use [Black](https://black.readthedocs.io/) for code formatting
- We use [isort](https://pycqa.github.io/isort/) for import sorting
- We use [flake8](https://flake8.pycqa.org/) for linting
- We use [mypy](https://mypy.readthedocs.io/) for type checking
- We follow [Google Style](https://google.github.io/styleguide/pyguide.html) for docstrings

### Testing

- Write tests for new functionality
- Maintain or improve test coverage
- Use pytest for testing
- Use pytest-asyncio for async tests
- Mock external dependencies

### Documentation

- Update README.md if needed
- Add docstrings to new functions/classes
- Update API documentation
- Add examples for new features

## Security Considerations

Fleet-Mind coordinates drone swarms, so security is paramount:

- Never commit API keys, credentials, or secrets
- Validate all inputs, especially from network sources
- Use secure communication protocols
- Follow principle of least privilege
- Report security vulnerabilities privately to security@terragon.ai

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issue tracker](https://github.com/terragon-labs/fleet-mind/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/terragon-labs/fleet-mind/issues/new).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## License

By contributing, you agree that your contributions will be licensed under its MIT License.

## References

This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md).