# Contributing to NagaNLP

Thank you for your interest in contributing to NagaNLP! We're excited to have you on board. This document will guide you through the process of contributing to our project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Issues](#reporting-issues)
  - [Feature Requests](#feature-requests)
  - [Bug Fixes](#bug-fixes)
  - [Documentation](#documentation)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to agnivamaiti.official@gmail.com.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment (see below)
4. Create a new branch for your changes
5. Make your changes and test them
6. Submit a pull request

## How to Contribute

### Reporting Issues

Before creating a new issue, please check if a similar issue already exists. When creating an issue, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Any relevant error messages or logs
- Screenshots if applicable

### Feature Requests

We welcome new feature suggestions! Please include:
- A clear description of the feature
- The problem it solves
- Any alternative solutions considered
- Additional context or examples

### Bug Fixes

1. Check if the bug has already been reported
2. If not, create an issue describing the bug
3. Fork the repository and create a new branch for your fix
4. Write tests that demonstrate the bug
5. Implement your fix
6. Ensure all tests pass
7. Submit a pull request with a clear description of the fix

### Documentation

Improvements to documentation are always welcome! This includes:
- Fixing typos and grammatical errors
- Adding missing docstrings
- Improving existing documentation
- Translating documentation to other languages

## Development Setup

### Prerequisites
- Python 3.8+
- pip
- git

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/AgnivaMaiti/naga-nlp.git
   cd naga-nlp
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Testing

Run the test suite:
```bash
pytest tests/
```

To run tests with coverage:
```bash
pytest --cov=naganlp tests/
```

## Code Style

We use the following tools to maintain code quality:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

Run the formatters and linters:
```bash
black .
isort .
flake8
```

## Pull Request Process

1. Ensure your fork is up to date with the main branch
2. Create a new branch for your changes
3. Make your changes following the code style guidelines
4. Add or update tests as needed
5. Update documentation if necessary
6. Run the test suite and ensure all tests pass
7. Commit your changes with a clear, descriptive message
8. Push your branch to your fork
9. Submit a pull request to the main repository

In your pull request, please include:
- A clear description of the changes
- Any related issues or pull requests
- Screenshots or gifs for UI changes
- Any breaking changes or migration steps

## Review Process

1. A maintainer will review your PR as soon as possible
2. We may request changes or additional information
3. Once approved, your PR will be merged into the main branch

## License

By contributing to NagaNLP, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
