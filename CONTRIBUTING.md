# Contributing to NagaNLP

Thank you for your interest in contributing to NagaNLP! We welcome contributions from the community.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## How to Contribute

### Reporting Issues
- Check if the issue has already been reported
- Provide a clear and descriptive title
- Include steps to reproduce the issue
- Describe the expected and actual behavior
- Include any relevant error messages

### Making Changes
1. Fork the repository
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number-short-description
   ```
3. Make your changes
4. Add tests for your changes
5. Run the test suite and ensure all tests pass
6. Commit your changes with a descriptive message
7. Push your branch and create a pull request

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/naga-nlp.git
   cd naga-nlp
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Running Tests

Run the test suite:
```bash
pytest tests/
```

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting

Run the formatters and linters:
```bash
black .
isort .
flake8
```

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, including new environment variables, exposed ports, useful file locations, and container parameters.
3. Increase the version number in `naganlp/__init__.py` and the README.md to the new version that this Pull Request would represent.
4. The PR will be reviewed by the maintainers. We may suggest changes or improvements.

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
