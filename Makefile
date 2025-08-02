.PHONY: test test-cov lint type-check format install-dev clean

# Install development dependencies
install-dev:
	pip install -e .
	pip install -r requirements-test.txt

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=naganlp --cov-report=term-missing --cov-report=xml

# Run linter
lint:
	flake8 naganlp/

# Run type checking
type-check:
	mypy naganlp/

# Format code
format:
	black naganlp/ tests/
	isort naganlp/ tests/

# Clean up build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*~' -delete
	find . -type f -name '*.py[co]' -delete
