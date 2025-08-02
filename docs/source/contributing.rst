.. _contributing:

Contributing
============

We welcome contributions to NagaNLP! Here's how you can help:

Ways to Contribute
-----------------

- Report bugs
- Fix bugs
- Add features
- Improve documentation
- Write tests
- Improve performance
- Share your ideas

Development Setup
----------------

1. Fork the repository
2. Clone your fork:
   .. code-block:: bash
      git clone https://github.com/your-username/naga-nlp.git
      cd naga-nlp
3. Create a virtual environment:
   .. code-block:: bash
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
4. Install development dependencies:
   .. code-block:: bash
      pip install -e .[dev]
5. Install pre-commit hooks:
   .. code-block:: bash
      pre-commit install

Code Style
----------

- Follow PEP 8
- Use type hints
- Write docstrings following NumPy style
- Keep lines under 88 characters
- Use Black for code formatting
- Use isort for import sorting

Testing
-------

Run the test suite:
.. code-block:: bash
   pytest tests/

Submit a Pull Request
---------------------
1. Create a new branch
2. Make your changes
3. Add tests
4. Run tests and linters
5. Update documentation
6. Submit a PR

For more details, see :ref:`CONTRIBUTING.md <contributing-md>`.
