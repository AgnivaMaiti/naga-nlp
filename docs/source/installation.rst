.. _installation:

Installation
============

Using pip
--------

.. code-block:: bash

    pip install naganlp

From source
-----------

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/your-username/naga-nlp.git
       cd naga-nlp

2. Install with pip in development mode:

   .. code-block:: bash

       pip install -e .[dev]

Dependencies
------------

- Python 3.8+
- PyTorch
- Transformers
- NLTK
- Other dependencies listed in requirements.txt

Verifying Installation
---------------------

After installation, you can verify that NagaNLP is properly installed by running:

.. code-block:: python

    import naganlp
    print(naganlp.__version__)

This should print the version of NagaNLP that you installed.
