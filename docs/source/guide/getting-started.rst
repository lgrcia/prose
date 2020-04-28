.. _getting-started:

Installation
============

|prose| runs more safely in its own `virtual environment`_ and is tested on Python 3.6.

.. _virtual environment: https://docs.python.org/3/tutorial/venv.html

OSX
---

create your virtualenv_

.. _virtualenv: https://docs.python.org/3/tutorial/venv.html and activate it

.. code-block:: sh

    python3.6 -m venv prose_env
    source prose_env/bin/activate.bin

Then to locally install |prose|

.. code-block:: sh

    git clone https://github.com/LionelGarcia/prose.git

    cd prose_env
    python3.6 -m pip install -e ../prose


Linux
-----

.. note::

    Linux users will need `sudo apt-get install python-virtualenv`


.. code-block:: sh

    virtualenv -p python3.6 prose_env
    source prose_env/bin/activate

Then to locally install |prose|

.. code-block:: sh

    git clone https://github.com/LionelGarcia/prose.git

    cd prose_env
    python3.6 -m pip install -e ../prose


You can now access the ``prose`` command. To quit the virtual environment run ``deactivate``.

