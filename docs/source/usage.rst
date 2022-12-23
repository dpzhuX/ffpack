Usage
=====

.. _environment:

Environment
------------

We encourage to use conda for environment management but it is not necessary.

To create an environment with a specific Python version,

.. code-block:: bash

   $ conda create -n ffpackEnv python=3.9
   $ conda activate ffpackEnv

Installation
------------

To use FFPack, install it using pip,

.. code-block:: bash

   $ pip install ffpack


Develop version
---------------

To try the develop version, clone the source code from the GitHub,

.. code-block:: bash

   $ git clone https://github.com/dpzhuX/ffpack.git

Then install it with `pip`,

.. code-block:: bash

   $ pip install -e .

We recommend to run the develop version in a seperate environment.