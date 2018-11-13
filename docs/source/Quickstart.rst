.. _Github: https://github.com/aaronyaosmith/HG_marker_detection

.. _Python: https://www.python.org/downloads/

.. _website: http://www.cometsc.com/login

Installation/Quickstart
========================

To skip installation on your own machine and run COMET on your data on COMET's servers, check out our website_!

To install COMET's Python implementation on your own machine, you should first have Python_ version 3 installed.

Once that is done, clone the COMET source from our Github_, navigate to the cloned directory, then install it using Python's pip tool with the following three respective commands:

.. code-block:: console

   $ git clone https://github.com/aaronyaosmith/HG_marker_detection.git
   $ cd HG_marker_detection/
   $ pip install .

Now, run COMET on your data. Give the directory where your data is located as the first argument, and your desired output directory as your second argument.

In this alpha stage of COMET's development, your data must be formatted into three files in your input directory: 'markers.txt', 'cluster.txt', and 'tsne.txt'. See the :doc:`Manual<manual>` for more information.

In this example, we have our data located in ``input/``. ``output/`` is the directory where COMET's output will be stored.

.. code-block:: console
		
   $ hgmd input/ output/

After this command is entered, COMET will run in the terminal, processing your data. See :doc:`Examples<examples>` for details on what this should look like.

.. toctree::
