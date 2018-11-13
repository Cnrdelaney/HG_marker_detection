.. |cluster| image:: _static/cluster_format.png

.. |markers| image:: _static/markers_format.png

.. |tsne| image:: _static/tsne_format.png

Usage
======

COMET can be executed with a single terminal command:

.. code-block:: console

   $ hgmd [-h] [-X [X]] [-L [L]] input_path output_path

* The ``-h`` flag prints a help message, explaining each argument.
* The ``-X`` and ``-L`` flags accept arguments ``X`` and ``L``. These are the bounds used in calculation of the gene expression cutoff value. The ``X`` argument specifies the minimum number of expressing cells that should be in the cluster; the cutoff value is selected such that there are at minimum ``X`` cells both in the cluster and expressing any gene of interest. Similar, the ``L`` argument is the maximum number of expressing cells across the *entire* population; the cutoff value is selected such that no greater than ``L`` cells express any gene of interest. These flags may be omitted, or used one at a time. They default to having no bound.
* The ``input_path`` argument specifies the directory where input files are found. See below for instructions on formatting these files.
* The ``output_path`` argument specifies the directory where COMET will put its output files.

For examples of COMET usage and output, see :doc:`Examples<examples>`.

Input formatting
-----------------

Before you use COMET with gene expression data, your data should be formatted into 3 files:

* ``markers.txt``: a table in CSV format. The first row of the table lists gene names, while the first column lists cell names (the cell at the very top left should be blank). Each element in the rest of the table should contain a numerical gene expression value, corresponding to the row/cell and column/gene of the element.
  
  |markers|
  
* ``tsne.txt``: a table in CSV format, of three columns without column labels. The first column is cell name (the same as those in ``markers.txt``), the second is the tSNE_1 value for the cell, and the third is the tSNE_2 value for the cell.
  
  |tsne|
  
* ``cluster.txt``: a table in CSV format, of two columns without column labels. The first column is cell name (consistent with ``markers.txt`` and ``tsne.txt``), and the second is the cluster of which the cell is a member.
  
  |cluster|
  





.. toctree::
                                                                  
