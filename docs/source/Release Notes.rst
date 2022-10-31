Release Notes
=============

v2.0.0
------
Stabilized-ica is now compatible with scikit-learn API, meaning that you can use the base class as a sklearn transformer and include it in complex ML pipelines (see `this tutorial <https://github.com/ncaptier/stabilized-ica/blob/master/examples/MNIST_classification.ipynb>`_ for an illustration).

**sica.annotate** and **sica.singlecell** modules have been removed from **stabilized-ica** and integrated into a complementary python toolbox called `sica-omics <https://github.com/ncaptier/sica-omics>`_ . **stabilized-ica** no longer contains dependencies specific to omics data analysis.

**Fixed bugs:**

    * `svd_solver` default value (parameter of sica._whitening.whitening) was changed from `full` (i.e full svd decomposition) to `auto` (i.e selection of most efficient solver for the size of the given dataset). This significantly speeds up the computation for large datasets.

**New features:**

    * **sica.base.MSTD** has new `fun` and `algorithm` parameters so that the user can specify the ICA algorithm and the non-linearity function to use (for the previous version only `algorithm = fastica_par` and `fun = 'logcosh'` were available).