**Welcome to the stabilized-ica documentation !** We hope it will guide you through the use of our package for any of your project.

If you have any question about stabilized-ica, or if you want to report an issue, please do not hesitate to `contact us <about.html#contact-us>`_ or raise an issue on `GitHub <https://github.com/ncaptier/stabilized-ica/issues>`_.

What is stabilized-ica ?
------------------------
**stabilized-ica** proposes a python implementation of a stabilized algorithm for Independent Component Analysis (ICA). ICA aims to linearly decompose a multivariate signal into statistically independent components (also called sources or latent variables). 
A major problem is that independent components are usually not unique. Indeed, most ICA solvers converge towards different local minima and thus give different results when run multiple times. This is particularly striking when it is applied to real data 
that do not necessarily follow the ICA model and whose limited sample size induces statistical errors. These inconsistencies negatively impact the reproductibilty of the conclusions that one may draw from the ICA decomposition. 

In 2003, J. Himberg and A. Hyvarinen [1] tackled this problem with a stabilization process. Not only their method derives more reliable independent components but it also gives a stability argument to assess their significance. The **stabilized-ica** 
package is built around our own python implementation of their method. It also provides a bunch of tools to visualize, interpret and assess the significance and reproductibility of the results. Our implementation works for any numerical multivariate data set. 
However, some tools are specifically dedicated to the application of this stabilized process to the deconvolution of omics data with ICA (ex : bulk or single-cell RNA-seq data).


[1] : J. Himberg and A. Hyvarinen, "Icasso: software for investigating the reliability of ICA estimates by clustering and visualization," 2003 IEEE XIII Workshop on Neural Networks for Signal Processing (IEEE Cat. No.03TH8718), 2003, pp. 259-268, doi: 10.1109/NNSP.2003.1318025.


