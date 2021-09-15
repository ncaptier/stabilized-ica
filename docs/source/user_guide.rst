User Guide
==========

Methods and Algorithms
----------------------

Icasso algorithm
^^^^^^^^^^^^^^^^
The main algorithm of **stabilized-ica**, `sica.base.StabilizedICA <modules/generated/sica.base.StabilizedICA.html#sica.base.StabilizedICA>`_, solves the ICA problem 
and stabilizes the resulting components through multiple runs of the ICA solver and a final aggregation step. Our implementation is mainly based on the Icasso method 
which was developped by J. Himberg and A. Hyvarinen in 2003 [1]. There are three main steps : 

1. An ICA solver is chosen and its hyperparameters are fixed. The user also sets the number of components to extract (we will call it `n_components`). Then, the solver is run `n_runs` times with different initializations. At the end, `n_components*n_runs` are extracted.  
2. The `n_components*n_runs` components are clustered into `n_components` clusters using hierarchical agglomerative clustering with average linkage criterion. The similarity metric is the absolute value of the pearson correlation coefficient between estimates of the ICA components.   
3. Finally, the centrotype of each cluster is computed and returned as a stabilized independent component. The centrotype is the point in the cluster which has the maximum sum of similarities to other points in the cluster. Besides, for each cluster, an index which quantifies its compactness and isolation is computed in order to assess the quality of the associated stabilized components (i.e its centrotype). It is computed as the difference between the average intra-cluster similarities and average extra-cluster similarities.

Two ICA solvers: FastICA and Infomax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To define ICA, we use a statistical "latent variables" model (cf. section 2 [2]). We assume that we observe n linear mixtures :math: 'x_1, ... , x_n' of latent sources :math: 's_1,...,s_n' :

..math::
    x_j = a_{j1}s_1 + a_{j2}s_2 + ... + a_{jn}s_n \quad \text{for all j}

It is convenient to use a vector-matrix notation introducing the observed random vector :math: '\boldsymbol{x}  \, \in \, \mathbb{R}^{n}', the latent random vector :math: '\boldsymbol{s}  \, \in \, \mathbb{R}^{n}' and the unknown mixing matrix :math: '\boldsymbol{A} \, \in \, \mathbb{R}^{n \times n}' :
..math::
    \boldsymbol{x} = \boldsymbol{A} \boldsymbol{s}


Assess the reproductibility via Mutual Nearest Neighbors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Omics data analysis
-------------------

Enrichment analysis for annotating components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Single-cell data
^^^^^^^^^^^^^^^^


References
----------

[1] : J. Himberg and A. Hyvarinen, "Icasso: software for investigating the reliability of ICA estimates by clustering and visualization," 2003 IEEE XIII Workshop on Neural Networks for Signal Processing (IEEE Cat. No.03TH8718), 2003, pp. 259-268, doi: 10.1109/NNSP.2003.1318025.