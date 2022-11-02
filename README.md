# stabilized-ica

<p align="center">
    <img src="https://github.com/ncaptier/stabilized-ica/blob/master/docs/source/images/full_logo.png" width="400" height="400" />
</p>

[![Documentation Status](https://readthedocs.org/projects/stabilized-ica/badge/?version=latest)](https://stabilized-ica.readthedocs.io/en/latest/?badge=latest) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository proposes a python implementation for stabilized ICA decomposition algorithm. Most of the technical
details can be found in the references [1], [2],[3], [4] and [5].

Our algorithm integrates two well-known methods to solve the ICA problem :

* FastICA (implemented in scikit-learn)
* Preconditioned ICA for Real Data - Picard (implemented in [picard package](https://pierreablin.github.io/picard/))

We also propose an implementation of the Mutual Nearest Neighbors method as well as a visualization tool to draw the associated network. It is used to study the stability of the ICA components through different datasets.   

Stabilized-ica is now compatible with scikit-learn API, meaning that you can use the base class as a sklearn transformer and include it in complex ML pipelines. 

## Install

### Install from source
```
pip install git+https://github.com/ncaptier/stabilized-ica.git
```

## Experiments

We provide three jupyter notebooks for an illustration with transcriptomic data :

* [ICA decomposition with stabilized ICA](https://github.com/ncaptier/stabilized-ica/blob/master/examples/transcriptomic_ICA.ipynb)
* [Stability of ICA components accross several NSCLC cohorts](https://github.com/ncaptier/stabilized-ica/blob/master/examples/stability_study.ipynb)
* [Stabilized ICA for single-cell expression data (cell cycle)](https://github.com/ncaptier/stabilized-ica/blob/master/examples/cell_cycle_ICA.ipynb)

We provide one jupyter notebook for an illustration with EEG/MEG data :

* [Detecting artifacts and biological phenomena on MEG data with stabilized-ica](https://github.com/ncaptier/stabilized-ica/blob/master/examples/sica_MEG.ipynb)

We provide one jupyter notebook for an illustration of the integration of stabilized-ica into scikit-learn Machine learning pipelines:   

* [MNIST classification with stabilized-ica and multinomial logistic regression](https://github.com/ncaptier/stabilized-ica/blob/master/examples/MNIST_classification.ipynb)

## Data

The data set which goes with the jupyter
notebook ["ICA decomposition with stabilized ICA"](https://github.com/ncaptier/stabilized-ica/blob/master/examples/transcriptomic_ICA.ipynb)
can be found in the .zip
file [data.zip](https://github.com/ncaptier/stabilized-ica/blob/master/examples/data.zip).
Please extract locally the data set before running the notebook.

For the jupyter
notebooks ["Stability of ICA components accross several NSCLC cohorts"](https://github.com/ncaptier/stabilized-ica/blob/master/examples/stability_study.ipynb)
and ["Stabilized ICA for single-cell expression data (cell cycle)"](https://github.com/ncaptier/stabilized-ica/blob/master/examples/cell_cycle_ICA.ipynb)
please note that you will have to load the data yourself in order to run them (all the necessary links are reported on
the notebooks).

## Stabilized ICA for omics data

stabilized-ica was originally developped to deconvolute omics data into reproducible biological sources. We provide two
additional computational tools to use stabilized-ica with omics data and interpret the extacted stable sources:

* [sica-omics](https://github.com/ncaptier/sica-omics) is a Python toolbox which complements stabilized-ica for the
  analysis of omics data. In particular, it proposes annotation functions to decipher the biological meaning of the
  extracted ica sources, as well as a wrapper to adapt stabilized-ica base code to the special case
  of [Anndata](https://anndata.readthedocs.io/en/latest/) format which is popular for dealing with single-cell gene
  expression data.
* [BIODICA](https://sysbio-curie.github.io/biodica-environment/) is a computational environment for application of
  independent component analysis (ICA) to bulk and single-cell molecular profiles, interpretation of the results in
  terms of biological functions and correlation with metadata. It uses the stabilized-ica package as its computational
  core. In particular, it comes with Graphical User interface providing a no-code access to all of its functionnalities.
    ```
  If you use BIODICA in a scientific publication, we would appreciate citation to the following paper:
  
    Nicolas Captier, Jane Merlevede, Askhat Molkenov, Ainur Seisenova, Altynbek Zhubanchaliyev, Petr V Nazarov, Emmanuel Barillot, Ulykbek Kairov, Andrei Zinovyev, BIODICA: a computational environment for Independent Component Analysis of omics data, Bioinformatics, Volume 38, Issue 10, 15 May 2022, Pages 2963–2964, https://doi.org/10.1093/bioinformatics/btac204
  ```

## Examples

#### Stabilized ICA method

```python
import pandas as pd
from sica.base import StabilizedICA

df = pd.read_csv("data.csv", index_col=0)

sICA = StabilizedICA(n_components=45, n_runs=30 ,plot=True, n_jobs = -1)
sICA.fit(df)

Metagenes = pd.DataFrame(sICA.S_ , columns = df.index , index = ['metagene ' + str(i) for i in range(sICA.S_.shape[0])])
Metagenes.head()
```

#### Mutual Nearest Neighbors method

```python
from sica.mutualknn import MNNgraph

cg = MNNgraph(data = [df1 , df2 , df3] , names=['dataframe1' , 'dataframe2' , 'dataframe3'] , k=1)
cg.draw(colors = ['r', 'g' , 'b'] , spacing = 2)

cg.export_json("example.json")
```

## Acknowledgements

This package was created as a part of the PhD project of Nicolas Captier in the [Computational Systems Biology of Cancer group](http://sysbio.curie.fr) of Institut Curie.

## References

[1] "Determining the optimal number of independent components for reproducible transcriptomic data analysis" - Kairov et
al. 2017   
[2] "Assessing reproducibility of matrix factorization methods in independent transcriptomes" - Cantini et al. 2019    
[3] "Icasso: software for investigating the reliability of ICA estimates by clustering and visualization" - Himberg et
al. 2003   
[4] "Faster independent component analysis by preconditioning with Hessian approximations" - Ablin et al. 2018   
[5] "BIODICA: a computational environment for Independent Component Analysis of omics data" - Captier et al. 2022