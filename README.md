# stabilized-ica
[![PyPI version](https://badge.fury.io/py/stabilized-ica.svg)](https://badge.fury.io/py/stabilized-ica)
[![Documentation Status](https://readthedocs.org/projects/stabilized-ica/badge/?version=latest)](https://stabilized-ica.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://pepy.tech/badge/stabilized-ica)](https://pepy.tech/project/stabilized-ica)

<p align="center">
    <img src="https://github.com/ncaptier/stabilized-ica/blob/master/docs/source/images/full_logo.png" width="400" height="400" />
</p>

This repository proposes a python implementation for stabilized ICA decomposition algorithm. Most of the technical details can be found in the references [1], [2] and [3].    

Our algorithm integrates two well-known methods to solve the ICA problem :
* FastICA (implemented in scikit-learn)
* Infomax and extensions (implemented in [picard package](https://pierreablin.github.io/picard/))     
   
We also propose an implementation of the Mutual Nearest Neighbors method as well as a visualization tool to draw the associated network. It is used to study the stability of the ICA components through different datasets.   

Finally, we propose an adaptation of our algorithm for the special case of [AnnData](https://anndata.readthedocs.io/en/latest/anndata.AnnData.html) format. Our module sica.singlecell is modeled after the [scanpy package](https://scanpy.readthedocs.io/en/stable/) that deals with single-cell gene expression data.

**Note :** This project was originally developped to provide a reproducible an biologically meaningful python algorithm for the deconvolution of "omics" data. Several modules like sica.singlecell or sica.annotate are specifically dedicated to this task. However, the sica.base module which contains the main computations for the stabilization of ICA or the sica.mutualknn module which draws a mutual k-nearest neighbors graph to assess the reproducibility of the ICA components could work perfectly with any other data type.   

### Documentation

<https://stabilized-ica.readthedocs.io/en/latest/>

### Install the latest stable version with PyPi
```
pip install stabilized-ica
```

### Install from source
```
pip install git+https://github.com/ncaptier/stabilized-ica
```

## Experiments

We provide three jupyter notebooks for an illustration with transcriptomic data :
* [ICA decomposition with stabilized ICA](https://github.com/ncaptier/stabilized-ica/blob/master/examples/transcriptomic_ICA.ipynb)
* [Stability of ICA components accross several NSCLC cohorts](https://github.com/ncaptier/stabilized-ica/blob/master/examples/stability_study.ipynb)
* [Stabilized ICA for single-cell expression data (cell cycle)](https://github.com/ncaptier/stabilized-ica/blob/master/examples/cell_cycle_ICA.ipynb)   

We provide one jupyter notebook for an illustration with EEG/MEG data :
* [Detecting artifacts and biological phenomena on MEG data with stabilized-ica](https://github.com/ncaptier/stabilized-ica/blob/master/examples/sica_MEG.ipynb)

## Data

The data set which goes with the jupyter notebook ["ICA decomposition with stabilized ICA"](https://github.com/ncaptier/stabilized-ica/blob/master/examples/transcriptomic_ICA.ipynb) can be found in the .zip file [data.zip](https://github.com/ncaptier/stabilized-ica/blob/master/examples/data.zip). Please extract locally the data set before running the notebook.   

For the jupyter notebooks ["Stability of ICA components accross several NSCLC cohorts"](https://github.com/ncaptier/stabilized-ica/blob/master/examples/stability_study.ipynb) and ["Stabilized ICA for single-cell expression data (cell cycle)"](https://github.com/ncaptier/stabilized-ica/blob/master/examples/cell_cycle_ICA.ipynb) please note that you will have to load the data yourself in order to run them (all the necessary links are reported on the notebooks).   

## Examples 

#### Stabilized ICA method

```python
import pandas as pd
from sica.base import StabilizedICA

df = pd.read_csv("data.csv" , index_col = 0).transpose()

sICA = StabilizedICA(n_components = 45 , max_iter = 2000 , n_jobs = -1)
sICA.fit(df , n_runs = 30 , plot = True , normalize = True)

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

#### Application to single-cell data

```python
import scanpy
from sica.singlecell import ica

adata = scanpy.read_h5ad('GSE90860_3.h5ad')
adata.X -= adata.X.mean(axis =0)

ica(adata , observations = 'genes' , n_components = 30 , n_runs = 100)
```

## Acknowledgements

This package was created as a part of the PhD project of Nicolas Captier in the [Computational Systems Biology of Cancer group](http://sysbio.curie.fr) of Institut Curie.

## References

[1] "Determining the optimal number of independent components for reproducible transcriptomic data analysis" - Kairov et al. 2017   
[2] "Assessing reproducibility of matrix factorization methods in independent transcriptomes" - Cantini et al. 2019    
[3] "Icasso: software for investigating the reliability of ICA estimates by clustering and visualization" - Himberg et al. 2003
