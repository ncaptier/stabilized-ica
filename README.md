# stabilized-ica

This repository proposes a python implementation for stabilized ICA decomposition algorithm. Most of the technical details can be found in the references [1], [2] and [3].    

Our algorithm integrates two well-known methods to solve the ICA problem :
* FastICA (implemented in scikit-learn)
* Infomax and extensions (implemented in [picard package](https://pierreablin.github.io/picard/))     

We propose a brief review of these two methods in [supplementary_material.pdf](documentation/supplementary_material.pdf) so that the user can knowingly decide which of them is best suited for his problem.   
   
We also propose an implementation of the Mutual Nearest Neighbors method as well as a visualization tool to draw the associated network. It is used to study the stability of the ICA components through different datasets.   

Finally, we propose an adaptation of our algorithm for the special case of [AnnData](https://anndata.readthedocs.io/en/latest/anndata.AnnData.html) format. Our module sica.singlecell is modeled after the [scanpy package](https://scanpy.readthedocs.io/en/stable/) that deals with single-cell gene expression data.

## Installation

```
$ pip install git+https://github.com/ncaptier/stabilized-ica#egg=sica
```

## Experiments

We provide two jupyter notebooks for an illustration with transcriptomic data :
* [ICA decomposition with stabilized ICA](examples/transcriptomic_ICA.ipynb)
* [Stability of ICA components accross several NSCLC cohorts](examples/stability_study.ipynb)

## Data

The data set which goes with the jupyter notebook ["ICA decomposition with stabilized ICA"](transcriptomic_ICA.ipynb) can be found in the .zip file [examples/data.zip](data.zip). Please extract locally the data set before running the notebook.   

For the jupyter notebook ["Stability of ICA components accross several NSCLC cohorts"](examples/stability_study.ipynb) please note that you will have to load the data yourself in order to run it (all the necessary links are reported on the notebook).

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

This package was created as a part of Master internship by Nicolas Captier in the [Computational Systems Biology of Cancer group](http://sysbio.curie.fr) of Institut Curie.

## References

[1] "Determining the optimal number of independent components for reproducible transcriptomic data analysis" - Kairov et al. 2017   
[2] "Assessing reproducibility of matrix factorization methods in independent transcriptomes" - Cantini et al. 2019    
[3] "Icasso: software for investigating the reliability of ICA estimates by clustering and visualization" - Himberg et al. 2003
