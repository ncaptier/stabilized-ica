# Stabilized_ICA

This repository proposes a python implementation for stabilized ICA decomposition algorithm. Most of the technical details can be found in the references [1], [2] and [3]

## Experiments

We provide a jupyter notebook for an illustration with transcriptomic data:
* [ICA decomposition with stabilized ICA](transcriptomic_ICA.ipynb)

## Requirements

To run this algorithm as well as the jupyter notebook, one will need the following python packages:
* numpy
* matplotlib.pyplot
* pandas
* scikit-learn
* tdqm 

## References

[1] "Determining the optimal number of independent components for reproducible transcriptomic data analysis" - Kairov et al. 2017   
[2] "Assessing reproducibility of matrix factorization methods in independent transcriptomes" - Cantini et al. 2019 
[3] "Icasso: software for investigating the reliability of ICA estimates by clustering and visualization" - Himberg et al. 2003