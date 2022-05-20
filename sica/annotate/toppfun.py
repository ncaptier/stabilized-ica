import numpy as np
import requests
import json
import pandas as pd
import warnings
from ._utils import convert_to_entrez, get_top_genes, check_data, check_params

from typing import Union, Optional, NoReturn, List, Tuple


class ToppFunAnalysis(object):
    """ Provide tools for running Toppfun enrichment analysis for different metagenes.

    Parameters
    ----------
    data : 
        If ``pre_selected = False`` : pandas.DataFrame , shape (n_metagenes , n_genes) or pandas.Series, shape (n_genes)
                The column names (or the index keys for a serie) should be valid gene IDs.
        If ``pre_selected = True`` : pandas.Series , shape (n_metagenes)
                For each metagene the serie contains a list of the IDs of the extreme expressed
                genes.

    convert_ids : boolean, optional.
        If True gene ids will be converted to Entrez IDs. ToppFunAnalysis conversion tool can handle Entrez gene ids,
        Ensemble gene ids, NCBI RefSeq ids, official gene symbols, HUGO Gene Nomenclature, UniProt ids, Affymetrix
        probeset ids, and a mix of everything.
        If False ToppFunAnalysis will assume that the inputs are already Entrez IDs. No conversion will be performed.
        The default is True.
        
    pre_selected : boolean, optional.
        Indicate whether the extreme genes have already been selected (see above).
        The default is False.
        
    threshold : numeric or array-like of two numerics, optional
        See sica.annotate._utils.get_top_genes. The default is 3.
        
    method : {'quantile' , 'std'}, optional
        See sica.annotate._utils.get_top_genes. The default is 'std'.
        
    tail : {'left' , 'right' , 'both' , 'heaviest'}, optional
        See sica.annotate._utils.get_top_genes. The default is 'heaviest'.
    
    Attributes
    ----------    
    top_genes_ : pandas.DataFrame, shape (n_metagenes , 3)
        For each metagene the 'inputs' column contains a list of the IDs of the extreme expressed
        genes.
    
    References
    ----------
    For more details, please refer to the ToppGene API (see https://toppgene.cchmc.org/API/enrich. ).
    
    Examples
    --------
    >>> from sica.annotate import toppfun
    >>> annotations = toppfun.ToppFunAnalysis(data = Metagenes)
    >>> metagene7_annot = annotations.get_analysis(metagene = 'metagene 7')
    >>> metagene7_annot.head()
    """

    def __init__(
            self,
            data: Union[pd.DataFrame, pd.Series],
            convert_ids: Optional[bool] = True,
            pre_selected: Optional[bool] = False,
            threshold: Optional[Union[
                int, float, np.ndarray, List[Union[float, int]], Tuple[Union[float, int], Union[float, int]]]] = 3,
            method: Optional[str] = "std",
            tail: Optional[str] = "heaviest",
    ) -> NoReturn:

        # Check data
        data = check_data(data, pre_selected)

        self.convert_ids = convert_ids
        if not self.convert_ids:
            warnings.warn(
                "If convert_ids is False ToppFunAnalysis will assume that the inputs are already Entrez gene "
                "IDs. No conversion will be performed. "
            )

        # Initialization of selt.top_genes_ attribute
        self.top_genes_ = pd.DataFrame(
            {"inputs": None, "entrezgene": None, "notfound": None}, index=data.index
        )
        if pre_selected:
            self.top_genes_["inputs"] = data.copy()
        else:
            threshold = check_params(threshold, method, tail)
            self.top_genes_["inputs"] = data.apply(
                get_top_genes, threshold=threshold, method=method, tail=tail, axis=1
            )

    def convert_metagenes(self, idx: Union[str, object, list]) -> None:
        """ Convert the IDs of the most expressed genes contained in ``top_genes_``.
        
        Parameters
        ----------
        idx : {"all" , object , list of objects}
        
            If ``idx = "all"`` all the metagenes will be converted. 
            
            Otherwise, only the metagenes associated with ``idx`` will be converted. In that case, ``idx`` must
            correspond to valid indexes of the input data.
 
        Returns
        -------
        None
        
        """

        # Define the function that will be applied to the rows of self.top_genes_ dataframe
        def fun(row):
            if row["entrezgene"] is None:
                return convert_to_entrez(row["inputs"])[:2]
            else:
                return row["entrezgene"], row["notfound"]

        # Apply func to the rows of self.top_genes_ associated with the metagenes parameter
        if idx == "all":

            warnings.warn(
                "idx = 'all' : this operation can take quite some time depending on the number of metagenes and the "
                "number of most expressed genes. "
            )
            self.top_genes_[["entrezgene", "notfound"]] = self.top_genes_.apply(
                fun, axis=1, result_type="expand"
            )

        elif isinstance(idx, list):

            warnings.warn(
                "metagenes is a list : this operation can take quite some time depending on the number of metagenes "
                "and the number of most expressed genes. "
            )
            self.top_genes_.loc[idx, ["entrezgene", "notfound"]] = (
                self.top_genes_.loc[idx].apply(fun, axis=1, result_type="expand")
            ).values

        else:

            self.top_genes_.loc[idx, ["entrezgene", "notfound"]] = np.array(
                fun(self.top_genes_.loc[idx]), dtype="object"
            )

        return

    def get_analysis(
            self,
            metagene: object,
            type_list: Optional[List[str]] = None,
            p_value: Optional[float] = 0.05,
            min_entities: Optional[int] = 10,
            max_entities: Optional[int] = 500,
            maxres: Optional[int] = 10,
            correct: Optional[str] = "FDR",
    ) -> pd.DataFrame:
        """ Return the ToppFun enrichment analysis of a given metagene.
        
        Parameters
        ----------
        metagene : object
            It must correspond to a valid index of the input data.
            
        type_list: list of strings, optional
            List of features to perform enrichment tests. If None, all the available features
            will be used (see sica.annotate.toppfun._get_analysis). The default is None.
        
        p_value: float in (0 , 1), optional
            P-value maximal threshold to accept enrichments. The default is 0.05
            
        min_entities: int, optional
            Minimal number of gene hits to accept in feature categories. The default is 10.
    
        max_entities: int, optional
            Maximal number of gene hits to accept in feature categories. The default is 500.
        
        maxres: int, optional
            Number of top enrichments to show for each feature. The default is 10.
        
        correct: str {'none', 'FDR', 'Bonferroni'}, optional
            P-value correction methods. FDR refers to the Benjamini and Hochberg method.
            The default is 'FDR'.
            
        Returns
        -------
        pandas.DataFrame
            Results of the Toppfun enrichment analysis for the given metagene.

        """

        if self.convert_ids:
            self.convert_metagenes(idx=metagene)
            entrez_dict = {
                "Genes": [int(k) for k in self.top_genes_.loc[metagene, "entrezgene"]]
            }
        else:
            entrez_dict = {
                "Genes": [int(k) for k in self.top_genes_.loc[metagene, "inputs"]]
            }

        results = []
        annotations = _get_analysis(
            entrez_dict, type_list, p_value, min_entities, max_entities, maxres, correct
        ).json()["Annotations"]

        for element in annotations:
            gene_symbol_list = [gene["Symbol"] for gene in element["Genes"]]
            element["Gene_Symbol"] = ",".join(gene_symbol_list)
            element.pop("Genes", None)
            results.append(element)

        return pd.DataFrame(results)


def _get_analysis(
        entrez_dict: dict,
        type_list: List[str],
        p_value: float,
        min_entities: int,
        max_entities: int,
        maxres: int,
        correct: str
) -> requests.models.Response:
    """ Call TOPPGENE API to detect functional enrichments of a gene list (https://toppgene.cchmc.org/API/enrich.)
    
    Parameters
    ----------
    entrez_dict: dict
        Dictionary { "Genes" : list of genes in Entrez IDs}.

    Returns
    -------
    response: requests.models.Response
        Response from "https://toppgene.cchmc.org/API/enrich" API call.
    
    """
    if type_list is None:
        type_list = [
            "GeneOntologyMolecularFunction",
            "GeneOntologyBiologicalProcess",
            "GeneOntologyCellularComponent",
            "HumanPheno",
            "MousePheno",
            "Domain",
            "Pathway",
            "Pubmed",
            "Interaction",
            "Cytoband",
            "TFBS",
            "GeneFamily",
            "Coexpression",
            "CoexpressionAtlas",
            "GeneFamily",
            "Computational",
            "MicroRNA",
            "Drug",
            "Disease",
        ]

    url = "https://toppgene.cchmc.org/API/enrich"
    headers = {"Content-Type": "text/json"}
    parameters = {"Categories": []}
    for type_id in type_list:
        parameters["Categories"].append(
            {
                "Type": type_id,
                "Pvalue": p_value,
                "MinGenes": min_entities,
                "MaxGenes": max_entities,
                "MaxResults": maxres,
                "Correction": correct,
            }
        )
    data_all = {"Genes": entrez_dict["Genes"], "Categories": parameters["Categories"]}
    response = requests.post(url, headers=headers, data=json.dumps(data_all))
    if response.status_code == 200:
        print("Enrichment analysis success!")
    else:
        print(
            "Something went wrong during enrichment... Status code:",
            response.status_code,
        )
    return response
