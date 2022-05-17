import numpy as np
import pandas as pd
from reactome2py import analysis
import warnings
import webbrowser
from ._utils import convert_to_entrez, get_top_genes, check_data, check_params


class ReactomeAnalysis(object):
    """ Provide tools for running reactome enrichment analysis for different metagenes.
    
    Parameters
    ----------
    data : 
        If ``pre_selected = False`` : pandas.DataFrame , shape (n_metagenes , n_genes) or pandas.Series, shape (n_genes)
                The column names (or the index keys for a serie) should be valid gene IDs.
        If ``pre_selected = True`` : pandas.Series , shape (n_metagenes)
                For each metagene the serie contains a list of the IDs of the extreme expressed
                genes.
                
    convert_ids : boolean, optional.
        If True gene ids will be converted to Entrez gene ids. If False, gene ids should be valid ids for Reactome
        analysis (e.g. HUGO gene symbols, EntrezGene , Uniprot ...).
        See https://reactome.org/userguide/analysis for more details. The default is True.

    pre_selected : boolean , optional.
        Indicate whether the extreme genes have already been selected (see above).
        The default is False.
        
    threshold : numeric or array-like of two numerics , optional
        See sica.annotate._utils.get_top_genes. The default is 3.
        
    method : {'quantile' , 'std'} , optional
        See sica.annotate._utils.get_top_genes. The default is 'std'.
        
    tail : {'left' , 'right' , 'both' , 'heaviest'} , optional
        See sica.annotate._utils.get_top_genes. The default is 'heaviest'.
    
    Attributes
    ----------    
    top_genes_ : pandas.DataFrame, shape (n_metagenes , 3)
        For each metagene the 'inputs' column contains a list of the IDs of the extreme expressed
        genes.
        
    References
    ----------
    Please refer to the package reactome2py for more details (see https://github.com/reactome/reactome2py) .
    
    If you want to better understand how reactome works, please see https://reactome.org/userguide .
    
    Examples
    --------
    >>> from sica.annotate import reactome
    >>> annotations = reactome.ReactomeAnalysis(data = Metagenes)
    >>> metagene7_annot = annotations.get_analysis(metagene = 'metagene 7')
    >>> metagene7_annot.head()
    """

    def __init__(
            self,
            data,
            convert_ids=True,
            pre_selected=False,
            threshold=3,
            method="std",
            tail="heaviest",
    ):

        # Check data
        check_data(data, pre_selected)

        self.convert_ids = convert_ids
        if not self.convert_ids:
            warnings.warn(
                "If convert_ids is False ReactomeAnalysis will assume that the inputs are valid gene ids for "
                "Reactome analysis ((e.g HUGO gene symbols, EntrezGene , Uniprot ...). No conversion will be performed. "
                "See https://reactome.org/userguide/analysis for more details. "
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

        # Initialization of self.tokens dictionary. Keep tokens once the reactome analysis has been performed
        # to avoid repeating the whole analysis when calling get_analysis or open_analysis a second time.
        self.tokens = {ind: None for ind in data.index}

    def convert_metagenes(self, idx):
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

        # Apply func to the rows of self.top_genes_ associated with idx
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
                "idx is a list : this operation can take quite some time depending on the number of metagenes and the "
                "number of most expressed genes. "
            )
            self.top_genes_.loc[idx, ["entrezgene", "notfound"]] = (
                self.top_genes_.loc[idx].apply(fun, axis=1, result_type="expand")
            ).values

        else:

            self.top_genes_.loc[idx, ["entrezgene", "notfound"]] = np.array(
                fun(self.top_genes_.loc[idx]), dtype="object"
            )

        return

    def open_full_analysis(self, metagene):
        """ Browse the analysis for the given metagene in reactome web portal.
        
        Parameters
        ----------
        metagene : object
            It must correspond to a valid index of the input data.

        Returns
        -------
        None.

        """
        if self.tokens[metagene] is not None:
            token = self.tokens[metagene]
        else:
            if self.convert_ids:
                self.convert_metagenes(idx=metagene)
                token = _get_token(self.top_genes_.loc[metagene, "entrezgene"])
            else:
                token = _get_token(self.top_genes_.loc[metagene, "inputs"])
            self.tokens[metagene] = token

        url = "https://reactome.org/PathwayBrowser/#/DTAB=AN&ANALYSIS=" + token
        webbrowser.open(url)
        return

    def get_analysis(
            self,
            metagene,
            species="Homo sapiens",
            sort_by="Entities FDR",
            ascending=True,
            p_value=0.05,
            min_entities=10,
            max_entities=500,
    ):
        """ Return the reactome enrichment analysis of a given metagene.
        
        Parameters
        ----------
        metagene : object
            It must correspond to a valid index of the input data.

        species : string or list of string, optional
            List of species to filter the result. The default is 'Homo sapiens'.
            
        sort_by : {None , '#Entities found' , '#Entities total' , 'Entities ratio' , 'Entities pValue' , 'Entities FDR ' , '#Reactions found' , '#Reactions total' , 'Reactions ratio'}, optional
            How to sort the result. The default is 'Entities FDR'.
        
        ascending : boolean, optional
            Sort ascending vs. descending if ``sort_by is not None``.The default is True.
        
        p_value : float in (0 , 1), optional
            Only hit pathway with pValue equals or below ``p_value`` will be returned. The default is 0.05.
            
        min_entities : int >= 0, optional
             Minimum number of contained entities per pathway. The default is 10.
            
        max_entities : int > 0, optional
             Maximum number of contained entities per pathway. The default is 500.

        Returns
        -------
        pandas.DataFrame
            Results of the reactome enrichment analysis for the given metagene.

        """
        if self.tokens[metagene] is not None:
            token = self.tokens[metagene]
        else:
            if self.convert_ids:
                self.convert_metagenes(idx=metagene)
                token = _get_token(self.top_genes_.loc[metagene, "entrezgene"])
            else:
                token = _get_token(self.top_genes_.loc[metagene, "inputs"])
            self.tokens[metagene] = token

        df = analysis.pathway2df(token)
        df.iloc[:, 2:10] = df.iloc[:, 2:10].apply(pd.to_numeric)

        mask = pd.Series(True, index=df.index)
        if species is not None:
            mask = mask & (df["Species name"] == species)
        if p_value is not None:
            mask = mask & (df["Entities pValue"] <= p_value)
        if min_entities is not None:
            mask = mask & (df["#Entities total"] >= min_entities)
        if max_entities is not None:
            mask = mask & (df["#Entities total"] <= max_entities)

        if sort_by is not None:
            return df[mask].sort_values(by=sort_by, ascending=ascending)
        else:
            return df[mask]


def _get_token(ids):
    """ Return the token associated with the reactome enrichment analysis.

    Parameters
    ----------
    ids : comma separated list of genes IDs in string format.
        The type of ID should be accepted by the Reactome analysis tool 
        (see https://reactome.org/userguide/analysis for more details).

    Returns
    -------
    string
        The token associated with the data result - analysis Web-Service is token based, so for every analysis
        request, a token is associated with the result.

    """
    result = analysis.identifiers(ids=",".join(ids))
    return result["summary"]["token"]
