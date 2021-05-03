import numpy as np
import requests
import json
import pandas as pd
import warnings
from ._utils import convert_to_entrez , get_top_genes , check_data , check_params

class ToppFunAnalysis(object):    
    """ Provide tools for running Toppfun enrichment analysis for different metagenes.
    
    Parameters
    ----------
    data : If pre_selected = False : 
                pandas.DataFrame , shape (n_metagenes , n_genes) or pandas.Series, shape (n_genes)
                The column names (or the index keys for a serie) should be valid gene IDs.
           If pre_selected = True :
                pandas.Series , shape (n_metagenes)
                For each metagene the serie contains a list of the IDs of the extreme expressed
                genes.
                
    input_type : string, optional.
        Type of input gene IDs. Common types are 'entrezgene' , 'symbol' , 'uniprot' , 'ensembl.gene' , 'refseq'...
        For the complete list of available types, see https://docs.mygene.info/en/latest/doc/query_service.html#available_fields 
        If input_type is None, conversion will not be possible and input IDs will be assumed to be Entrez IDs.
        The default is None.
        
    pre_selected : boolean , optional.
        Indicate whether the extreme genes have already been selected (see above!).
        The default is False.
        
    threshold : numeric or array-like of two numerics , optional
        See get_top_genes. The default is 3.
        
    method : {'quantile' , 'std'} , optional
        See get_top_genes. The default is 'std'.
        
    tail : {'left' , 'right' , 'both' , 'heaviest'} , optional
        See get_top_genes. The default is 'heaviest'.
    
    Attributes
    --------
    
    top_genes_ : pandas.DataFrame, shape (n_metagenes , 3)
        For each metagene the 'inputs' column contains a list of the IDs of the extreme expressed
        genes.
        
    """
    
    def __init__(self , data , input_type = None , pre_selected = False, threshold = 3 , method = 'std' , tail = 'heaviest'):
        
        # Check data
        check_data(data , pre_selected)   
        
        self.input_type = input_type
        if self.input_type is None :
            warnings.warn("If input_type is None the conversion of input IDs to Entrez ids will not be possible. ToppFunAnalysis will assume that the inputs are already Entrez IDs.")
            
        # Initialization of selt.top_genes_ attribute
        self.top_genes_ = pd.DataFrame({'inputs' : None , 'entrezgene' : None , 'notfound' : None} , index = data.index)        
        if pre_selected :
            self.top_genes_['inputs'] = data.copy()
        else :
            threshold = check_params(threshold , method, tail)  
            self.top_genes_['inputs'] = data.apply(get_top_genes , threshold = threshold , method = method , tail = tail , axis = 1)   
        
            
    def convert_metagenes(self , idx):
        """ Convert the IDs of the most expressed genes contained in self.top_genes_.
        
        Parameters
        ----------
        idx : {"all" , string , list of strings}
            If idx = "all" all the metagenes will be converted. Otherwise, only the metagenes
            associated with idx will be converted. In that case, idx must correspond to valid 
            indexes of the input data.
 
        Returns
        -------
        None
        
        """
        
        if self.input_type is None :
            raise ValueError("Conversion is not possible with self.input_type = None.")
        
        # Define the function that will be applied to the rows of self.top_genes_ dataframe
        def fun(row):
            if row['entrezgene'] is None :
                return convert_to_entrez(row['inputs'] , self.input_type)[:2]
            else :
                return row['entrezgene'] , row['notfound']
            
        # Apply func to the rows of self.top_genes_ associated with the metagenes parameter 
        if idx == 'all' :
            
            warnings.warn("idx = 'all' : this operation can take quite some time depending on the number of metagenes and the number of most expressed genes.")
            self.top_genes_[['entrezgene' , 'notfound']] = self.top_genes_.apply(fun , axis = 1 , result_type = 'expand')
            
        elif isinstance(idx , list) :
            
            warnings.warn("metagenes is a list : this operation can take quite some time depending on the number of metagenes and the number of most expressed genes.")
            self.top_genes_.loc[idx , ['entrezgene' , 'notfound']] = (self.top_genes_.loc[idx].apply(fun , axis = 1 , result_type = 'expand')).values
            
        else :
            
            self.top_genes_.loc[idx, ['entrezgene' , 'notfound']] = np.array(fun(self.top_genes_.loc[idx]) , dtype="object")
            
        return
    
    
    def get_analysis(self , metagene , type_list = None , p_value=0.05 , min_entities=10, max_entities=500 , maxres=10, correct="FDR"):
        """ Return the ToppFun enrichment analysis of a given metagene.
        
        Parameters
        ----------
        metagene : object
            It must correspond to a valid index of the input data.
            
        type_list: list of strings, optional
            List of features to perform enrichment tests. If None, all the available features
            will be used (see _get_analysis). The default is None.
        
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
                    
        if self.input_type is None :
            entrez_dict = {'Genes' : [int(id) for id in self.top_genes_.loc[metagene , 'inputs']]}
        else :
            self.convert_metagenes(idx = metagene)
            entrez_dict = {'Genes' : [int(id) for id in self.top_genes_.loc[metagene , 'entrezgene']]}
            
        results = []
        annotations = _get_analysis(entrez_dict, type_list , p_value , min_entities 
                                    , max_entities , maxres , correct).json()["Annotations"]
        
        for element in annotations :
            gene_symbol_list = [gene['Symbol'] for gene in element['Genes']]
            element["Gene_Symbol"] = ','.join(gene_symbol_list)
            element.pop("Genes", None)
            results.append(element)
            
        return pd.DataFrame(results)
    
    
def _get_analysis(entrez_dict, type_list, p_value , min_entities , max_entities , maxres , correct ):
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
    if type_list is None :
        type_list = ["GeneOntologyMolecularFunction","GeneOntologyBiologicalProcess", "GeneOntologyCellularComponent","HumanPheno", "MousePheno",
                     "Domain" , "Pathway", "Pubmed", "Interaction", "Cytoband", "TFBS", "GeneFamily", "Coexpression", "CoexpressionAtlas","GeneFamily",
                     "Computational","MicroRNA","Drug","Disease"]
        
    url = "https://toppgene.cchmc.org/API/enrich"
    headers = {'Content-Type': 'text/json'}
    parameters = {}
    parameters["Categories"] = []
    for type_id in type_list:
        parameters["Categories"].append({"Type":type_id,
                                        "Pvalue":p_value,
                                        "MinGenes":min_entities,
                                        "MaxGenes":max_entities,
                                        "MaxResults":maxres,
                                        "Correction":correct})
    data_all = {}
    data_all["Genes"] = entrez_dict["Genes"]
    data_all["Categories"] = parameters["Categories"]
    response = requests.post(url,headers=headers,data=json.dumps(data_all))
    if response.status_code == 200:
        print("Enrichment analysis success!")
    else:
        print("Something went wrong during enrichment... Status code:", response.status_code)
    return response
