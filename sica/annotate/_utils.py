import numpy as np
import mygene
import pandas as pd
import scipy.stats as stats


def _convert_geneID(genes_list , input_type , output_type):
    """ Call mygene querymany() function to convert gene IDs
    https://docs.mygene.info/projects/mygene-py/en/latest/#mygene.MyGeneInfo.querymany

    Parameters
    ----------
    genes_list : List of strings
        List containing gene IDs.
        
    input_type: string
        Type of input gene IDs.
        For available types, see: https://docs.mygene.info/en/latest/doc/query_service.html#available_fields

    output_type: string
        Type of output gene IDs.
        For available types, see: https://docs.mygene.info/en/latest/doc/data.html#available-fields

    Returns
    -------
    pandas.DataFrame
        Dataframe containing query gene IDs as index. See mygene documentation for more details
        (https://docs.mygene.info/projects/mygene-py/en/latest/)

    """
    
    #Initiate mygene
    mg = mygene.MyGeneInfo()
    #Perform Conversion
    conversion_df = mg.querymany(genes_list, scopes=input_type, fileds = output_type , verbose=False, as_dataframe=True)
    
    return conversion_df[~conversion_df.index.duplicated(keep='first')] #remove potential duplicates
    
def convert_to_entrez(genes_list , input_type):
    """ Convert input gene IDs to Entrez gene IDs with mygene conversion tools (see _convert_geneID function).
    
    Parameters
    ----------
    genes_list : List of strings
        List containing gene IDs.
        
    input_type: string
        Type of input gene IDs.
        For available types, see: https://docs.mygene.info/en/latest/doc/query_service.html#available_fields

    Returns
    -------
    entrez : List of strings
        List containing Entrez gene IDs corresponding to input IDs from genes_list.
        
    notfound : List of strings or None
        List of input IDs which were not found by mygene conversion tools.
        
    df : pandas.DataFrame
        Dataframe containing query gene IDs as index returned by _convert_geneID.

    """
    
    df = _convert_geneID(genes_list , input_type = input_type , output_type = 'entrezgene')

    bool_mask = df['entrezgene'].isna()
    entrez = list(df[~bool_mask]['entrezgene'])
    notfound = list(df[bool_mask].index)
        
    return entrez , notfound , df

def get_top_genes(metagene , threshold , method , tail):
    """ Select the extreme expressed genes for a given metagene.
    
    Parameters
    ----------
    metagene : pandas.Series, shape (n_genes)
        The indexes should be valid gene IDs (e.g HUGO gene symbols, EntrezGene...)
        
    threshold : numeric or array-like of two numerics
        Used for selecting the extreme expressed genes in the metagene (i.e most expressed 
        and/or least expressed genes). If a tuple is passed, different thresholds are used 
        for the left tail and the right tail of the metagene distribution.
        
    method : {'quantile' , 'std'}
        Method for selecting the extreme expressed genes in the metagene.
        - 'quantile' will select genes above and/or under certain quantiles of the metagene
          distribution.
        - 'std' will select genes above mu + k*std and/or under mu - l*std (with k and l defined by the
          threshold parameter and mu and std the mean and the standard deviation of the metagene distribution).
        
    tail : {'left' , 'right' , 'both' , 'heaviest'}
        Define the tail of the metagene distribution to look at for selecting the extreme expressed genes.

    Returns
    -------
    top_genes : list of strings
        List of the IDs of the extreme expressed genes for the given metagene.

    """
    if method == 'quantile':
        t_left = metagene.quantile(q = threshold[0])
        t_right = metagene.quantile(q = 1 - threshold[1])            
    elif method == 'std':
        mu , std = metagene.mean() , metagene.std()
        t_left = mu - threshold[0]*std
        t_right = mu + threshold[1]*std
    
    S_l = metagene[metagene <= t_left].sort_values(ascending = False)
    S_r = metagene[metagene >= t_right].sort_values(ascending = False) 
    
    if tail == 'left' :
        top_genes = list(S_l.index)
    elif tail == 'right' :
        top_genes = list(S_r.index)
    elif tail == 'both' : 
        top_genes = list(S_l.index) + list(S_r.index)
    elif tail == 'heaviest' :
        if stats.skew(metagene)> 0:
            top_genes = list(S_r.index)
        else : 
            top_genes = list(S_l.index)
            
    return top_genes

def check_data(data , pre_selected):
    
    if pre_selected :
       if not isinstance(data , pd.Series):
        raise ValueError("When pre_selected is True, data parameter must be a pandas.Series of shape (n_metagenes) containing list of extreme expressed genes")    
    else :
        if isinstance(data , pd.Series):
            data = pd.DataFrame(data).transpose()
        elif not isinstance(data , pd.DataFrame):
            raise ValueError("When pre_selected is False, data parameter must be a pandas.Series of shape (n_genes) or a pandas.DataFrame of shape (n_metagenes , n_genes)")         
    return
    
def check_params(threshold , method , tail):
    
    if isinstance(threshold, (int , float)) :
        threshold = np.array([threshold]*2)
    elif not isinstance(threshold, (tuple , list , np.array)) or len(threshold) != 2 :
        raise ValueError("threshold must be either a numeric or an array-like of two numerics") 
        
    if not method in ['quantile' , 'std'] :
        raise ValueError("method parameter value must be either 'quantile' or 'std' ")
        
    if not tail in ['left' , 'right' , 'both' , 'heaviest'] :
        raise ValueError("tail parameter value must be 'left', 'right', 'both' or 'heaviest'")
        
    return threshold