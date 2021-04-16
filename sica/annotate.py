import numpy as np
import pandas as pd
from reactome2py import analysis
import scipy.stats as stats
import webbrowser

def _check_data(data , pre_selected):
    
    if pre_selected :
       if not isinstance(data , pd.Series):
        raise ValueError("When pre_selected is True, data parameter must be a pandas.Series of shape (n_metagenes) containing list of extreme expressed genes")    
    else :
        if isinstance(data , pd.Series):
            data = pd.DataFrame(data).transpose()
        elif not isinstance(data , pd.DataFrame):
            raise ValueError("When pre_selected is False, data parameter must be a pandas.Series of shape (n_genes) or a pandas.DataFrame of shape (n_metagenes , n_genes)")         
    return
    
def _check_threshold(threshold , method , tail):
    
    if isinstance(threshold, (int , float)) :
        threshold = np.array([threshold]*2)
    elif not isinstance(threshold, (tuple , list , np.array)) or len(threshold) != 2 :
        raise ValueError("threshold must be either a numeric or an array-like of two numerics") 
        
    if not method in ['quantile' , 'std'] :
        raise ValueError("method parameter value must be either 'quantile' or 'std' ")
        
    if not tail in ['left' , 'right' , 'both' , 'heaviest'] :
        raise ValueError("tail parameter value must be 'left', 'right', 'both' or 'heaviest'")
        
    return threshold

def _get_top_genes(metagene , threshold , method , tail):
    """ Select the extreme expressed genes for a given metagene.
    
    Parameters
    ----------
    metagene : pandas.Series, shape (n_genes)
        The index names should be valid gene symbols (e.g HUGO gene symbols, EntrezGene...)
        
    threshold : numeric or array-like of two numerics
        Used for selecting the extreme expressed genes in the metagene (i.e most expressed 
        and/or least expressed genes).
        If a tuple is passed, different thresholds are used for the left tail and the right tail
        of the metagene distribution.
        
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
        List of the names of the extreme expressed genes for the given metagene.

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


def _get_token(ids):
    """ Return the token associated with the reactome enrichment analysis of the
        list of gene names given by ids.

    Parameters
    ----------
    ids : comma seperated list of genes identifiers symbol in string format.
        The symbols should be accepted by the Reactome analysis tool 
        (see https://reactome.org/userguide/analysis for more details).

    Returns
    -------
    string
        The token associated with the data result - analysis Web-Service is token based, so for every analysis
        request a TOKEN is associated to the result

    """
    result = analysis.identifiers(ids=','.join(ids))
    return result['summary']['token']



class ReactomeAnalysis(object):
    """ Provide tools for running reactome enrichment analysis for different metagenes
    computed with the stabilized ICA algorithm.
    
    Parameters
    ----------
    data : If pre_selected = False : 
                pandas.DataFrame , shape (n_metagenes , n_genes) or pandas.Series, shape (n_genes)
                The column names (or the index names for a serie) should be valid gene symbols 
                accepted by the reactome analysis tool (e.g HUGO gene symbols, EntrezGene...).
                See https://reactome.org/userguide/analysis for more details.
           If pre_selected = True :
                pandas.Series , shape (n_metagenes)
                For each metagene the serie contains a list of the names of the extreme expressed
                genes. The gene names should be valid gene symbols accepted by the reactome analysis tool.
            
    pre_selected : boolean , optional.
        Indicate whether the extreme genes have already been selected (see above!).
        The default is True.
        
    threshold : numeric or array-like of two numerics
        See _get_top_genes.
        
    method : {'quantile' , 'std'} , optional
        See _get_top_genes. The default is 'quantile'.
        
    tail : {'left' , 'right' , 'both' , 'heaviest'} , optional
        See _get_top_genes. The default is 'heaviest'.
    
    Attributes
    --------
    
    top_genes_ : pandas.Series, shape (n_metagenes)
        For each metagene the serie contains a list of the names of the extreme expressed
        genes.
        
    """
    
    def __init__(self, data , pre_selected = False, threshold = 0.01, method = 'quantile' , tail = 'heaviest'):
        
        _check_data(data , pre_selected)
        
        if pre_selected :
            self.top_genes_ = data.copy()
        else :
            threshold = _check_threshold(threshold , method, tail)  
            self.top_genes_ = data.apply(_get_top_genes , threshold = threshold , method = method , tail = tail , axis = 1)
            
        self.tokens = {ind : None for ind in data.index}     
        
    def open_full_analysis(self , metagene): 
        """ Browse the analysis for the given metagene in reactome web portal.
        
        Parameters
        ----------
        metagene : metagene name
            It must corresponds to a valid index key of the dataframe self.data.

        Returns
        -------
        None.

        """
        if self.tokens[metagene] is not None :
            token = self.tokens[metagene]
        else : 
            token = _get_token(self.top_genes_.loc[metagene])
            self.tokens[metagene] = token
            
        url = 'https://reactome.org/PathwayBrowser/#/DTAB=AN&ANALYSIS=' + token
        webbrowser.open(url)
        return 
    
    def get_analysis(self , metagene , species='Homo sapiens', sort_by='Entities FDR', 
                     ascending = True, p_value=0.05 , min_entities=10, max_entities=500) :
        """ Create a dataframe for the results of the reactome enrichment analysis of 
        a given metagene (i.e a row of self.data).
        
        Parameters
        ----------
        metagene : metagene name
            It must corresponds to a valid index key of the dataframe self.data.
            
        species : string or list of string, optional
            List of species to filter the result. The default is 'Homo sapiens'.
            
        sort_by : {None , '#Entities found' , '#Entities total' , 'Entities ratio' , 'Entities pValue' , 'Entities FDR ' , '#Reactions found' , '#Reactions total' , 'Reactions ratio'}, optional
            How to sort the result. The default is 'Entities FDR'.
        
        ascending : boolean, optional
            Sort ascending vs. descending if sort_by is not None.The default is True.
        
        p_value : float in (0 , 1), optional
            Only hit pathway with pValue equals or below p_value will be returned. The default is 0.05.
            
        min_entities : int >= 0, optional
             Minimum number of contained entities per pathway. The default is 10.
            
        max_entities : int > 0, optional
             Maximum number of contained entities per pathway. The default is 500.

        Returns
        -------
        pandas.DataFrame
            Results of the reactome enrichment analysis for the given metagene.

        """
        if self.tokens[metagene] is not None :
            token = self.tokens[metagene]
        else : 
            token = _get_token(self.top_genes_.loc[metagene])
            self.tokens[metagene] = token
        
        df = analysis.pathway2df(token)
        df.iloc[: , 2:10] = df.iloc[: , 2:10].apply(pd.to_numeric)

        mask= pd.Series(True,index=df.index)
        if species is not None :
            mask = mask & (df['Species name'] == species)
        if p_value is not None :
            mask = mask & (df['Entities pValue'] <= p_value)
        if min_entities is not None :
            mask = mask & (df['#Entities total'] >= min_entities)
        if max_entities is not None :
            mask = mask & (df['#Entities total'] <= max_entities)
            
        if sort_by is not None :
            return df[mask].sort_values(by = sort_by , ascending = ascending)
        else : 
            return df[mask]