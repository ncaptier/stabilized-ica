import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import json

"""
This python code provides an implementation for the Mutual Nearest Neighbors method as well as
a Networkx tool to display the associated graph.
"""

class MNN(object):
    """Given two arrays X and Y or a precomputed distance matrix computes the undirected adjacency matrix
    using the Mutual Nearest Neighbors method.
    
    Parameters
    ----------
    X : 2D array of shape (n_components_1 , n_features_1), 2D array of shape (n_components_1 , n_components_2) if metric = "precomputed".
    
    Y : 2D array of shape (n_components_2 , n_features_2), optional
        The default is None.
        
    k : int > 0
        Parameter for the Mutual Nearest Neighbor method (i.e number of neighbors that we consider).
        
    metric : string
        Metric for the computation of the adjacency matrix (e.g "pearson" , "spearman" 
        or the metrics accepted by ``scipy.spatial.distance.cdsit``).
        
    Notes
    -----  
    - In the case where the distance matrix is not precomputed, we compute the distance between each rows of X and Y.
    
    - In the case X and Y are dataframes, we consider only the common columns of X and Y. Otherwise, we assume that the columns are the same for X and Y (i.e ``n_features_1 = n_features_2``)
    
    """
    def __init__(self , k , metric , X , Y = None):
        self.X = X
        self.Y = Y
        self.k = k
        self.metric = metric 
        if metric == "precomputed":
            self.distance = X
        else:
            self.distance = self.compute_distance(self.X , self.Y , self.metric)
    
    @staticmethod
    def compute_distance(X , Y , metric):
        """Compute the distance between each pair of rows of X and Y
        
        Parameters
        ----------
        X : 2D array of shape (n_components_1 , n_features_1)

        Y : 2D array of shape (n_components_2 , n_features_2)

        metric : string

        Returns
        -------
        2D array of shape (n_components_1 , n_components_2)

        """
        #Consider only the common columns of dataframes X and Y
        if isinstance(Y, pd.DataFrame) : 
            common_features = set(X.columns) & set(Y.columns)
            X = X[common_features]
            Y = Y[common_features]
        else :
            X = pd.DataFrame(X)
            Y = pd.DataFrame(Y)
            
        if metric in ['pearson' , 'spearman' , 'kendall'] :
            corr = (pd.concat([X, Y], keys=['X', 'Y']).T).corr(method = metric)
            return 1 - np.abs((corr.loc['X' , 'Y']).values)
        else:
            return cdist(X , Y , metric = metric)
    
    def adjacency_matrix(self , weighted):
        """Compute the undirected adjacency matrix with the Mutual Nearest Neighbors method (``k`` neighbors)

        Parameters
        ----------        
        weighted : boolean
            If True each coefficient of the adjacency matrix is weighted by ``1 - distance``, otherwise the coefficient are 0 or 1. 

        Returns
        -------
        2D array of shape (n_components_1 , n_components_2)

        """
        bool_mask = (self.distance <= np.sort(self.distance , axis = 1)[: , self.k-1].reshape(-1 , 1)) * \
                    (self.distance <= np.sort(self.distance , axis = 0)[self.k-1, :].reshape(1, - 1)) 
        
        if weighted:           
            return bool_mask*(1 - self.distance)
        else:
            return bool_mask*np.ones(bool_mask.shape)

########################################################################################################################

def _pairs(items):
    """Return a list with all the pairs formed by two different elements of a list "items"
    
    Note : This function is a useful tool for the building of the MNN graph.
    
    Parameters
    ----------
    items : list

    Returns
    -------
    list
        List of pairs formed by two different elements of the items

    """
    return [(items[i],items[j]) for i in range(len(items)) for j in range(i+1, len(items))]

        
class MNNgraph(object):
    """ Given a list of data sets, draws the MNN graph with a networkx object (compatible with the software Cytoscape)
    
    Parameters
    ----------   
    data : list of 2D data sets of shape (n_components_i , n_features_i)
        
    names : list of strings
        Names of the data sets.
    
    k : int > 0
        Parameter for the Mutual Nearest Neighbors method.
    
    metric : string, optional
        Metric for the computation of the adjacency matrices.
        The default is "pearson".
        
    weighted : boolean, optional
        If True each edge is associated with a weight ``1 - distance`` (cf. adjacency_matrix())
        The default is True.
        
    Attributes
    ----------   
    graph_ : networkx object
    
    Note
    ----   
    If the elements of data are not dataframes, we assume that they all share the same features.
    
    Examples
    --------
    >>> from sica.mutualknn import MNNgraph
    >>> cg = MNNgraph(data = [df1 , df2 , df3] , names=['dataframe1' , 'dataframe2' , 'dataframe3'] , k=1)
    >>> cg.draw(colors = ['r', 'g' , 'b'] , spacing = 2)    
    >>> cg.export_json("example.json")
   
    """
    def __init__(self, data , names , k , metric = 'pearson' , weighted = True):
        self.data = data
        self.names = names
        self.n_sets = len(names)
        self.graph_ = self.create_graph(self.data , self.names , k , metric , weighted)    
    
    @staticmethod
    def create_graph(data , names , k , metric , weighted):
        """Create the MNN graph associated to the list of data sets. Two situations are 
        distinguished : one with only two data sets and another with more than two data sets.
                
        Parameters
        ----------
        data : list of 2D arrays of shape (n_components_i , n_features_i) 
            
        names : list of strings
            Names of the data sets.
        
        k : integer >= 1
            Parameter for the Mutual Nearest Neighbors Method.
        
        metric : string
            Metric for the computation of the adjacency matrices.
        
        weighted : boolean
            If True each edge is associated with a weight ``1 - distance``.

        Returns
        -------
        G : graph (networkx object)
            MNN graph for the data sets contained in the list "data".

        """        
        G = nx.Graph()
        
        if len(names) <= 2:
            
            #for each MNN link between the two data sets, add two nodes and an edge to the graph G
            #the pos attribute for each node will be useful for further drawing
            if metric == "precomputed":
                h = MNN(X = data , k = k, metric = metric).adjacency_matrix(weighted)
            else:
                h = MNN(X = data[0], Y = data[1], k = k, metric = metric).adjacency_matrix(weighted)
            count = 0
            for u in range(h.shape[0]):
                for v in range(h.shape[1]):
                    if h[u , v] > 0:
                        n1 , n2 = names[0] + ' ' + str(u + 1) , names[1] + ' ' + str(v + 1)
                        G.add_node(n1 , weight = 1 , data_set = names[0] , pos = [-1 , count] , label = str(u+1))
                        G.add_node(n2 , weight = 1 , data_set = names[1] , pos = [1 , count] , label = str(v+1))
                        G.add_edge(n1 , n2 , weight=h[u , v] , label = str(np.round(h[u , v] ,2)))  
                        count += -5
        else:
            
            #for each pair of data sets and for each MNN link between two data sets, add two nodes 
            #and an edge to the graph G 
            P , L = _pairs(data) , _pairs(names)      
            for i in range(len(L)):
                h = MNN(X = P[i][0], Y = P[i][1], k = k, metric = metric).adjacency_matrix(weighted)
                for u in range(h.shape[0]):
                    for v in range(h.shape[1]):
                        if h[u , v] > 0:
                            n1 , n2 = L[i][0] + ' ' + str(u + 1) , L[i][1] + ' ' + str(v + 1)
                            G.add_node(n1 , weight = 1 , data_set = L[i][0])
                            G.add_node(n2 , weight = 1 , data_set = L[i][1])
                            G.add_edge(n1 , n2 , weight=h[u , v])         
        return G
    
    def draw(self, bipartite_graph = False , ax = None , colors = None  , spacing = 1):
        """Draw the MNN graph.
        
        Parameters
        ----------        
        bipartite_graph : boolean, optional
            If True a custom bipartite layout is used (only with two data sets). The default is False
            
        ax : matplotlib.axes, optional
            The default is None.
            
        colors : list of matplotlib.colors, optional
            List of colors you want each data set to be associated with. The default is None.
            
        spacing : float >= 1, optional
            Deal with the space between nodes. Increase this value to move nodes farther apart.

        Returns
        -------
        None.

        """
        
        #1) Draw the nodes of the graph with a different color for each data set. In case of bipartite
        #graph a custom-made layout is used (attribute pos), otherwise the spring layout is used.
        
        if bipartite_graph:
            
            left_nodes = set(n for n,d in self.graph_.nodes(data=True) if d['data_set']==self.names[0])
            right_nodes = set(self.graph_) - left_nodes
            pos = nx.get_node_attributes(self.graph_ , 'pos') 
            nx.draw_networkx_nodes(self.graph_, pos , nodelist=left_nodes , node_color='r', node_size=350, alpha = 0.8 ,  label = self.names[0] , ax = ax)
            nx.draw_networkx_nodes(self.graph_, pos , nodelist=right_nodes , node_color='b', node_size=350, alpha=0.8 ,  label = self.names[1] , ax = ax)
            nx.draw_networkx_labels(self.graph_, pos, labels =nx.get_node_attributes(self.graph_ , 'label') , font_size=12 , ax=ax)
            
        else:
            
            pos = nx.spring_layout(self.graph_ , weight = 'weight' , k = spacing*(1/np.sqrt(self.graph_.number_of_nodes())))
            
            if colors is None:
                cmap = plt.get_cmap('gist_rainbow' , self.n_sets)
    
            for i in range(self.n_sets):
                nodelist = [e[0] for e in list(self.graph_.nodes(data='data_set')) if e[1] == self.names[i]]               
                nx.draw_networkx_nodes(self.graph_, pos,
                                        nodelist=nodelist,
                                        node_size=50, 
                                        node_color= np.array([cmap(i)]) if colors is None else colors[i],
                                        label = self.names[i],
                                        ax=ax)
                
        #2) Draw the edges of the graph with width proportionnal to their
        #weight. In case of bipartite graph, edge labels are also displayed.
                
        width = np.array([self.graph_[e[0]][e[1]]['weight'] for e in self.graph_.edges()])
        temp = np.max(width)       
        width = (width/temp)*3  if temp > 0 else 1.0       
        
        nx.draw_networkx_edges(self.graph_ , pos , width = width , ax=ax)
        
        if bipartite_graph:
            nx.draw_networkx_edge_labels(self.graph_ , pos,
                                         edge_labels=nx.get_edge_attributes(self.graph_ , 'label') , 
                                         font_size=16 , label_pos=0.3 , ax=ax)
            
        #3) Add a legend (for the colors of the nodes) and a title
            
        if ax is None:
            plt.legend()
            plt.title('MNN graph')
        else:
            ax.legend()
            ax.set_title('MNN graph')
        return  
    
    def export_json(self , file_name):
        """Save the graph in a json file adapted to cytoscape format
        
        Parameters
        ----------
        file_name : string
            Name of the json file.
    
        Returns
        -------
        None.
    
        """
        dic = nx.readwrite.json_graph.cytoscape_data(self.graph_)
        with open(file_name , 'w') as fp:
            json.dump(dic , fp )
        return
    