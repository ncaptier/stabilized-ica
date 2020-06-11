import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.spatial.distance import cdist
import json

"""
This python code provides an implementation for the Mutual Nearest Neighbors method as well as
a Networkx tool to display the associated graph.
"""

class MNN(object):
    
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
        X : 2D array, shape (n_components_1 , n_features)

        Y : 2D array, shape (n_components_2 , n_features)

        metric : string

        Returns
        -------
        2D array, shape (n_components_1 , n_components_2)

        """
        if metric == 'pearson':
            return 1 - np.abs(np.corrcoef(X , Y , rowvar=True)[:X.shape[0] , X.shape[0]:])
        elif metric == 'spearman':
            return 1 - np.abs((stats.spearmanr(a = X, b = Y , axis = 1)[0])[:X.shape[0] , X.shape[0]:])
        else:
            return cdist(X , Y , metric = metric)
    
    def adjacency_matrix(self):
        """Compute the adjacency matrix with the mutual nearest neighbors method (k = self.k)
           
           Note: If a correlation metric is used, the coefficients of the adjacency matrix correspond
                 to the absolute values of the correlation coefficient. Otherwise they are equal to 0
                 or 1.

        Returns
        -------
        2D array, shape (n_components_1 , n_components_2)

        """
        bool_mask = (self.distance <= np.sort(self.distance , axis = 1)[: , self.k-1].reshape(-1 , 1)) * \
                    (self.distance <= np.sort(self.distance , axis = 0)[self.k-1, :].reshape(1, - 1)) 
        
        if (self.metric == 'pearson') or (self.metric == 'spearman') or (self.metric == 'precomputed'):           
            return bool_mask*(1 - self.distance)
        else:
            return bool_mask*np.ones(bool_mask.shape)

        
class MNNgraph(object):
    
    def __init__(self, data , names , k , metric = 'pearson'):
        self.data = data
        self.names = names
        self.n_sets = len(names)
        self.graph = self.create_graph(self.data , self.names , k , metric)    
    
    @staticmethod
    def create_graph(data , names , k , metric):
        """Create the MNN graph associated to the list of data sets. Two situations are 
           distinguished: one with only two data sets and another with more than two data sets.
        
           Note : A precomputed distance matrix will only work with two data sets
                
        Parameters
        ----------
        if metric == "precomputed":
            data : 2D array, shape (n_components1 , n_components2)
              distance matrix between the two data sets
        else:
            data : list of 2D arrays with shape (n_components , n_features) 
            
        names : list of strings
            names of the data sets
        
        k : integer >= 1
            parameter for the Mutual Nearest Neighbors Method
        
        metric: string
            metric for the computation of the adjacency matrix

        Returns
        -------
        G : graph (networkx object)
            MNN graph for the data sets contained in the list (or the distance matrix) "data"

        """        
        G = nx.Graph()
        
        if (not isinstance(data, list)) or (len(data)==2):
            
            #for each MNN link between the two data sets, add two nodes and an edge to the graph G
            #the pos attribute for each node will be useful for further drawing
            if metric == "precomputed":
                h = MNN(X = data , k = k, metric = metric).adjacency_matrix()
            else:
                h = MNN(X = data[0], Y = data[1], k = k, metric = metric).adjacency_matrix()
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
            P , L = pairs(data) , pairs(names)      
            for i in range(len(L)):
                h = MNN(X = P[i][0], Y = P[i][1], k = k, metric = metric).adjacency_matrix()
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
            if True a custom bipartite layout is used (only with two data sets). The default is False
            
        ax : matplotlib.axes, optional
            The default is None.
            
        colors : list of matplotlib.colors, optional
            list of colors you want each data set to be associated with. The default is None.
            
        spacing : float >= 1, optional
            deal with the space between nodes. Increase this value to move nodes farther apart.

        Returns
        -------
        None.

        """
        
        #1) Draw the nodes of the graph with a different color for each data set. In case of bipartite
        #graph a custom-made layout is used (attribute pos), otherwise the spring layout is used.
        
        if bipartite_graph:
            
            left_nodes = set(n for n,d in self.graph.nodes(data=True) if d['data_set']==self.names[0])
            right_nodes = set(self.graph) - left_nodes
            pos = nx.get_node_attributes(self.graph , 'pos') 
            nx.draw_networkx_nodes(self.graph, pos , nodelist=left_nodes , node_color='r', node_size=350, alpha = 0.8 ,  label = self.names[0] , ax = ax)
            nx.draw_networkx_nodes(self.graph, pos , nodelist=right_nodes , node_color='b', node_size=350, alpha=0.8 ,  label = self.names[1] , ax = ax)
            nx.draw_networkx_labels(self.graph, pos, labels =nx.get_node_attributes(self.graph , 'label') , font_size=12 , ax=ax)
            
        else:
            
            pos = nx.spring_layout(self.graph , weight = 'weight' , k = spacing*(1/np.sqrt(self.graph.number_of_nodes())))
            
            if colors is None:
                cmap = plt.get_cmap('gist_rainbow' , self.n_sets)
    
            for i in range(self.n_sets):
                nodelist = [e[0] for e in list(self.graph.nodes(data='data_set')) if e[1] == self.names[i]]               
                nx.draw_networkx_nodes(self.graph, pos,
                                        nodelist=nodelist,
                                        node_size=50, 
                                        node_color= np.array([cmap(i)]) if colors is None else colors[i],
                                        label = self.names[i],
                                        ax=ax)
                
        #2) Draw the edges of the graph with width proportionnal to their
        #weight. In case of bipartite graph, edge labels are also displayed.
                
        width = np.array([self.graph[e[0]][e[1]]['weight'] for e in self.graph.edges()])
        temp = np.max(width)       
        width = (width/temp)*3  if temp > 0 else 1.0       
        
        nx.draw_networkx_edges(self.graph , pos , width = width , ax=ax)
        
        if bipartite_graph:
            nx.draw_networkx_edge_labels(self.graph , pos,
                                         edge_labels=nx.get_edge_attributes(self.graph , 'label') , 
                                         fontsize=16 , label_pos=0.3 , ax=ax)
            
        #3) Add a legend (for the colors of the nodes) and a title
            
        if ax is None:
            plt.legend()
            plt.title('MNN graph')
        else:
            ax.legend()
            ax.set_title('MNN graph')
        return  
    
    def get_average_clustering(self):
        """Compute the average clustering coefficient of the MNN graph.

        Returns
        -------
        Float
            average clustering coefficient of the MNN graph

        """
        return nx.average_clustering(self.graph , weight= 'weight')
    
    def export_json(self , file_name):
        """Save the graph in a json file adapted to cytoscape format
        
        Parameters
        ----------
        file_name : string
            name of the json file.
    
        Returns
        -------
        None.
    
        """
        dic = nx.readwrite.json_graph.cytoscape_data(self.graph)
        with open(file_name , 'w') as fp:
            json.dump(dic , fp )
        return
    
        
def pairs(items):
    """Return a list with all the pairs formed by two different elements of a list "items"
    
       Note : This function is a useful tool, used to build the MNN graph
    
    Parameters
    ----------
    items : list

    Returns
    -------
    list
        list of pairs formed by two different elements of the items

    """
    return [(items[i],items[j]) for i in range(len(items)) for j in range(i+1, len(items))]