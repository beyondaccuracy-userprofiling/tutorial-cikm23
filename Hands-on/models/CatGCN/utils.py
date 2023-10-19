import torch

import time
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from texttable import Texttable

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    t.set_precision(6)
    print(t.draw())

def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    return graph

def field_reader(path):
    """
    Function to read the field index from the path.
    :param path: Path to the field index.
    :return field_index: Numpy matrix of field index.
    """
    field_index = np.load(path).astype(np.int64)
    return field_index

def target_reader(path):
    """
    Reading the target vector from disk.
    :param path: Path to the target.
    :return target: Target vector.
    """
    target = np.array(pd.read_csv(path).iloc[:,1]).reshape(-1,1)
    return target

def label_reader(path):
    """
    Reading the user_label file from the path.
    :param path: Path to the label file
    :return user_labels: User labels DataFrame file.
    """
    user_labels = pd.read_csv(path)
    return user_labels