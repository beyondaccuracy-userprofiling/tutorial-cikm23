import torch

import pymetis as metis
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

class ClusteringMachine(object):
    """
    Clustering the graph, feature set, and target. If the graph is not huge enough, we suggest using 'none' here.
    """

    def __init__(self, args, graph, field_index, target):
        """
        :param args: Arguments object with parameters.
        :param graph: Networkx Graph.
        :param field_index: field_index matrix (ndarray).
        :param target: Target vector (ndarray).
        """
        self.args = args
        self.graph = graph
        self.field_index = field_index
        self.target = target
        self._set_sizes()
        self._set_loss_weight()

    def _set_sizes(self):
        """
        Setting the field and class count.
        """
        self.user_count = self.field_index.shape[0]
        self.field_count = np.max(self.field_index)+1
        self.field_size = self.field_index.shape[1]
        self.class_count = np.max(self.target)+1
        print("####\tData Info\t####")
        print("user count:\t", self.user_count)
        print("field count:\t", self.field_count)
        print("field size:\t", self.field_size)
        print("class count:\t", self.class_count)

    def _set_loss_weight(self):
        class_weight = self.target.shape[0] / (self.class_count * np.bincount(self.target.squeeze()))
        if self.args.weight_balanced == 'True':
            self.class_weight = torch.FloatTensor(class_weight)
        else:
            self.class_weight = torch.ones(self.class_count)

    def decompose(self):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        """
        if self.args.clustering_method == "none":
            print("\nWithout graph clustering.\n")
            self.clusters = [0]
            self.cluster_membership = {node: 0 for node in self.graph.nodes()}
        elif self.args.clustering_method == "metis":
            print("\nMetis graph clustering started.\n")
            self.metis_clustering()
        else:
            print("\nRandom graph clustering started.\n")
            self.random_clustering()
        self.generate_data_partitioning()
        self.transfer_edges_and_nodes()

    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = [cluster for cluster in range(self.args.cluster_number)]
        self.cluster_membership = {node: np.random.choice(self.clusters) for node in self.graph.nodes()}

    def metis_clustering(self):
        """
        Clustering the graph with Metis.
        """
        (st, parts) = metis.part_graph(self.graph, self.args.cluster_number, seed=self.args.seed)
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def generate_data_partitioning(self):
        """
        Creating data partitions and train-val-test splits.
        """
        if self.args.clustering_method != "metis":
            self.sg_nodes = {}
            self.sg_targets = {}
            self.sg_edges = {}
            self.sg_train_nodes = {}
            self.sg_val_nodes = {}
            self.sg_test_nodes = {}
            self.sg_field_index = {}
            for cluster in self.clusters:
                subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])
                self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
                self.sg_targets[cluster] = self.target[self.sg_nodes[cluster],:]
                mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
                self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] + [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
                self.sg_train_nodes[cluster], sg_val_test_nodes = \
                    train_test_split(list(mapper.values()), test_size = 1-self.args.train_ratio, random_state=self.args.seed, shuffle=True)
                self.sg_val_nodes[cluster], self.sg_test_nodes[cluster] = \
                    train_test_split(sg_val_test_nodes, test_size = 0.5, random_state=self.args.seed, shuffle=True)
                self.sg_train_nodes[cluster] = sorted(self.sg_train_nodes[cluster])
                self.sg_val_nodes[cluster] = sorted(self.sg_val_nodes[cluster])
                self.sg_test_nodes[cluster] = sorted(self.sg_test_nodes[cluster])
                self.sg_field_index[cluster] = self.field_index[self.sg_nodes[cluster],:]

    def transfer_edges_and_nodes(self):
        """
        Transfering the data to PyTorch format.
        """
        for cluster in self.clusters:
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster])
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t()
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            self.sg_val_nodes[cluster] = torch.LongTensor(self.sg_val_nodes[cluster])
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])
            self.sg_field_index[cluster] = torch.LongTensor(self.sg_field_index[cluster])