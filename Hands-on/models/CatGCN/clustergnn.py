import torch
import torch.nn.functional as F
from torch.autograd import Variable

import time
import numpy as np
from sklearn import metrics
from tqdm import trange, tqdm

from layers import StackedGNN

class ClusterGNNTrainer(object):
    """
    Training a huge graph cluster partition strategy.
    """
    def __init__(self, args, clustering_machine):
        self.args = args
        self.clustering_machine = clustering_machine
        self.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        self.class_weight = clustering_machine.class_weight.to(self.device)
        self.create_model()

    def create_model(self):
        """
        Creating a StackedGNN and transferring to CPU/GPU.
        """
        self.model = StackedGNN(self.args, self.clustering_machine.field_count, self.clustering_machine.field_size, self.clustering_machine.class_count)
        self.model = self.model.to(self.device)

    def generate_field_adjs(self, node_count):
        # Normalization by  P'' = Q^{-1/2}*P'*Q^{-1/2}, P' = P+probe*O.
        field_adjs = torch.ones((node_count, self.clustering_machine.field_size, self.clustering_machine.field_size))
        field_adjs += self.args.diag_probe * torch.eye(self.clustering_machine.field_size)
        row_sum = self.clustering_machine.field_size + self.args.diag_probe
        field_adjs = (1. / row_sum) * field_adjs
        return field_adjs

    def do_forward_pass(self, cluster):
        """
        Making a forward pass with data from a given partition.
        :param cluster: Cluster index.
        :return average_loss: Average loss on the cluster.
        :return node_count: Number of nodes.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        train_nodes = self.clustering_machine.sg_train_nodes[cluster].to(self.device)
        field_index = self.clustering_machine.sg_field_index[cluster].to(self.device)
        field_adjs = self.generate_field_adjs(field_index.shape[0]).to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        prediction = self.model(edges, field_index, field_adjs)
        average_loss = F.nll_loss(prediction[train_nodes], target[train_nodes], self.class_weight)
        node_count = train_nodes.shape[0]
        return average_loss, node_count

    def do_validation(self, cluster, epoch):
        """
        Making a validation with data from a given partition.
        :param cluster: Cluster index.
        :return average_loss: Average loss on the cluster.
        :return node_count: Number of nodes.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        val_nodes = self.clustering_machine.sg_val_nodes[cluster].to(self.device)
        field_index = self.clustering_machine.sg_field_index[cluster].to(self.device)
        field_adjs = self.generate_field_adjs(field_index.shape[0]).to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        prediction = self.model(edges, field_index, field_adjs)
        average_loss = F.nll_loss(prediction[val_nodes], target[val_nodes], self.class_weight)
        node_count = val_nodes.shape[0]
        return average_loss, node_count

    def do_prediction(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return average_loss: Average loss on the cluster.
        :return node_count: Number of nodes.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        test_nodes = self.clustering_machine.sg_test_nodes[cluster].to(self.device)
        field_index = self.clustering_machine.sg_field_index[cluster].to(self.device)
        field_adjs = self.generate_field_adjs(field_index.shape[0]).to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        prediction = self.model(edges, field_index, field_adjs)
        average_loss = F.nll_loss(prediction[test_nodes], target[test_nodes], self.class_weight)
        node_count = test_nodes.shape[0]
        target = target[test_nodes]
        prediction = prediction[test_nodes,:]

        return average_loss, node_count, prediction, target

    def update_average_loss(self, batch_average_loss, node_count):
        """
        Updating the average loss in the epoch.
        :param batch_average_loss: Loss of the cluster. 
        :param node_count: Number of nodes in currently processed cluster.
        :return average_loss: Average loss in the epoch.
        """
        self.accumulated_loss = self.accumulated_loss + batch_average_loss.item()*node_count
        self.node_count_seen = self.node_count_seen + node_count
        average_loss = self.accumulated_loss / self.node_count_seen
        return average_loss

    def train_val_test(self):
        """
        Training, validation, and test a model per epoch.
        """
        print("Training, validation, and test started.\n")
        train_start_time = time.perf_counter()
        bad_counter = 0
        best_loss = np.inf
        best_epoch = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        for epoch in range(1, self.args.epochs+1):
            epoch_start_time = time.time()
            np.random.shuffle(self.clustering_machine.clusters)
            self.model.train()
            self.node_count_seen = 0
            self.accumulated_loss = 0
            for cluster in self.clustering_machine.clusters:
                self.optimizer.zero_grad()
                batch_average_loss, node_count = self.do_forward_pass(cluster)
                batch_average_loss.backward()
                self.optimizer.step()
                average_loss = self.update_average_loss(batch_average_loss, node_count)
            train_loss = average_loss

            self.model.eval()
            self.node_count_seen = 0
            self.accumulated_loss = 0
            for cluster in self.clustering_machine.clusters:
                batch_average_loss, node_count = self.do_validation(cluster, epoch)
                average_loss = self.update_average_loss(batch_average_loss, node_count)
            val_loss = average_loss

            print("Epoch: {:04d}".format(epoch), 
                "||",
                "time cost: {:.2f}s".format(time.time() - epoch_start_time), 
                "||",
                "train loss: {:.4f}".format(train_loss),
                "val loss: {:.4f}".format(val_loss))

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                bad_counter = 0
                best_model_state = self.model.state_dict()
            else:
                bad_counter += 1

            if bad_counter == self.args.patience:
                break

        self.model.load_state_dict(best_model_state)
        self.model.eval()
        self.node_count_seen = 0
        self.accumulated_loss = 0
        self.predictions = []
        self.targets = []
        for cluster in self.clustering_machine.clusters:
            batch_average_loss, node_count, prediction, target = self.do_prediction(cluster)
            average_loss = self.update_average_loss(batch_average_loss, node_count)
            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        test_loss = average_loss
        self.targets = np.concatenate(self.targets)
        self.predictions = np.concatenate(self.predictions).argmax(1)
        acc_score = metrics.accuracy_score(self.targets, self.predictions)
        macro_f1 = metrics.f1_score(self.targets, self.predictions, average="macro")
        classification_report = metrics.classification_report(self.targets, self.predictions, digits=4)
        print(classification_report)
        # Confusion matrics and AUC
        confusion_matrix = metrics.confusion_matrix(self.targets, self.predictions)
        print(confusion_matrix)
        fpr, tpr, _ = metrics.roc_curve(self.targets, self.predictions)
        auc = metrics.auc(fpr, tpr)
        print("AUC:", auc)

        elaps_time = (time.perf_counter() - train_start_time)/60

        print("Optimization Finished!")
        print("Total time elapsed: {:.2f}min".format(elaps_time))
        print("Best Result:\n",
            "best epoch: {:04d}".format(best_epoch),
            "||",
            "test loss: {:.4f}".format(test_loss),
            "||",
            "accuracy: {:.4f}".format(acc_score),
            "macro-f1: {:.4f}".format(macro_f1))        
