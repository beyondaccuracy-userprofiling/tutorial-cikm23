import torch

import numpy as np
from parser import parameter_parser
from clustering import ClusteringMachine
from clustergnn import ClusterGNNTrainer

from utils import tab_printer, graph_reader, field_reader, target_reader, label_reader

from fairness import Fairness

def main():
    """
    Parsing command line parameters, reading data, graph decomposition, fitting and scoring the model.
    """
    args = parameter_parser()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    tab_printer(args)
    graph = graph_reader(args.edge_path)
    field_index = field_reader(args.field_path)
    target = target_reader(args.target_path)
    user_labels = label_reader(args.labels_path)

    clustering_machine = ClusteringMachine(args, graph, field_index, target)
    clustering_machine.decompose()
    gnn_trainer = ClusterGNNTrainer(args, clustering_machine)
    gnn_trainer.train_val_test()

    if args.compute_fairness:
        ## Compute fairness metrics
        print("Fairness metrics on sensitive attributes '{}':".format(args.sens_attr))
        fair_obj = Fairness(user_labels, clustering_machine.sg_test_nodes[0], gnn_trainer.targets, gnn_trainer.predictions, args.sens_attr)
        fair_obj.statistical_parity()
        fair_obj.equal_opportunity()
        fair_obj.overall_accuracy_equality()
        fair_obj.treatment_equality()

if __name__ == "__main__":
    main()
