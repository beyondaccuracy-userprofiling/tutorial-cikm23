import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description = "Run CatGCN.")

    parser.add_argument("--gpu",
                        type = int,
                        default = 0,
                        help = "GPU device")

    parser.add_argument("--edge-path",
                        nargs = "?",
                        default = "./input/user_edge.csv",
	                    help = "Edge list csv.")

    parser.add_argument("--field-path",
                        nargs = "?",
                        default = "./input/user_field.npy",
	                    help = "Field npy.")

    parser.add_argument("--target-path",
                        nargs = "?",
                        default = "./input/user_age.csv",
	                    help = "Target classes csv.")

    parser.add_argument("--clustering-method",
                        nargs = "?",
                        default = "none",
	                    help = "Clustering method for graph decomposition, use 'metis', 'random', or 'none'.")

    parser.add_argument('--graph-refining', 
                        nargs = "?",
                        default='agc', 
                        help="Optimize the field feature, use 'agc', 'fignn', or 'none'.")

    parser.add_argument('--aggr-pooling', 
                        nargs = "?", 
                        default='mean',
                        help="Aggregate the field feature. Default is 'mean'.")

    parser.add_argument('--bi-interaction', 
                        nargs = "?",
                        default='nfm', 
                        help="Compute the user feature with nfm, use 'nfm' or 'none'.")

    parser.add_argument('--aggr-style', 
                        nargs = "?",
                        default='sum',
                        help="Aggregate the user feature, use 'sum' or 'none'.")

    parser.add_argument('--graph-layer', 
                        nargs = "?",
                        default='sgc', 
                        help="Optimize the user feature, use 'pna', 'sgc', 'appnp', etc.")

    parser.add_argument('--weight-balanced', 
                        nargs = "?", 
                        default='True', 
                        help="Adjust weights inversely proportional to class frequencies.")

    parser.add_argument("--epochs",
                        type = int,
                        default = 9999,
	                    help = "Number of training epochs. Default is 9999.")

    parser.add_argument("--patience",
                        type = int,
                        default = 10,
	                    help = "Number of training patience. Default is 10.")

    parser.add_argument("--seed",
                        type = int,
                        default = 42,
	                    help = "Random seed for train-test split. Default is 42.")

    parser.add_argument("--train-ratio",
                        type = float,
                        default = 0.8,
	                    help = "Train data ratio. Default is 0.8.")
    
    parser.add_argument("--balance-ratio",
                        type = float,
                        default = 0.5,
	                    help = "Balance ratio parameter when aggr_style is 'sum'. Default is 0.5.")

    parser.add_argument("--dropout",
                        type = float,
                        default = 0.5,
	                    help = "Dropout parameter. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.1,
	                    help = "Learning rate. Default is 0.1.")

    parser.add_argument('--weight-decay', 
                        type=float, 
                        default=1e-5, 
                        help='Weight decay (L2 loss on parameters).')
    
    parser.add_argument("--diag-probe",
                        type = float,
                        default = 1.,
	                    help = "Diag probe coefficient. Default is 1.0.")

    parser.add_argument("--cluster-number",
                        type = int,
                        default = 100,
                        help = "Number of clusters extracted. Default is 100.")

    parser.add_argument("--field-dim",
                        type = int,
                        default = 64,
	                    help = "Number of field dims. Default is 64.")
    
    parser.add_argument("--nfm-units",
                        type=str, 
                        default="64", 
                        help="Hidden units for local interaction modeling, splitted with comma, maybe none.")

    parser.add_argument("--grn-units",
                        type=str, 
                        default="64", 
                        help="Hidden units for global interaction modeling, splitted with comma, maybe none.")

    parser.add_argument("--gnn-units",
                        type=str, 
                        default="64",
                        help="Hidden units for baseline models, splitted with comma, maybe none.")

    parser.add_argument("--gnn-hops",
                        type = int,
                        default = 1,
                        help = "Hops number of pure neighborhood aggregation. Default is 1.")

    parser.add_argument("--num-steps",
                        type = int,
                        default = 2,
                        help = "GRU steps for FiGNN. Default is 2.")
    
    parser.add_argument("--multi-heads",
                        type=str,
                        default="8,1", 
                        help="Multi heads in each gat layer, splitted with comma.")
    
    parser.add_argument("--alpha",
                        type = float,
                        default = 0.5,
	                    help = "Alpha coefficient for GCNII. Default is 0.5.")

    parser.add_argument("--theta",
                        type = float,
                        default = 0.5,
	                    help = "Theta coefficient for GCNII. Default is 0.5.")

    parser.add_argument("--gat-units",
                        type=str, 
                        default="64",
                        help="Hidden units for global gat part, splitted with comma, maybe none.")

    # New parameter for computing fairness
    parser.add_argument("--compute-fairness",
                        type = bool,
                        default = False,
                        help = "Whether to compute fairness metrics.")

    parser.add_argument("--labels-path",
                        nargs="?",
                        default="./input/user_labels.csv",
                        help="Labels csv path.")

    parser.add_argument("--sens-attr",
                        type=str,
                        default="gender",
                        help="Sensitive attribute for fairness computation.")

    parser.add_argument("--label",
                        type=str,
                        default="",
                        help="Classification label.")

    return parser.parse_args()
