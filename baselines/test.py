from util.dataMR import split_data
from util.options import args_parser
args = args_parser()
benchmark_dataset, fliter_dataset_train, fliter_dataset_test = split_data(args);


