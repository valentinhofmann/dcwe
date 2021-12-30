import argparse
import logging

from data_helpers import *


def main():

    logging.disable(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str, required=True, help='Data directory.')
    parser.add_argument('--data', default=None, type=str, required=True, help='Name of data.')
    parser.add_argument('--task', default=None, type=str, required=True, help='Name of task.')
    parser.add_argument('--social_dim', default=0, type=int, required=True, help='Size of social embeddings.')
    args = parser.parse_args()

    for split in ['train', 'dev', 'test']:
        if args.task == 'sa':
            dataset_split = SADataset(args.data, split=split, social_dim=args.social_dim, data_dir=args.data_dir)
        elif args.task == 'mlm':
            dataset_split = MLMDataset(args.data, split=split, social_dim=args.social_dim, data_dir=args.data_dir)
        with open('{}/{}_{}_{}_{}.p'.format(args.data_dir, args.task, args.data, args.social_dim, split), 'wb') as f:
            pickle.dump(dataset_split, f)


if __name__ == '__main__':
    main()
