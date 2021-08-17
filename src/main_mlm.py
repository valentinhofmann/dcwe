import argparse
import logging
import pickle
import random
import time

import torch
from torch import optim
from torch.utils.data import DataLoader

from data_helpers import *
from model import MLMModel


def main():

    logging.disable(logging.WARNING)

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str, required=True, help='Data directory.')
    parser.add_argument('--results_dir', default=None, type=str, required=True, help='Results directory.')
    parser.add_argument('--trained_dir', default=None, type=str, required=True, help='Trained model directory.')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')
    parser.add_argument('--lr', default=None, type=float, required=True, help='Learning rate.')
    parser.add_argument('--n_epochs', default=None, type=int, required=True, help='Number of epochs.')
    parser.add_argument('--lambda_a', default=0, type=float, required=True, help='Regularization constant a.')
    parser.add_argument('--lambda_w', default=0, type=float, required=True, help='Regularization constant w.')
    parser.add_argument('--device', default=None, type=int, required=True, help='Selected CUDA device.')
    parser.add_argument('--data', default=None, type=str, required=True, help='Name of data.')
    parser.add_argument('--social_dim', default=50, type=int, help='Size of social embeddings.')
    parser.add_argument('--gnn', default=None, type=str, help='Type of graph neural network.')
    parser.add_argument('--social_only', default=False, action='store_true', help='Only use social information.')
    parser.add_argument('--time_only', default=False, action='store_true', help='Only use temporal information.')
    args = parser.parse_args()

    print('Load training data...')
    with open('{}/mlm_{}_{}_train.p'.format(args.data_dir, args.data, args.social_dim), 'rb') as f:
        train_dataset = pickle.load(f)
    print('Load development data...')
    with open('{}/mlm_{}_{}_dev.p'.format(args.data_dir, args.data, args.social_dim), 'rb') as f:
        dev_dataset = pickle.load(f)
    print('Load test data...')
    with open('{}/mlm_{}_{}_test.p'.format(args.data_dir, args.data, args.social_dim), 'rb') as f:
        test_dataset = pickle.load(f)

    print('Lambda a: {:.0e}'.format(args.lambda_a))
    print('Lambda w: {:.0e}'.format(args.lambda_w))
    print('Social embeddings dimensionality: {}'.format(args.social_dim))
    print('Number of time units: {}'.format(train_dataset.n_times))
    print('Number of vocabulary items: {}'.format(len(train_dataset.filter_tensor)))

    collator = MLMCollator(train_dataset.user2id, train_dataset.tok)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collator)

    filename = 'mlm_{}'.format(args.data)
    filename += '_{}'.format(args.social_dim)
    if args.social_only:
        filename += '_s'
    elif args.time_only:
        filename += '_t'

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model = MLMModel(
        n_times=train_dataset.n_times,
        social_dim=args.social_dim,
        gnn=args.gnn,
        social_only=args.social_only,
        time_only=args.time_only
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = model.to(device)
    graph_data = train_dataset.graph_data.to(device)
    vocab_filter = train_dataset.filter_tensor.to(device)

    best_result = get_best('{}/{}.txt'.format(args.results_dir, filename), metric='perplexity')
    if best_result:
        best_perplexity = best_result[0]
    else:
        best_perplexity = None
    print('Best perplexity so far: {}'.format(best_perplexity))

    print('Train model...')
    for epoch in range(1, args.n_epochs + 1):

        model.train()

        for i, batch in enumerate(train_loader):

            if i % 1000 == 0:
                print('Processed {} examples...'.format(i * args.batch_size))

            labels, users, times, years, months, days, reviews, masks, segs = batch

            labels = labels.to(device)
            users = users.to(device)
            reviews = reviews.to(device)
            masks = masks.to(device)
            segs = segs.to(device)

            optimizer.zero_grad()

            offset_t0, offset_t1, loss = model(labels, reviews, masks, segs, users, graph_data, times, vocab_filter)

            loss += args.lambda_a * torch.norm(offset_t1, dim=-1).pow(2).mean()

            if not args.social_only:
                loss += args.lambda_w * torch.norm(offset_t1 - offset_t0, dim=-1).pow(2).mean()

            loss.backward()

            optimizer.step()

        print('Evaluate model...')
        model.eval()

        losses = list()

        with torch.no_grad():

            for batch in dev_loader:

                labels, users, times, years, months, days, reviews, masks, segs = batch

                labels = labels.to(device)
                users = users.to(device)
                reviews = reviews.to(device)
                masks = masks.to(device)
                segs = segs.to(device)

                offset_t0, offset_t1, loss = model(labels, reviews, masks, segs, users, graph_data, times, vocab_filter)

                losses.append(loss.item())

        perplexity_dev = np.exp(np.mean(losses))

        losses = list()

        with torch.no_grad():

            for batch in test_loader:

                labels, users, times, years, months, days, reviews, masks, segs = batch

                labels = labels.to(device)
                users = users.to(device)
                reviews = reviews.to(device)
                masks = masks.to(device)
                segs = segs.to(device)

                offset_t0, offset_t1, loss = model(labels, reviews, masks, segs, users, graph_data, times, vocab_filter)

                losses.append(loss.item())

        perplexity_test = np.exp(np.mean(losses))

        print(perplexity_dev, perplexity_test)

        with open('{}/{}.txt'.format(args.results_dir, filename), 'a+') as f:
            f.write('{}\t{}\t{:.0e}\t{:.0e}\t{:.0e}\n'.format(
                perplexity_dev, perplexity_test, args.lr, args.lambda_a, args.lambda_w))

        if best_perplexity is None or perplexity_dev < best_perplexity:

            best_perplexity = perplexity_dev
            torch.save(model.state_dict(), '{}/{}.torch'.format(args.trained_dir, filename))


if __name__ == '__main__':

    start_time = time.time()

    main()

    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()
