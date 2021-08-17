import argparse
import logging
import random
import time

import torch
from sklearn.metrics import f1_score
from torch import optim, nn
from torch.utils.data import DataLoader

from data_helpers import *
from model import SABert


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
    parser.add_argument('--device', default=None, type=int, required=True, help='Selected CUDA device.')
    parser.add_argument('--data', default=None, type=str, required=True, help='Name of data.')
    args = parser.parse_args()

    print('Load training data...')
    with open('{}/sa_{}_50_train.p'.format(args.data_dir, args.data), 'rb') as f:
        train_dataset = pickle.load(f)
    print('Load development data...')
    with open('{}/sa_{}_50_dev.p'.format(args.data_dir, args.data), 'rb') as f:
        dev_dataset = pickle.load(f)
    print('Load test data...')
    with open('{}/sa_{}_50_test.p'.format(args.data_dir, args.data), 'rb') as f:
        test_dataset = pickle.load(f)

    collator = SACollator(train_dataset.user2id)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collator)

    filename = 'sa_{}_bert'.format(args.data)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model = SABert().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    best_result = get_best('{}/{}.txt'.format(args.results_dir, filename), metric='f1')
    if best_result:
        best_f1 = best_result[0]
    else:
        best_f1 = None
    print('Best F1 so far: {}'.format(best_f1))

    print('Train model...')
    for epoch in range(1, args.n_epochs + 1):

        model.train()

        for i, batch in enumerate(train_loader):

            if i % 1000 == 0:
                print('Processed {} examples...'.format(i * args.batch_size))

            labels, users, times, years, months, days, reviews, masks, segs = batch

            labels = labels.to(device)
            reviews = reviews.to(device)
            masks = masks.to(device)
            segs = segs.to(device)

            optimizer.zero_grad()

            output = model(reviews, masks, segs)

            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

        print('Evaluate model...')
        model.eval()

        y_true = list()
        y_pred = list()

        with torch.no_grad():

            for batch in dev_loader:

                labels, users, times, years, months, days, reviews, masks, segs = batch

                labels = labels.to(device)
                reviews = reviews.to(device)
                masks = masks.to(device)
                segs = segs.to(device)

                output = model(reviews, masks, segs)

                y_true.extend(labels.tolist())
                y_pred.extend(torch.round(output).tolist())

        f1_dev = f1_score(y_true, y_pred, average='macro')

        y_true = list()
        y_pred = list()

        with torch.no_grad():

            for batch in test_loader:

                labels, users, times, years, months, days, reviews, masks, segs = batch

                labels = labels.to(device)
                reviews = reviews.to(device)
                masks = masks.to(device)
                segs = segs.to(device)

                output = model(reviews, masks, segs)

                y_true.extend(labels.tolist())
                y_pred.extend(torch.round(output).tolist())

        f1_test = f1_score(y_true, y_pred, average='macro')

        print(f1_dev, f1_test)

        with open('{}/{}.txt'.format(args.results_dir, filename), 'a+') as f:
            f.write('{}\t{}\t{:.0e}\n'.format(f1_dev, f1_test, args.lr))

        if best_f1 is None or f1_dev > best_f1:

            best_f1 = f1_dev
            torch.save(model.state_dict(), '{}/{}.torch'.format(args.trained_dir, filename))


if __name__ == '__main__':

    start_time = time.time()

    main()

    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()
