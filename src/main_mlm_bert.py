import argparse
import logging
import random
import time

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM

from data_helpers import *


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
    with open('{}/mlm_{}_50_train.p'.format(args.data_dir, args.data), 'rb') as f:
        train_dataset = pickle.load(f)
    print('Load development data...')
    with open('{}/mlm_{}_50_dev.p'.format(args.data_dir, args.data), 'rb') as f:
        dev_dataset = pickle.load(f)
    print('Load test data...')
    with open('{}/mlm_{}_50_test.p'.format(args.data_dir, args.data), 'rb') as f:
        test_dataset = pickle.load(f)

    collator = MLMCollator(train_dataset.user2id, train_dataset.tok)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collator)

    filename = 'mlm_{}_bert'.format(args.data)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
    emb_layer = model.get_input_embeddings().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
            reviews = reviews.to(device)
            masks = masks.to(device)
            segs = segs.to(device)

            optimizer.zero_grad()

            embs = emb_layer(reviews)

            loss = model(inputs_embeds=embs, attention_mask=masks, token_type_ids=segs, masked_lm_labels=labels)[0]

            loss.backward()
            optimizer.step()

        print('Evaluate model...')
        model.eval()

        losses = list()

        with torch.no_grad():

            for batch in dev_loader:

                labels, users, times, years, months, days, reviews, masks, segs = batch

                labels = labels.to(device)
                reviews = reviews.to(device)
                masks = masks.to(device)
                segs = segs.to(device)

                loss = model(reviews, attention_mask=masks, token_type_ids=segs, masked_lm_labels=labels)[0]

                losses.append(loss.item())

        perplexity_dev = np.exp(np.mean(losses))

        losses = list()

        with torch.no_grad():

            for batch in test_loader:

                labels, users, times, years, months, days, reviews, masks, segs = batch

                labels = labels.to(device)
                reviews = reviews.to(device)
                masks = masks.to(device)
                segs = segs.to(device)

                loss = model(reviews, attention_mask=masks, token_type_ids=segs, masked_lm_labels=labels)[0]

                losses.append(loss.item())

        perplexity_test = np.exp(np.mean(losses))

        print(perplexity_dev, perplexity_test)

        with open('{}/{}.txt'.format(args.results_dir, filename), 'a+') as f:
            f.write('{}\t{}\t{:.0e}\n'.format(perplexity_dev, perplexity_test, args.lr))

        if best_perplexity is None or perplexity_dev < best_perplexity:

            best_perplexity = perplexity_dev
            torch.save(model.state_dict(), '{}/{}.torch'.format(args.trained_dir, filename))


if __name__ == '__main__':

    start_time = time.time()

    main()

    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()
