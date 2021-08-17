import pickle
from collections import defaultdict, Counter

import networkx as nx
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformers import BertTokenizer

stops = set(stopwords.words('english'))


class MLMDataset(Dataset):
    """Dataset class for masked language modeling."""

    def __init__(self, name, split, social_dim, data_dir):

        self.tok = BertTokenizer.from_pretrained('bert-base-uncased')

        data = pd.read_csv('{}/{}_{}.csv'.format(data_dir, name, split), parse_dates=['time'])

        data.dropna(inplace=True)
        data.time = pd.to_datetime(data.time)
        data.reset_index(inplace=True, drop=True)

        self.users = list(data.user)
        self.years = list(data.year)
        self.months = list(data.month)
        self.days = list(data.day)

        if name == 'reddit':
            self.times = list(data.month.apply(convert_times, name=name))
        elif name == 'arxiv' or name == 'ciao' or name == 'yelp':
            self.times = list(data.year.apply(convert_times, name=name))
        self.n_times = len(set(self.times))

        vocab = defaultdict(Counter)
        for text, time in zip(data.text, self.times):
            vocab[time].update(text.strip().split())
        for time in vocab:
            total = sum(vocab[time].values())
            vocab[time] = {w: count / total for w, count in vocab[time].items()}
        w_counts = dict()
        for time in vocab:
            for w in vocab[time]:
                w_counts[w] = w_counts.get(w, 0) + vocab[time][w]
        w_top = sorted(w_counts.keys(), key=lambda x: w_counts[x], reverse=True)[:100000]
        filter_list = [w for w in w_top if w not in stops and w in self.tok.vocab and w.isalpha()]
        self.filter_tensor = torch.tensor([t for t in self.tok.encode(filter_list) if t >= 2100])

        self.reviews = list(data.text.apply(self.tok.encode, add_special_tokens=True))
        self.reviews = truncate(self.reviews)

        self.user2id, self.graph_data = load_external_data(name, social_dim, data_dir)

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):

        user = self.users[idx]
        year = self.years[idx]
        month = self.months[idx]
        day = self.days[idx]
        time = self.times[idx]
        review = self.reviews[idx]

        return user, time, year, month, day, review


class SADataset(Dataset):
    """Dataset class for sentiment analysis."""

    def __init__(self, name, split, social_dim, data_dir):

        self.tok = BertTokenizer.from_pretrained('bert-base-uncased')

        data = pd.read_csv('{}/{}_{}.csv'.format(data_dir, name, split), parse_dates=['time'])

        data.dropna(inplace=True)
        data.time = pd.to_datetime(data.time)
        data.reset_index(inplace=True, drop=True)

        self.labels = list(data.label)
        self.users = list(data.user)
        self.years = list(data.year)
        self.months = list(data.month)
        self.days = list(data.day)

        self.times = list(data.year.apply(convert_times, name=name))
        self.n_times = len(set(self.times))

        vocab = defaultdict(Counter)
        for text, time in zip(data.text, self.times):
            vocab[time].update(text.strip().split())
        for time in vocab:
            total = sum(vocab[time].values())
            vocab[time] = {w: count / total for w, count in vocab[time].items()}
        w_counts = dict()
        for time in vocab:
            for w in vocab[time]:
                w_counts[w] = w_counts.get(w, 0) + vocab[time][w]
        w_top = sorted(w_counts.keys(), key=lambda x: w_counts[x], reverse=True)[:100000]
        filter_list = [w for w in w_top if w not in stops and w in self.tok.vocab and w.isalpha()]
        self.filter_tensor = torch.tensor([t for t in self.tok.encode(filter_list) if t >= 2100])

        self.reviews = list(data.text.apply(self.tok.encode, add_special_tokens=True))
        self.reviews = truncate(self.reviews)

        self.user2id, self.graph_data = load_external_data(name, social_dim, data_dir)

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):

        label = self.labels[idx]
        user = self.users[idx]
        year = self.years[idx]
        month = self.months[idx]
        day = self.days[idx]
        time = self.times[idx]
        review = self.reviews[idx]

        return label, user, time, year, month, day, review


class MLMCollator:
    """Collator class for masked language modeling."""

    def __init__(self, user2id, tok, mlm_p=0.15):

        self.user2id = user2id
        self.tok = tok
        self.mlm_p = mlm_p

    def __call__(self, batch):

        batch_size = len(batch)

        users = torch.tensor([self.user2id[u] for u, _, _, _, _, _ in batch]).long()
        times = torch.tensor([t for _, t, _, _, _, _ in batch]).long()
        years = torch.tensor([y for _, _, y, _, _, _ in batch]).long()
        months = torch.tensor([m for _, _, _, m, _, _ in batch]).long()
        days = torch.tensor([d for _, _, _, _, d, _ in batch]).long()
        reviews = [r for _, _, _, _, _, r in batch]

        max_len = max(len(r) for r in reviews)
        reviews_pad = torch.zeros((batch_size, max_len)).long()
        masks_pad = torch.zeros((batch_size, max_len)).long()
        segs_pad = torch.zeros((batch_size, max_len)).long()

        for i, r in enumerate(reviews):
            reviews_pad[i, :len(r)] = torch.tensor(r)
            masks_pad[i, :len(r)] = 1

        labels = reviews_pad.clone()
        p_matrix = torch.full(labels.shape, self.mlm_p)

        special_mask = [self.tok.get_special_tokens_mask(l, already_has_special_tokens=True) for l in labels.tolist()]
        p_matrix.masked_fill_(torch.tensor(special_mask, dtype=torch.bool), value=0.0)
        padding_mask = labels.eq(self.tok.pad_token_id)
        p_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(p_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        reviews_pad[indices_replaced] = self.tok.convert_tokens_to_ids(self.tok.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tok), labels.shape, dtype=torch.long)
        reviews_pad[indices_random] = random_words[indices_random]

        return labels, users, times, years, months, days, reviews_pad, masks_pad, segs_pad


class SACollator:
    """Collator class for sentiment analysis."""

    def __init__(self, user2id):

        self.user2id = user2id

    def __call__(self, batch):

        batch_size = len(batch)

        labels = torch.tensor([l for l, _, _, _, _, _, _ in batch]).float()
        users = torch.tensor([self.user2id[u] for _, u, _, _, _, _, _ in batch]).long()
        times = torch.tensor([t for _, _, t, _, _, _, _ in batch]).long()
        years = torch.tensor([y for _, _, _, y, _, _, _ in batch]).long()
        months = torch.tensor([m for _, _, _, _, m, _, _ in batch]).long()
        days = torch.tensor([d for _, _, _, _, _, d, _ in batch]).long()
        reviews = [r for _, _, _, _, _, _, r in batch]

        max_len = max(len(r) for r in reviews)
        reviews_pad = torch.zeros((batch_size, max_len)).long()
        masks_pad = torch.zeros((batch_size, max_len)).long()
        segs_pad = torch.zeros((batch_size, max_len)).long()

        for i, r in enumerate(reviews):
            reviews_pad[i, :len(r)] = torch.tensor(r)
            masks_pad[i, :len(r)] = 1

        return labels, users, times, years, months, days, reviews_pad, masks_pad, segs_pad


def convert_times(time, name):

    if name == 'arxiv':
        return time - 2001

    elif name == 'ciao':
        return time - 2000

    elif name == 'yelp':
        return time - 2010

    elif name == 'reddit':
        return time - 9


def truncate(reviews):

    truncated = list()

    for r in reviews:
        if len(r) > 512:
            r = r[:256] + r[-256:]
        truncated.append(r)

    return truncated


def load_external_data(name, social_dim, data_dir):
    """Function to load and preprocess graph data and node2vec input embeddings."""

    with open('{}/{}_edges.p'.format(data_dir, name), 'rb') as f:
        edge_set = pickle.load(f)
        if name == 'reddit' or name == 'arxiv':
            edge_set = set(e[:2] for e in edge_set if e[2] > 0.01)

    with open('{}/{}_users.p'.format(data_dir, name), 'rb') as f:
        users = pickle.load(f)

    if name == 'arxiv' or name == 'reddit':
        graph = nx.Graph()
    else:
        graph = nx.DiGraph()

    graph.add_nodes_from(users)
    graph.add_edges_from(edge_set)

    assert graph.number_of_nodes() == len(users)
    assert graph.number_of_edges() == len(edge_set)

    user2id = {u: i for i, u in enumerate(users)}

    vectors = dict()

    with open('{}/{}_vectors_{}.txt'.format(data_dir, name, social_dim), 'r') as f:

        for i, l in enumerate(f):

            if i == 0:
                continue

            if l.strip() == '':
                continue

            if name == 'ciao':
                vectors[int(l.strip().split()[0])] = np.array(l.strip().split()[1:], dtype=float)
            elif name == 'arxiv' or name == 'reddit' or name == 'yelp':
                vectors[str(l.strip().split()[0])] = np.array(l.strip().split()[1:], dtype=float)

    vector_matrix = np.zeros((len(users), social_dim))

    for i, n in enumerate(users):
        vector_matrix[i, :] = vectors[n]

    x = torch.tensor(vector_matrix, dtype=torch.float)

    a = nx.adjacency_matrix(graph, nodelist=users)
    edge_index = torch.tensor(np.stack((a.tocoo().row, a.tocoo().col)).astype(np.int32), dtype=torch.long)

    return user2id, Data(edge_index=edge_index, x=x)


def get_best(file, metric):

    try:
        results = list()
        with open(file, 'r') as f:
            for l in f:
                if l.strip() == '':
                    continue
                results.append(tuple([float(v) for v in l.strip().split()]))
        if metric == 'perplexity':
            return min(results)
        elif metric == 'f1':
            return max(results)

    except FileNotFoundError:
        return None


def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)
