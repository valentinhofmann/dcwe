import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GATConv
from transformers import BertModel, BertForMaskedLM

from data_helpers import isin


class MLMModel(nn.Module):
    """"Class to train dynamic contextualized word embeddings with masked language modeling."""

    def __init__(self, n_times=1, social_dim=50, gnn=None, social_only=False, time_only=False):
        """Initialize dynamic contextualized word embeddings model.

        Args:
            n_times: number of time points (only relevant if time is not ablated)
            social_dim: dimensionality of social embeddings
            gnn: type of GNN (currently 'gat' and 'gcn' are possible)
            social_only: use only social information (temporal ablation)
            time_only: use only temporal information (social ablation)
        """

        super(MLMModel, self).__init__()

        # For ablated models
        self.social_only = social_only
        self.time_only = time_only

        # Contextualizing component
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.bert_emb_layer = self.bert.get_input_embeddings()

        # Dynamic component
        if self.social_only:
            self.social_component = SocialComponent(social_dim, gnn)
        elif self.time_only:
            self.offset_components = nn.ModuleList([OffsetComponent() for _ in range(n_times)])
        else:
            self.social_components = nn.ModuleList([SocialComponent(social_dim, gnn) for _ in range(n_times)])

    def forward(self, labels, reviews, masks, segs, users, g_data, times, vocab_filter, embs_only=False):
        """Perform forward pass.

        Args:
            labels: tensor of masked language modeling labels
            reviews: tensor of tokenized reviews
            masks: tensor of attention masks
            segs: tensor of segment indices
            users: tensor of batch user indices
            g_data: graph data object
            times: tensor of batch time points
            vocab_filter: tensor with word types for dynamic component
            embs_only: only compute dynamic type-level embeddings
        """

        # Retrieve BERT input embeddings
        bert_embs = self.bert_emb_layer(reviews)

        # Temporal ablation
        if self.social_only:
            offset_last = None  # No need to compute embeddings at last time point for temporal ablation
            offset_now = torch.cat(
                [self.social_component(bert_embs[i], users[i], g_data) for i, j in enumerate(times)],
                dim=0
            )
            offset_now = offset_now * isin(reviews, vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)

        # Social ablation
        elif self.time_only:
            offset_last = torch.cat(
                [self.offset_components[j](bert_embs[i]) for i, j in enumerate(F.relu(times - 1))],
                dim=0
            )
            offset_now = torch.cat(
                [self.offset_components[j](bert_embs[i]) for i, j in enumerate(times)],
                dim=0
            )
            offset_last = offset_last * isin(reviews, vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)
            offset_now = offset_now * isin(reviews, vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)

        # Full dynamic component
        else:
            offset_last = torch.cat(
                [self.social_components[j](bert_embs[i], users[i], g_data) for i, j in enumerate(F.relu(times - 1))],
                dim=0
            )
            offset_now = torch.cat(
                [self.social_components[j](bert_embs[i], users[i], g_data) for i, j in enumerate(times)],
                dim=0
            )
            offset_last = offset_last * isin(reviews, vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)
            offset_now = offset_now * isin(reviews, vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)

        # Compute dynamic type-level embeddings (input to contextualizing component)
        input_embs = bert_embs + offset_now

        # Only compute dynamic type-level embeddings (not fed into contextualizing component)
        if embs_only:
            return bert_embs, input_embs

        # Pass through contextualizing component
        output = self.bert(inputs_embeds=input_embs, attention_mask=masks, token_type_ids=segs, masked_lm_labels=labels)

        return offset_last, offset_now, output[0]


class SAModel(nn.Module):
    """"Class to train dynamic contextualized word embeddings for sentiment analysis."""

    def __init__(self, n_times=1, social_dim=50, gnn=None):
        """Initialize dynamic contextualized word embeddings model.

        Args:
            n_times: number of time points
            social_dim: dimensionality of social embeddings
            gnn: type of GNN (currently 'gat' and 'gcn' are possible)
        """

        super(SAModel, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_emb_layer = self.bert.get_input_embeddings()
        self.social_components = nn.ModuleList([SocialComponent(social_dim, gnn) for _ in range(n_times)])
        self.linear_1 = nn.Linear(768, 100)
        self.linear_2 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, reviews, masks, segs, users, g_data, times, vocab_filter, embs_only=False):
        """Perform forward pass.

        Args:
            reviews: tensor of tokenized reviews
            masks: tensor of attention masks
            segs: tensor of segment indices
            users: tensor of batch user indices
            g_data: graph data object
            times: tensor of batch time points
            vocab_filter: tensor with word types for dynamic component
            embs_only: only compute dynamic type-level embeddings
        """

        # Retrieve BERT input embeddings
        bert_embs = self.bert_emb_layer(reviews)
        offset_last = torch.cat(
            [self.social_components[j](bert_embs[i], users[i], g_data) for i, j in enumerate(F.relu(times - 1))],
            dim=0
        )
        offset_now = torch.cat(
            [self.social_components[j](bert_embs[i], users[i], g_data) for i, j in enumerate(times)],
            dim=0
        )
        offset_last = offset_last * isin(reviews, vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)
        offset_now = offset_now * isin(reviews, vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)

        # Compute dynamic type-level embeddings (input to contextualizing component)
        input_embs = bert_embs + offset_now

        # Only compute dynamic type-level embeddings (not fed into contextualizing component)
        if embs_only:
            return bert_embs, input_embs

        # Pass through contextualizing component
        output_bert = self.dropout(self.bert(inputs_embeds=input_embs, attention_mask=masks, token_type_ids=segs)[1])
        h = self.dropout(torch.tanh(self.linear_1(output_bert)))
        output = torch.sigmoid(self.linear_2(h)).squeeze(-1)

        return offset_last, offset_now, output


class SocialComponent(nn.Module):
    """"Class implementing the social part of the dynamic component."""

    def __init__(self, social_dim=50, gnn=None):
        super(SocialComponent, self).__init__()
        self.gnn_component = GNNComponent(social_dim, gnn)
        self.linear_1 = nn.Linear(768 + social_dim, 768)
        self.linear_2 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)

    def forward(self, embs, users, graph_data):
        user_output = self.gnn_component(users, graph_data)
        user_output = user_output.unsqueeze(0).expand(embs.size(0), -1)
        h = torch.cat((embs, user_output), dim=-1)
        h = self.dropout(torch.tanh(self.linear_1(h)))
        offset = self.linear_2(h).unsqueeze(0)
        return offset


class GNNComponent(nn.Module):
    """"Class implementing the GNN of the dynamic component."""

    def __init__(self, social_dim=50, gnn=None):
        super(GNNComponent, self).__init__()
        self.social_dim = social_dim
        self.gnn = gnn
        if self.gnn == 'gcn':
            self.conv_1 = GCNConv(self.social_dim, self.social_dim)
            self.conv_2 = GCNConv(self.social_dim, self.social_dim)
        elif self.gnn == 'gat':
            self.conv_1 = GATConv(self.social_dim, self.social_dim, heads=4, dropout=0.6, concat=False)
            self.conv_2 = GATConv(self.social_dim, self.social_dim, heads=4, dropout=0.6, concat=False)
        else:
            self.linear_1 = nn.Linear(self.social_dim, self.social_dim)
            self.linear_2 = nn.Linear(self.social_dim, self.social_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, users, graph_data):
        if self.gnn == 'gcn' or self.gnn == 'gat':
            h = self.dropout(torch.tanh(self.conv_1(graph_data.x, graph_data.edge_index)))
            return self.dropout(torch.tanh(self.conv_2(h, graph_data.edge_index)))[users]
        else:
            h = self.dropout(torch.tanh(self.linear_1(graph_data.x[users])))
            return self.dropout(torch.tanh(self.linear_2(h)))


class OffsetComponent(nn.Module):
    """"Class implementing the dynamic component for social ablation."""

    def __init__(self):
        super(OffsetComponent, self).__init__()
        self.linear_1 = nn.Linear(768, 768)
        self.linear_2 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)

    def forward(self, embs):
        h = self.dropout(torch.tanh(self.linear_1(embs)))
        offset = self.linear_2(h).unsqueeze(0)
        return offset


class SABert(nn.Module):
    """"Class to train non-dynamic contextualized word embeddings (BERT) for sentiment analysis."""

    def __init__(self):
        super(SABert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear_1 = nn.Linear(768, 100)
        self.linear_2 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, reviews, masks, segs):
        output_bert = self.dropout(self.bert(reviews, attention_mask=masks, token_type_ids=segs)[1])
        h = self.dropout(torch.tanh(self.linear_1(output_bert)))
        output = torch.sigmoid(self.linear_2(h)).squeeze(-1)
        return output
