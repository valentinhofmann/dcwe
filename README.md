# Dynamic Contextualized Word Embeddings

This repository contains the code to train **Dynamic Contextualized Word Embeddings** (DCWEs), proposed in the ACL paper [Dynamic Contextualized Word Embeddings](https://aclanthology.org/2021.acl-long.542.pdf).
DCWEs represent the meaning of words as a function of both linguistic and extralinguistic (social and temporal) context.

# Dependencies

The code requires `Python>=3.6`, `numpy>=1.18`, `torch>=1.2`, `torch_geometric>=1.6`, and `transformers>=2.5`.

# Usage

## Training of DCWEs

To train DCWEs using masked language modeling, run the following command:

    python src/main_mlm.py \
	--data_dir $DATA_DIR \
	--results_dir $RESULTS_DIR \
	--trained_dir $TRAINED_DIR \
	--data $DATA \
	--device $DEVICE \
	--batch_size $BATCH_SIZE \
	--lr $LR \
	--n_epochs $N_EPOCHS \
	--lambda_a $LAMBDA_A \
	--lambda_w $LAMBDA_W \
	--social_dim $SOCIAL_DIM \
	--gnn $GNN

The parameters have the following meaning:

- `$DATA_DIR`: directory containing the dataset (split into train, dev, and test) on which to train DCWEs
- `$RESULTS_DIR`, `$TRAINED_DIR`: directories for storing the results and trained embedding models
- `$DATA`: name of dataset
- `$DEVICE`: CUDA device on which to train DCWEs
- `$BATCH_SIZE`: batch size for training DCWEs
- `$LR`: learning rate
- `$N_EPOCHS`: number of epochs
- `$LAMBDA_A`: regularization constant for anchoring prior
- `$LAMBDA_W`: regularization constant for random walk prior
- `$SOCIAL_DIM`: dimensionality of social embeddings
- `$GNN`: type of GNN for dynamic component (currently GAT and GCN are possible)

In addition, you can use the parameters `--social_only` and `--time_only` to train models that use only social information (temporal ablation) or temporal information (social ablation).

DCWEs can also be trained on downstream tasks. The repository contains example code for sentiment analysis:

    python src/main_sa.py \
	--data_dir $DATA_DIR \
	--results_dir $RESULTS_DIR \
	--trained_dir $TRAINED_DIR \
	--data $DATA \
	--device $DEVICE \
	--batch_size $BATCH_SIZE \
	--lr $LR \
	--n_epochs $N_EPOCHS \
	--lambda_a $LAMBDA_A \
	--lambda_w $LAMBDA_W \
	--social_dim $SOCIAL_DIM \
	--gnn $GNN

The parameters are as for training DCWEs on masked language modeling.

We also provide code to train baseline models (BERT without dynamic component) in `src/main_mlm_bert.py` and `src/main_sa_bert.py`.

## Data Preparation

The file `src/data_helpers.py` contains PyTorch dataset classes that can be used for data managing and loading.
For large amounts of data, we found it useful to first create and pickle PyTorch datasets, which can then directly be loaded for training.
The file `src/pickle_data.py` contains code to do so.
As an example for how to preprocess the data and train the node2vec input embeddings, the repository contains code for the YELP dataset in `data`.

Depending on your data, it might be necessary to adapt these parts of the code.

# Citation

If you use the code in this repository, please cite the following paper:

```
@inproceedings{hofmann2021dcwe,
    title = {Dynamic Contextualized Word Embeddings},
    author = {Hofmann, Valentin and Pierrehumbert, Janet and Sch{\"u}tze, Hinrich},
    booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
    year = {2021}
}
```