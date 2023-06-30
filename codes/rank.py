#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from model import KGEModel
from dataloader import TestDataset

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Given an model_path and optional data_path, load model and test set and for every link prediction triple save top k predictions to model_path/ranking.csv ')
    parser.add_argument('model_path', type=Path, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--seed', type=int, default=None, help='Randomgenerator seed for reproducibility.')
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-k', '--top_k', type=int, default=20, help='Number of top k ranks saved.')
    return parser.parse_args(args)


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def main(args):
    torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    start_datetime = datetime.datetime.now()

    # Get model args from saved config
    with open(os.path.join(args.model_path, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.countries = argparse_dict['countries'] # TODO
    if args.data_path is None:                  # Makes it possible to change the dataset
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    args.gamma = argparse_dict['gamma']
    args.regularization = argparse_dict['regularization']
    args.negative_sample_size = argparse_dict['negative_sample_size']
    args.max_steps = argparse_dict['max_steps']

    print(args)

    # Load datasets and entity dicts
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        id2entity = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
            id2entity[int(eid)] = entity

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))


    # All true triples
    all_true_triples = train_triples + valid_triples + test_triples

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )

    logging.info('Loading checkpoint %s...' % args.model_path)
    checkpoint = torch.load(os.path.join(args.model_path, 'checkpoint'))
    kge_model.load_state_dict(checkpoint['model_state_dict'])

    logging.info('Model Parameter Configuration:')

    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()

    kge_model.eval()


    test_dataloader_head = DataLoader(
        TestDataset(
            test_triples,
            all_true_triples,
            args.nentity,
            args.nrelation,
            'head-batch'
        ),
        batch_size=args.test_batch_size,
    num_workers=max(1, args.cpu_num//2),
    collate_fn=TestDataset.collate_fn
    )

    test_dataloader_tail = DataLoader(
    TestDataset(
    test_triples,
    all_true_triples,
    args.nentity,
    args.nrelation,
    'tail-batch'
),
    batch_size=args.test_batch_size,
    num_workers=max(1, args.cpu_num//2),
    collate_fn=TestDataset.collate_fn
    )
    test_dataset_list = [test_dataloader_head, test_dataloader_tail]

    logs = []

    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])

    top5 = dict()
    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                if args.cuda:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                batch_size = positive_sample.size(0)

                score = kge_model((positive_sample, negative_sample), mode)

                score += filter_bias

                # Explicitly sort all the entities to ensure that there is no test exposure bias
                # Get indeces of scores sorted descending
                argsort = torch.argsort(score, dim = 1, descending=True)

                # Get true entity for head/tail query
                if mode == 'head-batch':
                    positive_arg = positive_sample[:, 0]
                elif mode == 'tail-batch':
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError('mode %s not supported' % mode)


                for i in range(batch_size):
                    # Notice that argsort is not ranking
                    # Look where the index of the true entity is equal to the indexes stored in argsort,
                    # This results in boolean a tensor of size num_entities with ONE True entry
                    # Get the index of the True entry with .nonzero()
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1


                    #ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    # logs.append({
                    #     'MRR': 1.0/ranking,
                    #     'MR': float(ranking),
                    #     'HITS@1': 1.0 if ranking <= 1 else 0.0,
                    #     'HITS@3': 1.0 if ranking <= 3 else 0.0,
                    #     'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    # })


                    # keys of top5 dict consists of query and ground truth because query wasn't unique
                    cur_positive_sample = positive_sample[i]
                    if mode == 'head-batch':
                        cur_query = (None, cur_positive_sample[1].item(), cur_positive_sample[2].item())
                        cur_top5 = argsort[i, :args.top_k].tolist()
                        if cur_query in top5.keys():
                            print('Warning: Query was already in toy_5 dict.')
                        top5[(cur_query, cur_positive_sample[0].item()) ] = cur_top5
                    elif mode == 'tail-batch':
                        cur_query = (cur_positive_sample[0].item(), cur_positive_sample[1].item(), None)
                        cur_top5 = argsort[i, :args.top_k].tolist()
                        if cur_query in top5.keys():
                            print('Warning: Query was already in toy_5 dict.')
                        top5[(cur_query, cur_positive_sample[2].item())] = cur_top5

                step += 1

    columns = ['query_head', 'query_relation', 'query_tail', 'ground_truth'] + [f'top_{k}' for k in range(1, args.top_k+1)]
    data = []
    for key in top5:
        ((query_head, query_relation, query_tail), answer) = key
        topk = top5[key]
        data.append([query_head, query_relation, query_tail, answer]+topk)

    df = pd.DataFrame(data=data, columns=columns, dtype=pd.Int64Dtype())

    output_path = args.model_path.joinpath('ranking.csv')

    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
