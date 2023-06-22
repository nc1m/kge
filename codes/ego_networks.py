"""Module docstring.

continue
"""
import logging
import argparse
from pathlib import Path
import datetime
import json

import torch
import pandas as pd

import const

def parse_args():
    """Parse arguments.

    continue
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('data_path', type=Path, help='Path to dataset dir.')
    parser.add_argument('-s', action='store_true', help='Set to store metadate in data_path/metadata.json.')
    parser.add_argument('-f', action='store_true', help='Set to NOT add reverse edges')
    parser.add_argument('--mode', default='jaccard', type=str,
                        choices=const.EGO_NETWORK_MODES, help='Choos the set metric.')
    parser.add_argument('--check', '-c', action='store_true', help='Check if triple are true negatives.')
    return parser.parse_args()


def read_triple(file_path, entity2id, relation2id):
    """
    Read triples and map them into ids.
    """
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def find_neighbor_sets(tensor_triples, nentity):
    """Compute neighbor sets.


    """
    neighbor_sets = {}
    for e in range(nentity):
        neighbor_sets[e] = tensor_triples[tensor_triples[:, 0]== e][:, 2].numpy()
    return neighbor_sets


def set_similarity(set1, set2, mode):
    if ((len(set1) == 0) and (len(set2) == 0)):
        return 0
    if mode == "count":
        return len(list(set(set1).intersection(set2)))
    if mode == "jaccard":
        intersection = set(set1).intersection(set2)
        return len(intersection) / (len(set1) + len(set2) - len(intersection))


def find_hard_negatives(entity, neighbor_sets, nentity, mode):
    '''
    finds a list of candidate entities that share common neighbours with the given entity.
    List is ordered based on the score of the similarity.
    Goal is to find the most probable hard negative entities
    Returns a list of tuple of entities and their scores. List is sorted based on score in a descending way
    '''
    hard_negatives = []
    for candidate_entity in range(nentity):
        score = set_similarity(
            neighbor_sets[entity], neighbor_sets[candidate_entity], mode=mode)
        if score > 0:
            if candidate_entity != entity:
                hard_negatives.append((candidate_entity, score))
    hard_negatives.sort(key=lambda tup: tup[1], reverse=True)
    '''
    TODO: eliminate original triple if its contained in the set.
    currently it is handled while returning the hard negatives as [1:]
    '''
    return hard_negatives


def find_all_hard_negatives_per_entity(neighbor_sets, nentity, mode):
    entity_to_hard_negatives = {}
    count = 0
    # ent_s = len(all_entities)
    for ent in range(nentity):
        count += 1
        if count % 500 == 0:
            print("Percentage: ", str((count) * 100 / nentity))
        # TODO: BE VERY CAREFUL ABOUT WHICH METHOD YOU APPLY BELOW
        # this method uses local variables
        entity_to_hard_negatives[ent] = find_hard_negatives(
            ent, neighbor_sets, nentity, mode)
        # this method uses global variable
        # entity_to_hard_negatives[ent] = find_hard_negatives_without_variables(ent, mode)
    return entity_to_hard_negatives


def main(args):
    """
    """
    start_datetime = datetime.datetime.now()

    print(f'Arguments passed to the script: {args}')


    # Load datasets and entity dicts
    with open(args.data_path.joinpath('entities.dict')) as fin:
        entity2id = dict()
        id2entity = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
            id2entity[int(eid)] = entity

    with open(args.data_path.joinpath('relations.dict')) as fin:
        relation2id = dict()
        id2relation = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
            id2relation[int(rid)] = relation

    nentity = len(entity2id)
    # nrelation = len(relation2id)

    train_triples = read_triple(args.data_path.joinpath(
        'train.txt'), entity2id, relation2id)
    print('#train: %d', len(train_triples))
    valid_triples = read_triple(args.data_path.joinpath(
        'valid.txt'), entity2id, relation2id)
    print('#valid: %d', len(valid_triples))
    test_triples = read_triple(args.data_path.joinpath(
        'test.txt'), entity2id, relation2id)
    print('#test: %d', len(test_triples))

    all_true_triples = train_triples + valid_triples + test_triples

    triple_df = pd.DataFrame(all_true_triples, columns=[
        "head", "relation", "tail"], dtype=int)
    tensor_heads = torch.tensor(triple_df["head"])
    tensor_tails = torch.tensor(triple_df["tail"])
    tensor_relations = torch.tensor(triple_df['relation'])
    if args.f:
        tensor_triples = torch.stack([tensor_heads, tensor_relations, tensor_tails], dim=1)
    else:
        # add inverse triples
        tensor_triples = torch.cat((torch.stack([tensor_heads, tensor_relations, tensor_tails], dim=1), torch.stack([tensor_tails, tensor_relations, tensor_heads], dim=1)), dim=0)

    print("triples are converted in a tensor form")
    neighbor_sets = find_neighbor_sets(tensor_triples, nentity)
    print("neighboring sets are calculated")
    all_hard_negatives_t = find_all_hard_negatives_per_entity(
        neighbor_sets, nentity, mode=args.mode)
    print("all_hard_negatives are created")

    if args.check:
        # Check if triples are true negatives
        print('Checking for false negatives.')
        num_false_neg = 0
        num_neg = 0
        for head, relation, tail in all_true_triples:
            # head corruption
            head_candidates = all_hard_negatives_t[head]
            for candidate in head_candidates:
                num_neg += 1
                if (candidate[0], relation, tail) in all_true_triples:
                    num_false_neg += 1

            # tail corruption
            tail_candidates = all_hard_negatives_t[tail]
            for candidate in tail_candidates:
                num_neg += 1
                if (head, relation, candidate[0]) in all_true_triples:
                    num_false_neg += 1
        print(args.data_path)
        print(f'Number of false negatives: {num_false_neg}/{num_neg} ({num_false_neg/num_neg*100:.2f}%).')


    # /data/yamanishi/processed/enzyme/1/
    # /data/yamanishi/processed/enzyme/metadata.json
    if args.s:
        output_path = args.data_path.joinpath('metadata.json')
    else:
        output_path = args.data_path.parents[0].joinpath('metadata.json')

    # load existing data
    if output_path.exists():
        with open(output_path, mode='r') as json_fp:
            metadata = json.load(json_fp)
            # print(metadata)
    else:
        metadata = dict()

    # append metadata
    if 'ego_networks' not in metadata:
        metadata['ego_networks'] = dict()
    metadata['ego_networks'][args.mode] = all_hard_negatives_t

    # write metadata
    with open(output_path, mode='w') as json_fp:
        print(f'Appending ego network to {output_path}.')
        json.dump(metadata, json_fp)

    print(f'Script runtime: {datetime.datetime.now() - start_datetime}')


if __name__ == '__main__':
    main(parse_args())
