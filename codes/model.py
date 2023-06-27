#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from pathlib import Path
import json
import os
import time
from datetime import timedelta
import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from dataloader import TestDataset
import const


__similarity_cache__ = dict()

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')


    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:,0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:,1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:,2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        test_time = datetime.datetime.now()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        # print(f'Sample retrieval time: {datetime.datetime.now() - test_time}')

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2

        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 +
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}

        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
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

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        # print(mode)
                        # print(positive_sample.shape)
                        # print(negative_sample.shape)


                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

            # Compute auc metrics
            # generate negative samples
            startTime = time.time()

            if args.metadata_path is None:
                similarityPath = Path(args.data_path).parents[0].joinpath('metadata.json')
                assert similarityPath.exists(), f'{similarityPath} does not exist use metadata_path argument to specify the path to yamanishi_similarity_data.json'
            else:
                similarityPath = Path(args.metadata_path)

            with open(similarityPath, 'r') as jsonFp:
                similarityData = json.load(jsonFp)

            with open(os.path.join(args.data_path, 'entities.dict')) as fin:
                entity2id = dict()
                id2entity = dict()
                for line in fin:
                    eid, entity = line.strip().split('\t')
                    entity2id[entity] = int(eid)
                    id2entity[int(eid)] = entity

            samples, gt = append_negative_samples(test_triples, all_true_triples, similarityData, id2entity, entity2id, 5, args.eval_neg_sampling_method, seed=42)
            samples = torch.LongTensor(samples)

            if args.cuda:
                samples = samples.cuda()

            with torch.no_grad():
                y_score = model(samples).squeeze(1).cpu().numpy()

            y_true = np.array(gt)

            precision, recall, thresholds = precision_recall_curve(y_true, y_score)

            auc_pr = auc(recall, precision)
            # ignore last element of precision since: "Precision: Precision values such that element i is the precision of predictions with score >= thresholds[i] and the last element is 1."
            # TODO: Error? computes value greater than 1
            # auc_p = auc(thresholds, precision[:len(precision)-1])
            metrics['auc_pr'] = auc_pr

            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            metrics['auc_roc'] = roc_auc
            avg_precision = average_precision_score(y_true, y_score)
            metrics['avg_precision'] = avg_precision
            # metrics['auc_p'] = auc_p
            print(f'auc time: {str(timedelta(seconds=(time.time() - startTime)))}')

        return metrics



TARGET_TYPES = ['enzyme', 'ion_channel', 'gpcr', 'nuclear_receptor']
def get_entity_type(entity, similarityData):
    """
    Returns the type of the entity and if it's a drug.
    """
    isTarget = True
    if entity.startswith('D'):
        isTarget = False
    for eType in TARGET_TYPES:
        allEntities = similarityData[eType]['target' if isTarget else 'drug']['index']
        if entity in allEntities:
            return eType, isTarget
    assert False, f'entity type not found'


def get_negative_sample_entity(entity2corrupt, rng, similarityData, negative_sampling_method, entity2id):
    """
    Returns either a random entity emb id based on uniform/type based sampling
    or a list of entity emb ids sorted by similarity
    or a list of entity emb ids sorted by ego network similarity
    """
    if negative_sampling_method == const.NEGATIVE_SAMPLING_METHOD_UNIFORM:
        return rng.integers(0, max(entity2id.values()))
    elif negative_sampling_method == const.NEGATIVE_SAMPLING_METHOD_TYPE:
        eType, isTarget = get_entity_type(entity2corrupt, similarityData)
        allEntities = similarityData[eType]['target' if isTarget else 'drug']['index']
        rEntity = rng.integers(0, len(allEntities))
        rEntity = allEntities[rEntity]
        return entity2id[rEntity]
    elif negative_sampling_method == const.NEGATIVE_SAMPLING_METHOD_SIMILARITY:
        global __similarity_cache__
        if entity2corrupt in __similarity_cache__:
            return __similarity_cache__[entity2corrupt]
        else:
            eType, isTarget = get_entity_type(entity2corrupt, similarityData)
            index = similarityData[eType]['target' if isTarget else 'drug']['index']
            simMat = np.array(similarityData[eType]['target' if isTarget else 'drug']['similarity_matrix'])
            eIndex = index.index(entity2corrupt) # index from target as str to target index in similarty matrix
            eSimilarities = np.argsort(simMat[eIndex])
            eSimilarities = np.flip(eSimilarities)
            # TODO: Just store top k most similar entities?
            eSimilarities = eSimilarities[1:] # most similar entity is the entity itself => remove it
            candidateEntities = [index[x] for x in eSimilarities]
            candidateEntities = [entity2id[x] for x in candidateEntities]
            __similarity_cache__[entity2corrupt] = candidateEntities
            return candidateEntities
    elif negative_sampling_method in [const.NEGATIVE_SAMPLING_METHOD_EGO_NETWORK_COUNT, const.NEGATIVE_SAMPLING_METHOD_EGO_NETWORK_JACCARD]:
        # Get list of ego network candidates with their similarity score
        candidates = similarityData['ego_networks'][negative_sampling_method][str(entity2id[entity2corrupt])] # TODO: Through the json encoding decoding the dictionary keys are strings.
        # Remove similarity score
        candidates = [x[0] for x in candidates]
        return candidates


def append_negative_samples(positive_triples, all_true_triples, similarityData, id2entity, entity2id, k, negative_sampling_method, seed=None):
    all_true_triples = set(all_true_triples)
    samples = []
    gt = []
    rng = np.random.default_rng(42)
    for head, relation, tail in positive_triples:
        samples.append((head, relation, tail))
        gt.append(1)
        for _ in range(k):
            if rng.random() < 0.5:
                headCorruption = True
            else:
                headCorruption = False

            if headCorruption:
                rEntity = get_negative_sample_entity(id2entity[head], rng, similarityData, negative_sampling_method, entity2id)
                if negative_sampling_method in [const.NEGATIVE_SAMPLING_METHOD_UNIFORM, const.NEGATIVE_SAMPLING_METHOD_TYPE]:
                    while (rEntity, relation, tail) in all_true_triples or (rEntity, relation, tail) in samples:
                        rEntity = get_negative_sample_entity(id2entity[head], rng, similarityData, negative_sampling_method, entity2id)
                    samples.append((rEntity, relation, tail))
                    gt.append(0)
                elif negative_sampling_method in [const.NEGATIVE_SAMPLING_METHOD_SIMILARITY, const.NEGATIVE_SAMPLING_METHOD_EGO_NETWORK_COUNT, const.NEGATIVE_SAMPLING_METHOD_EGO_NETWORK_JACCARD]:
                    for eId in rEntity:
                        if (eId, relation, tail) not in all_true_triples and (eId, relation, tail) not in samples:
                            samples.append((eId, relation, tail))
                            gt.append(0)
                            break

            else:
                rEntity = get_negative_sample_entity(id2entity[tail], rng, similarityData, negative_sampling_method, entity2id)
                if negative_sampling_method in [const.NEGATIVE_SAMPLING_METHOD_UNIFORM, const.NEGATIVE_SAMPLING_METHOD_TYPE]:
                    while (head, relation, rEntity) in all_true_triples or (head, relation, rEntity) in samples:
                        rEntity = get_negative_sample_entity(id2entity[tail], rng, similarityData, negative_sampling_method, entity2id)
                    samples.append((head, relation, rEntity))
                    gt.append(0)
                elif negative_sampling_method in [const.NEGATIVE_SAMPLING_METHOD_SIMILARITY, const.NEGATIVE_SAMPLING_METHOD_EGO_NETWORK_COUNT, const.NEGATIVE_SAMPLING_METHOD_EGO_NETWORK_JACCARD]:
                    for eId in rEntity:
                        if (head, relation, eId) not in all_true_triples and (head, relation, eId) not in samples:
                            samples.append((head , relation, eId))
                            gt.append(0)
                            break
    return samples, gt
