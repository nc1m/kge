#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import datetime
import time
from pathlib import Path
import sys

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from model import append_negative_samples
from model import KGEModel
import const

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('model_path', type=str, default=None)

    parser.add_argument('--seed', type=int, default=None, help='Randomgenerator seed for reproducibility.')
    parser.add_argument('--offline', action='store_true', help='Set if you do not want to sync to wandb.')
    parser.add_argument('--auc_sampling', default='uniform', type=str, choices=const.NEGATIVE_SAMPLING_METHODS, help='Choose how the negative samples are sampled for the auc score metric.') # difficult to rename, hardcoded in append_negative_samples()
    parser.add_argument('--auc_path', default=None, type=str, help='Path to the metainformation used for type/simiarity based sampling')

    parser.add_argument('--kegg', action='store_true', help='Set to perform evaluation on novel kegg interactions between yamanishi entities.')
    parser.add_argument('--kegg_path', default=None, type=Path, help='Use custome interaction .txt')
    parser.add_argument('-v','--verbose', action='count', default=0, help='Choose verbosity levels, based on how often the argument is given.')
    parser.add_argument('--eval_neg_sample_ratio', type=int, default=5, help='Numbe of negative samples per positive validation/test sample.')
    parser.add_argument('--scale_emb', action='store_true', help='Scale (using standard scalar) the embeddings before handing them to the classifiers.')
    parser.add_argument('--svc_prob', action='store_true', help='If set enable probability estimates for svc, but will slow down the training.')
    parser.add_argument('--no_clf', action='store_true', help='If set skip classifier training/evaluation.')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)


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

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    dateTime = datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')
    logLevel = 10*(3-max(0,min(args.verbose,2)))
    log_file = os.path.join(args.model_path, f'{dateTime}-post.log') # TODO change log file name
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logLevel,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logLevel)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def build_classifier_dataset(model, sample):
    """Given a model and sample triples get the embeddings of the models for the triples.
    """
    sample = torch.LongTensor(sample)

    head = torch.index_select(
        model.entity_embedding,
        dim=0,
        index=sample[:,0]
    ).unsqueeze(1)

    relation = torch.index_select(
        model.relation_embedding,
        dim=0,
        index=sample[:,1]
    ).unsqueeze(1)

    tail = torch.index_select(
        model.entity_embedding,
        dim=0,
        index=sample[:,2]
    ).unsqueeze(1)

    data = torch.cat((head, relation, tail), dim=2)
    data = torch.squeeze(data)
    data = data.detach().numpy()

    return data


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

    # Update wandb config based on loaded arguments
    wandb.config.update({'model': args.model,
                         'data_path': args.data_path,
                         'hidden_dim': args.hidden_dim,
                         'seed': args.seed,
                         'neg_sampling': args.auc_sampling,
                         'gamma': args.gamma,
                         'regularization': args.regularization,
                         'negative_sample_size': args.negative_sample_size,
                         'max_steps': args.max_steps,
                         'eval_neg_sample_ratio': args.eval_neg_sample_ratio,
                         'scale_emb': args.scale_emb,
                         'kegg': args.kegg})

    # Config logging
    set_logger(args)

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

    # TODO: Remove?
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

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
    if args.kegg:
        if args.kegg_path:
            valid_path = args.kegg_path
        else:
            valid_path = Path(args.data_path).parents[0].joinpath('kegg_interactions.txt')
        valid_triples = read_triple(valid_path, entity2id, relation2id)

        # add validation set to test set so they are in all_true_triples and filtered in negative sampling
        test_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
        test_triples = test_triples + read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    else:
        valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
        test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
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


    # Load similarity data
    if args.auc_path is None:
        similarityPath = Path(args.data_path).parents[0].joinpath('metadata.json')
        assert similarityPath.exists(), f'{similarityPath} does not exist use auc_path argument to specify the path to metadata.json'
    else:
        similarityPath = Path(args.auc_path)

    with open(similarityPath, 'r') as jsonFp:
        similarityData = json.load(jsonFp)

    # Add negative samples to validation data
    samplesValid, gtValid = append_negative_samples(valid_triples, all_true_triples, similarityData, id2entity, entity2id, args.eval_neg_sample_ratio, args.auc_sampling, seed=42)
    samplesValid = torch.LongTensor(samplesValid)

    if args.cuda:
        samplesValid = samplesValid.cuda()

    # Predict scores and evaluate raw kge model
    with torch.no_grad():
        predictionsValid = kge_model(samplesValid).squeeze(1).cpu().numpy()


    fpr, tpr, thresholds = roc_curve(gtValid, predictionsValid)
    roc_auc = auc(fpr, tpr)
    rocDisplay = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=args.model)
    rocDisplay.plot()

    precision, recall, thresholds = precision_recall_curve(gtValid, predictionsValid)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(gtValid, predictionsValid)
    prDisplay = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=avg_precision, estimator_name=args.model)
    prDisplay.plot()


    # Log raw kge model metrics
    wandb.log({"kge/roc": rocDisplay.figure_,
               "kge/pr": prDisplay.figure_,
               "kge/roc_auc": roc_auc,
               "kge/avg_precision": avg_precision,
               'kge/pr_auc': pr_auc})


    if args.no_clf:
        sys.exit()

    # Create classifier train data
    samplesTrain, gtTrain = append_negative_samples(train_triples, all_true_triples, similarityData, id2entity, entity2id, args.eval_neg_sample_ratio, args.auc_sampling, seed=args.seed)
    samplesTrain = build_classifier_dataset(kge_model, samplesTrain)
    gtTrain = np.array(gtTrain) # TODO: needed?

    print(f'Training classifiers with {gtTrain.shape[0]}/{len(train_triples)*(args.eval_neg_sample_ratio+1)} samples.')

    # Add embeddings to validation data
    samplesValid = build_classifier_dataset(kge_model, samplesValid)
    # TODO: shuffle training samples?

    # Scale data
    if args.scale_emb:
        print('Scaling data.')
        standardScalar = StandardScaler()
        samplesTrain = standardScalar.fit_transform(X=samplesTrain, y=gtTrain) # TODO: y=gtTrain?
        samplesValid = standardScalar.transform(X=samplesValid)

    svc = SVC(probability=args.svc_prob, random_state=args.seed)
    decisionTreeClf = DecisionTreeClassifier(random_state=args.seed)
    xgbClf = xgb.XGBClassifier(objective="binary:logistic", random_state=args.seed)
    logRegClf = LogisticRegression(random_state=args.seed)
    mlpClf = MLPClassifier(random_state=args.seed)

    clf_id = [('svc', svc), ('decision_tree', decisionTreeClf), ('xgb', xgbClf), ('logistic_regression', logRegClf), ('mlp', mlpClf)]


    for name, clf in clf_id:
        clf = clf.fit(samplesTrain, gtTrain)
        predValid = clf.predict(samplesValid)
        acc = accuracy_score(gtValid, predValid)
        log_data = {f'{name}/acc': acc}
        # Check if clf has predict_proba() method for "threshold based metrics"
        if hasattr(clf, 'predict_proba'):
            probaValid = clf.predict_proba(samplesValid)
            fpr, tpr, thresholds = roc_curve(gtValid, probaValid[:,1])
            roc_auc = auc(fpr, tpr)
            # rocDisplay = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=args.model)
            # rocDisplay.plot()

            precision, recall, thresholds = precision_recall_curve(gtValid, probaValid[:,1])
            pr_auc = auc(recall, precision)
            avg_precision = average_precision_score(gtValid, predValid)
            # prDisplay = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=avg_precision, estimator_name=args.model)
            # prDisplay.plot()
            # log_data[f'{name}/roc'] = rocDisplay.figure_
            # log_data[f'{name}/pr'] = prDisplay.figure_
            log_data[f'{name}/roc'] = wandb.plot.roc_curve(gtValid, probaValid)
            log_data[f'{name}/pr'] = wandb.plot.pr_curve(gtValid, probaValid)
            log_data[f'{name}/roc_auc'] = roc_auc
            log_data[f'{name}/avg_precision'] = avg_precision
            log_data[f'{name}/pr_auc'] = pr_auc
        wandb.log(log_data)




if __name__ == '__main__':
    args = parse_args()
    wandb.init(entity=args.wandb_entity if args.wandb_entity else 'l3s-future-lab', project=args.wandb_project, mode='offline' if args.offline else 'online')
    main(args)
