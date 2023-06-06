import argparse
from pathlib import Path
import random
import json
import re

import requests

from bioservices.kegg import KEGG
from bioservices.kegg import KEGGParser

YAMANICHI_INTERACTION_PREDICATE = 'interacts_with'
IS_DRUG_PREDICATE = 'isDrug'
IS_TARGET_PREDICATE = 'isTarget'

ENZYME_URL = 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/bind_orfhsa_drug_e.txt'
ION_CHANNEL_URL = 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/bind_orfhsa_drug_ic.txt'
GPCR_URL = 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/bind_orfhsa_drug_gpcr.txt'
NUCLEAR_RECEPTOR_URL = 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/bind_orfhsa_drug_nr.txt'

def parse_args():
    parser = argparse.ArgumentParser(description='''This script generates a tab seperated file of target drug interactions between all entities in the yamanishi dataset that are found in the kegg database and are not already in the yamanishi dataset.
The raw yamanishi dataset and kegg data will be downloaded to [dataDir]/raw.
The kegg interactions will be written to [dataDir]/kegg_interactions.txt.''')
    parser.add_argument('dataDir', type=Path, help='Directory for the dataset.')
    parser.add_argument('-d', '--download', action='store_true', help='Set flag if yout want to download the kegg data for the yamanishi dataset')
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


def download_url(url, dirPath):
    """Downloads the file at the url, if it doesn't alread exists.
    """
    filename = url.rpartition('/')[2]
    path = dirPath.joinpath(filename)
    if path.exists():
        return path
    else:
        print(f'Downloading {url}')

    r = requests.get(url)
    if r.status_code == requests.codes.ok:
        with open(path, mode='w') as fp:
            fp.write(r.text)
    else:
        print(f'Download of url {url} failed with http response {r.status_code}')
    return path

def read_yamanishi_file(filePath):
    """Read the yamanishi file and returns a list of targets, drugs and edges.
    """
    targets = set()
    targetDrugs = set()
    edges = set()
    with open(filePath, 'r') as fp:
        for line in fp:
            target, drug = line.split('\t')
            drug = drug.rstrip()
            targets.add(target)
            targetDrugs.add(drug)
            edges.add((target, drug))

    return targets, targetDrugs, edges


def main(args):
    print(args)

    args.dataDir.mkdir(parents=True, exist_ok=True)
    rawDir = args.dataDir.joinpath('raw')
    rawDir.mkdir(exist_ok=True)

    keggDataPath = rawDir.joinpath('yamanishi_kegg_data.json')

    outputPath = args.dataDir.joinpath('kegg_interactions.txt')

    targets = set()
    drugs = set()
    yamanishi_interactions = set()
    new_interactions = set()

    rawEnzymePath = download_url(ENZYME_URL, rawDir)
    rawIonChPath = download_url(ION_CHANNEL_URL, rawDir)
    rawGpcrPath = download_url(GPCR_URL, rawDir)
    rawNuclRecepPath = download_url(NUCLEAR_RECEPTOR_URL, rawDir)

    newTargets, newDrugs, newInteractions = read_yamanishi_file(rawEnzymePath)
    targets.update(newTargets)
    drugs.update(newDrugs)
    yamanishi_interactions.update(newInteractions)

    newTargets, newDrugs, newInteractions = read_yamanishi_file(rawIonChPath)
    targets.update(newTargets)
    drugs.update(newDrugs)
    yamanishi_interactions.update(newInteractions)

    newTargets, newDrugs, newInteractions= read_yamanishi_file(rawGpcrPath)
    targets.update(newTargets)
    drugs.update(newDrugs)
    yamanishi_interactions.update(newInteractions)

    newTargets, newDrugs, newInteractions = read_yamanishi_file(rawNuclRecepPath)
    targets.update(newTargets)
    drugs.update(newDrugs)
    yamanishi_interactions.update(newInteractions)

    if args.download:
        kegg = KEGG()
        parser = KEGGParser()

        yamanishiData = dict()

        numEnt = len(targets)
        counter = 0
        for entity in targets:
            data = kegg.get(entity)
            if isinstance(data, int):
                print(f'No KEGG entry found for {entity}! ERRORCODE: {data}')
                if 'ERROR' not in yamanishiData:
                    yamanishiData['ERROR'] = []
                    yamanishiData['ERROR'].append((entity, data))
                else:
                    yamanishiData['ERROR'].append((entity, data))
                continue
            data = parser.parse(data)
            data[IS_YAMANISHI_DATA_PREDICATE] = True
            data[IS_DRUG_PREDICATE] = False
            data[IS_TARGET_PREDICATE] = True
            yamanishiData[entity] = data
            counter += 1
            print(f'Entity {counter}/{numEnt-1}')

        numEnt = len(drugs)
        counter = 0
        for entity in drugs:
            data = kegg.get(entity)
            if isinstance(data, int):
                print(f'No KEGG entry found for {entity}! ERRORCODE: {data}')
                if 'ERROR' not in yamanishiData:
                    yamanishiData['ERROR'] = []
                    yamanishiData['ERROR'].append((entity, data))
                else:
                    yamanishiData['ERROR'].append((entity, data))
                continue
            data = parser.parse(data)
            data[IS_YAMANISHI_DATA_PREDICATE] = True
            data[IS_DRUG_PREDICATE] = True
            data[IS_TARGET_PREDICATE] = False
            yamanishiData[entity] = data
            counter += 1
            print(f'Entity {counter}/{numEnt-1}')

        with open(keggDataPath, mode='w') as jsonFp:
            json.dump(yamanishiData, jsonFp)
    else:
        assert keggDataPath.exists(), f'No yamanishi kegg data found, run the script with -d to download the data.'
        with open(keggDataPath, mode='r') as jsonFp:
            yamanishiData = json.load(jsonFp)

    drug_pattern = re.compile('(?:D)\\d{5}')
    target_pattern = re.compile('(?:hsa:|HSA:)\\d{2,5}')

    for entity in yamanishiData:
        data = yamanishiData[entity]
        # case: entity is target
        if 'DRUG_TARGET' in data:
            entry = yamanishiData[entity]['DRUG_TARGET']
            dataIds = drug_pattern.findall(str(entry))
            for drug in dataIds:
                if entity in targets and drug in drugs and (entity, drug) not in yamanishi_interactions:
                    new_interactions.add((entity, drug))
        # case: entity is drug
        elif 'TARGET' in data:
            entry = yamanishiData[entity]['TARGET']
            dataIds = target_pattern.findall(str(entry))
            for target in dataIds:
                target = target.lower()
                # if ':' not in target:
                #     target = target[0:3] + ':' + target[3:]
                if target in targets and entity in drugs and (target, entity) not in yamanishi_interactions:
                    new_interactions.add((target, entity))


    new_interactions = sorted(list(new_interactions))
    new_interactions = [f'{x[0]}\t{YAMANICHI_INTERACTION_PREDICATE}\t{x[1]}' for x in new_interactions]

    print(f'Writing {len(new_interactions)} edges to {outputPath}.')
    with open(outputPath, 'w') as fp:
        fp.writelines(new_interactions)

if __name__ == '__main__':
    main(parse_args())
