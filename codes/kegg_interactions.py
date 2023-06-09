import argparse
from pathlib import Path
import json
import re

from bioservices.kegg import KEGG
from bioservices.kegg import KEGGParser

from create_datasets import download_url
from create_datasets import read_yamanishi_file

YAMANICHI_INTERACTION_PREDICATE = 'interacts_with'
IS_DRUG_PREDICATE = 'isDrug'
IS_TARGET_PREDICATE = 'isTarget'
IS_YAMANISHI_DATA_PREDICATE = 'isYamanishi'

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


# def download_url(url, dirPath):
#     """Downloads the file at the url, if it doesn't alread exists.
#     """
#     filename = url.rpartition('/')[2]
#     path = dirPath.joinpath(filename)
#     if path.exists():
#         return path
#     else:
#         print(f'Downloading {url}')

#     r = requests.get(url)
#     if r.status_code == requests.codes.ok:
#         with open(path, mode='w') as fp:
#             fp.write(r.text)
#     else:
#         print(f'Download of url {url} failed with http response {r.status_code}')
#     return path

# def read_yamanishi_file(filePath):
#     """Read the yamanishi file and returns a list of targets, drugs and edges.
#     """
#     targets = set()
#     targetDrugs = set()
#     edges = set()
#     with open(filePath, 'r') as fp:
#         for line in fp:
#             target, drug = line.split('\t')
#             drug = drug.rstrip()
#             targets.add(target)
#             targetDrugs.add(drug)
#             edges.add((target, drug))

#     return targets, targetDrugs, edges

def get_new_interactions(yamanishi_targets, yamanishi_drugs, yamanishi_interactions, kegg_data):
    new_interactions = set()
    drug_pattern = re.compile('(?:D)\\d{5}')
    target_pattern = re.compile('(?:hsa:|HSA:)\\d{2,5}')

    for entity in yamanishi_targets + yamanishi_drugs:
        if entity not in kegg_data:
            continue
        data = kegg_data[entity]
        # case: entity is target
        if 'DRUG_TARGET' in data:
            entry = kegg_data[entity]['DRUG_TARGET']
            dataIds = drug_pattern.findall(str(entry))
            for drug in dataIds:
                if entity in yamanishi_targets and drug in yamanishi_drugs and (entity, drug) not in yamanishi_interactions:
                    new_interactions.add((entity, drug))
        # case: entity is drug
        elif 'TARGET' in data:
            entry = kegg_data[entity]['TARGET']
            dataIds = target_pattern.findall(str(entry))
            for target in dataIds:
                target = target.lower()
                # if ':' not in target:
                #     target = target[0:3] + ':' + target[3:]
                if target in yamanishi_targets and entity in yamanishi_drugs and (target, entity) not in yamanishi_interactions:
                    new_interactions.add((target, entity))

    new_interactions = list(new_interactions)
    new_interactions.sort()
    return new_interactions


def write_new_interactions(directory, new_interactions):
    kegg_path = directory.joinpath('kegg_interactions.txt')
    new_interactions = [f'{x[0]}\t{YAMANICHI_INTERACTION_PREDICATE}\t{x[1]}\n' for x in new_interactions]
    print(f'Writing {len(new_interactions)} edges to {kegg_path}.')
    with open(kegg_path, 'w') as fp:
        fp.writelines(new_interactions)


def main(args):
    print(args)

    args.dataDir.mkdir(parents=True, exist_ok=True)
    rawDir = args.dataDir.joinpath('raw')
    rawDir.mkdir(exist_ok=True)

    keggDataPath = rawDir.joinpath('yamanishi_kegg_data.json')

    targets = []
    drugs = []
    yamanishi_interactions = []

    rawEnzymePath = download_url(ENZYME_URL, rawDir)
    rawIonChPath = download_url(ION_CHANNEL_URL, rawDir)
    rawGpcrPath = download_url(GPCR_URL, rawDir)
    rawNuclRecepPath = download_url(NUCLEAR_RECEPTOR_URL, rawDir)

    enzymes, enzyme_drugs, enzyme_interactions = read_yamanishi_file(rawEnzymePath)
    # remove duplicates (see read_yamanishi_file() doc string) while preserving order
    enzymes = list(dict.fromkeys(enzymes))
    enzyme_drugs = list(dict.fromkeys(enzyme_drugs))
    enzyme_interactions = list(dict.fromkeys(enzyme_interactions))
    targets.extend(enzymes)
    drugs.extend(enzyme_drugs)
    yamanishi_interactions.extend(enzyme_interactions)

    ion_channels, ion_channel_drugs, ion_channel_interactions = read_yamanishi_file(rawIonChPath)
    ion_channels = list(dict.fromkeys(ion_channels))
    ion_channel_drugs = list(dict.fromkeys(ion_channel_drugs))
    ion_channel_interactions = list(dict.fromkeys(ion_channel_interactions))
    targets.extend(ion_channels)
    drugs.extend(ion_channel_drugs)
    yamanishi_interactions.extend(ion_channel_interactions)

    gpcrs, gpcr_drugs, gpcr_interactions= read_yamanishi_file(rawGpcrPath)
    gpcrs = list(dict.fromkeys(gpcrs))
    gpcr_drugs = list(dict.fromkeys(gpcr_drugs))
    gpcr_interactions = list(dict.fromkeys(gpcr_interactions))
    targets.extend(gpcrs)
    drugs.extend(gpcr_drugs)
    yamanishi_interactions.extend(gpcr_interactions)

    nuclear_receptors, nuclear_receptor_drugs, nuclear_receptor_interactions = read_yamanishi_file(rawNuclRecepPath)
    nuclear_receptors = list(dict.fromkeys(nuclear_receptors))
    nuclear_receptor_drugs = list(dict.fromkeys(nuclear_receptor_drugs))
    nuclear_receptor_interactions = list(dict.fromkeys(nuclear_receptor_interactions))
    targets.extend(nuclear_receptors)
    drugs.extend(nuclear_receptor_drugs)
    yamanishi_interactions.extend(nuclear_receptor_interactions)

    # remove duplicates
    targets = list(dict.fromkeys(targets))
    drugs =  list(dict.fromkeys(drugs))
    yamanishi_interactions = list(dict.fromkeys(yamanishi_interactions))

    if args.download:
        kegg = KEGG()
        parser = KEGGParser()

        yamanishiData = {}

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


    processed_dir = args.dataDir.joinpath('processed')
    processed_dir.mkdir(exist_ok=True)


    # TODO: generate new interactions text.txt for every dataset
    print(type(enzymes))
    print(type(targets))
    enzyme_dir = processed_dir.joinpath('enzyme')
    enzyme_dir.mkdir(exist_ok=True)
    new_enzyme_interactions = get_new_interactions(enzymes, enzyme_drugs, enzyme_interactions, yamanishiData)
    write_new_interactions(enzyme_dir, new_enzyme_interactions)

    ion_channel_dir = processed_dir.joinpath('ion_channel')
    ion_channel_dir.mkdir(exist_ok=True)
    new_ion_channel_interactions = get_new_interactions(ion_channels, ion_channel_drugs, ion_channel_interactions, yamanishiData)
    write_new_interactions(ion_channel_dir, new_ion_channel_interactions)

    gpcr_dir = processed_dir.joinpath('gpcr')
    gpcr_dir.mkdir(exist_ok=True)
    new_gpcr_interactions = get_new_interactions(gpcrs, gpcr_drugs, gpcr_interactions, yamanishiData)
    write_new_interactions(gpcr_dir, new_gpcr_interactions)

    nuclear_receptor_dir = processed_dir.joinpath('nuclear_receptor')
    nuclear_receptor_dir.mkdir(exist_ok=True)
    new_nuclear_receptor_interactions = get_new_interactions(nuclear_receptors, nuclear_receptor_drugs, nuclear_receptor_interactions, yamanishiData)
    write_new_interactions(nuclear_receptor_dir, new_nuclear_receptor_interactions)

    whole_yamanishi_dir = processed_dir.joinpath('whole_yamanishi')
    whole_yamanishi_dir.mkdir(exist_ok=True)
    whole_yamanishi_typed_dir = processed_dir.joinpath('whole_yamanishi_typed')
    whole_yamanishi_typed_dir.mkdir(exist_ok=True)
    new_yamanishi_interactions = get_new_interactions(targets, drugs, yamanishi_interactions, yamanishiData)
    write_new_interactions(whole_yamanishi_dir, new_yamanishi_interactions)
    write_new_interactions(whole_yamanishi_typed_dir, new_yamanishi_interactions)


if __name__ == '__main__':
    main(parse_args())
