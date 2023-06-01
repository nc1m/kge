import argparse
from pathlib import Path
from pprint import pprint
from collections import Counter
import random
import json

import requests

ENZYME_INTERACTION_URL = 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/bind_orfhsa_drug_e.txt'
ION_CHANNEL_INTERACTION_URL = 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/bind_orfhsa_drug_ic.txt'
GPCR_INTERACTION_URL = 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/bind_orfhsa_drug_gpcr.txt'
NUCLEAR_RECEPTOR_INTERACTION_URL = 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/bind_orfhsa_drug_nr.txt'

ENZYME_DRUG_SIM_URL = 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/e_simmat_dc.txt'
ION_CHANNEL_DRUG_SIM_URL = 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/ic_simmat_dc.txt'
GPCR_DRUG_SIM_URL = 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/gpcr_simmat_dc.txt'
NUCLEAR_RECEPTOR_DRUG_SIM_URL = 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/nr_simmat_dc.txt'

ENZYME_TARGET_SIM_URL = 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/e_simmat_dg.txt'
ION_CHANNEL_TARGET_SIM_URL = 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/ic_simmat_dg.txt'
GPCR_TARGET_SIM_URL = 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/gpcr_simmat_dg.txt'
NUCLEAR_RECEPTOR_TARGET_SIM_URL = 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/nr_simmat_dg.txt'

YAMANISHI_REALTION = 'interacts_with'
TYPE_RELATION = 'isa'

DRUG_TYPE = 'drug'
ENZYME_TYPE = 'enzyme'
ION_CHANNEL_TYPE = 'ion_channel'
GPCR_TYPE = 'gpcr'
NUCLEAR_RECEPTOR_TYPE = 'nuclear_receptor'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataDir', type=Path, help='Directory for the dataset.')
    parser.add_argument('--numSplits', type=int, default=5, help='The number dataset split variations. (Shuould be greater than 2 to get a reasonable k split cross validation split for the type edges)')
    parser.add_argument('--minOcc', type=int, default=3, help='The number of occurences an entity must have to be considered for a train/test/val split')
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
    targets = list()
    targetDrugs = list()
    edges = list()
    with open(filePath, 'r') as fp:
        for line in fp:
            target, drug = line.split('\t')
            drug = drug.rstrip()
            targets.append(target)
            targetDrugs.append(drug)
            edges.append((target, drug))

    return targets, targetDrugs, edges


def create_entity2id_dict(targets, drugs):
    """Creates a dictionary of entities and gives them a unique id.
    """
    entity2id = dict()
    curEntityId = 0
    entities = list(set(targets)) + list(set(drugs))
    entities.sort()
    for e in entities:
        if e not in entity2id:
            entity2id[e] = curEntityId
            curEntityId += 1
    return entity2id


def write_id_dicts(dirPath, entity2id, relation2id):
    """Writes the entity id and relation id mappings to disk.
    """
    entityPath = dirPath.joinpath('entities.dict')
    with open(entityPath, 'w') as fp:
        for entity, entId in entity2id.items():
            fp.write(f'{entId}\t{entity}\n')

    relationPath = dirPath.joinpath('relations.dict')
    with open(relationPath, 'w') as fp:
        for relation, entId in relation2id.items():
            fp.write(f'{entId}\t{relation}\n')

def create_split(edges, rareEntities, p):
    """TODO
    """
    trainEntities = set()
    trainEdges = list()
    valTestEdges = list()
    random.shuffle(edges)
    for target, drug in edges:
        # If entities not in train set or rare => add to train set
        if target not in trainEntities or drug not in trainEntities or target in rareEntities or drug in rareEntities:
            trainEdges.append(f'{target}\t{YAMANISHI_REALTION}\t{drug}\n')
            trainEntities.add(target)
            trainEntities.add(drug)
        # else use choose randomly
        else:
            if random.random() < p:
                trainEdges.append(f'{target}\t{YAMANISHI_REALTION}\t{drug}\n')
                trainEntities.add(target)
                trainEntities.add(drug)
            else:
                valTestEdges.append(f'{target}\t{YAMANISHI_REALTION}\t{drug}\n')

        random.shuffle(valTestEdges)
        halfLen = int(len(valTestEdges)/2)
        valEdges = valTestEdges[halfLen:]
        testEdges = valTestEdges[:halfLen]
    return trainEdges, valEdges, testEdges

def create_type_edges(drugs, enzymes, ionChannels, gpcrs, nuclReceptors):
    typeEdges = []
    seenEntities = set()
    print(f'number of drugs: {len(drugs)}')
    for e in set(drugs):
        if e not in seenEntities:
            typeEdges.append(f'{e}\t{TYPE_RELATION}\t{DRUG_TYPE}\n')
            seenEntities.add(e)

    for e in enzymes:
        if e not in seenEntities:
            typeEdges.append(f'{e}\t{TYPE_RELATION}\t{ENZYME_TYPE}\n')
            seenEntities.add(e)

    for e in ionChannels:
        if e not in seenEntities:
            typeEdges.append(f'{e}\t{TYPE_RELATION}\t{ION_CHANNEL_TYPE}\n')
            seenEntities.add(e)

    for e in gpcrs:
        if e not in seenEntities:
            typeEdges.append(f'{e}\t{TYPE_RELATION}\t{GPCR_TYPE}\n')
            seenEntities.add(e)

    for e in nuclReceptors:
        if e not in seenEntities:
            typeEdges.append(f'{e}\t{TYPE_RELATION}\t{NUCLEAR_RECEPTOR_TYPE}\n')
            seenEntities.add(e)
    print(f'number of entities in yamanishi: {len(seenEntities)}')
    return typeEdges

def create_chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def read_similarity_matrix(path, isTarget):
    simMat = []
    simIndex = []
    with open(path, mode='r') as fp:
        info = next(fp)
        check = info
        info = info.split('\t')
        info = info[1:]         # remove enmpty string at the beginning
        for entity in info:
            entity = entity.rstrip()
            if isTarget:
                entity = entity[:3] + ':' + entity[3:]
            simIndex.append(entity) # now entity 0 is at index 0 in list
        for line in fp:
            row = line.split('\t')
            row = row[1:] # remove index column
            row = [x.strip() for x in row]
            simMat.append(row)

    return simIndex, simMat


def build_metadata(targetSimIndex, targetSimMat, drugSimIndex, drugSimMat):
    metaInf = dict()

    targetInf = dict()
    targetInf['index'] = targetSimIndex
    targetInf['similarity_matrix'] = targetSimMat
    metaInf['target'] = targetInf

    drugInf = dict()
    drugInf['index'] = drugSimIndex
    drugInf['similarity_matrix'] = drugSimMat
    metaInf['drug'] = drugInf

    return metaInf


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
    print(args)

    if not args.dataDir.is_dir():
        args.dataDir.mkdir(parents=True)
    rawDir = args.dataDir.joinpath('raw')
    processedDir = args.dataDir.joinpath('processed')
    rawDir.mkdir(exist_ok=True)
    processedDir.mkdir(exist_ok=True)

    relation2id = {YAMANISHI_REALTION:0}

    allDrugs = list()
    allEdges = list()

    # Createenzyme datasets
    rawEnzymePath = download_url(ENZYME_INTERACTION_URL, rawDir)
    enzymeDataPath = processedDir.joinpath('enzyme')
    enzymeDataPath.mkdir(exist_ok=True)

    enzymes, enzymeDrugs, enzymeEdges = read_yamanishi_file(rawEnzymePath)
    allDrugs.extend(enzymeDrugs)
    allEdges.extend(enzymeEdges)

    enzymeDrugEntity2Id = create_entity2id_dict(enzymes, enzymeDrugs)

    enzymeCounts = Counter(enzymes)
    # commonEnzymes = {x for x, count in enzymeCounts.items() if count >= args.minOcc}
    rareEnzymeEntities = {x for x, count in enzymeCounts.items() if count < args.minOcc}
    drugCounts = Counter(enzymeDrugs)
    # commonEnzymeDrugs = {x for x, count in drugCounts.items() if count >= args.minOcc}
    rareEnzymeEntities.update({x for x, count in drugCounts.items() if count < args.minOcc})
    print(f'Creating enzyme datasets at {enzymeDataPath}')
    print(f'Number of rare entities: {len(rareEnzymeEntities)}')
    for i in range(1, args.numSplits+1):
        if len(str(args.numSplits)) > 1:
            curDataDir = enzymeDataPath.joinpath(str(i).zfill(len(str(args.numSplits))))
        else:
            curDataDir = enzymeDataPath.joinpath(str(i))
        curDataDir.mkdir(exist_ok=True)
        write_id_dicts(curDataDir, enzymeDrugEntity2Id, relation2id)

        trainEdges, valEdges, testEdges = create_split(enzymeEdges, rareEnzymeEntities, 0.68)
        print(f'Split {i}:')
        print(f'Train set: {len(trainEdges)/len(enzymeEdges)}')
        print(f'Val set: {len(valEdges)/len(enzymeEdges)}')
        print(f'Test set: {len(testEdges)/len(enzymeEdges)}')
        trainPath = curDataDir.joinpath('train.txt')
        with open(trainPath, mode='w') as fp:
            fp.writelines(trainEdges)
        valPath = curDataDir.joinpath('valid.txt')
        with open(valPath, mode='w') as fp:
            fp.writelines(valEdges)
        testPath = curDataDir.joinpath('test.txt')
        with open(testPath, mode='w') as fp:
            fp.writelines(testEdges)



    # create ion channel datasets
    rawIonChPath = download_url(ION_CHANNEL_INTERACTION_URL, rawDir)
    ionChDataPath = processedDir.joinpath('ion_channel')
    ionChDataPath.mkdir(exist_ok=True)

    ionChannels, ionChDrugs, ionChEdges = read_yamanishi_file(rawIonChPath)
    allDrugs.extend(ionChDrugs)
    allEdges.extend(ionChEdges)

    ionChDrugEntity2Id = create_entity2id_dict(ionChannels, ionChDrugs)

    ionChCounts = Counter(ionChannels)
    rareIonChEntities = {x for x, count in ionChCounts.items() if count < args.minOcc}
    drugCounts = Counter(ionChDrugs)
    rareIonChEntities.update({x for x, count in drugCounts.items() if count < args.minOcc})

    print(f'Creating ion channel datasets at {ionChDataPath}')
    print(f'Number of rare entities: {len(rareIonChEntities)}')
    for i in range(1, args.numSplits+1):
        if len(str(args.numSplits)) > 1:
            curDataDir = ionChDataPath.joinpath(str(i).zfill(len(str(args.numSplits))))
        else:
            curDataDir = ionChDataPath.joinpath(str(i))
        curDataDir.mkdir(exist_ok=True)
        write_id_dicts(curDataDir, ionChDrugEntity2Id, relation2id)

        trainEdges, valEdges, testEdges = create_split(ionChEdges, rareIonChEntities, 0.73)
        print(f'Split {i}:')
        print(f'Train set: {len(trainEdges)/len(ionChEdges)}')
        print(f'Val set: {len(valEdges)/len(ionChEdges)}')
        print(f'Test set: {len(testEdges)/len(ionChEdges)}')
        trainPath = curDataDir.joinpath('train.txt')
        with open(trainPath, mode='w') as fp:
            fp.writelines(trainEdges)
        valPath = curDataDir.joinpath('valid.txt')
        with open(valPath, mode='w') as fp:
            fp.writelines(valEdges)
        testPath = curDataDir.joinpath('test.txt')
        with open(testPath, mode='w') as fp:
            fp.writelines(testEdges)



    # create gpcr datasets
    rawGpcrPath = download_url(GPCR_INTERACTION_URL, rawDir)
    gpcrDataPath = processedDir.joinpath('gpcr')
    gpcrDataPath.mkdir(exist_ok=True)

    gpcrs, gpcrDrugs, gpcrEdges = read_yamanishi_file(rawGpcrPath)
    allDrugs.extend(gpcrDrugs)
    allEdges.extend(gpcrEdges)

    gpcrDrugEntity2Id = create_entity2id_dict(gpcrs, gpcrDrugs)

    gpcrCounts = Counter(gpcrs)
    rareGpcrEntities = {x for x, count in gpcrCounts.items() if count < args.minOcc}
    drugCounts = Counter(gpcrDrugs)
    rareGpcrEntities.update({x for x, count in drugCounts.items() if count < args.minOcc})

    print(f'Creating gpcr datasets at {gpcrDataPath}')
    print(f'Number of rare entities: {len(rareGpcrEntities)}')
    for i in range(1, args.numSplits+1):
        if len(str(args.numSplits)) > 1:
            curDataDir = gpcrDataPath.joinpath(str(i).zfill(len(str(args.numSplits))))
        else:
            curDataDir = gpcrDataPath.joinpath(str(i))
        curDataDir.mkdir(exist_ok=True)
        write_id_dicts(curDataDir, gpcrDrugEntity2Id, relation2id)

        trainEdges, valEdges, testEdges = create_split(gpcrEdges, rareGpcrEntities, 0.66)
        print(f'Split {i}:')
        print(f'Train set: {len(trainEdges)/len(gpcrEdges)}')
        print(f'Val set: {len(valEdges)/len(gpcrEdges)}')
        print(f'Test set: {len(testEdges)/len(gpcrEdges)}')
        trainPath = curDataDir.joinpath('train.txt')
        with open(trainPath, mode='w') as fp:
            fp.writelines(trainEdges)
        valPath = curDataDir.joinpath('valid.txt')
        with open(valPath, mode='w') as fp:
            fp.writelines(valEdges)
        testPath = curDataDir.joinpath('test.txt')
        with open(testPath, mode='w') as fp:
            fp.writelines(testEdges)



    # create nuclear receptor datasets
    rawNuclRecepPath = download_url(NUCLEAR_RECEPTOR_INTERACTION_URL, rawDir)
    nuclRecepDataPath = processedDir.joinpath('nuclear_receptor')
    nuclRecepDataPath.mkdir(exist_ok=True)

    nuclReceptors, nuclRecepDrugs, nuclRecepEdges = read_yamanishi_file(rawNuclRecepPath)
    allDrugs.extend(nuclRecepDrugs)
    allEdges.extend(nuclRecepEdges)

    nuclRecepDrugEntity2Id = create_entity2id_dict(nuclReceptors, nuclRecepDrugs)

    nuclRecepCounts = Counter(nuclReceptors)
    rareNuclRecepEntities = {x for x, count in nuclRecepCounts.items() if count < args.minOcc}
    drugCounts = Counter(nuclRecepDrugs)
    rareNuclRecepEntities.update({x for x, count in drugCounts.items() if count < args.minOcc})

    print(f'Creating nuclear receptor datasets at {nuclRecepDataPath}')
    print(f'Number of entities: {len((set(nuclReceptors) | set(nuclRecepDrugs)))}')
    print(f'Number of rare entities: {len(rareNuclRecepEntities)}')
    for i in range(1, args.numSplits+1):
        if len(str(args.numSplits)) > 1:
            curDataDir = nuclRecepDataPath.joinpath(str(i).zfill(len(str(args.numSplits))))
        else:
            curDataDir = nuclRecepDataPath.joinpath(str(i))
        curDataDir.mkdir(exist_ok=True)
        write_id_dicts(curDataDir, nuclRecepDrugEntity2Id, relation2id)

        trainEdges, valEdges, testEdges = create_split(nuclRecepEdges, rareNuclRecepEntities, 0.37)
        print(f'Split {i}:')
        print(f'Train set: {len(trainEdges)/len(nuclRecepEdges)}')
        print(f'Val set: {len(valEdges)/len(nuclRecepEdges)}')
        print(f'Test set: {len(testEdges)/len(nuclRecepEdges)}')
        trainPath = curDataDir.joinpath('train.txt')
        with open(trainPath, mode='w') as fp:
            fp.writelines(trainEdges)
        valPath = curDataDir.joinpath('valid.txt')
        with open(valPath, mode='w') as fp:
            fp.writelines(valEdges)
        testPath = curDataDir.joinpath('test.txt')
        with open(testPath, mode='w') as fp:
            fp.writelines(testEdges)



    # create create whole yamanishi and whole typed yamanishi  datasets
    yamDataPath = processedDir.joinpath('whole_yamanishi')
    yamDataPath.mkdir(exist_ok=True)
    typedDataPath = processedDir.joinpath('whole_yamanishi_typed')
    typedDataPath.mkdir(exist_ok=True)

    allTargets = enzymes + ionChannels + gpcrs + nuclReceptors

    allEntity2Id = create_entity2id_dict(allTargets, allDrugs)
    typedEntity2Id = create_entity2id_dict(allTargets + [DRUG_TYPE, ENZYME_TYPE, ION_CHANNEL_TYPE, GPCR_TYPE, NUCLEAR_RECEPTOR_TYPE], allDrugs)

    typedRelation2id = relation2id.copy()
    typedRelation2id.update({TYPE_RELATION: 1})

    typeEdges = create_type_edges(allDrugs, enzymes, ionChannels, gpcrs, nuclReceptors)

    allCounts = Counter(allTargets)
    rareEntities = {x for x, count in allCounts.items() if count < args.minOcc}
    drugCounts = Counter(allDrugs)
    rareEntities.update({x for x, count in drugCounts.items() if count < args.minOcc})

    print(f'Creating yamanishi datasets at {yamDataPath}')
    print(f'Creating typed yamanishi datasets at {typedDataPath}')
    print(f'Number of rare entities: {len(rareEntities)}')
    for i in range(1, args.numSplits+1):
        if len(str(args.numSplits)) > 1:
            curYamDataDir = yamDataPath.joinpath(str(i).zfill(len(str(args.numSplits))))
            curTypedDataDir = typedDataPath.joinpath(str(i).zfill(len(str(args.numSplits))))

        else:
            curYamDataDir = yamDataPath.joinpath(str(i))
            curTypedDataDir = typedDataPath.joinpath(str(i))
        curYamDataDir.mkdir(exist_ok=True)
        curTypedDataDir.mkdir(exist_ok=True)
        write_id_dicts(curYamDataDir, allEntity2Id, relation2id)
        write_id_dicts(curTypedDataDir, typedEntity2Id, typedRelation2id)

        trainEdges, valEdges, testEdges = create_split(allEdges, rareEntities, 0.7)
        print(f'Split {i}:')
        print(f'Train set: {len(trainEdges)/len(allEdges)}')
        print(f'Val set: {len(valEdges)/len(allEdges)}')
        print(f'Test set: {len(testEdges)/len(allEdges)}')
        trainPath = curYamDataDir.joinpath('train.txt')
        with open(trainPath, mode='w') as fp:
            fp.writelines(trainEdges)
        valPath = curYamDataDir.joinpath('valid.txt')
        with open(valPath, mode='w') as fp:
            fp.writelines(valEdges)
        testPath = curYamDataDir.joinpath('test.txt')
        with open(testPath, mode='w') as fp:
            fp.writelines(testEdges)

        trainPath = curTypedDataDir.joinpath('train.txt')
        with open(trainPath, mode='w') as fp:
            fp.writelines(trainEdges + typeEdges)
        valPath = curTypedDataDir.joinpath('valid.txt')
        with open(valPath, mode='w') as fp:
            fp.writelines(valEdges)
        testPath = curTypedDataDir.joinpath('test.txt')
        with open(testPath, mode='w') as fp:
            fp.writelines(testEdges)

    # create k-fold cross validation split for edge type prediction
    typeSplitDataPath = processedDir.joinpath('type_split')
    typeSplitDataPath.mkdir(exist_ok=True)

    # typedEntity2id
    # TypedRelation2id
    random.shuffle(typeEdges)
    chunks = list(create_chunks(typeEdges, args.numSplits))

    trainIndxs = []
    valIndxs = []
    testIndxs = []
    for i_split in range(args.numSplits):
        cur_trainIndx = []
        for i_train in range(args.numSplits-2):
            cur_trainIndx.append(((i_split+i_train) % args.numSplits))
        trainIndxs.append(cur_trainIndx)
        valIndxs.append((i_split+args.numSplits-2)%args.numSplits)
        testIndxs.append((i_split+args.numSplits-1)%args.numSplits)

    print(f'Creating type split datasets at {typeSplitDataPath}')
    # print(f'Number of rare entities: {len(rareIonChEntities)}')
    for i in range(1, args.numSplits+1):
        if len(str(args.numSplits)) > 1:
            curDataDir = typeSplitDataPath.joinpath(str(i).zfill(len(str(args.numSplits))))
        else:
            curDataDir = typeSplitDataPath.joinpath(str(i))
        curDataDir.mkdir(exist_ok=True)
        write_id_dicts(curDataDir, typedEntity2Id, typedRelation2id)

        pythonIndex = i-1

        cur_trainChunks = [chunks[x] for x in trainIndxs[pythonIndex]]
        trainEdges = []
        for l in cur_trainChunks:
            trainEdges.extend(l)
        valEdges = chunks[valIndxs[pythonIndex]]
        testEdges = chunks[testIndxs[pythonIndex]]
        print(f'Split {i}:')
        print(f'Train set: {len(trainEdges)/len(typeEdges)}')
        print(f'Val set: {len(valEdges)/len(typeEdges)}')
        print(f'Test set: {len(testEdges)/len(typeEdges)}')
        trainEdges = trainEdges + [f'{target}\t{YAMANISHI_REALTION}\t{drug}\n' for target, drug in allEdges]
        trainPath = curDataDir.joinpath('train.txt')
        with open(trainPath, mode='w') as fp:
            fp.writelines(trainEdges)
        valPath = curDataDir.joinpath('valid.txt')
        with open(valPath, mode='w') as fp:
            fp.writelines(valEdges)
        testPath = curDataDir.joinpath('test.txt')
        with open(testPath, mode='w') as fp:
            fp.writelines(testEdges)


    # Create metainformation (target to type and similarity matrices)
    metaInformation = dict()


    rawEnzymeSimPath = download_url(ENZYME_TARGET_SIM_URL, rawDir)
    targetSimIndex, targetSimMat = read_similarity_matrix(rawEnzymeSimPath, isTarget=True)
    # print(len(simIndex))
    # print(len(set(enzymes)))
    rawEnzymeDrugSimPath = download_url(ENZYME_DRUG_SIM_URL, rawDir)
    drugSimIndex, drugSimMat = read_similarity_matrix(rawEnzymeDrugSimPath, isTarget=False)
    metaInformation['enzyme'] = build_metadata(targetSimIndex, targetSimMat, drugSimIndex, drugSimMat)


    rawIonChSimPath = download_url(ION_CHANNEL_TARGET_SIM_URL, rawDir)
    targetSimIndex, targetSimMat = read_similarity_matrix(rawIonChSimPath, isTarget=True)
    # print(len(simIndex))
    # print(len(set(ionChannels)))
    rawIonChDrugSimPath = download_url(ION_CHANNEL_DRUG_SIM_URL, rawDir)
    drugSimIndex, drugSimMat = read_similarity_matrix(rawIonChDrugSimPath, isTarget=False)
    metaInformation['ion_channel'] = build_metadata(targetSimIndex, targetSimMat, drugSimIndex, drugSimMat)


    rawGpcrSimPath = download_url(GPCR_TARGET_SIM_URL, rawDir)
    targetSimIndex, targetSimMat = read_similarity_matrix(rawGpcrSimPath, isTarget=True)
    # print(len(simIndex))
    # print(len(set(gpcrs)))
    rawGpcrDrugSimPath = download_url(GPCR_DRUG_SIM_URL, rawDir)
    drugSimIndex, drugSimMat = read_similarity_matrix(rawGpcrDrugSimPath, isTarget=False)
    metaInformation['gpcr'] = build_metadata(targetSimIndex, targetSimMat, drugSimIndex, drugSimMat)

    rawNuclRecepSimPath = download_url(NUCLEAR_RECEPTOR_TARGET_SIM_URL, rawDir)
    targetSimIndex, targetSimMat = read_similarity_matrix(rawNuclRecepSimPath, isTarget=True)
    # print(len(simIndex))
    # print(len(set(nuclReceptors)))
    rawNuclRecepDrugSimPath = download_url(NUCLEAR_RECEPTOR_DRUG_SIM_URL, rawDir)
    drugSimIndex, drugSimMat = read_similarity_matrix(rawNuclRecepDrugSimPath, isTarget=False)
    metaInformation['nuclear_receptor'] = build_metadata(targetSimIndex, targetSimMat, drugSimIndex, drugSimMat)

    # for key in metaInformation:
    #     print(key)
    #     for kkey in metaInformation[key]:
    #         print(f'\t{kkey}')
    #         for kkkey in metaInformation[key][kkey]:
    #             print(f'\t\t{kkkey}')


    metaInfPath = processedDir.joinpath('yamanishi_similarity_data.json')
    with open(metaInfPath, mode='w') as jsonFp:
        json.dump(metaInformation, jsonFp)


if __name__ == '__main__':
    args = parse_args()
    main(args)


    # def check_adj_file(path):
    #     with open(path, mode='r') as fp:
    #         info = next(fp)
    #         check = info
    #         check = check.split('\t')[1:]
    #         check = [x.strip() for x in check]
    #         ocheck = []
    #         for line in fp:
    #             first = line.split('\t')[0].strip()
    #             ocheck.append(first)

    #         same = True
    #         for x,y in zip(check, ocheck):
    #             if x != y:
    #                 print(x)
    #                 print(y)
    #                 same = False
    #     return same

    # print(check_adj_file(rawEnzymeSimPath))
