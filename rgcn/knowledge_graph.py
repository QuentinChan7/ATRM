import os
import numpy as np

class RGCNLinkDataset:
    def __init__(self, name, dir):
        self.name = name
        self.dir = os.path.join(dir, name)

    def load(self, load_time=True):
        entity_path = os.path.join(self.dir, 'entity2id.txt')
        relation_path = os.path.join(self.dir, 'relation2id.txt')
        train_path = os.path.join(self.dir, 'train.txt')
        valid_path = os.path.join(self.dir, 'valid.txt')
        test_path = os.path.join(self.dir, 'test.txt')
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        self.train = np.array(_read_triplets_as_list(train_path, entity_dict, relation_dict, load_time))
        self.valid = np.array(_read_triplets_as_list(valid_path, entity_dict, relation_dict, load_time))
        self.test = np.array(_read_triplets_as_list(test_path, entity_dict, relation_dict, load_time))
        self.num_nodes = len(entity_dict)
        print("# Sanity Check:  entities: {}".format(self.num_nodes))
        self.num_rels = len(relation_dict)
        self.relation_dict = relation_dict
        self.entity_dict = entity_dict
        print("# Sanity Check:  relations: {}".format(self.num_rels))
        print("# Sanity Check:  edges: {}".format(len(self.train)))

def load_from_local(dir, dataset):
    data = RGCNLinkDataset(dataset, dir)
    # if "-d" in dataset:
    #     data.load(load_time=True)
    # else:
    #
    data.load()
    return data

def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            d[int(line[1])] = line[0]
    return d

def _read_triplets(filename):
    with open(filename, 'r+', encoding='utf-8') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line

def _read_triplets_as_list(filename, entity_dict, relation_dict, load_time):
    l = []
    for triplet in _read_triplets(filename):
        s = int(triplet[0])
        r = int(triplet[1])
        o = int(triplet[2])
        if load_time:
            st = int(triplet[3])
            # et = int(triplet[4])
            # l.append([s, r, o, st, et])
            l.append([s, r, o, st])
        else:
            l.append([s, r, o])
    return l
