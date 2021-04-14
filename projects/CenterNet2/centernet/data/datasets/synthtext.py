from detectron2.structures import BoxMode
from scipy.io import loadmat
from os.path import join
from detectron2.data import MetadataCatalog, DatasetCatalog
from sklearn.model_selection import train_test_split
import numpy as np
import json
import cv2
import re

ROOT = '/media/sergiev/painkiller/downloads/SynthText/SynthText/'
CACHED = True
ORD_MIN = 33

categories = [{'id': 0, 'name': '!'}, {'id': 1, 'name': '"'}, {'id': 2, 'name': '#'},
              {'id': 3, 'name': '$'}, {'id': 4, 'name': '%'}, {'id': 5, 'name': '&'},
              {'id': 6, 'name': "'"}, {'id': 7, 'name': '('}, {'id': 8, 'name': ')'},
              {'id': 9, 'name': '*'}, {'id': 10, 'name': '+'}, {'id': 11, 'name': ','},
              {'id': 12, 'name': '-'}, {'id': 13, 'name': '.'}, {'id': 14, 'name': '/'},
              {'id': 15, 'name': '0'}, {'id': 16, 'name': '1'}, {'id': 17, 'name': '2'},
              {'id': 18, 'name': '3'}, {'id': 19, 'name': '4'}, {'id': 20, 'name': '5'},
              {'id': 21, 'name': '6'}, {'id': 22, 'name': '7'}, {'id': 23, 'name': '8'},
              {'id': 24, 'name': '9'}, {'id': 25, 'name': ':'}, {'id': 26, 'name': ';'},
              {'id': 27, 'name': '<'}, {'id': 28, 'name': '='}, {'id': 29, 'name': '>'},
              {'id': 30, 'name': '?'}, {'id': 31, 'name': '@'}, {'id': 32, 'name': 'A'},
              {'id': 33, 'name': 'B'}, {'id': 34, 'name': 'C'}, {'id': 35, 'name': 'D'},
              {'id': 36, 'name': 'E'}, {'id': 37, 'name': 'F'}, {'id': 38, 'name': 'G'},
              {'id': 39, 'name': 'H'}, {'id': 40, 'name': 'I'}, {'id': 41, 'name': 'J'},
              {'id': 42, 'name': 'K'}, {'id': 43, 'name': 'L'}, {'id': 44, 'name': 'M'},
              {'id': 45, 'name': 'N'}, {'id': 46, 'name': 'O'}, {'id': 47, 'name': 'P'},
              {'id': 48, 'name': 'Q'}, {'id': 49, 'name': 'R'}, {'id': 50, 'name': 'S'},
              {'id': 51, 'name': 'T'}, {'id': 52, 'name': 'U'}, {'id': 53, 'name': 'V'},
              {'id': 54, 'name': 'W'}, {'id': 55, 'name': 'X'}, {'id': 56, 'name': 'Y'},
              {'id': 57, 'name': 'Z'}, {'id': 58, 'name': '['}, {'id': 59, 'name': '\\'},
              {'id': 60, 'name': ']'}, {'id': 61, 'name': '^'}, {'id': 62, 'name': '_'},
              {'id': 63, 'name': '`'}, {'id': 90, 'name': '{'}, {'id': 91, 'name': '|'},
              {'id': 92, 'name': '}'}, {'id': 93, 'name': '~'}]


def get_synthtext(split_name, mat=None):
    if mat is None:
        mat = loadmat(join(ROOT, 'gt.mat'))
    with open(join(ROOT, split_name + '.json'), 'r') as idx_file:
        idx = json.load(idx_file)
    synthtext = []
    for i in idx:
        record = {'image_id': i, 'file_name': join(ROOT, mat['imnames'][0][i][0])}
        h, w = cv2.imread(record['file_name']).shape[:2]
        record |= {'height': h, 'width': w}
        quadrangles = mat['charBB'][0][i].T
        line = re.sub(r'\s', '', mat['txt'][0][i])
        assert len(line) != quadrangles.shape[0], exit(
            f'chars: {len(line)}, boxes: {quadrangles.shape[9]}, idx: {i}')
        record['anno'] = []
        for char, quad in zip(line, quadrangles):
            obj = {'bbox': [np.min(quad)[:, 0], np.min(quad)[:, 1],
                            np.max(quad)[:, 0], np.max(quad)[:, 1], ],
                   'bbox_mode': BoxMode.XYXY_ABS,
                   'category_id': ord(char.upper()) - ORD_MIN}
            record['anno'].append(obj)
        synthtext.append(record)
    return synthtext


def idx_to_json(idx, save_path):
    with open(join(ROOT, save_path), 'w') as file:
        json.dump(sorted(idx.tolist()), file)


def split():
    size = 858750 if CACHED else len(loadmat(join(ROOT, 'gt.mat'))['imnames'][0])
    indices = np.arange(size)
    train_idx, test_idx = train_test_split(indices, test_size=.3)
    val_idx, test_idx = train_test_split(test_idx, test_size=.5)
    print([len(i) for i in (val_idx, test_idx, train_idx)])
    idx_to_json(train_idx, 'st_train.json')
    idx_to_json(val_idx, 'st_val.json')
    idx_to_json(test_idx, 'st_test.json')


if __name__ == '__main__':
    mat = loadmat(join(ROOT, 'gt.mat'))
    for d in ['train', 'val', 'test']:
        DatasetCatalog.register("st_" + d, lambda d=d: get_synthtext(d))
        MetadataCatalog.get('st_' + d).set(thing_classes=[i['name'] for i in categories])
    st_metadata = MetadataCatalog.get('st_train')
