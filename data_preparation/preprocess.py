import json
import logging
import argparse
from pathlib import Path
from pprint import pprint

import scipy.io as sio
import numpy as np
import pandas as pd

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Fill activity related information to the annotation files.')
    # general
    parser.add_argument('--dpath',
                        help='Path of the files.',
                        default='../data/mpii/annot/',
                        type=str)

    args = parser.parse_args()

    return args


def load_data(data_path):
    print(f'Trying to load data from {Path(data_path, "json.mat")}')
    data = sio.loadmat(Path(data_path, 'json.mat'))
    data_json = json.loads(data['txt'][0])
    annotations = data_json['annolist']
    activity_information = data_json['act']
    image_train = data_json['img_train']

    # clean MATLAB's data structure inconsistency, i.e.,
    # one element is left as is (the direct element), but multiple elements are converted to a list of element.
    for obj in annotations:
        if isinstance(obj['annorect'], dict):
            obj['annorect'] = [obj['annorect']]

    return annotations, activity_information, image_train


def create_object_instances(annotations, activity_information, image_train):
    instances = []
    for idx, obj in tqdm(enumerate(annotations)):
        for item in obj['annorect']:
            proc_obj = {}
            if 'annopoints' in item and len(item['annopoints']) > 0:
                try:
                    proc_obj['image'] = obj['image']['name']
                    proc_obj['scale'] = item['scale']
                    proc_obj['center'] = [item['objpos']['x'], item['objpos']['y']]
                    proc_obj['bbox'] = (item['x1'], item['y1'], item['x2'], item['y2'])
                    proc_obj['is_train'] = image_train[idx]
                    df = pd.DataFrame.from_dict(sorted(item['annopoints']['point'], key=lambda x: x['id']))
                    joints = np.array([df.x.tolist(), df.y.tolist()]).T.tolist()
                    joints_vis = (df.is_visible == True).astype(int).tolist()
                    proc_obj['joints'] = joints
                    proc_obj['joints_vis'] = joints_vis
                    proc_obj['cat_name'] = activity_information[idx]['cat_name']
                    proc_obj['act_name'] = activity_information[idx]['act_name']
                    proc_obj['act_id'] = activity_information[idx]['act_id']
                    instances.append(proc_obj)
                except:
                    pprint(item)
    print('Total # of BBox on People:', len(instances))
    return instances


def add_info_to_json(instances, json_fname, data_path='./'):
    with open(Path(data_path, json_fname), 'r') as f:
        data = json.loads(f.read())

    for idx in tqdm(range(len(data))):
        dp = data[idx]
        for item in instances:
            if item['image'] == dp['image'] and abs(item['scale'] - dp['scale']) < 0.00001:
                dp['bbox'] = item['bbox']
                dp['cat_name'] = item['cat_name']
                dp['act_name'] = item['act_name']
                dp['act_id'] = item['act_id']

    with open(Path(data_path, f'proc_{json_fname}'), 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    args = parse_args()

    annotations, activity_information, image_train = load_data(args.dpath)
    instances = create_object_instances(annotations, activity_information, image_train)

    add_info_to_json(instances, 'train.json', data_path=args.dpath)
    add_info_to_json(instances, 'valid.json', data_path=args.dpath)
    add_info_to_json(instances, 'trainval.json', data_path=args.dpath)
