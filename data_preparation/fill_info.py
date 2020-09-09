import json
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
                        required=True,
                        type=str)

    args = parser.parse_args()

    return args


def add_info_to_json(json_fname, data_path='./'):
    with open(Path(data_path, json_fname), 'r') as f:
        data = json.loads(f.read())

    for idx in tqdm(range(len(data))):
        dp = data[idx]
        for item in result:
            if item['image'] == dp['image'] and abs(item['scale'] - dp['scale']) < 0.00001:
                dp['bbox'] = item['bbox']
                dp['cat_name'] = item['cat_name']
                dp['act_name'] = item['act_name']
                dp['act_id'] = item['act_id']

    with open(Path(data_path, f'proc_{json_fname}'), 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    args = parse_args()
    DATA_PATH = args.dpath

    data = sio.loadmat(Path(DATA_PATH, 'json.mat'))
    data_json = json.loads(data['txt'][0])
    my_data = data_json['annolist']
    act = data_json['act']
    img_train = data_json['img_train']

    # clean MATLAB's data structure inconsistency, i.e.,
    # one element is left as is (the direct element), but multiple elements are converted to a list of element.
    for obj in my_data:
        if isinstance(obj['annorect'], dict):
            obj['annorect'] = [obj['annorect']]

    result = []
    for idx, obj in enumerate(my_data):
        for item in obj['annorect']:
            proc_obj = {}
            if 'annopoints' in item and len(item['annopoints']) > 0:
                try:
                    proc_obj['image'] = obj['image']['name']
                    proc_obj['scale'] = item['scale']
                    proc_obj['center'] = [item['objpos']['x'], item['objpos']['y']]
                    proc_obj['bbox'] = (item['x1'], item['y1'], item['x2'], item['y2'])
                    proc_obj['is_train'] = img_train[idx]
                    df = pd.DataFrame.from_dict(sorted(item['annopoints']['point'], key=lambda x: x['id']))
                    joints = np.array([df.x.tolist(), df.y.tolist()]).T.tolist()
                    joints_vis = (df.is_visible == True).astype(int).tolist()
                    proc_obj['joints'] = joints
                    proc_obj['joints_vis'] = joints_vis
                    proc_obj['cat_name'] = act[idx]['cat_name']
                    proc_obj['act_name'] = act[idx]['act_name']
                    proc_obj['act_id'] = act[idx]['act_id']
                    result.append(proc_obj)
                except:
                    pprint(item)

    print('Total # of BBox on People:', len(result))

    add_info_to_json('train.json')
    add_info_to_json('valid.json')
    add_info_to_json('trainval.json')
