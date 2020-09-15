import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def parse_args():
    parser = argparse.ArgumentParser(description='Fill activity related information to the annotation files.')
    # general
    parser.add_argument('--dpath',
                        help='Path of the files.',
                        default='../data/mpii/annot/',
                        type=str)

    parser.add_argument('--seed',
                        help='The random seed used.',
                        default=2020,
                        type=int)

    args = parser.parse_args()

    return args


def df2json(fname, df):
    """Save rows in a DataFrame as elements in a JSON list."""
    l = []
    for _, row in df.iterrows():
        l.append(row.to_dict())
    with open(fname, 'w') as f:
        json.dump(l, f)
    return df


def sample_n_points_by_act_name(df, act_name, n=10):
    sub_df = df[df.act_name == act_name]
    return sub_df.sample(n)


def load_and_clean(dpath):
    with open(Path(dpath, 'proc_trainval.json'), 'r') as f:
        all_data = json.loads(f.read())
    print('# data point before cleaning:', len(all_data))

    new_all_data = []
    for idx, data in enumerate(all_data):
        if 'act_id' not in data:
            continue
        else:
            new_all_data.append(data)
    all_data = sorted(new_all_data, key=lambda x: x['act_id'])

    with open(Path(dpath, 'proc_all.json'), 'w') as f:
        json.dump(all_data, f)

    print('# data point after cleaning:', len(all_data))

    return all_data


if __name__ == '__main__':
    args = parse_args()
    data = load_and_clean(args.dpath)
    df = pd.DataFrame.from_dict(data)

    gen2fine = defaultdict(set)
    fine2gen = dict()
    for idx, row in df.iterrows():
        gen2fine[row.cat_name].add(row.act_name)
        fine2gen[row.act_name] = row.cat_name

    dynamic_cats = ('water activities', 'sports', 'bicycling', 'winter activities', 'running')
    ndynamic_cats = set(df.cat_name.unique()).difference(dynamic_cats)
    while True:
        # select action categories which contain more than "this value" frames
        act_threshold = int(
            input('Action frame # threshold (actions that contain less than this number of frame will be ignored): '))

        d = {}
        ct = Counter(df.act_name)
        for item in ct:
            if len(item) and ct[item] > act_threshold:
                d[item] = ct[item]

        dynamic_candidates = dict()
        ndynamic_candidates = dict()
        for item in d:
            if fine2gen[item] in dynamic_cats:
                dynamic_candidates[item] = d[item]
            else:
                ndynamic_candidates[item] = d[item]

        print(f'With threshold {act_threshold}, '
              f'now we have {len(dynamic_candidates)} dynamic categories, '
              f'and {len(ndynamic_candidates)} non-dynamic categories')

        dfs = []
        for cat in dynamic_candidates:
            dfs.append(df[df.act_name == cat])
        dfs = pd.concat(dfs).reset_index(drop=True)

        ndfs = []
        for cat in ndynamic_candidates:
            ndfs.append(df[df.act_name == cat])
        ndfs = pd.concat(ndfs).reset_index(drop=True)

        print(f'Dynamic # frames: {dfs.shape[0]} || Non-dynamic # frames: {ndfs.shape[0]}')

        if input('Good with the current setting? (Y/n)').lower() == 'y':
            break

    output_folder = Path(args.dpath, f'split_at_{act_threshold}')
    output_folder.mkdir(exist_ok=True)
    print(f'Making dir at {output_folder} ... Done!')

    sss = StratifiedShuffleSplit(n_splits=1, test_size=3000 / ndfs.shape[0], random_state=args.seed)
    for tr_idx, val_idx in sss.split(ndfs.act_name.tolist(), ndfs.act_name.tolist()):
        break
    used_ndynamic = ndfs.iloc[val_idx]
    unused_ndynamic = ndfs.iloc[tr_idx]
    unused_ndynamic = df2json(Path(output_folder, f'pretrain_unused.json'), unused_ndynamic)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=500 / 3000, random_state=args.seed)
    for tr_idx, val_idx in sss.split(used_ndynamic.act_name.tolist(), used_ndynamic.act_name.tolist()):
        break

    train = df2json(Path(output_folder, f'pretrain_train.json'), used_ndynamic.iloc[tr_idx])
    valid = df2json(Path(output_folder, f'pretrain_valid.json'), used_ndynamic.iloc[val_idx])
    print(f'Used train # non-dynamic instances is {train.shape[0]}, '
          f'cover {len(Counter(train.act_name))} fine-grained action categories.')
    print(f'Used valid # non-dynamic instances is {valid.shape[0]}, '
          f'cover {len(Counter(valid.act_name))} fine-grained action categories.')
    print(f'Used # non-dynamic instances is {used_ndynamic.shape[0]}, '
          f'cover {len(Counter(used_ndynamic.act_name))} fine-grained action categories.')
    print(f'Unused # non-dynamic instances is {unused_ndynamic.shape[0]}, '
          f'cover {len(Counter(unused_ndynamic.act_name))} fine-grained action categories.')

    used_dynamic = []
    for act in dfs.act_name.unique():
        sub_df = sample_n_points_by_act_name(dfs, act, act_threshold)
        used_dynamic.append(sub_df)

    used_dynamic = pd.concat(used_dynamic)
    print(f'We have {used_dynamic.shape[0]} frames, '
          f'cover {len(used_dynamic.act_name.unique())} dynamic action categories, are available to use.')

    dfs.index = list(dfs.index)
    unused_dynamic = dfs.drop(used_dynamic.index)

    used_dynamic = df2json(Path(output_folder, f'finetune.json'), used_dynamic)
    unused_dynamic = df2json(Path(output_folder, f'finetune_test.json'), unused_dynamic)

    print(f'Used finetune # dynamic instances is {used_dynamic.shape[0]}, '
          f'cover {len(Counter(used_dynamic.act_name))} fine-grained action categories.')
    print(f'Used test # dynamic instances is {unused_dynamic.shape[0]}, '
          f'cover {len(Counter(unused_dynamic.act_name))} fine-grained action categories.')
