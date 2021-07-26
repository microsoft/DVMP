import io
import os
import argparse
import re
from molecule.deepchem_dataloader import load_molnet_dataset, MOLNET_DIRECTORY


def main(dataset, save_dir):
    tasks, dfs, _ = load_molnet_dataset(dataset)
    save_dir = os.path.join(save_dir, dataset)
    os.makedirs(save_dir, exist_ok=True)

    subsets = ['train', 'dev', 'test']
    for subset, df in zip(subsets, dfs):
        smiles = df['text'].values.tolist()
        labels = df['labels'].values.tolist()
        assert len(smiles) == len(labels), \
            'len(smiles) {} != len(labels) {}'.format(len(smiles), len(labels))

        for task_id, task in enumerate(tasks):
            task = re.sub(' ', '_', task)
            task = re.sub('\W', '', task)
            task_save_dir = os.path.join(save_dir, task)
            os.makedirs(task_save_dir, exist_ok=True)

            f1 = io.open(os.path.join(task_save_dir, subset + '.input0'),
                        'w', encoding='utf8', newline='\n')
            f2 = io.open(os.path.join(task_save_dir, subset + '.label'),
                        'w', encoding='utf8', newline='\n')
            
            for smi, label in zip(smiles, labels):
                f1.write(smi+'\n')
                f2.write("{}\n".format(label[task_id]))
            f1.close()
            f2.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,)
    parser.add_argument('--save-dir', type=str, default=os.path.abspath(os.path.dirname(__file__)))
    args = parser.parse_args()
    dataset = args.dataset
    save_dir = args.save_dir
    main(dataset, save_dir)