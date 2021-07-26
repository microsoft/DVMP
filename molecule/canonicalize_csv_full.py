import argparse
from tqdm import tqdm

from rdkit import Chem
import io
import multiprocessing
import pandas as pd
import re


def rm_map_number(smiles):
    t = re.sub(':\d*', '', smiles)
    return t


def canonicalize(tsmiles):
    reactions, keep_atommap = tsmiles
    reactants, products = reactions.strip().split('>>')
    if not keep_atommap:
        reactants = rm_map_number(reactants)
        products = rm_map_number(products)

    reactants_mol = Chem.MolFromSmiles(reactants)
    products_mol = Chem.MolFromSmiles(products)

    if (reactants_mol is None) or (products_mol is None):
        return None
    else:
        return Chem.MolToSmiles(reactants_mol), Chem.MolToSmiles(products_mol)


def main(args):
    input_fn = args.fn
    if args.output_fn is None:
        output_fn = input_fn.replace('.csv', '')
    else:
        output_fn = args.output_fn
    prod_output_fn = output_fn + '.prod'
    reac_output_fn = output_fn + '.reac'

    df = pd.read_csv(input_fn)
    total_cnt = df.shape[0]
    invalid_cnt = 0

    pool = multiprocessing.Pool(args.workers)

    def input_args():
        for f in df[args.field]:
            yield f, args.keep_atommapnum

    reactants_smiles_list = []
    products_smiles_list = []

    for res in tqdm(pool.imap(canonicalize, input_args(), chunksize=1000), total=total_cnt):
        if res is None:
            invalid_cnt += 1
        else:
            reactants_smiles, products_smiles = res
            reactants_smiles_list.append("{}\n".format(reactants_smiles))
            products_smiles_list.append("{}\n".format(products_smiles))
    io.open(prod_output_fn, 'w', encoding='utf8', newline='\n').writelines(products_smiles_list)
    io.open(reac_output_fn, 'w', encoding='utf8', newline='\n').writelines(reactants_smiles_list)
    print("Invalid: {}/{}".format(invalid_cnt, total_cnt))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fn', type=str)
    parser.add_argument('--field', type=str, default='rxn_smiles')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output-fn', type=str, default=None)
    parser.add_argument('--keep-atommapnum', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
