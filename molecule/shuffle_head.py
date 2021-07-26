import io
import random
import argparse
import os


def main(args):
    input_fn = args.fn
    lines = io.open(input_fn, 'r', encoding='utf8', newline='\n').readlines()
    random.shuffle(lines)
    assert args.head_num > 0
    if args.output_fn is None:
        prefix, ext = os.path.splitext(input_fn)
        output_fn = '{}.{}{}'.format(prefix, args.head_num, ext)
    else:
        output_fn = args.output_fn

    io.open(output_fn, 'w', encoding='utf8', newline='\n').writelines(lines[:args.head_num])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fn', type=str)
    parser.add_argument('head_num', type=int)
    parser.add_argument('--output-fn', type=str, default=None)
    args = parser.parse_args()
    main(args)
