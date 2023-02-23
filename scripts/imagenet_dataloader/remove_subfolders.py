from tqdm import tqdm

import os
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='../../../datasets/imagenet/val', help='the root folder')
    args = parser.parse_args()
    root = args.folder #'../../../datasets/imagenet/val'
    subfolders = os.listdir(root)
    real_subfolders = []
    for sub in subfolders:
        if os.path.isdir(os.path.join(root, sub)):
            real_subfolders.append(sub)

    print('Find %d subfolders in %s' % (len(real_subfolders), root))

    for sub in tqdm(real_subfolders):
        files = os.listdir(os.path.join(root, sub))
        for f in files:
            shutil.move( os.path.join(root, sub, f) , os.path.join(root, f) )
        shutil.rmtree(os.path.join(root, sub))

    files = os.listdir(root)
    print('Now there are %d files is %s in total' % (len(files), root))
            