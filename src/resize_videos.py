import argparse
import glob
import os
import subprocess

exts = ['avi', 'mp4', 'mkv', 'webm']

parser = argparse.ArgumentParser()
parser.add_argument('-vi', '--in_dir', type=str, help='input directory')
parser.add_argument('-vo', '--out_dir', type=str, help='output directory')
parser.add_argument('-s', '--size', type=int, default=288, help='short edge')
parser.add_argument('-fps', '--fps', type=float, default=None, help='frame rate')
args = parser.parse_args()

assert os.path.exists(args.in_dir), 'video directory does not exist'
os.makedirs(args.out_dir, exist_ok=True)

filepaths = []
for ext in exts:
    filepaths += sorted(
        glob.glob(os.path.join(args.in_dir, '*.{:s}'.format(ext)))
    )

for path in filepaths:
    name = os.path.splitext(os.path.basename(path))[0]
    print('Processing {:s} ...'.format(name))
    out_path = os.path.join(args.out_dir, name + '.mp4')
    if os.path.exists(out_path):
        continue

    cmd = "ffmpeg -i {:s} -c:v libx264 -crf 17 " \
          "-vf scale='if(lte(iw\,ih)\,{:d}\,-2)':'if(lte(iw\,ih)\,-2\,{:d})'" \
          "".format(path, args.size, args.size)
    if args.fps is not None:
        cmd += ",fps={:f}".format(args.fps)
    cmd += " " + out_path
    subprocess.call(cmd, shell=True)