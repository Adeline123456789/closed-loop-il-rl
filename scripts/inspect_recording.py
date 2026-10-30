"""Inspect a real episode and optionally export RGB to ordinary PNG / MP4."""
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('episode', type=Path)
parser.add_argument('--output', type=Path)
parser.add_argument('--fps', type=float, default=15)
args = parser.parse_args()
for path in sorted(args.episode.glob('*.npy')):
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    print(f'{path.name:27} shape={str(array.shape):23} dtype={array.dtype}')
if args.output:
    from PIL import Image
    import imageio.v2 as imageio
    args.output.mkdir(parents=True, exist_ok=True)
    frames = np.load(args.episode / 'rgb.npy', mmap_mode='r', allow_pickle=False)
    for t in (0, len(frames)//2, len(frames)-1):
        Image.fromarray(frames[t]).save(args.output / f'frame_{t:03}.png')
    with imageio.get_writer(str(args.output / 'rgb.mp4'), fps=args.fps, codec='libx264', quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    print('Exported genuine RGB frames:', args.output)
