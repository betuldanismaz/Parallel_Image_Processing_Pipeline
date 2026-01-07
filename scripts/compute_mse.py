import os
import sys
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def load_image(path):
    if not os.path.exists(path):
        return None
    img = mpimg.imread(path)
    # If image is uint8 in [0,255], convert to float in [0,255]
    if img.dtype == np.float32 or img.dtype == np.float64:
        # PNGs read by matplotlib often give floats in [0,1]
        if img.max() <= 1.0:
            img = (img * 255.0).astype(np.float32)
        else:
            img = img.astype(np.float32)
    else:
        img = img.astype(np.float32)
    return img


def compute_mse(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError(f'image shapes differ: {img1.shape} vs {img2.shape}')
    diff = img1 - img2
    mse = np.mean(np.square(diff))
    return mse, diff


def save_diff(diff, outpath):
    # Normalize absolute diff for visualization
    absdiff = np.abs(diff)
    # Collapse color channels by max
    if absdiff.ndim == 3:
        vis = np.max(absdiff, axis=2)
    else:
        vis = absdiff
    # Normalize to 0-1
    vmax = vis.max() if vis.max() > 0 else 1.0
    vis_norm = vis / vmax
    plt.imsave(outpath, vis_norm, cmap='hot')


def main():
    serial_path = os.path.join('results', 'output_serial_7x7.png')
    openmp_path = os.path.join('results', 'output_openmp_7x7.png')
    mpi_path = os.path.join('output_mpi_7x7.png')

    serial = load_image(serial_path)
    if serial is None:
        print(f'Serial output not found: {serial_path}', file=sys.stderr)
        sys.exit(1)

    pairs = [
        ('OpenMP', openmp_path),
        ('MPI', mpi_path),
    ]

    for name, path in pairs:
        img = load_image(path)
        if img is None:
            print(f'{name} output not found: {path} — skipping')
            continue
        try:
            mse, diff = compute_mse(serial, img)
        except ValueError as e:
            print(f'Error computing MSE for {name}: {e}')
            continue
        print(f'{name} vs Serial — MSE: {mse:.6f}')
        outdiff = os.path.join('results', f'diff_serial_{name.lower()}.png')
        save_diff(diff, outdiff)
        print(f'Diff image saved: {outdiff}')


if __name__ == '__main__':
    main()
