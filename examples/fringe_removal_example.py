#!/usr/bin/env python3
"""
Standalone script for testing and visualizing inpainting-based fringe removal
on a real image.

Runs stdpipe.fringe_removal.remove_fringes() on a FITS image and displays the
original image, the reconstructed fringe pattern, and the corrected image side
by side, along with some basic statistics.

Usage:
    python3 fringe_removal_example.py image.fits
    python3 fringe_removal_example.py image.fits --scale 4 --iterations 4
    python3 fringe_removal_example.py image.fits -o result.png --save-fits corrected.fits
"""

import argparse
import os
import sys
import time

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std

from stdpipe.fringe_removal import remove_fringes


def main():
    parser = argparse.ArgumentParser(
        description='Test and visualize fringe removal on a FITS image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('filename', help='Input FITS image')
    parser.add_argument('--scale', type=float, default=6.0,
                        help='Sigma of Gaussian smoothing kernel, pixels')
    parser.add_argument('--bg-size', type=int, default=256,
                        help='Mesh size of the coarse sky background, pixels')
    parser.add_argument('--threshold', type=float, default=2.0,
                        help='Source masking threshold, sigmas')
    parser.add_argument('--dilate', type=int, default=2,
                        help='Source mask dilation iterations')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of refinement iterations')
    parser.add_argument('--halo-sn', type=float, default=20.0,
                        help='Peak significance above which the faint circumstellar '
                             'structure (halos, ghosts) is modeled radially and '
                             'subtracted along with the fringes, sigmas (0 to disable)')
    parser.add_argument('--ext', type=int, default=0,
                        help='FITS extension to read')
    parser.add_argument('-o', '--output', default=None,
                        help='Save the figure to this file instead of displaying it')
    parser.add_argument('--save-fits', default=None,
                        help='Save the corrected image to this FITS file')
    parser.add_argument('--save-model', default=None,
                        help='Save the fringe model to this FITS file')
    args = parser.parse_args()

    if not os.path.exists(args.filename):
        sys.exit(f"File not found: {args.filename}")

    # Load the image
    image = fits.getdata(args.filename, args.ext).astype(np.float64)
    header = fits.getheader(args.filename, args.ext)
    print(f"Loaded {args.filename}: {image.shape[1]}x{image.shape[0]} pixels")

    # Non-finite pixels are masked from the fringe estimation
    mask = ~np.isfinite(image)
    if np.any(mask):
        print(f"Masking {np.sum(mask)} non-finite pixels")

    # Run fringe removal
    t0 = time.time()
    corrected, model = remove_fringes(
        image,
        mask=mask if np.any(mask) else None,
        scale=args.scale,
        bg_size=args.bg_size,
        threshold=args.threshold,
        dilate=args.dilate,
        iterations=args.iterations,
        halo_sn=args.halo_sn if args.halo_sn > 0 else None,
        get_fringe_model=True,
        verbose=True,
    )
    print(f"Fringe removal took {time.time() - t0:.1f} s")

    # Statistics
    good = ~mask
    med = np.median(image[good])
    rms = mad_std(image[good])
    model_amp = mad_std(model[good])
    print(f"\nImage median: {med:.2f}, pixel rms (mad_std): {rms:.2f}")
    print(f"Fringe model amplitude (mad_std): {model_amp:.2f} "
          f"({100 * model_amp / rms:.1f}% of pixel rms)")

    # Optionally save FITS outputs
    if args.save_fits:
        fits.writeto(args.save_fits, corrected.astype(np.float32), header, overwrite=True)
        print(f"Corrected image saved to {args.save_fits}")
    if args.save_model:
        fits.writeto(args.save_model, model.astype(np.float32), header, overwrite=True)
        print(f"Fringe model saved to {args.save_model}")

    # Visualization
    import matplotlib
    if args.output:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    panels = [
        (image, 'Original', med, rms),
        (model, f'Fringe model (mad_std = {model_amp:.2f})',
         np.median(model[good]), max(2.5 * model_amp, 1e-10)),
        (corrected, 'Corrected', med, rms),
    ]

    for ax, (data, title, m, r) in zip(axes, panels):
        im = ax.imshow(data, origin='lower', cmap='gray',
                       vmin=m - 2 * r, vmax=m + 2 * r,
                       interpolation='nearest')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{os.path.basename(args.filename)} — "
        f"scale={args.scale} bg_size={args.bg_size} threshold={args.threshold} "
        f"dilate={args.dilate} iterations={args.iterations} halo_sn={args.halo_sn}",
        fontsize=10,
    )
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
