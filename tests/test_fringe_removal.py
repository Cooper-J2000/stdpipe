#!/usr/bin/env python3
"""
Tests for inpainting-based fringe removal (fringe_removal.remove_fringes).

Uses curved, spatially varying fringe patterns - the realistic case where
frequency-domain methods fail - and verifies that the fringe map is correctly
continued ("inpainted") under the stars.
"""

import numpy as np
import pytest

from stdpipe.fringe_removal import remove_fringes
from test_background_fringes import add_random_stars


@pytest.fixture
def test_config():
    """Standard test configuration."""
    return {
        'size': 512,
        'sky': 100.0,
        'noise_std': 5.0,
        'nstars': 50,
        'flux_range': (2000, 20000),
        'fwhm': 3.0,
        'fringe_amplitude': 10.0,
        'fringe_period': 40.0,
        'seed': 42,
    }


@pytest.fixture
def rng(test_config):
    return np.random.RandomState(test_config['seed'])


def create_curved_fringes(size, amplitude, period, center_frac=(-0.3, 1.2)):
    """
    Curved, non-stationary fringe pattern: chirped concentric arcs around an
    off-image center, so both the orientation and the local period vary
    across the field - mimicking realistic thin-film interference fringes.
    """
    y, x = np.mgrid[:size, :size]
    cy, cx = center_frac[0] * size, center_frac[1] * size
    r = np.hypot(x - cx, y - cy)
    # Local period shrinks gradually across the field (chirp)
    phase = 2 * np.pi * r / period * (1 + 0.3 * r / (2 * size))
    return amplitude * np.sin(phase)


@pytest.fixture
def fringed_scene(test_config, rng):
    """Scene with sky + curved fringes + stars + noise, and its components."""
    c = test_config
    size = c['size']

    fringes = create_curved_fringes(size, c['fringe_amplitude'], c['fringe_period'])

    clean = np.full((size, size), c['sky'])
    add_random_stars(clean, c['nstars'], c['flux_range'], c['fwhm'], rng)
    clean += rng.normal(0, c['noise_std'], (size, size))

    return {'image': clean + fringes, 'clean': clean, 'fringes': fringes}


# Central region slice to avoid convolution edge effects in metrics
def central(a, margin=32):
    return a[margin:-margin, margin:-margin]


@pytest.mark.unit
def test_curved_fringes_removed(test_config, fringed_scene):
    """The recovered fringe model should match the injected curved pattern."""
    corrected, model = remove_fringes(
        fringed_scene['image'], get_fringe_model=True
    )

    fringes = fringed_scene['fringes']
    resid = central(model - fringes)
    resid -= np.median(resid)  # constant offsets are degenerate with the sky

    rms_in = np.std(central(fringes))
    rms_out = np.std(resid)

    print(f"\nCurved fringes: rms {rms_in:.2f} -> residual {rms_out:.2f}")
    assert rms_out < 0.35 * rms_in


@pytest.mark.unit
def test_reconstruction_identity(test_config, fringed_scene):
    """corrected + model must reproduce the input exactly."""
    corrected, model = remove_fringes(
        fringed_scene['image'], get_fringe_model=True
    )
    assert np.allclose(corrected + model, fringed_scene['image'])


@pytest.mark.unit
def test_fringe_model_inpainted_under_stars(test_config, fringed_scene, rng):
    """
    The key "inpainting" property: the fringe model under the stars should
    follow the true fringe pattern, not the stellar flux.
    """
    from scipy.ndimage import gaussian_filter

    c = test_config
    size = c['size']

    # Add a few additional bright stars at known positions
    image = fringed_scene['image'].copy()
    xs, ys = [100, 250, 400], [150, 300, 100]
    star = np.zeros_like(image)
    for x0, y0 in zip(xs, ys):
        star[y0, x0] = 1e5
    image += gaussian_filter(star, c['fwhm'] / 2.355) * (c['fwhm'] / 2.355)**2 * 2 * np.pi

    corrected, model = remove_fringes(image, get_fringe_model=True)

    fringes = fringed_scene['fringes']
    for x0, y0 in zip(xs, ys):
        stamp_model = model[y0 - 3 : y0 + 4, x0 - 3 : x0 + 4]
        stamp_true = fringes[y0 - 3 : y0 + 4, x0 - 3 : x0 + 4]
        diff = np.mean(stamp_model - stamp_true)
        print(f"star at ({x0},{y0}): model-true = {diff:.2f} ADU")
        # Error well below fringe amplitude => star did not leak into the map
        assert abs(diff) < 0.5 * c['fringe_amplitude']


@pytest.mark.unit
def test_star_flux_preserved(test_config, fringed_scene):
    """Aperture photometry should be unaffected by fringe removal."""
    from stdpipe import photometry
    from scipy.spatial import cKDTree

    corrected = remove_fringes(fringed_scene['image'])

    obj_clean = photometry.get_objects_sep(fringed_scene['clean'], thresh=5.0)
    obj_corr = photometry.get_objects_sep(corrected, thresh=5.0)

    tree = cKDTree(np.column_stack([obj_clean['x'], obj_clean['y']]))
    dist, idx = tree.query(np.column_stack([obj_corr['x'], obj_corr['y']]))
    matched = dist < 2.0
    ratio = obj_corr['flux'][matched] / obj_clean['flux'][idx[matched]]

    print(f"\nMatched {np.sum(matched)} stars, median flux ratio {np.median(ratio):.4f}")
    assert np.sum(matched) > 0.8 * len(obj_clean)
    assert abs(np.median(ratio) - 1) < 0.02


@pytest.mark.unit
def test_mask_and_nan_handling(test_config, fringed_scene, rng):
    """NaN pixels and an explicit mask should not break the fringe estimate."""
    c = test_config
    image = fringed_scene['image'].copy()

    # Saturated column and a NaN blob
    image[:, 200:203] = 1e6
    image[50:80, 50:80] = np.nan
    mask = np.zeros_like(image, dtype=bool)
    mask[:, 200:203] = True

    corrected, model = remove_fringes(image, mask=mask, get_fringe_model=True)

    assert np.all(np.isfinite(model))

    fringes = fringed_scene['fringes']
    good = central(np.isfinite(image) & ~mask)
    resid = central(model - fringes)[good]
    resid = resid - np.median(resid)
    assert np.std(resid) < 0.5 * np.std(central(fringes))


@pytest.mark.unit
def test_pure_noise_no_overfitting(test_config, rng):
    """Without fringes, the model should stay much smaller than the noise."""
    from astropy.stats import mad_std

    c = test_config
    image = c['sky'] + rng.normal(0, c['noise_std'], (c['size'], c['size']))

    corrected, model = remove_fringes(image, get_fringe_model=True)

    amp = mad_std(central(model))
    print(f"\nPure noise: model amplitude {amp:.3f} vs noise {c['noise_std']}")
    assert amp < 0.2 * c['noise_std']


@pytest.mark.unit
def test_bg_size_none(test_config, fringed_scene):
    """Global-median coarse background mode should still remove fringes."""
    corrected, model = remove_fringes(
        fringed_scene['image'], bg_size=None, get_fringe_model=True
    )

    fringes = fringed_scene['fringes']
    resid = central(model - fringes)
    resid -= np.median(resid)
    assert np.std(resid) < 0.5 * np.std(central(fringes))


@pytest.mark.unit
def test_circumstellar_structure_removed(test_config, rng):
    """
    A bright star with an extended scattered-light halo and a ghost
    reflection ring: with the radial circumstellar modeling (default), both
    are captured by ring medians and subtracted along with the fringes,
    while the fringe structure passing through them is still measured from
    the data and the star core itself is left untouched.

    Uses a larger canvas so the ghost lies within the radial model reach
    (min(shape)/4) but outside the protected stellar wing zone.
    """
    c = test_config
    size = 1024
    x0 = y0 = size // 2

    fringes = create_curved_fringes(size, c['fringe_amplitude'], c['fringe_period'])
    clean = np.full((size, size), c['sky'])
    add_random_stars(clean, 4 * c['nstars'], c['flux_range'], c['fwhm'], rng)
    clean += rng.normal(0, c['noise_std'], (size, size))

    yy, xx = np.mgrid[:size, :size]
    r2 = (xx - x0) ** 2 + (yy - y0) ** 2
    rr = np.sqrt(r2)
    # Saturated-like core, a wide halo (sigma 25 px, peak 6x noise) and a
    # faint sub-threshold ghost ring (donut at r=140, peak 1.5x noise),
    # placed outside the protected stellar wing zone
    star = 5e4 * np.exp(-r2 / (2 * 2.0**2))
    halo = 6 * c['noise_std'] * np.exp(-r2 / (2 * 25.0**2))
    ghost = 1.5 * c['noise_std'] * np.exp(-((rr - 140.0) ** 2) / (2 * 10.0**2))
    image = clean + fringes + star + halo + ghost

    corrected, model = remove_fringes(image, get_fringe_model=True)

    # Outside the protected star wing zone, the corrected image should
    # reproduce the clean scene plus the star: fringes and ghost removed
    ring = (r2 >= 120**2) & (r2 < 200**2)
    err = (corrected - clean - star)[ring]

    print(f"\nCircumstellar residual: median {np.median(err):.2f}, "
          f"structure rms {np.std(err - np.median(err)):.2f} "
          f"(fringes {np.std(fringes[ring]):.2f}, "
          f"halo+ghost peak {np.max((halo + ghost)[ring]):.1f})")

    assert abs(np.median(err)) < 1.0
    assert np.std(err - np.median(err)) < 0.5 * np.std(fringes[ring])

    # The fringe structure in the model must come from the actual data
    # (inpainted through the halo), not be a smooth plateau
    err_model = (model - fringes - halo - ghost)[ring]
    struct_err = np.std(err_model - np.median(err_model))
    assert struct_err < 0.6 * np.std(fringes[ring])

    # The star core itself must stay in the corrected image
    assert corrected[y0, x0] > 0.5 * star[y0, x0]

    # The model around the star must be radially smooth - no pedestal
    # disks, collars or sharp rings (regression test for the various
    # transition artifacts of the circumstellar modeling)
    rbins = np.arange(6, 150, 3)
    prof = np.array([
        np.median(model[(rr >= a) & (rr < b)])
        for a, b in zip(rbins[:-1], rbins[1:])
    ])
    max_step = np.max(np.abs(np.diff(prof)))
    print(f"Max radial step in model around star: {max_step:.2f}")
    assert max_step < 0.3 * c['fringe_amplitude']


@pytest.mark.unit
def test_iterative_refinement(test_config, fringed_scene):
    """
    A single smoothing pass only partially captures fringes narrower than a
    few `scale`; additive refinement iterations should recover the rest.
    """
    def resid_rms(iterations):
        _, model = remove_fringes(
            fringed_scene['image'], iterations=iterations, get_fringe_model=True
        )
        resid = central(model - fringed_scene['fringes'])
        return np.std(resid - np.median(resid))

    rms_in = np.std(central(fringed_scene['fringes']))
    rms1, rms3 = resid_rms(1), resid_rms(3)

    print(f"\nResidual rms: input {rms_in:.2f}, 1 iter {rms1:.2f}, 3 iters {rms3:.2f}")
    assert rms1 < 0.8 * rms_in
    assert rms3 < 0.6 * rms1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
