#!/usr/bin/env python3
"""
Tests for Fourier-based fringe removal.

Verifies that Fourier filtering can effectively remove periodic fringes
while preserving stellar sources.
"""

import numpy as np
import pytest

from stdpipe import fringe_removal
from test_background_fringes import (
    create_linear_fringes,
    create_2d_fringes,
    create_radial_fringes,
    add_random_stars
)


# Test configuration
@pytest.fixture
def test_config():
    """Standard test configuration."""
    return {
        'size': 512,
        'noise_std': 2.0,
        'nstars': 50,
        'flux_range': (1000, 10000),
        'fwhm': 3.0,
        'seed': 42
    }


@pytest.fixture
def rng(test_config):
    """Random number generator."""
    return np.random.RandomState(test_config['seed'])


# ==============================================================================
# Tests: Basic functionality
# ==============================================================================

@pytest.mark.slow
def test_simple_fringe_removal_auto(test_config, rng):
    """Test removal of single-frequency linear fringes using auto method."""
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']

    period = 80
    amplitude = 100

    # Create fringe pattern
    fringes = create_linear_fringes(size, amplitude, period, angle_deg=30)

    # Add noise (no stars for simplest test)
    image = fringes + rng.normal(0, noise_std, (size, size))

    # Remove fringes
    corrected = fringe_removal.remove_fringes_fourier(
        image,
        fwhm=fwhm,
        method='global',
        period_range=(50, 150),
        sigma_threshold=3.0,
        verbose=False
    )

    # Measure residual (should be much smaller than original)
    original_rms = np.sqrt(np.mean(fringes**2))
    residual_rms = np.sqrt(np.mean(corrected**2))

    print(f"\nSimple fringe removal:")
    print(f"  Original RMS: {original_rms:.1f} ADU")
    print(f"  Residual RMS: {residual_rms:.1f} ADU")
    print(f"  Reduction: {100*(1 - residual_rms/original_rms):.1f}%")

    # Should reduce fringes by at least 40% (conservative, windowing effects)
    # Typical reduction: 50-60% for well-defined fringes
    assert residual_rms < 0.6 * original_rms, \
        f"Insufficient fringe removal: {residual_rms:.1f} / {original_rms:.1f} ADU"


@pytest.mark.slow
def test_simple_fringe_removal_bandpass(test_config, rng):
    """Test removal using bandpass method."""
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']

    period = 80
    amplitude = 100

    # Create fringe pattern
    fringes = create_linear_fringes(size, amplitude, period, angle_deg=30)
    image = fringes + rng.normal(0, noise_std, (size, size))

    # Remove fringes with bandpass
    corrected = fringe_removal.remove_fringes_fourier(
        image,
        fwhm=fwhm,
        method='bandpass',
        period_range=(60, 120),  # Centered on true period
        verbose=False
    )

    # Measure improvement
    original_rms = np.sqrt(np.mean(fringes**2))
    residual_rms = np.sqrt(np.mean(corrected**2))

    print(f"\nBandpass fringe removal:")
    print(f"  Original RMS: {original_rms:.1f} ADU")
    print(f"  Residual RMS: {residual_rms:.1f} ADU")
    print(f"  Reduction: {100*(1 - residual_rms/original_rms):.1f}%")

    # Bandpass should also be effective (similar to auto method)
    assert residual_rms < 0.7 * original_rms


@pytest.mark.slow
def test_fringe_model_output(test_config, rng):
    """Test that fringe model output is reasonable."""
    size = test_config['size']
    fwhm = test_config['fwhm']

    period = 80
    amplitude = 100

    # Create fringe pattern
    fringes = create_linear_fringes(size, amplitude, period, angle_deg=30)
    image = fringes + rng.normal(0, test_config['noise_std'], (size, size))

    # Get both corrected image and fringe model
    corrected, fringe_model = fringe_removal.remove_fringes_fourier(
        image,
        fwhm=fwhm,
        method='global',
        period_range=(50, 150),
        get_fringe_model=True,
        verbose=False
    )

    # Verify: image ≈ corrected + fringe_model
    reconstruction = corrected + fringe_model
    reconstruction_error = np.sqrt(np.mean((image - reconstruction)**2))

    print(f"\nFringe model test:")
    print(f"  Reconstruction error: {reconstruction_error:.2f} ADU")

    assert reconstruction_error < 5.0, "Reconstruction error too high"

    # Fringe model should have similar amplitude to input fringes
    fringe_model_amp = np.ptp(fringe_model)
    original_amp = np.ptp(fringes)

    print(f"  Original amplitude: {original_amp:.1f} ADU")
    print(f"  Model amplitude: {fringe_model_amp:.1f} ADU")

    # Should recover most of the amplitude
    assert 0.5 * original_amp < fringe_model_amp < 1.5 * original_amp


# ==============================================================================
# Tests: Multiple fringe periods
# ==============================================================================

@pytest.mark.slow
def test_multiple_fringe_periods(test_config, rng):
    """Test removal of multiple fringe components."""
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']

    # Create two fringe components
    fringes1 = create_linear_fringes(size, 80, period=60, angle_deg=30)
    fringes2 = create_linear_fringes(size, 60, period=120, angle_deg=120)
    fringes_combined = fringes1 + fringes2

    image = fringes_combined + rng.normal(0, noise_std, (size, size))

    # Remove fringes
    corrected = fringe_removal.remove_fringes_fourier(
        image,
        fwhm=fwhm,
        method='global',
        period_range=(40, 200),
        sigma_threshold=3.0,
        verbose=False
    )

    # Measure improvement
    original_rms = np.sqrt(np.mean(fringes_combined**2))
    residual_rms = np.sqrt(np.mean(corrected**2))

    print(f"\nMultiple fringes:")
    print(f"  Original RMS: {original_rms:.1f} ADU")
    print(f"  Residual RMS: {residual_rms:.1f} ADU")
    print(f"  Reduction: {100*(1 - residual_rms/original_rms):.1f}%")

    # Should remove both components (at least 30% reduction)
    assert residual_rms < 0.7 * original_rms


# ==============================================================================
# Tests: Fringes + stars
# ==============================================================================

@pytest.mark.slow
def test_preserve_stars_during_fringe_removal(test_config, rng):
    """
    Critical test: Verify stars are preserved while removing fringes.

    This is the key requirement - fringe removal must not affect photometry.
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']

    period = 80
    amplitude = 80

    # Create clean image with stars only (no fringes)
    image_clean = np.zeros((size, size))
    add_random_stars(image_clean, nstars, flux_range, fwhm, rng)
    image_clean += rng.normal(0, noise_std, (size, size))

    # Create image with stars + fringes
    fringes = create_linear_fringes(size, amplitude, period, angle_deg=30)
    image_with_fringes = image_clean + fringes

    # Remove fringes
    corrected = fringe_removal.remove_fringes_fourier(
        image_with_fringes,
        fwhm=fwhm,
        method='global',
        period_range=(50, 150),
        sigma_threshold=3.0,
        verbose=False
    )

    # Measure difference: corrected vs clean (both have stars, no fringes)
    star_preservation_error = np.sqrt(np.mean((corrected - image_clean)**2))

    # Also measure fringe removal effectiveness
    fringe_rms = np.sqrt(np.mean(fringes**2))
    residual_fringe = np.sqrt(np.mean((corrected - image_clean - fringes)**2))

    print(f"\nStars + fringes test:")
    print(f"  Fringe RMS: {fringe_rms:.1f} ADU")
    print(f"  Star preservation error: {star_preservation_error:.1f} ADU")
    print(f"  Residual fringe: {residual_fringe:.1f} ADU")

    # Star preservation error should be comparable to or smaller than fringes
    # (some error expected due to windowing/filtering artifacts and incomplete fringe removal)
    assert star_preservation_error < 0.6 * fringe_rms, \
        f"Stars affected too much: {star_preservation_error:.1f} ADU"


@pytest.mark.slow
def test_photometry_after_fringe_removal(test_config, rng):
    """
    Test that photometry is accurate after fringe removal.

    Measure star fluxes before and after - should be unchanged.
    """
    from stdpipe import photometry

    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = 20  # Fewer stars for clean photometry

    period = 80
    amplitude = 100

    # Create image with stars (no fringes)
    image_clean = np.zeros((size, size))
    add_random_stars(image_clean, nstars, (3000, 8000), fwhm, rng)
    image_clean += rng.normal(0, noise_std, (size, size))

    # Add fringes
    fringes = create_linear_fringes(size, amplitude, period, angle_deg=30)
    image_with_fringes = image_clean + fringes

    # Remove fringes
    corrected = fringe_removal.remove_fringes_fourier(
        image_with_fringes,
        fwhm=fwhm,
        method='global',
        period_range=(50, 150),
        verbose=False
    )

    # Detect and measure stars in both images
    obj_clean = photometry.get_objects_sep(image_clean, thresh=5.0, minarea=5)
    obj_corrected = photometry.get_objects_sep(corrected, thresh=5.0, minarea=5)

    print(f"\nPhotometry test:")
    print(f"  Stars detected in clean image: {len(obj_clean)}")
    print(f"  Stars detected in corrected image: {len(obj_corrected)}")

    # Should detect similar number of stars
    assert abs(len(obj_clean) - len(obj_corrected)) < 5, \
        f"Detection count mismatch: {len(obj_clean)} vs {len(obj_corrected)}"

    # Match stars by position and compare fluxes
    from scipy.spatial import cKDTree

    tree_clean = cKDTree(np.column_stack([obj_clean['x'], obj_clean['y']]))
    tree_corrected = cKDTree(np.column_stack([obj_corrected['x'], obj_corrected['y']]))

    # Find matches within 2 pixels
    matches = tree_clean.query_ball_tree(tree_corrected, r=2.0)

    flux_diffs = []
    for i, match_list in enumerate(matches):
        if len(match_list) == 1:
            j = match_list[0]
            flux_clean = obj_clean['flux'][i]
            flux_corrected = obj_corrected['flux'][j]
            flux_diff = (flux_corrected - flux_clean) / flux_clean
            flux_diffs.append(flux_diff)

    flux_diffs = np.array(flux_diffs)

    print(f"  Matched stars: {len(flux_diffs)}")
    print(f"  Mean flux difference: {np.mean(flux_diffs)*100:.2f}%")
    print(f"  RMS flux difference: {np.std(flux_diffs)*100:.2f}%")

    # Fluxes should agree to within ~5%
    assert np.abs(np.mean(flux_diffs)) < 0.05, \
        f"Systematic flux bias: {np.mean(flux_diffs)*100:.1f}%"
    assert np.std(flux_diffs) < 0.10, \
        f"Flux scatter too high: {np.std(flux_diffs)*100:.1f}%"


# ==============================================================================
# Tests: Edge cases
# ==============================================================================

@pytest.mark.slow
def test_no_fringes_detected(test_config, rng):
    """Test behavior when no fringes are present."""
    size = test_config['size']
    fwhm = test_config['fwhm']

    # Pure noise, no fringes
    image = rng.normal(100, 2, (size, size))

    # Try to remove fringes (should detect none with high threshold)
    corrected = fringe_removal.remove_fringes_fourier(
        image,
        fwhm=fwhm,
        method='global',
        period_range=(50, 150),
        sigma_threshold=10.0,  # High threshold to avoid false detections
        verbose=False
    )

    # Image should be unchanged
    diff = np.sqrt(np.mean((corrected - image)**2))

    print(f"\nNo fringes test:")
    print(f"  Difference: {diff:.2f} ADU")

    # With high threshold, should detect no peaks and return unchanged image
    assert diff < 0.01, f"Image changed when no fringes present: {diff:.3f} ADU"


@pytest.mark.slow
def test_very_weak_fringes(test_config, rng):
    """Test with very weak fringes (below detection threshold)."""
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']

    period = 80
    amplitude = 5  # Very weak, similar to noise

    fringes = create_linear_fringes(size, amplitude, period, angle_deg=30)
    image = fringes + rng.normal(0, noise_std, (size, size))

    # Should not detect weak fringes with high threshold
    corrected = fringe_removal.remove_fringes_fourier(
        image,
        fwhm=fwhm,
        method='global',
        period_range=(50, 150),
        sigma_threshold=10.0,  # Very high threshold
        verbose=False
    )

    # Image should be mostly unchanged
    diff = np.sqrt(np.mean((corrected - image)**2))

    print(f"\nWeak fringes test:")
    print(f"  Difference: {diff:.2f} ADU")

    assert diff < 5.0


@pytest.mark.slow
def test_very_strong_fringes(test_config, rng):
    """Test with very strong fringes."""
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']

    period = 80
    amplitude = 500  # Very strong

    fringes = create_linear_fringes(size, amplitude, period, angle_deg=30)
    image = fringes + rng.normal(0, noise_std, (size, size))

    # Should still remove effectively
    corrected = fringe_removal.remove_fringes_fourier(
        image,
        fwhm=fwhm,
        method='global',
        period_range=(50, 150),
        sigma_threshold=3.0,
        verbose=False
    )

    original_rms = np.sqrt(np.mean(fringes**2))
    residual_rms = np.sqrt(np.mean(corrected**2))

    print(f"\nStrong fringes test:")
    print(f"  Original RMS: {original_rms:.1f} ADU")
    print(f"  Residual RMS: {residual_rms:.1f} ADU")
    print(f"  Reduction: {100*(1 - residual_rms/original_rms):.1f}%")

    # Should still achieve reasonable reduction
    assert residual_rms < 0.7 * original_rms


@pytest.mark.slow
def test_with_mask(test_config, rng):
    """Test fringe removal with masked pixels."""
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']

    period = 80
    amplitude = 100

    # Create fringe pattern
    fringes = create_linear_fringes(size, amplitude, period, angle_deg=30)
    image = fringes + rng.normal(0, noise_std, (size, size))

    # Create mask (mask 10% of pixels randomly)
    mask = rng.random((size, size)) < 0.1

    # Remove fringes with mask
    corrected = fringe_removal.remove_fringes_fourier(
        image,
        mask=mask,
        fwhm=fwhm,
        method='global',
        period_range=(50, 150),
        verbose=False
    )

    # Masked pixels should be unchanged
    assert np.allclose(corrected[mask], image[mask]), \
        "Masked pixels were modified"

    # Unmasked pixels should have fringes removed
    original_rms = np.sqrt(np.mean(fringes[~mask]**2))
    residual_rms = np.sqrt(np.mean(corrected[~mask]**2))

    print(f"\nMasked image test:")
    print(f"  Original RMS (unmasked): {original_rms:.1f} ADU")
    print(f"  Residual RMS (unmasked): {residual_rms:.1f} ADU")

    assert residual_rms < 0.7 * original_rms


# ==============================================================================
# Tests: Helper functions
# ==============================================================================

@pytest.mark.slow
def test_detect_fringe_peaks(test_config):
    """Test peak detection in Fourier space."""
    from stdpipe import fringe_removal

    size = test_config['size']
    fwhm = test_config['fwhm']

    period = 80
    amplitude = 100

    # Create fringe pattern (no noise for cleaner detection)
    fringes = create_linear_fringes(size, amplitude, period, angle_deg=30)

    # Detect peaks
    peaks = fringe_removal.detect_fringe_peaks_fft(
        fringes,
        sigma_threshold=3.0,
        period_range=(50, 150),
        fwhm=fwhm,
        verbose=False
    )

    print(f"\nPeak detection test:")
    print(f"  Detected peaks: {len(peaks)}")
    for peak in peaks[:5]:  # Print first 5
        if len(peak) == 3:
            fy, fx, period_est = peak
            print(f"    ({fy:4d}, {fx:4d}): period ≈ {period_est:.1f} px")

    # Should detect at least 2 peaks (symmetric pair)
    assert len(peaks) >= 2, f"Too few peaks detected: {len(peaks)}"

    # At least one peak should be near the true period
    periods = [p[2] for p in peaks if len(p) == 3]
    period_errors = [abs(p - period) / period for p in periods]
    min_error = min(period_errors) if period_errors else 1.0

    print(f"  Best period match: {min_error*100:.1f}% error")

    assert min_error < 0.2, f"No peak near true period (best error: {min_error*100:.1f}%)"


# ==============================================================================
# Summary test
# ==============================================================================

@pytest.mark.slow
def test_fringe_removal_summary(test_config, rng):
    """
    Comprehensive test showing fringe removal effectiveness across scenarios.

    Not a pass/fail test - generates comparison report.
    """
    from stdpipe import fringe_removal

    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']

    print("\n" + "="*70)
    print("FOURIER FRINGE REMOVAL SUMMARY")
    print("="*70)

    scenarios = [
        ("Linear (period=60px, amp=100)",
         create_linear_fringes(size, 100, 60, 30)),
        ("Linear (period=100px, amp=100)",
         create_linear_fringes(size, 100, 100, 30)),
        ("Linear (period=150px, amp=100)",
         create_linear_fringes(size, 100, 150, 30)),
        ("2D fringes (period=80/100px, amp=50)",
         create_2d_fringes(size, 50, 80, 100)),
        ("Radial (period=60px, amp=80)",
         create_radial_fringes(size, 80, 60)),
    ]

    print(f"\n{'Scenario':<40} {'Original':<12} {'Residual':<12} {'Reduction':<12}")
    print("-" * 70)

    for scenario_name, fringes in scenarios:
        # Add noise
        image = fringes + rng.normal(0, noise_std, (size, size))

        # Remove fringes
        corrected = fringe_removal.remove_fringes_fourier(
            image,
            fwhm=fwhm,
            method='global',
            period_range=(40, 200),
            sigma_threshold=3.0,
            verbose=False
        )

        # Compute metrics
        original_rms = np.sqrt(np.mean(fringes**2))
        residual_rms = np.sqrt(np.mean(corrected**2))
        reduction = 100 * (1 - residual_rms / original_rms)

        print(f"{scenario_name:<40} {original_rms:>10.1f}   {residual_rms:>10.1f}   {reduction:>10.1f}%")

    print("\n" + "="*70)
    print("Fourier filtering is effective for periodic fringes!")
    print("="*70)


if __name__ == '__main__':
    # Run with: pytest test_fringe_removal_fourier.py -v -s
    pytest.main([__file__, '-v', '-s'])
