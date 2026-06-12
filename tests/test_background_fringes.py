#!/usr/bin/env python3
"""
Tests for background estimation with fringe-like structures.

Fringes are periodic patterns in the background, common in near-IR imaging
due to interference effects. This tests how different background estimation
methods handle various fringe periods and amplitudes.
"""

import numpy as np
import pytest

from stdpipe import photometry


# Test configuration
@pytest.fixture
def test_config():
    """Standard test configuration."""
    return {
        'size': 512,
        'noise_std': 2.0,
        'nstars': 100,
        'flux_range': (1000, 10000),
        'fwhm': 3.0,
        'seed': 42
    }


@pytest.fixture
def rng(test_config):
    """Random number generator."""
    return np.random.RandomState(test_config['seed'])


# ==============================================================================
# Fringe pattern generators
# ==============================================================================

def create_linear_fringes(size, amplitude, period, angle_deg=0, phase=0):
    """
    Create linear fringe pattern (parallel stripes).

    Parameters
    ----------
    size : int
        Image size (square)
    amplitude : float
        Fringe amplitude (peak-to-peak)
    period : float
        Fringe period in pixels
    angle_deg : float
        Fringe orientation angle in degrees (0 = horizontal)
    phase : float
        Phase offset in radians

    Returns
    -------
    fringe : ndarray
        Fringe pattern
    """
    y, x = np.mgrid[:size, :size]

    # Rotate coordinates
    angle = np.deg2rad(angle_deg)
    x_rot = x * np.cos(angle) - y * np.sin(angle)

    # Create sinusoidal pattern
    fringe = amplitude * np.sin(2 * np.pi * x_rot / period + phase)

    return fringe


def create_2d_fringes(size, amplitude, period_x, period_y, phase_x=0, phase_y=0):
    """
    Create 2D fringe pattern (interference pattern).

    Creates more complex interference pattern from two orthogonal components.

    Parameters
    ----------
    size : int
        Image size (square)
    amplitude : float
        Fringe amplitude (peak-to-peak for each component)
    period_x : float
        Fringe period in x direction (pixels)
    period_y : float
        Fringe period in y direction (pixels)
    phase_x, phase_y : float
        Phase offsets in radians

    Returns
    -------
    fringe : ndarray
        2D fringe pattern
    """
    y, x = np.mgrid[:size, :size]

    # Two orthogonal components
    fringe_x = amplitude * np.sin(2 * np.pi * x / period_x + phase_x)
    fringe_y = amplitude * np.sin(2 * np.pi * y / period_y + phase_y)

    # Combine (interference)
    fringe = fringe_x + fringe_y

    return fringe


def create_radial_fringes(size, amplitude, period, center=None):
    """
    Create radial fringe pattern (concentric circles).

    Can represent Newton rings or other radial interference patterns.

    Parameters
    ----------
    size : int
        Image size (square)
    amplitude : float
        Fringe amplitude (peak-to-peak)
    period : float
        Fringe period in pixels (radial)
    center : tuple, optional
        Center position (y, x). Default: image center

    Returns
    -------
    fringe : ndarray
        Radial fringe pattern
    """
    if center is None:
        center = (size / 2, size / 2)

    y, x = np.mgrid[:size, :size]

    # Distance from center
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Radial sinusoidal pattern
    fringe = amplitude * np.sin(2 * np.pi * r / period)

    return fringe


# ==============================================================================
# Helper functions (reuse from test_background_estimation.py)
# ==============================================================================

def add_random_stars(image, nstars, flux_range, fwhm, rng):
    """Add random stars to image."""
    from scipy.ndimage import gaussian_filter

    size = image.shape[0]
    sigma = fwhm / 2.355

    for _ in range(nstars):
        x = rng.uniform(50, size - 50)
        y = rng.uniform(50, size - 50)
        flux = rng.uniform(*flux_range)

        # Create Gaussian PSF
        yy, xx = np.mgrid[:int(6*sigma), :int(6*sigma)]
        yy = yy - 3*sigma
        xx = xx - 3*sigma
        psf = flux * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        psf /= psf.sum()
        psf *= flux

        # Add to image
        x0 = int(x - 3*sigma)
        y0 = int(y - 3*sigma)
        x1 = x0 + psf.shape[1]
        y1 = y0 + psf.shape[0]

        if x0 >= 0 and y0 >= 0 and x1 < size and y1 < size:
            image[y0:y1, x0:x1] += psf


def compute_background_rms(estimated_bg, true_bg):
    """Compute RMS difference between estimated and true background."""
    return np.sqrt(np.mean((estimated_bg - true_bg)**2))


def compute_background_bias(estimated_bg, true_bg):
    """Compute mean bias between estimated and true background."""
    return np.mean(estimated_bg - true_bg)


def compute_fringe_recovery_fraction(estimated_bg, true_bg):
    """
    Compute fraction of fringe amplitude recovered.

    Returns ratio of (estimated amplitude) / (true amplitude).
    1.0 = perfect recovery, 0.0 = complete smoothing, >1.0 = over-estimation
    """
    true_amp = np.ptp(true_bg)  # peak-to-peak
    est_amp = np.ptp(estimated_bg)

    return est_amp / true_amp if true_amp > 0 else 0.0


# ==============================================================================
# Tests: Linear fringes with different periods
# ==============================================================================

@pytest.mark.slow
@pytest.mark.parametrize("period", [10, 20, 50, 100, 200])
@pytest.mark.parametrize("amplitude", [50, 100])
def test_linear_fringes_period_scan(test_config, rng, period, amplitude):
    """
    Test background estimation with linear fringes of different periods.

    Key question: At what period do methods start to recover vs smooth fringes?
    """
    size = test_config['size']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    fwhm = test_config['fwhm']

    # Create fringe background
    true_bg = create_linear_fringes(size, amplitude, period, angle_deg=30)

    # Add noise and stars
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Test SEP method (grid-based, size=128)
    bg_sep = photometry.get_background(image, method='sep', size=128)
    rms_sep = compute_background_rms(bg_sep, true_bg)
    recovery_sep = compute_fringe_recovery_fraction(bg_sep, true_bg)

    # Test morphology method (non-grid)
    bg_morph = photometry.get_background(image, method='morphology', size=25)
    rms_morph = compute_background_rms(bg_morph, true_bg)
    recovery_morph = compute_fringe_recovery_fraction(bg_morph, true_bg)

    # Expected behavior:
    # - Very short periods (< mesh size): should be smoothed out (low recovery)
    # - Longer periods (> mesh size): may be partially recovered
    # - Grid methods limited by mesh size (128 pixels)
    # - Morphology limited by structure size (25 pixels)

    print(f"\nLinear fringes: period={period}px, amplitude={amplitude} ADU")
    print(f"  SEP (grid=128):    RMS={rms_sep:6.1f} ADU, recovery={recovery_sep:.2f}")
    print(f"  Morphology (sz=25): RMS={rms_morph:6.1f} ADU, recovery={recovery_morph:.2f}")

    # Sanity checks (not strict, as fringes are inherently difficult)
    assert rms_sep < 2 * amplitude, "SEP RMS should be within ~2x of fringe amplitude"
    assert rms_morph < 2 * amplitude, "Morphology RMS should be within ~2x of fringe amplitude"


@pytest.mark.slow
@pytest.mark.parametrize("method,size", [
    ('sep', 64),
    ('sep', 128),
    ('sep', 256),
    ('morphology', 15),
    ('morphology', 25),
    ('morphology', 35),
])
def test_linear_fringes_method_comparison(test_config, rng, method, size):
    """
    Compare different methods and sizes on standard fringe pattern.

    Fixed fringe: period=80px, amplitude=100 ADU
    """
    img_size = test_config['size']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    fwhm = test_config['fwhm']

    period = 80
    amplitude = 100

    # Create fringe background
    true_bg = create_linear_fringes(img_size, amplitude, period, angle_deg=45)

    # Add noise and stars
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (img_size, img_size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate background
    bg = photometry.get_background(image, method=method, size=size)

    # Compute metrics
    rms = compute_background_rms(bg, true_bg)
    bias = compute_background_bias(bg, true_bg)
    recovery = compute_fringe_recovery_fraction(bg, true_bg)

    print(f"\n{method} (size={size}): RMS={rms:.1f} ADU, bias={bias:+.1f} ADU, recovery={recovery:.2f}")

    # Fringes are challenging - just verify reasonable behavior
    assert rms < 200, f"RMS too high: {rms:.1f} ADU"


# ==============================================================================
# Tests: 2D fringe patterns
# ==============================================================================

@pytest.mark.slow
@pytest.mark.parametrize("period", [50, 100, 150])
def test_2d_fringes(test_config, rng, period):
    """
    Test with 2D interference patterns (orthogonal fringes).

    These are more challenging than 1D fringes as they vary in both directions.
    """
    size = test_config['size']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    fwhm = test_config['fwhm']

    amplitude = 50  # per component (total can be ~100)

    # Create 2D fringe pattern
    true_bg = create_2d_fringes(size, amplitude, period, period * 1.3)

    # Add noise and stars
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Test multiple methods
    methods = [
        ('sep', 128),
        ('morphology', 25),
        ('percentile', 31),
    ]

    print(f"\n2D fringes: period={period}px (and {period*1.3:.0f}px), amplitude={amplitude} ADU/component")

    for method, method_size in methods:
        bg = photometry.get_background(image, method=method, size=method_size)
        rms = compute_background_rms(bg, true_bg)
        recovery = compute_fringe_recovery_fraction(bg, true_bg)

        print(f"  {method:12s} (size={method_size:3d}): RMS={rms:6.1f} ADU, recovery={recovery:.2f}")

        # 2D fringes are very challenging - relaxed checks
        assert rms < 300, f"{method}: RMS too high ({rms:.1f} ADU)"


# ==============================================================================
# Tests: Radial fringes (Newton rings)
# ==============================================================================

@pytest.mark.slow
@pytest.mark.parametrize("period", [40, 80, 120])
def test_radial_fringes(test_config, rng, period):
    """
    Test with radial fringe patterns (concentric circles).

    These represent Newton rings or other radial interference patterns.
    Very challenging as they vary radially from center.
    """
    size = test_config['size']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    fwhm = test_config['fwhm']

    amplitude = 80

    # Create radial fringe pattern
    true_bg = create_radial_fringes(size, amplitude, period)

    # Add noise and stars
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Test multiple methods
    methods = [
        ('sep', 64),
        ('sep', 128),
        ('morphology', 25),
    ]

    print(f"\nRadial fringes: period={period}px, amplitude={amplitude} ADU")

    for method, method_size in methods:
        bg = photometry.get_background(image, method=method, size=method_size)
        rms = compute_background_rms(bg, true_bg)
        recovery = compute_fringe_recovery_fraction(bg, true_bg)

        print(f"  {method:12s} (size={method_size:3d}): RMS={rms:6.1f} ADU, recovery={recovery:.2f}")

        # Radial fringes very challenging - verify it doesn't crash and produces something
        assert rms < 500, f"{method}: RMS extremely high ({rms:.1f} ADU)"
        assert not np.any(np.isnan(bg)), f"{method}: Background contains NaN values"


# ==============================================================================
# Test: Fringe + gradient combination
# ==============================================================================

@pytest.mark.slow
def test_fringes_plus_gradient(test_config, rng):
    """
    Test with realistic case: fringes superimposed on gradient.

    This is common in real data where you have both large-scale gradients
    (vignetting, scattered light) and small-scale fringes.
    """
    size = test_config['size']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    fwhm = test_config['fwhm']

    # Create gradient + fringes
    from test_background_estimation import create_linear_gradient

    gradient = create_linear_gradient(size, amplitude=300, angle_deg=45)
    fringes = create_linear_fringes(size, amplitude=50, period=80, angle_deg=135)
    true_bg = gradient + fringes

    # Add noise and stars
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Test methods
    methods = [
        ('sep', 128),
        ('sep', 64),
        ('morphology', 25),
    ]

    print("\nGradient (300 ADU) + Fringes (50 ADU, period=80px)")

    for method, method_size in methods:
        bg = photometry.get_background(image, method=method, size=method_size)
        rms = compute_background_rms(bg, true_bg)

        print(f"  {method:12s} (size={method_size:3d}): RMS={rms:6.1f} ADU")

        # Combined case is very challenging
        assert rms < 400, f"{method}: RMS too high ({rms:.1f} ADU)"


# ==============================================================================
# Summary test: Generate comparison report
# ==============================================================================

@pytest.mark.slow
def test_fringe_summary(test_config, rng):
    """
    Generate comprehensive comparison report for different fringe scenarios.

    This is not a pass/fail test, but a data-gathering exercise.
    """
    size = test_config['size']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    fwhm = test_config['fwhm']

    print("\n" + "="*80)
    print("FRINGE BACKGROUND ESTIMATION SUMMARY")
    print("="*80)

    scenarios = [
        ("Linear fringe (period=50px)",
         lambda: create_linear_fringes(size, 100, 50, 30)),
        ("Linear fringe (period=100px)",
         lambda: create_linear_fringes(size, 100, 100, 30)),
        ("Linear fringe (period=200px)",
         lambda: create_linear_fringes(size, 100, 200, 30)),
        ("2D fringes (period=80px)",
         lambda: create_2d_fringes(size, 50, 80, 100)),
        ("Radial fringes (period=60px)",
         lambda: create_radial_fringes(size, 80, 60)),
    ]

    methods = [
        ('SEP (grid=64)', 'sep', 64),
        ('SEP (grid=128)', 'sep', 128),
        ('Morphology (25)', 'morphology', 25),
    ]

    for scenario_name, bg_func in scenarios:
        print(f"\n{scenario_name}")
        print("-" * 80)
        print(f"{'Method':<20} {'RMS (ADU)':<12} {'Bias (ADU)':<13} {'Recovery':<10}")
        print("-" * 80)

        # Create background
        true_bg = bg_func()

        # Add noise and stars
        image = true_bg.copy()
        image += rng.normal(0, noise_std, (size, size))
        add_random_stars(image, nstars, flux_range, fwhm, rng)

        for method_label, method_name, method_size in methods:
            bg = photometry.get_background(image, method=method_name, size=method_size)
            rms = compute_background_rms(bg, true_bg)
            bias = compute_background_bias(bg, true_bg)
            recovery = compute_fringe_recovery_fraction(bg, true_bg)

            print(f"{method_label:<20} {rms:>10.1f}   {bias:>+11.1f}   {recovery:>8.2f}")

    print("\n" + "="*80)
    print("KEY OBSERVATIONS:")
    print("="*80)
    print("""
1. PERIOD DEPENDENCE:
   - Very short periods (< grid/structure size): Smoothed out (low recovery)
   - Intermediate periods (~ grid/structure size): Partially recovered
   - Long periods (>> grid/structure size): Better recovered but still challenging

2. METHOD COMPARISON:
   - SEP (grid-based): Limited by mesh size (64-128 pixels)
     * Cannot recover fringes with period < mesh size
     * Larger mesh = more smoothing
   - Morphology (non-grid): Limited by structure size (25 pixels)
     * Can handle shorter periods than SEP with large mesh
     * But still smooths structures < 25 pixels

3. FRINGE TYPES:
   - 1D linear fringes: Easiest to handle
   - 2D interference patterns: More challenging
   - Radial patterns: Most challenging (vary in all directions)

4. PRACTICAL IMPLICATIONS:
   - Background estimation methods are NOT designed to preserve fringes
   - They intentionally smooth small-scale structure (including fringes)
   - For fringe removal: Use dedicated fringe subtraction techniques
   - Background estimation should focus on large-scale structure only

5. RECOMMENDATIONS:
   - If fringes present: Use dedicated fringe correction first
   - Then apply background estimation for large-scale structure
   - Do not expect background estimators to handle both
""")


if __name__ == '__main__':
    # Run with: pytest test_background_fringes.py -v -s
    pytest.main([__file__, '-v', '-s'])
