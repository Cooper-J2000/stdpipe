"""
Fringe removal for astronomical images.

The main routine, :func:`remove_fringes`, models the fringe pattern as a
smooth two-dimensional field at intermediate spatial scales (between stellar
FWHM and the coarse sky background) and estimates it directly in image space:
sources are masked, and the masked pixels are transparently "inpainted" by
NaN-aware Gaussian smoothing, so the fringe structure is continued under the
stars. This works for realistic curved, non-stationary fringes where
frequency-domain methods fail.

The module also provides Fourier notch filtering
(:func:`remove_fringes_fourier`) for genuinely *periodic* patterns such as
electronic pickup noise, and :func:`visualize_fourier_spectrum` for
diagnostics.
"""

import numpy as np
from astropy.stats import mad_std


__all__ = [
    'remove_fringes',
    'remove_fringes_fourier',
    'detect_fringe_peaks_fft',
    'create_notch_filter',
    'create_bandpass_filter',
    'visualize_fourier_spectrum',
]


def _find_bright_sources(sn, core, halo_sn=20.0, min_rmax=0.0, max_sources=100):
    """Find halo-bearing sources: components that both have a bright peak
    (> `halo_sn` sigmas) and an extended thresholded footprint (internal
    radius > `min_rmax`). A footprint much wider than the PSF means the wings
    are above the threshold, implying yet fainter wings extend proportionally
    further below it.

    The cores are labeled at the `halo_sn` level: fringe crests never reach
    it, so bright stars are not merged with the crest networks they touch.

    Parameters
    ----------
    sn : ndarray
        Per-pixel significance map (residual / rms)
    core : ndarray of bool
        Low-threshold outlier mask used to measure the footprint extent
    halo_sn : float
        Peak significance above which a component may be halo-bearing
    min_rmax : float
        Minimum footprint internal radius (pixels)
    max_sources : int
        Keep at most this many sources (brightest first)

    Returns
    -------
    stars : list of (cy, cx, rmax) tuples, sorted by decreasing peak
    """
    from scipy import ndimage

    lab, n = ndimage.label(sn > halo_sn)
    if n == 0:
        return []

    idx = np.arange(1, n + 1)
    # Local footprint radius of each bright core: max internal distance to
    # the edge of the low-threshold mask over the core pixels. Measures how
    # far the wings of this particular source stay above the threshold,
    # regardless of what other structures the footprint is connected to
    edt_in = ndimage.distance_transform_edt(core)
    rmax = ndimage.maximum(edt_in, lab, idx)
    keep = np.where(rmax > min_rmax)[0]
    if len(keep) == 0:
        return []

    peaks = ndimage.maximum(sn, lab, idx)
    keep = keep[np.argsort(peaks[keep])[::-1]][:max_sources]
    centers = ndimage.maximum_position(sn, lab, [ci + 1 for ci in keep])

    return [(cy, cx, rmax[ci]) for (cy, cx), ci in zip(centers, keep)]


def _radial_geometry(stars, shape, max_radius=None, dr=2.0):
    """Precompute the per-star ring geometry (bounding box, radii, ring-sorted
    pixel order and ring boundaries) used by :func:`_radial_profile_field`.
    The geometry only depends on the star positions, so it can be reused for
    profiling different arrays."""
    H, W = shape
    if max_radius is None:
        max_radius = min(H, W) / 4

    geos = []
    for cy, cx, rm in stars:
        R = min(max_radius, 16 * rm)
        y0, y1 = max(0, int(cy - R)), min(H, int(cy + R + 1))
        x0, x1 = max(0, int(cx - R)), min(W, int(cx + R + 1))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(yy - cy, xx - cx).astype(np.float32)

        rbin = (rr / dr).astype(np.int32).ravel()
        nb = int(R / dr) + 1
        inside = np.flatnonzero(rbin < nb)
        order = inside[np.argsort(rbin[inside], kind='stable')]
        edges = np.searchsorted(rbin[order], np.arange(nb + 1))

        geos.append({
            'sly': slice(y0, y1),
            'slx': slice(x0, x1),
            'rr': rr,
            'rm': rm,
            'order': order,
            'edges': edges,
            'rgrid': (np.arange(nb) + 0.5) * dr,
        })

    return geos


def _pava_decreasing(y):
    """Isotonic regression for a non-increasing sequence (pool adjacent
    violators)."""
    vals, wts = [], []
    for v in y:
        vals.append(float(v))
        wts.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            vals[-2] = (vals[-1] * wts[-1] + vals[-2] * wts[-2]) / (wts[-1] + wts[-2])
            wts[-2] += wts[-1]
            vals.pop()
            wts.pop()
    return np.repeat(vals, wts)


def _radial_profile_field(data, geos, mask=None):
    """Sum of the non-monotone parts of azimuthal-median radial profiles
    around the stars whose geometry was precomputed by
    :func:`_radial_geometry`.

    For each star, ring medians of `data` (minus the already accumulated
    contributions of brighter stars) capture the azimuthally symmetric
    circumstellar structure while averaging the non-symmetric fringes
    passing through to nearly zero. The monotone-decreasing positive
    envelope of the profile - the stellar wings and the smooth halo - is
    then removed from it (isotonic fit), leaving only the detached
    structures: ghost reflection rings, background estimation bowls. Those
    are exactly what isotropic smoothing cannot model with circular
    fidelity; the smooth envelope part is left for the smoothing to absorb,
    so the field carries no imprint of the star itself.

    If `mask` is given (source mask: pixels above the detection threshold),
    masked pixels are excluded from the ring medians, and rings dominated by
    them are bridged by interpolation from the unmasked zones.
    """
    from scipy.ndimage import gaussian_filter1d

    out = np.zeros_like(data)

    for g in geos:
        vals = (data[g['sly'], g['slx']] - out[g['sly'], g['slx']]).ravel()[g['order']]
        if mask is not None:
            mvals = mask[g['sly'], g['slx']].ravel()[g['order']]
        edges = g['edges']
        nb = len(g['rgrid'])
        prof = np.full(nb, np.nan)
        for b in range(nb):
            v = vals[edges[b]:edges[b + 1]]
            if mask is not None:
                v = v[~mvals[edges[b]:edges[b + 1]]]
            # Require a mostly unmasked ring for a trustworthy median
            if len(v) > 0.5 * (edges[b + 1] - edges[b]):
                prof[b] = np.median(v)

        valid = np.isfinite(prof)
        if not np.any(valid):
            continue
        # Bridge masked rings by linear interpolation / constant extension
        prof = np.interp(np.arange(nb), np.flatnonzero(valid), prof[valid])

        # Outer baseline: the profile must approach zero far from the star
        prof = prof - np.median(prof[int(0.75 * nb):])

        # Mild radial smoothing against per-ring median noise that would
        # otherwise imprint concentric ring patterns on the image
        prof = gaussian_filter1d(prof, 1.0, mode='nearest')

        # Remove the monotone-decreasing positive envelope (stellar wings
        # and smooth halo), keeping only the detached non-monotone
        # structures. The envelope is smooth and sub-threshold, so the
        # multi-scale smoothing absorbs it instead, with no edge anywhere
        prof = prof - np.clip(_pava_decreasing(prof), 0.0, None)

        # Smooth taper to zero over the outer quarter to avoid a circular
        # edge at the field boundary
        prof = prof * np.clip((nb - 1 - np.arange(nb)) / (0.25 * nb), 0.0, 1.0)

        out[g['sly'], g['slx']] += np.interp(g['rr'], g['rgrid'], prof, right=0.0)

    return out


def _binned_gaussian(data, sigma, binning):
    """Gaussian smoothing computed on a block-averaged copy of the array and
    interpolated back - much faster for wide kernels, accurate when `sigma`
    is well above `binning`."""
    from scipy.ndimage import gaussian_filter, zoom

    H, W = data.shape
    Hp = (H + binning - 1) // binning * binning
    Wp = (W + binning - 1) // binning * binning
    padded = np.zeros((Hp, Wp), dtype=np.float64)
    padded[:H, :W] = data
    binned = padded.reshape(Hp // binning, binning, Wp // binning, binning).mean(axis=(1, 3))
    smoothed = gaussian_filter(binned, sigma / binning, mode='constant', cval=0.0)
    return zoom(smoothed, binning, order=1)[:H, :W]


def _smooth_inpaint(flat, mask, scale, levels=(1, 2, 4, 8)):
    """Mask-aware Gaussian smoothing with seamless multi-scale inpainting.

    Computes the normalized masked convolution (smoothed data divided by
    smoothed mask support) on a ladder of scales `levels` * `scale`. Each
    pixel is filled by the finest scale that still has meaningful unmasked
    support there - so masked star footprints are smoothly continued from
    the surrounding fringe structure rather than replaced by a flat
    large-scale average. The blending weight is a smooth function of the
    local support, so no seams appear at the transitions. Regions beyond
    the reach of the coarsest kernel smoothly approach zero (no correction).
    """
    from scipy.ndimage import gaussian_filter

    data0 = np.where(mask, 0.0, flat)
    support = (~mask).astype(np.float64)

    out = np.zeros_like(data0)
    wsum = np.zeros_like(data0)

    for level in levels:
        sigma = level * scale
        if sigma < 16:
            num = gaussian_filter(data0, sigma, mode='constant', cval=0.0)
            den = gaussian_filter(support, sigma, mode='constant', cval=0.0)
        else:
            # Wide kernels: compute on binned arrays, they have no fine
            # structure by construction
            binning = max(1, int(sigma / 8))
            num = _binned_gaussian(data0, sigma, binning)
            den = _binned_gaussian(support, sigma, binning)

        est = num / np.maximum(den, 1e-10)
        # Trust the estimate where >~15% of the kernel sees unmasked pixels;
        # the Gaussian tails still provide a meaningful weighted average of
        # the surrounding structure well inside masked holes
        w = np.clip(den / 0.15, 0.0, 1.0) ** 2
        out = out + est * w * (1.0 - wsum)
        wsum = wsum + w * (1.0 - wsum)

    return out


def remove_fringes(
    image,
    mask=None,
    scale=6.0,
    bg_size=256,
    threshold=2.0,
    dilate=2,
    iterations=3,
    halo_sn=20.0,
    get_fringe_model=False,
    verbose=False,
):
    """
    Remove fringes by estimating a smooth intermediate-scale background map
    with sources masked and inpainted.

    The algorithm exploits the scale separation between stars (~FWHM, a few
    pixels), fringes (tens of pixels) and the sky background (hundreds of
    pixels):

    1. Subtract a coarse sky background (SEP mesh of `bg_size` pixels)
    2. Around bright stars (peaks above `halo_sn` sigmas with footprints
       wider than the smoothing kernel), model the non-monotone azimuthally
       symmetric circumstellar structure - ghost reflection rings,
       background estimation bowls - as azimuthal-median radial profiles
       with their smooth monotone envelope (the stellar wings and halo)
       removed, and subtract it from the working residual. This captures
       circular artifacts with full circular fidelity, which isotropic
       smoothing cannot do
    3. Mask pixels deviating by more than `threshold` sigma (sources and
       outliers), grow the mask by `dilate` pixels
    4. Smooth the masked residual with a Gaussian of sigma `scale` pixels
       using mask-aware normalized convolution - masked pixels are filled by
       the weighted average of the surrounding fringe structure (inpainting),
       each region by the finest scale of a multi-scale ladder that still
       has unmasked support there. The smoothing also absorbs the smooth
       sub-threshold envelope of stellar halos and wings
    5. Iterate: re-mask on the fringe-subtracted residual (so that fringe
       crests are not mistaken for sources), smooth it again and add to the
       fringe map. The additive refinement also recovers the part of the
       fringe structure attenuated by the smoothing, so narrower fringes are
       progressively captured with each iteration

    The radial circumstellar fields are included in the returned fringe map
    and thus subtracted from the image along with the fringes.

    The resulting fringe map is defined everywhere, including under the
    stars, and is subtracted from the original image. The coarse sky
    background is left in the image.

    Parameters
    ----------
    image : ndarray
        Input image
    mask : ndarray, optional
        Boolean mask (True = masked pixels to exclude from fringe estimation)
    scale : float
        Sigma of the Gaussian smoothing kernel in pixels. Should be of order
        the stellar FWHM or somewhat larger, and well below the narrowest
        fringe period. Smaller values track narrower fringes at the cost of
        slightly more noise and source flux absorbed into the map.
        Default: 6
    bg_size : int or None
        Mesh size in pixels for the coarse sky background subtracted before
        fringe estimation. Should be much larger than the fringe scale.
        If None, the global median is used instead. Default: 256
    threshold : float
        Source masking threshold in sigmas of the local background RMS.
        Applied symmetrically (both positive and negative outliers are
        masked) to avoid biasing the fringe map. Default: 2.0
    dilate : int
        Number of binary dilation iterations applied to the source mask to
        cover the wings of the sources. Default: 2
    iterations : int
        Number of refinement iterations. Each iteration re-derives the
        source mask from the fringe-subtracted residual (the first one has
        to mask on the raw residual where fringe crests may exceed the
        threshold) and accumulates the smoothed residual into the fringe
        map. More iterations capture narrower fringes but absorb more noise.
        Default: 3
    halo_sn : float or None
        Peak significance (in sigmas) above which a source is considered
        bright enough to have halos and gets the radial halo treatment.
        Should be well above the peak significance of the fringe crests.
        Only sources whose thresholded footprint radius also exceeds `scale`
        are treated (ordinary stars have compact footprints and are not
        affected). None disables the halo modeling. Default: 20
    get_fringe_model : bool
        If True, return (corrected_image, fringe_model). Default: False
    verbose : bool or callable
        Whether to show verbose messages during the run. May be either
        boolean, or a `print`-like function.

    Returns
    -------
    corrected_image : ndarray
        Image with the fringe map subtracted
    fringe_model : ndarray, optional
        The subtracted fringe map (if get_fringe_model=True)

    Notes
    -----
    Circumstellar structures - halos, ghost rings, background bowls - are
    locally indistinguishable from fringes (smooth structures at similar
    scales). The smooth ones are absorbed by the masked multi-scale
    smoothing along with the fringes; sharply circular ones (ghost donuts,
    bowls) are additionally captured with full circular fidelity by
    azimuthal-median radial profiles around bright stars (the `halo_sn`
    mechanism), which average the oscillating fringes passing through to
    nearly zero. Everything smooth or circular around bright stars is thus
    subtracted, including the sub-threshold part of their PSF wings; the
    star cores and everything above the masking threshold stay in the
    image. Use halo_sn=None to disable the radial modeling (isotropic
    smoothing only).

    Faint extended sources (low surface brightness galaxies, nebulae) whose
    peaks stay below `halo_sn` remain partially degenerate with the fringe
    pattern: the part below the masking threshold gets absorbed into the
    fringe map and over-subtracted. For fields dominated by such sources,
    lower `halo_sn`, increase `scale`, or mask the sources explicitly via
    the `mask` argument.

    Examples
    --------
    >>> corrected = remove_fringes(image, mask=mask, scale=6)
    >>> corrected, fringes = remove_fringes(image, get_fringe_model=True)
    """
    import sep

    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    image = np.asarray(image, dtype=np.float64)

    # Base mask: user mask plus non-finite pixels
    mask_bad = ~np.isfinite(image)
    if mask is not None:
        mask_bad |= mask.astype(bool)

    img = image.copy()
    if np.any(mask_bad):
        img[mask_bad] = np.median(image[~mask_bad])

    # Coarse sky background, to be preserved in the output
    if bg_size is not None:
        bg = sep.Background(
            np.ascontiguousarray(img, dtype=np.float32),
            mask=np.ascontiguousarray(mask_bad),
            bw=bg_size,
            bh=bg_size,
        )
        resid = img - bg.back().astype(np.float64)
        rms = bg.rms().astype(np.float64)
        log(
            'Coarse background: %d px mesh, global level %.2f rms %.2f'
            % (bg_size, bg.globalback, bg.globalrms)
        )
    else:
        resid = img - np.median(img[~mask_bad])
        rms = np.full_like(img, mad_std(resid[~mask_bad]))
        log('Coarse background: global median, rms %.2f' % rms.flat[0])

    from scipy.ndimage import binary_dilation

    fringe_model = np.zeros_like(img)
    geos = None

    for i in range(max(1, int(iterations))):
        flat = resid - fringe_model
        sn = np.abs(flat) / rms

        # Radial models of the circumstellar structure around bright stars
        # (halos, ghost rings, background bowls), which is locally degenerate
        # with the fringes. Ring medians capture it while averaging the
        # fringes to nearly zero; the fringes under it remain measurable.
        # The star list is detected once: the halo_sn gate is far above the
        # fringe crests, so it is stable across iterations
        if halo_sn is not None and geos is None:
            stars = _find_bright_sources(
                sn, sn > threshold, halo_sn=halo_sn, min_rmax=scale
            )
            geos = _radial_geometry(stars, img.shape) if stars else []
            log('Bright stars with circumstellar structure modeled: %d' % len(stars))

        rings = None
        if geos:
            # The radial fields carry only the non-monotone circumstellar
            # structure (ghost rings, background bowls), which the isotropic
            # smoothing cannot model with circular fidelity. The smooth
            # wings and halos are absorbed by the smoothing itself
            rings = _radial_profile_field(
                flat, geos, mask=(sn > threshold) | mask_bad
            )
            flat = flat - rings

        # Symmetric outlier masking on the ring- and fringe-subtracted
        # residual
        mask_src = (np.abs(flat) > threshold * rms) | mask_bad
        if dilate > 0:
            mask_src = binary_dilation(mask_src, iterations=dilate)
        mask_src |= mask_bad

        log(
            'Iteration %d: %.1f%% pixels masked' % (i + 1, 100 * np.mean(mask_src))
        )

        # Additive refinement: smooth the current residual and accumulate.
        # A single Gaussian pass attenuates structure of period P by
        # exp(-2 pi^2 scale^2 / P^2); repeating on the residual recovers the
        # attenuated part (effective response 1 - (1 - G)^n).
        # The ring fields are part of the model and get subtracted as well
        fringe_model = fringe_model + _smooth_inpaint(flat, mask_src, scale)
        if rings is not None:
            fringe_model = fringe_model + rings

        log(
            'Iteration %d: fringe map amplitude (mad_std) %.2f'
            % (i + 1, mad_std(fringe_model))
        )

    corrected = np.ascontiguousarray(image - fringe_model)

    if get_fringe_model:
        return corrected, np.ascontiguousarray(fringe_model)
    else:
        return corrected


def detect_fringe_peaks_fft(
    image,
    mask=None,
    sigma_threshold=5.0,
    period_range=(30, 200),
    fwhm=None,
    protect_low_freq=True,
    protect_high_freq=True,
    verbose=False,
):
    """
    Detect peaks in the Fourier power spectrum corresponding to periodic
    patterns (e.g. electronic pickup noise).

    Note that realistic sky fringes are usually *not* periodic - their power
    is spread over a broad annulus in frequency space with no distinct peaks.
    Use :func:`remove_fringes` for those.

    Parameters
    ----------
    image : ndarray
        Input image
    mask : ndarray, optional
        Boolean mask (True = masked pixels)
    sigma_threshold : float
        Detection threshold in sigma above median power
    period_range : tuple
        (min_period, max_period) in pixels
    fwhm : float, optional
        Stellar FWHM for protecting high frequencies
    protect_low_freq : bool
        Exclude very low frequencies (large-scale background)
    protect_high_freq : bool
        Exclude high frequencies (stellar content); requires `fwhm`
    verbose : bool
        Print diagnostic information

    Returns
    -------
    peaks : list of tuples
        List of (fy, fx, period) peak positions in Fourier space (pixel
        offsets relative to the spectrum center) and the corresponding
        spatial period in pixels
    """
    img = image.copy()
    ny, nx = img.shape

    # Handle masked pixels by filling with global median (robust for clustered NaNs)
    if mask is not None and np.any(mask):
        median_val = np.nanmedian(img)
        img[mask] = median_val

    # Compute FFT and power spectrum
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted) ** 2

    # Log scale for better detection
    log_power = np.log10(power + 1)

    # Create frequency grids (cycles per pixel, per-axis normalization)
    fy = np.fft.fftshift(np.fft.fftfreq(ny))
    fx = np.fft.fftshift(np.fft.fftfreq(nx))
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    freq_radius = np.sqrt(FY**2 + FX**2)

    # Convert period_range to frequency range
    freq_min = 1.0 / period_range[1]  # Large period -> low frequency
    freq_max = 1.0 / period_range[0]  # Small period -> high frequency

    # Create search mask for peak detection
    search_mask = np.ones_like(power, dtype=bool)

    # Exclude DC component (central 5x5 pixels)
    search_mask[ny // 2 - 2 : ny // 2 + 3, nx // 2 - 2 : nx // 2 + 3] = False

    # Exclude frequencies outside period_range
    search_mask[freq_radius < freq_min] = False
    search_mask[freq_radius > freq_max] = False

    # Protect high frequencies (stellar content) if FWHM provided
    if protect_high_freq and fwhm is not None:
        freq_stellar = 1.0 / (2.0 * fwhm)  # Nyquist for FWHM
        search_mask[freq_radius > freq_stellar] = False

    # Protect very low frequencies (large-scale background)
    if protect_low_freq:
        freq_background = 4.0 / min(ny, nx)  # Period > image_size/4
        search_mask[freq_radius < freq_background] = False

    # Find peaks in search region
    search_power = log_power.copy()
    search_power[~search_mask] = np.nan

    # Compute threshold: median + sigma_threshold * MAD
    valid_power = search_power[np.isfinite(search_power)]
    if len(valid_power) == 0:
        if verbose:
            print("No valid frequency range to search")
        return []

    median_power = np.median(valid_power)
    mad_power = mad_std(valid_power)
    threshold = median_power + sigma_threshold * mad_power

    if verbose:
        print(f"Power spectrum statistics:")
        print(f"  Median: {median_power:.2f}")
        print(f"  MAD: {mad_power:.2f}")
        print(f"  Threshold: {threshold:.2f}")

    # Detect local maxima above threshold
    from scipy.ndimage import maximum_filter

    local_max = maximum_filter(search_power, size=5)
    peaks_mask = (
        (search_power == local_max) & (search_power > threshold) & np.isfinite(search_power)
    )

    # Extract peak positions
    peak_coords = np.argwhere(peaks_mask)

    if verbose:
        print(f"Detected {len(peak_coords)} peaks before symmetry")

    # Convert to frequency coordinates (relative to center)
    peaks = []
    for py, px in peak_coords:
        fy_peak = py - ny // 2
        fx_peak = px - nx // 2

        # Compute period for this peak (per-axis frequency normalization)
        freq = np.sqrt((fy_peak / ny) ** 2 + (fx_peak / nx) ** 2)
        if freq > 0:
            period = 1.0 / freq
        else:
            period = np.inf

        peaks.append((fy_peak, fx_peak, period))

        if verbose:
            print(f"  Peak at ({fy_peak:4d}, {fx_peak:4d}): period = {period:.1f} px")

    # Add symmetric peaks (Fourier transform of real signal is conjugate symmetric)
    symmetric_peaks = []
    for fy, fx, period in peaks:
        symmetric_peaks.append((-fy, -fx, period))

    peaks.extend(symmetric_peaks)

    # Remove duplicates
    peaks = list(set(peaks))

    if verbose:
        print(f"Total peaks (with symmetry): {len(peaks)}")

    return peaks


def create_notch_filter(shape, peaks, width=3.0, smooth=True):
    """
    Create notch filter to suppress specific Fourier peaks.

    Parameters
    ----------
    shape : tuple
        Image shape (ny, nx)
    peaks : list of tuples
        Peak positions [(fy1, fx1), (fy2, fx2), ...] or [(fy1, fx1, period1), ...]
    width : float
        Notch width in frequency pixels
    smooth : bool
        If True, use Gaussian notch; if False, use hard cutoff

    Returns
    -------
    filter : ndarray
        2D filter mask (1 = pass, 0 = suppress)
    """
    ny, nx = shape
    filter_mask = np.ones((ny, nx), dtype=float)

    # Create coordinate grids (centered at ny//2, nx//2)
    y = np.arange(ny) - ny // 2
    x = np.arange(nx) - nx // 2
    Y, X = np.meshgrid(y, x, indexing='ij')

    # Create notch for each peak
    for peak in peaks:
        # Handle both (fy, fx) and (fy, fx, period) formats
        if len(peak) == 3:
            fy, fx, period = peak
        else:
            fy, fx = peak

        if smooth:
            # Gaussian notch: gradually suppress around peak
            r2 = (Y - fy) ** 2 + (X - fx) ** 2
            notch = 1.0 - np.exp(-r2 / (2 * width**2))
        else:
            # Hard circular notch
            r = np.sqrt((Y - fy) ** 2 + (X - fx) ** 2)
            notch = np.where(r < width, 0.0, 1.0)

        filter_mask *= notch

    return filter_mask


def create_bandpass_filter(shape, period_range, smooth=True):
    """
    Create band-stop (notch) filter to suppress specific period range.

    This is actually a band-STOP filter that suppresses frequencies
    corresponding to the given period range while passing others.

    Parameters
    ----------
    shape : tuple
        Image shape (ny, nx)
    period_range : tuple
        (min_period, max_period) in pixels to SUPPRESS
    smooth : bool
        If True, use smooth Gaussian rolloff; if False, hard cutoff

    Returns
    -------
    filter : ndarray
        2D band-stop filter (1 = pass, 0 = suppress, smooth transition if smooth=True)
    """
    ny, nx = shape

    # Create frequency grids
    fy = np.fft.fftshift(np.fft.fftfreq(ny))
    fx = np.fft.fftshift(np.fft.fftfreq(nx))
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    freq_radius = np.sqrt(FY**2 + FX**2)

    # Convert periods to frequencies
    freq_center = 0.5 * (1.0 / period_range[0] + 1.0 / period_range[1])
    freq_width = abs(1.0 / period_range[0] - 1.0 / period_range[1])

    if smooth:
        # Gaussian notch centered on freq_center
        # Width chosen so 3-sigma spans the frequency range
        sigma = freq_width / 3.0
        bandstop = 1.0 - np.exp(-((freq_radius - freq_center) ** 2) / (2 * sigma**2))
    else:
        # Hard cutoff: suppress frequencies in range
        freq_min = 1.0 / period_range[1]  # Large period -> low frequency
        freq_max = 1.0 / period_range[0]  # Small period -> high frequency

        bandstop = np.where(
            (freq_radius >= freq_min) & (freq_radius <= freq_max),
            0.0,  # Suppress in band
            1.0,  # Pass outside band
        )

    return bandstop


def remove_fringes_fourier(
    image,
    mask=None,
    fwhm=None,
    method='global',
    period_range=(30, 200),
    sigma_threshold=2.0,
    notch_width=3.0,
    protect_low_freq=True,
    protect_high_freq=True,
    get_fringe_model=False,
    verbose=False,
):
    """
    Remove periodic patterns from image using Fourier notch filtering.

    Suitable for genuinely periodic, stationary patterns (electronic pickup
    noise, readout banding) that produce sharp peaks in Fourier space.
    Realistic sky fringes are usually curved and non-stationary, with power
    spread over a broad frequency annulus - use :func:`remove_fringes` for
    those instead.

    Parameters
    ----------
    image : ndarray
        Input image
    mask : ndarray, optional
        Boolean mask (True = masked pixels to exclude from analysis)
    fwhm : float, optional
        Stellar FWHM (pixels) for protecting high-frequency stellar content.
        If None, no high-frequency protection is applied.
    method : str
        Pattern removal method:
        - 'global': Detect and notch out power spectrum peaks (default)
        - 'bandpass': Suppress a whole frequency range given by period_range
    period_range : tuple
        Expected pattern period range (min, max) in pixels.
        Default: (30, 200)
    sigma_threshold : float
        Threshold for automatic peak detection (in sigma above median power).
        Higher = more conservative. Default: 2.0
    notch_width : float
        Width of notch filter in frequency pixels.
        Larger = more aggressive. Default: 3.0
    protect_low_freq : bool
        If True, don't filter very low frequencies (large-scale background).
        Default: True
    protect_high_freq : bool
        If True, protect high frequencies based on stellar FWHM.
        Requires `fwhm`. Default: True
    get_fringe_model : bool
        If True, return (corrected_image, fringe_model).
        Default: False
    verbose : bool or callable
        Print diagnostic information

    Returns
    -------
    corrected_image : ndarray
        Image with the periodic pattern removed
    fringe_model : ndarray, optional
        Removed pattern (if get_fringe_model=True)

    Examples
    --------
    # Automatic peak detection and notching
    corrected = remove_fringes_fourier(image, fwhm=3.0)

    # Bandpass filtering for known period range
    corrected = remove_fringes_fourier(
        image,
        method='bandpass',
        period_range=(50, 150),
        fwhm=3.0
    )
    """
    # Setup logging
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    log("Fourier notch pattern removal, method: %s" % method)

    # Prepare image
    img = image.copy()
    ny, nx = img.shape

    # Handle NaNs and masked pixels
    if mask is not None and np.any(mask):
        from scipy.ndimage import median_filter

        size = int(max(3, 2 * fwhm)) if fwhm is not None else 5
        img_filled = median_filter(img, size=size)
        img[mask] = img_filled[mask]

    # Compute FFT
    log("Computing FFT...")
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)

    # Create filter based on method
    if method == 'global':
        log(f"Detecting pattern peaks (threshold={sigma_threshold:.1f} sigma)...")

        # Detect peaks in power spectrum
        peaks = detect_fringe_peaks_fft(
            image,
            mask=mask,
            sigma_threshold=sigma_threshold,
            period_range=period_range,
            fwhm=fwhm,
            protect_low_freq=protect_low_freq,
            protect_high_freq=protect_high_freq,
            verbose=verbose,
        )

        if len(peaks) == 0:
            log("No pattern peaks detected.")
            if get_fringe_model:
                return image, np.zeros_like(image)
            else:
                return image

        log(f"Creating notch filter for {len(peaks)} peaks...")
        notch_filter = create_notch_filter(img.shape, peaks, width=notch_width, smooth=True)

    elif method == 'bandpass':
        log(f"Creating bandpass filter for period range {period_range} pixels...")
        notch_filter = create_bandpass_filter(img.shape, period_range, smooth=True)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply filter
    log("Applying filter...")
    fft_filtered = fft_shifted * notch_filter

    # Inverse FFT
    log("Computing inverse FFT...")
    fft_unshifted = np.fft.ifftshift(fft_filtered)
    img_corrected = np.real(np.fft.ifft2(fft_unshifted))

    # Restore masked pixels from original
    if mask is not None:
        img_corrected[mask] = image[mask]

    log("Done.")

    # Ensure C-contiguous array for compatibility with SEP and other tools
    img_corrected = np.ascontiguousarray(img_corrected)

    if get_fringe_model:
        fringe_model = image - img_corrected
        return img_corrected, fringe_model
    else:
        return img_corrected


def visualize_fourier_spectrum(
    image,
    peaks=None,
    period_range=None,
    fwhm=None,
    log_scale=True,
    vmax_percentile=99.9,
    save_path=None,
):
    """
    Visualize 2D Fourier power spectrum with detected peaks.

    Useful for debugging and parameter tuning.

    Parameters
    ----------
    image : ndarray
        Input image
    peaks : list of tuples, optional
        Detected peak positions [(fy, fx), ...]
    period_range : tuple, optional
        If provided, overlay search region
    fwhm : float, optional
        If provided, overlay high-frequency protection boundary
    log_scale : bool
        Use log scale for power spectrum
    vmax_percentile : float
        Percentile for color scale maximum
    save_path : str, optional
        If provided, save plot to this path

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    import matplotlib.pyplot as plt

    # Compute FFT and power
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted) ** 2

    if log_scale:
        power_plot = np.log10(power + 1)
        label = 'Log₁₀(Power + 1)'
    else:
        power_plot = power
        label = 'Power'

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot power spectrum
    im = ax.imshow(
        power_plot,
        origin='lower',
        cmap='viridis',
        vmin=0,
        vmax=np.percentile(power_plot, vmax_percentile),
    )
    plt.colorbar(im, ax=ax, label=label)

    ny, nx = image.shape
    center_y, center_x = ny // 2, nx // 2

    # Overlay period_range as circle
    if period_range is not None:
        # Convert periods to frequency radius (in pixels)
        freq_min = ny / period_range[1]  # Large period → small radius
        freq_max = ny / period_range[0]  # Small period → large radius

        circle_inner = plt.Circle(
            (center_x, center_y),
            freq_min,
            fill=False,
            edgecolor='red',
            linestyle='--',
            linewidth=2,
            label=f'Period range: {period_range[0]}-{period_range[1]} px',
        )
        circle_outer = plt.Circle(
            (center_x, center_y), freq_max, fill=False, edgecolor='red', linestyle='--', linewidth=2
        )
        ax.add_patch(circle_inner)
        ax.add_patch(circle_outer)

    # Overlay FWHM protection boundary
    if fwhm is not None:
        freq_stellar = ny / (2.0 * fwhm)
        circle_fwhm = plt.Circle(
            (center_x, center_y),
            freq_stellar,
            fill=False,
            edgecolor='yellow',
            linestyle=':',
            linewidth=2,
            label=f'FWHM protection (FWHM={fwhm:.1f} px)',
        )
        ax.add_patch(circle_fwhm)

    # Mark detected peaks
    if peaks is not None and len(peaks) > 0:
        for peak in peaks:
            if len(peak) == 3:
                fy, fx, period = peak
                label_text = f'{period:.0f}px'
            else:
                fy, fx = peak
                label_text = ''

            ax.plot(center_x + fx, center_y + fy, 'r+', markersize=15, markeredgewidth=2)
            if label_text:
                ax.text(center_x + fx + 5, center_y + fy + 5, label_text, color='red', fontsize=10)

    ax.set_title('2D Fourier Power Spectrum')
    ax.set_xlabel('Frequency X')
    ax.set_ylabel('Frequency Y')

    if period_range or fwhm:
        ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig
