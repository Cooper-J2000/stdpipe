"""
Image reprojection routines.

Provides :func:`reproject_swarp` (SWarp wrapper) and :func:`reproject_lanczos`
(pure-Python Lanczos interpolation with SWarp-style oversampling and Jacobian
flux conservation).
"""

import numpy as np

import os
import tempfile
import shlex
import time
import shutil
from concurrent.futures import ThreadPoolExecutor

from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits

from . import utils
from . import astrometry


def _pixel_to_pixel(wcs_from, wcs_to, x, y):
    """Map pixel coordinates from one WCS to another.

    Like :func:`astropy.wcs.utils.pixel_to_pixel` but uses ``quiet=True``
    for the inverse SIP transformation so that non-converging solutions do
    not raise an exception.  Sky positions outside the target projection
    domain map to NaN.  Note that pixels where the iterative SIP inversion
    fails to converge get best-effort (possibly inaccurate) coordinates
    rather than NaN; in practice such divergent solutions land far outside
    the image and are rejected by the bounds checks downstream.
    """
    sky = wcs_from.all_pix2world(x, y, 0)
    try:
        pix = wcs_to.all_world2pix(sky[0], sky[1], 0, quiet=True)
    except TypeError:
        # Very old astropy without quiet= support
        pix = wcs_to.all_world2pix(sky[0], sky[1], 0)
    return pix


# ---------------------------------------------------------------------------
# Lanczos helpers
# ---------------------------------------------------------------------------


def _lanczos_kernel(x, a):
    """Lanczos kernel of order *a*."""
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    mask = np.abs(x) < a
    zero = x == 0
    result[zero] = 1.0
    nonzero = mask & ~zero
    xn = x[nonzero]
    result[nonzero] = np.sin(np.pi * xn) * np.sin(np.pi * xn / a) / (np.pi * xn * np.pi * xn / a)
    return result


def _lanczos_map_coordinates(image, coords, a=3, cval=np.nan, weight=None, min_weight=0.5):
    """Interpolate *image* at fractional pixel coordinates using Lanczos kernel.

    Parameters
    ----------
    image : 2D array
    coords : (2, N) array of (row, col) coordinates
    a : int
        Lanczos kernel order (2, 3 or 4 typical).
    cval : float
        Fill value for out-of-bounds pixels.
    weight : 2D array or None
        Optional weight map (1.0 for valid, 0.0 for masked pixels).  If
        provided, ``image * weight`` and *weight* are interpolated with the
        same kernel and their ratio is returned (normalized convolution), so
        masked pixels are excluded from the interpolation.  The image is
        expected to have finite values (e.g. zero) at masked pixels.
    min_weight : float
        With *weight*: points receiving less than this fraction of their
        kernel mass from valid pixels are set to *cval*.

    Returns
    -------
    values : 1D array of interpolated values

    Notes
    -----
    Kernel taps falling outside the image are replicated from the nearest
    edge pixel (clamped indexing).  Without *weight*, NaN pixels in the
    input propagate to every output value whose ``2*a x 2*a`` kernel
    support includes them.
    """
    ny, nx = image.shape
    n_pts = coords.shape[1]
    result = np.full(n_pts, cval, dtype=np.float64)

    yr, xr = coords[0], coords[1]

    # Filter out-of-bounds
    valid = (yr >= -0.5) & (yr < ny - 0.5) & (xr >= -0.5) & (xr < nx - 0.5)

    yr_v = yr[valid]
    xr_v = xr[valid]

    if len(yr_v) == 0:
        return result

    # Integer and fractional parts
    iy = np.floor(yr_v).astype(int)
    ix = np.floor(xr_v).astype(int)
    fy = yr_v - iy
    fx = xr_v - ix

    # Kernel support: -a+1 to a
    offsets = np.arange(-a + 1, a + 1)

    # Precompute kernels: shape (n_valid, 2*a)
    ky = _lanczos_kernel(fy[:, None] - offsets[None, :], a)
    kx = _lanczos_kernel(fx[:, None] - offsets[None, :], a)

    # Normalize
    ky /= ky.sum(axis=1, keepdims=True)
    kx /= kx.sum(axis=1, keepdims=True)

    # Row and column indices
    row_idx = np.clip(iy[:, None] + offsets[None, :], 0, ny - 1)
    col_idx = np.clip(ix[:, None] + offsets[None, :], 0, nx - 1)

    # Apply separable kernel (vectorized: 2a iters instead of 4a²)
    vals = np.zeros(len(yr_v))
    if weight is None:
        for j in range(len(offsets)):
            row_pixels = image[row_idx[:, j][:, None], col_idx]  # (n_valid, 2a)
            vals += ky[:, j] * np.sum(kx * row_pixels, axis=1)

        result[valid] = vals
    else:
        # Normalized convolution: interpolate image*weight and weight with
        # the same kernel, take the ratio.  The denominator is the fraction
        # of kernel mass on valid pixels (exactly 1 where all are valid).
        dens = np.zeros(len(yr_v))
        for j in range(len(offsets)):
            rows = (row_idx[:, j][:, None], col_idx)
            row_weight = weight[rows]  # (n_valid, 2a)
            vals += ky[:, j] * np.sum(kx * image[rows] * row_weight, axis=1)
            dens += ky[:, j] * np.sum(kx * row_weight, axis=1)

        out = np.full(len(yr_v), cval, dtype=np.float64)
        good = dens >= min_weight
        out[good] = vals[good] / dens[good]
        result[valid] = out

    return result


def _local_area_ratio(x_in, y_in, fallback):
    """Per-pixel Jacobian area ratio of the output->input pixel mapping.

    Computes ``|det d(x_in, y_in) / d(x_out, y_out)|`` by finite differences
    of the coordinate grids, so it follows SIP distortion and
    projection-induced pixel scale variation across the field.

    Parameters
    ----------
    x_in, y_in : 2D arrays
        Input pixel coordinates sampled on the (unit-spaced) output grid.
    fallback : float
        Value used where the determinant cannot be computed (grid smaller
        than 2 pixels along an axis, or non-finite coordinates).

    Returns
    -------
    area : 2D array or float scalar
    """
    if x_in.shape[0] < 2 or x_in.shape[1] < 2:
        return fallback

    dx_dc = np.gradient(x_in, axis=1)
    dx_dr = np.gradient(x_in, axis=0)
    dy_dc = np.gradient(y_in, axis=1)
    dy_dr = np.gradient(y_in, axis=0)

    area = np.abs(dx_dc * dy_dr - dx_dr * dy_dc)
    area[~np.isfinite(area)] = fallback
    return area


def _reproject_single_flags(image, wcs_in, wcs_out, shape_out, oversamp=None):
    """Reproject a single integer flag image using nearest-neighbor sampling.

    When output pixels are larger than input ones (*oversamp* > 1, chosen
    automatically by default), each output pixel is sampled on an
    ``oversamp x oversamp`` sub-pixel grid and the flags of all sampled
    input pixels are combined with bitwise OR, so that isolated flagged
    pixels are not lost in downscaling (like SWarp RESAMPLING_TYPE=FLAGS).

    Returns
    -------
    result : 2D integer array (0 where no data)
    footprint : 2D float array (fractional coverage 0.0–1.0)
    """
    ny_out, nx_out = shape_out
    image = np.asarray(image)
    dtype = image.dtype
    ny_in, nx_in = image.shape

    if oversamp is None:
        area_ratio = np.prod(proj_plane_pixel_scales(wcs_out)) / np.prod(
            proj_plane_pixel_scales(wcs_in)
        )
        oversamp = max(1, int(np.sqrt(area_ratio) + 0.5))

    if np.issubdtype(dtype, np.floating):
        # Float "flags" cannot be combined bitwise - single-sample nearest
        oversamp = 1

    step = 1.0 / oversamp
    sub_offsets = np.arange(oversamp) * step + step / 2 - 0.5 if oversamp > 1 else [0.0]
    n_sub = len(sub_offsets) ** 2

    yy, xx = np.mgrid[0:ny_out, 0:nx_out]

    if np.issubdtype(dtype, np.floating):
        result = np.full(ny_out * nx_out, np.nan, dtype=dtype)
    else:
        result = np.zeros(ny_out * nx_out, dtype=dtype)
    count = np.zeros(ny_out * nx_out, dtype=np.int32)

    for dy_off in sub_offsets:
        for dx_off in sub_offsets:
            pixel_in = _pixel_to_pixel(
                wcs_out,
                wcs_in,
                (xx + dx_off).ravel().astype(float),
                (yy + dy_off).ravel().astype(float),
            )
            px = np.asarray(pixel_in[0])
            py = np.asarray(pixel_in[1])

            # NaN coordinates (outside projection domain) cannot be cast to int
            finite = np.isfinite(px) & np.isfinite(py)
            ix = np.full(px.shape, -1, dtype=int)
            iy = np.full(py.shape, -1, dtype=int)
            ix[finite] = np.round(px[finite]).astype(int)
            iy[finite] = np.round(py[finite]).astype(int)

            valid = (ix >= 0) & (ix < nx_in) & (iy >= 0) & (iy < ny_in)

            if np.issubdtype(dtype, np.floating):
                result[valid] = image[iy[valid], ix[valid]]
            else:
                result[valid] |= image[iy[valid], ix[valid]]
            count[valid] += 1

    footprint = (count.astype(np.float64) / n_sub).reshape(shape_out)
    return result.reshape(shape_out), footprint


def _reproject_chunk(
    image,
    wcs_in,
    wcs_out,
    row_start,
    row_end,
    nx_out,
    order,
    oversamp,
    conserve_flux,
    area_fallback,
    sub_offsets,
    weight=None,
):
    """Reproject a horizontal chunk of rows.  Used by :func:`_reproject_single`.

    Parameters
    ----------
    conserve_flux : bool
        If True, multiply by the local Jacobian area ratio (per pixel).
    area_fallback : float
        Global mean area ratio, used where the local Jacobian cannot be
        computed.
    weight : 2D array or None
        Optional weight map for masked input pixels, passed to
        :func:`_lanczos_map_coordinates`.

    Returns
    -------
    row_start : int
    result : 2D array (NaN where no data)
    footprint : 2D float array (fractional coverage 0.0–1.0)
    """
    ny_chunk = row_end - row_start
    chunk_shape = (ny_chunk, nx_out)

    yy, xx = np.mgrid[row_start:row_end, 0:nx_out]

    if oversamp <= 1:
        pixel_in = _pixel_to_pixel(
            wcs_out, wcs_in, xx.ravel().astype(float), yy.ravel().astype(float)
        )
        x_in = np.asarray(pixel_in[0]).reshape(chunk_shape)
        y_in = np.asarray(pixel_in[1]).reshape(chunk_shape)
        coords = np.array([y_in.ravel(), x_in.ravel()])
        values = _lanczos_map_coordinates(image, coords, a=order, weight=weight)

        area = _local_area_ratio(x_in, y_in, area_fallback) if conserve_flux else 1.0
        result = values.reshape(chunk_shape) * area
        footprint = np.isfinite(result).astype(np.float64)
        return row_start, result, footprint

    accumulator = np.zeros(chunk_shape, dtype=np.float64)
    count = np.zeros(chunk_shape, dtype=np.int32)
    n_sub = len(sub_offsets) ** 2
    area = None

    for dy_off in sub_offsets:
        for dx_off in sub_offsets:
            pixel_out_x = (xx + dx_off).ravel().astype(float)
            pixel_out_y = (yy + dy_off).ravel().astype(float)

            pixel_in = _pixel_to_pixel(wcs_out, wcs_in, pixel_out_x, pixel_out_y)
            x_in = np.asarray(pixel_in[0]).reshape(chunk_shape)
            y_in = np.asarray(pixel_in[1]).reshape(chunk_shape)

            if conserve_flux and area is None:
                # The Jacobian varies negligibly over a sub-pixel offset,
                # so one sub-grid is enough to estimate it
                area = _local_area_ratio(x_in, y_in, area_fallback)

            values = _lanczos_map_coordinates(
                image, np.array([y_in.ravel(), x_in.ravel()]), a=order, weight=weight
            )
            vals_2d = values.reshape(chunk_shape)

            valid = np.isfinite(vals_2d)
            accumulator[valid] += vals_2d[valid]
            count[valid] += 1

    if area is None:
        area = 1.0

    result = np.full(chunk_shape, np.nan, dtype=np.float64)
    good = count > 0
    area_good = area[good] if isinstance(area, np.ndarray) else area
    result[good] = (accumulator[good] / count[good]) * area_good
    footprint = count.astype(np.float64) / n_sub
    return row_start, result, footprint


def _reproject_single(
    image,
    wcs_in,
    wcs_out,
    shape_out,
    order,
    conserve_flux,
    oversamp,
    weight_nans=True,
    parallel=False,
):
    """Reproject a single image with Lanczos interpolation.

    Parameters
    ----------
    weight_nans : bool
        If True, handle NaN input pixels through a SWarp-style weight map
        (normalized convolution) instead of propagating them.
    parallel : bool or int
        If True, use threads (number chosen automatically).
        If int > 1, use that many threads.

    Returns
    -------
    result : 2D array (NaN where no data)
    footprint : 2D float array (fractional coverage 0.0–1.0)
    """
    ny_out, nx_out = shape_out
    image = np.asarray(image, dtype=np.float64)

    # SWarp-style weight map for masked (NaN) input pixels; skipped
    # entirely for clean images where it would be a no-op
    weight = None
    if weight_nans and not np.all(np.isfinite(image)):
        weight = np.isfinite(image).astype(np.float64)
        image = np.where(weight > 0, image, 0.0)

    # Global mean pixel area ratio (product of per-axis scales handles
    # non-square pixels); >1 means output pixels cover more sky
    area_fallback = np.prod(proj_plane_pixel_scales(wcs_out)) / np.prod(
        proj_plane_pixel_scales(wcs_in)
    )
    scale_ratio = np.sqrt(area_fallback)

    # Auto oversampling
    if oversamp is None:
        oversamp = max(1, int(scale_ratio + 0.5))

    # Sub-pixel offsets for oversampling
    if oversamp > 1:
        step = 1.0 / oversamp
        sub_offsets = np.arange(oversamp) * step + step / 2 - 0.5
    else:
        sub_offsets = None

    # Determine number of workers
    if parallel is True:
        n_workers = min(os.cpu_count() or 1, ny_out)
    elif isinstance(parallel, int) and parallel > 1:
        n_workers = min(parallel, ny_out)
    else:
        n_workers = 1

    if n_workers <= 1:
        # Sequential path
        _, result, footprint = _reproject_chunk(
            image,
            wcs_in,
            wcs_out,
            0,
            ny_out,
            nx_out,
            order,
            oversamp,
            conserve_flux,
            area_fallback,
            sub_offsets,
            weight,
        )
        return result, footprint

    # Threaded path — split by row chunks
    chunk_size = max(1, (ny_out + n_workers - 1) // n_workers)
    row_ranges = []
    for i in range(n_workers):
        rs = i * chunk_size
        re = min(rs + chunk_size, ny_out)
        if rs < re:
            row_ranges.append((rs, re))

    result = np.full(shape_out, np.nan, dtype=np.float64)
    footprint = np.zeros(shape_out, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=len(row_ranges)) as pool:
        # WCS transformations are not guaranteed thread-safe when sharing
        # a single object, so give each worker its own copies
        futures = [
            pool.submit(
                _reproject_chunk,
                image,
                wcs_in.deepcopy(),
                wcs_out.deepcopy(),
                rs,
                re,
                nx_out,
                order,
                oversamp,
                conserve_flux,
                area_fallback,
                sub_offsets,
                weight,
            )
            for rs, re in row_ranges
        ]
        for f in futures:
            row_start, chunk, fp_chunk = f.result()
            nrows = chunk.shape[0]
            result[row_start : row_start + nrows] = chunk
            footprint[row_start : row_start + nrows] = fp_chunk

    return result, footprint


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reproject_lanczos(
    input=None,
    wcs=None,
    shape=None,
    width=None,
    height=None,
    header=None,
    order=3,
    conserve_flux=True,
    oversamp=None,
    is_flags=False,
    use_nans=True,
    weight_nans=True,
    parallel=False,
    return_footprint=False,
    verbose=False,
):
    """Reproject images using Lanczos interpolation with automatic oversampling.

    Implements SWarp-style oversampling (sub-pixel averaging when output pixels
    are larger than input pixels) and Jacobian area scaling for flux
    conservation.

    Accepts the same input format as :func:`reproject_swarp`: a list of
    ``(image, header/WCS)`` tuples or a list of FITS filenames.  For multiple
    inputs the reprojected frames are averaged (simple coadd).

    Parameters
    ----------
    input : list or tuple
        List of ``(image, header_or_wcs)`` tuples or FITS filenames.
        A single ``(image, header_or_wcs)`` tuple is also accepted
        (wrapped into a list automatically for reproject compatibility).
    wcs : `~astropy.wcs.WCS`, optional
        Output WCS.  Overrides any WCS already present in *header*.
    shape : tuple, optional
        Output ``(height, width)``.
    width, height : int, optional
        Output dimensions (alternative to *shape*).
    header : `~astropy.io.fits.Header`, optional
        Output FITS header providing the WCS (unless *wcs* is given) and
        image dimensions (unless *shape*/*width*/*height* are given).
    order : int
        Lanczos kernel order (default 3).
    conserve_flux : bool
        If True (default), multiply by the local Jacobian area ratio of
        the pixel mapping (computed per output pixel by finite
        differences, so it follows SIP distortion and projection-induced
        scale variation across the field) so that *total flux* is
        conserved.  If False, *surface brightness* is conserved instead.
    oversamp : int or None
        Sub-pixel oversampling factor per axis.  ``None`` (default) selects
        automatically: ``max(1, round(output_scale / input_scale))``.
    is_flags : bool
        If True, treat input as integer flag/mask images: use
        nearest-neighbor resampling (no interpolation) and bitwise AND
        for combining multiple inputs.  Only frames actually covering a
        pixel participate in the AND.  When output pixels are larger
        than input ones, the flags of all contributing input pixels are
        combined with bitwise OR (controlled by *oversamp*, like SWarp
        RESAMPLING_TYPE=FLAGS), so isolated flagged pixels survive
        downscaling.  Overrides *order* and *conserve_flux*.
    use_nans : bool
        If True (default), regions with no input coverage are set to NaN
        for floating-point images, or have all flag bits set (``0xFFFF``
        for 16-bit integers) for flag images.  If False, they are set
        to zero instead.
    weight_nans : bool
        If True (default), NaN input pixels are handled SWarp-style
        through an internal weight map: the image (with masked pixels
        zeroed) and the weight are resampled with the same kernel and
        their ratio is taken (normalized convolution), so isolated masked
        pixels do not poison their whole kernel neighbourhood.  Output
        pixels receiving less than half of their kernel weight from valid
        pixels remain NaN.  If False, any NaN within the kernel support
        propagates to the output.  Ignored for flag images.
    parallel : bool or int
        If True, use threads for parallel interpolation (number chosen
        automatically).  If int > 1, use that many threads.  Gives
        ~3-4x speedup on multi-core machines.
    return_footprint : bool
        If True, return ``(coadd, footprint)`` where *footprint* is a
        float array with values between 0.0 (no coverage) and 1.0 (full
        coverage).  When oversampling is active, fractional values
        indicate partial sub-pixel coverage.  Default is False for
        backward compatibility.
    verbose : bool or callable
        Logging control.

    Returns
    -------
    coadd : 2D `~numpy.ndarray` or None
        Reprojected (and optionally coadded) image.
    footprint : 2D `~numpy.ndarray`
        Coverage map (only returned when ``return_footprint=True``).

    Notes
    -----
    NaN pixels in the input images are treated as masked.  With
    ``weight_nans=True`` (default) they are excluded from the
    interpolation through a SWarp-style weight map, and only output
    pixels dominated by masked input (less than half of the kernel
    weight on valid pixels) become NaN.  With ``weight_nans=False``
    every output pixel whose Lanczos kernel support (``2*order x
    2*order`` input pixels) contains a NaN becomes NaN.  NaN output
    pixels are excluded from the multi-frame average.
    """
    if input is None:
        input = []

    # Accept a single (image, header/WCS) tuple for reproject compatibility
    if isinstance(input, tuple) and len(input) == 2 and isinstance(input[0], np.ndarray):
        input = [input]

    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Resolve output geometry
    if header is not None:
        header = header.copy()
    else:
        header = fits.Header({'NAXIS': 2, 'BITPIX': -64, 'EQUINOX': 2000.0})

    if wcs is not None and wcs.is_celestial:
        astrometry.clear_wcs(header)
        header += wcs.to_header(relax=True)

    if (width is None or height is None) and shape is not None:
        height, width = shape

    if width is not None:
        header['NAXIS1'] = width
    if height is not None:
        header['NAXIS2'] = height

    wcs_out = WCS(header)
    if not wcs_out.is_celestial:
        log("Can't reproject without target WCS")
        return (None, None) if return_footprint else None

    if 'NAXIS1' not in header or 'NAXIS2' not in header:
        log("Can't reproject without output image dimensions")
        return (None, None) if return_footprint else None

    shape_out = (header['NAXIS2'], header['NAXIS1'])

    if is_flags:
        log('Input images will be handled as integer flags')

    # Collect input frames
    frames = []
    for item in input:
        if isinstance(item, str):
            hdulist = fits.open(item)
            img = hdulist[0].data
            if not is_flags:
                img = img.astype(np.float64)
            wcs_in = WCS(hdulist[0].header)
            hdulist.close()
        else:
            img, hdr_or_wcs = item
            if not is_flags:
                img = np.asarray(img, dtype=np.float64)
            if isinstance(hdr_or_wcs, WCS):
                wcs_in = hdr_or_wcs
            else:
                wcs_in = WCS(hdr_or_wcs)
        frames.append((img, wcs_in))

    if not frames:
        log("No input frames")
        if return_footprint:
            return None, None
        return None

    # Reproject and combine, frame by frame (flat memory)
    if is_flags:
        # Bitwise AND over covering frames (like SWarp COMBINE_TYPE=AND);
        # frames not covering a pixel do not clear its flags
        coadd = None
        footprint = None
        for i, (img, wcs_in) in enumerate(frames):
            log('Reprojecting frame %d/%d' % (i + 1, len(frames)))
            result, fp = _reproject_single_flags(
                img, wcs_in, wcs_out, shape_out, oversamp=oversamp
            )

            if coadd is None:
                if np.issubdtype(result.dtype, np.floating):
                    blank = np.nan
                else:
                    # All bits set: neutral element for AND, and the
                    # "no coverage" sentinel (0xFFFF for 16-bit types)
                    blank = np.invert(result.dtype.type(0))
                coadd = np.full(shape_out, blank, dtype=result.dtype)
                footprint = np.zeros(shape_out, dtype=np.float64)

            covered = fp > 0
            if np.issubdtype(coadd.dtype, np.floating):
                # Float "flags" cannot be combined bitwise; last frame wins
                coadd[covered] = result[covered]
            else:
                coadd[covered] &= result[covered]

            # Combined footprint: covered if any frame covers the pixel
            footprint = np.maximum(footprint, fp)

        if not use_nans:
            coadd[footprint == 0] = 0
    else:
        sum_img = np.zeros(shape_out, dtype=np.float64)
        count = np.zeros(shape_out, dtype=np.int32)
        fp_sum = np.zeros(shape_out, dtype=np.float64)
        for i, (img, wcs_in) in enumerate(frames):
            log('Reprojecting frame %d/%d' % (i + 1, len(frames)))
            result, fp = _reproject_single(
                img,
                wcs_in,
                wcs_out,
                shape_out,
                order,
                conserve_flux,
                oversamp,
                weight_nans=weight_nans,
                parallel=parallel,
            )
            valid = np.isfinite(result)
            sum_img[valid] += result[valid]
            count[valid] += 1
            fp_sum += fp

        coadd = np.full(shape_out, np.nan, dtype=np.float64)
        good = count > 0
        coadd[good] = sum_img[good] / count[good]
        # Average footprint across frames
        footprint = fp_sum / len(frames)

        if not use_nans:
            coadd[~good] = 0.0

    if return_footprint:
        return coadd, footprint
    return coadd


def reproject_swarp(
    input=None,
    wcs=None,
    shape=None,
    width=None,
    height=None,
    header=None,
    extra=None,
    is_flags=False,
    use_nans=True,
    get_weights=False,
    _workdir=None,
    _tmpdir=None,
    _exe=None,
    verbose=False,
):
    """
    Wrapper for running SWarp for re-projecting and mosaicking of images onto target WCS grid.

    It accepts as input either list of filenames, or list of tuples where first
    element is an image, and second one - either FITS header or WCS.

    If the input images are integer flags, set `is_flags=True` so that it will be handled
    by passing `RESAMPLING_TYPE=FLAGS` and `COMBINE_TYPE=AND`.

    If `use_nans=True`, the regions with zero weights will be filled with NaNs (or 0xFFFF).

    Any additional configuration parameter may be passed to SWarp through `extra` argument which
    should be the dictionary with parameter names as keys.

    """

    if input is None:
        input = []
    if extra is None:
        extra = {}

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Find the binary
    binname = None

    if _exe is not None:
        # Check user-provided binary path, and fail if not found
        if os.path.isfile(_exe):
            binname = _exe
    else:
        # Find SWarp binary in common paths
        for exe in ['swarp']:
            binname = shutil.which(exe)
            if binname is not None:
                break

    if binname is None:
        log("Can't find SWarp binary")
        return None
    # else:
    #     log("Using SWarp binary at", binname)

    if (width is None or height is None) and shape is not None:
        height, width = shape

    if header is None:
        # Construct minimal FITS header
        header = fits.Header(
            {
                'NAXIS': 2,
                'NAXIS1': width,
                'NAXIS2': height,
                'BITPIX': -64,
                'EQUINOX': 2000.0,
            }
        )
    else:
        header = header.copy()

    if wcs is not None and wcs.is_celestial:
        # Add WCS information to the header
        astrometry.clear_wcs(header)
        whdr = wcs.to_header(relax=True)

        if wcs.sip is not None:
            whdr = astrometry.wcs_sip2pv(whdr)

        # Here we will try to fix some common problems with WCS not supported by SWarp
        # FIXME: handle SIP distortions!
        if wcs.wcs.has_pc() and 'PC1_1' not in whdr:
            pc = wcs.wcs.get_pc()
            whdr['PC1_1'] = pc[0, 0]
            whdr['PC1_2'] = pc[0, 1]
            whdr['PC2_1'] = pc[1, 0]
            whdr['PC2_2'] = pc[1, 1]

        header += whdr
    else:
        wcs = WCS(header)

        if wcs is None or not wcs.is_celestial:
            log("Can't re-project without target WCS")
            return None

    workdir = _workdir if _workdir is not None else tempfile.mkdtemp(prefix='swarp', dir=_tmpdir)

    # Output coadd filename
    coaddname = os.path.join(workdir, 'coadd.fits')
    if os.path.exists(coaddname):
        os.unlink(coaddname)

    # Input header filename - the result will be re-projected to it
    headername = os.path.join(workdir, 'coadd.head')
    utils.file_write(headername, header.tostring(endcard=True, sep='\n'))

    # Output weights filename
    weightsname = os.path.join(workdir, 'coadd.weights.fits')

    # Dummy config filename, to prevent loading from current dir
    confname = os.path.join(workdir, 'empty.conf')
    utils.file_write(confname)

    xmlname = os.path.join(workdir, 'swarp.xml')

    opts = {
        'VERBOSE_TYPE': 'QUIET' if not verbose else 'NORMAL',
        'IMAGEOUT_NAME': coaddname,
        'WEIGHTOUT_NAME': weightsname,
        'c': confname,
        'XML_NAME': xmlname,
        'VMEM_DIR': workdir,
        'RESAMPLE_DIR': workdir,
        #
        'SUBTRACT_BACK': False,  # Do not subtract the backgrounds
        'FSCALASTRO_TYPE': 'VARIABLE',  # and not re-scale the images by default
    }

    if is_flags:
        log('The images will be handled as integer flags')
        opts['RESAMPLING_TYPE'] = 'FLAGS'
        opts['COMBINE_TYPE'] = 'AND'  # Use only common flags in overlapping masks

    opts.update(extra)

    # Handle input data
    filenames = []
    bzero = 0
    for i, item in enumerate(input):
        if isinstance(item, str):
            # Item is filename already
            filename = item
        elif len(item) == 2:
            # It should be a tuple of image plus header or WCS
            image = item[0]
            header = item[1]

            if image.dtype.name == 'bool':
                image = image.astype(np.int16)

            if isinstance(header, WCS):
                header = header.to_header(relax=True)

            # Convert SIP headers to TPV
            if WCS(header).sip is not None:
                header = astrometry.wcs_sip2pv(header)

            filename = os.path.join(workdir, 'image_%04d.fits' % i)
            fits.writeto(filename, image, header, overwrite=True)

        filenames.append(filename)
        bzero = max(
            bzero, fits.getheader(filename).get('BZERO', 0)
        )  # Keep the largest BZERO among input files

    # Build the command line
    command = (
        binname
        + ' '
        + utils.format_astromatic_opts(opts)
        + ' '
        + ' '.join([shlex.quote(_) for _ in filenames])
    )
    if not verbose:
        command += ' > /dev/null 2>/dev/null'
    log('Will run SWarp like that:')
    log(command)

    # Run the command!

    t0 = time.time()
    res = os.system(command)
    t1 = time.time()

    if res == 0 and os.path.exists(coaddname) and os.path.exists(weightsname):
        log('SWarp run successfully in %.2f seconds' % (t1 - t0))

        coadd = fits.getdata(coaddname)
        weights = fits.getdata(weightsname)

        # it seems SWarp adds BZERO to the output if inputs had them (e.g. unsigned ints do)
        # FIXME: this point needs further investigation!
        if np.issubdtype(coadd.dtype.type, int):
            coadd -= bzero

        if use_nans:
            if np.issubdtype(coadd.dtype, np.floating):
                coadd[weights == 0] = np.nan
            else:
                coadd[weights == 0] = 0xFFFF

    else:
        log('Error', res, 'running SWarp')
        coadd = None
        weights = None

    if _workdir is None:
        shutil.rmtree(workdir)

    if get_weights:
        return coadd, weights
    else:
        return coadd
