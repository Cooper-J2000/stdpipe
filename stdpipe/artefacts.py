# Various artefact filtering routines
#
# For more advanced real-bogus classification with cutout-based morphological
# features, see the realbogus_features module:
#
#   from stdpipe import realbogus_features as rbf
#   obj = rbf.classify(obj, image, method='hybrid', classifier='scoring')
#
# The realbogus_features module provides:
# - Catalog-only, cutout-only, or hybrid feature extraction
# - Multiple classifiers: scoring (no training), IsolationForest, RandomForest
# - Generalized trend removal for all features
# - Training utilities for custom classifiers

import numpy as np
from sklearn.ensemble import IsolationForest

from . import smoothing
from .realbogus_features import FLAG_BOGUS, FLAGS_UNRELIABLE_MASK


def _has_column(obj, name):
    """Check whether a table-like or dict-like catalog has a column."""
    colnames = getattr(obj, 'colnames', None)
    if colnames is not None:
        return name in colnames
    if hasattr(obj, 'keys'):
        return name in obj.keys()
    names = getattr(getattr(obj, 'dtype', None), 'names', None)
    return names is not None and name in names


def _get_catalog_features(obj):
    """
    Build the feature list from available catalog columns.

    Prefers SExtractor-style columns (FLUX_RADIUS, FLUX_MAX/FLUX_AUTO) and
    falls back to SEP-style ones (peak/flux) where possible, so the same
    filter works on both `get_objects_sextractor()` and `get_objects_sep()`
    catalogs.
    """
    features = []

    if _has_column(obj, 'FLUX_RADIUS'):
        features.append([np.asarray(obj['FLUX_RADIUS'], dtype=float), 'FLUX_RADIUS'])

    for name in ('fwhm', 'FWHM_IMAGE'):
        if _has_column(obj, name):
            features.append([np.asarray(obj[name], dtype=float), 'FWHM'])
            break

    for num, den in (('FLUX_MAX', 'FLUX_AUTO'), ('peak', 'flux')):
        if _has_column(obj, num) and _has_column(obj, den):
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.asarray(obj[num], dtype=float) / np.asarray(obj[den], dtype=float)
            features.append([ratio, f'{num} / {den}'])
            break

    return features


def filter_sextractor_detections(
    obj,
    trend_cols=['x', 'y', 'MAG_AUTO'],
    trend_scales=None,
    contamination='auto',
    return_features=False,
    return_classifier=False,
    random_state=0,
    verbose=True,
    **kwargs,
):
    """
    Flag detections likely to be artefacts using IsolationForest.

    Builds feature vectors from FLUX_RADIUS, FWHM, and peakiness
    (FLUX_MAX/FLUX_AUTO, or peak/flux for SEP catalogs), using whichever
    columns are present. Optionally removes smooth spatial trends (e.g.,
    across x/y/MAG_AUTO) via an approximate LOESS regressor before fitting
    the outlier model.

    Expected columns in `obj`
    -------------------------
    Required:
    - flags
    - at least one of the feature columns: FLUX_RADIUS; fwhm or FWHM_IMAGE;
      FLUX_MAX + FLUX_AUTO or peak + flux
    Trend columns (when `trend_cols` is set):
    - columns named in `trend_cols` (default: x, y, MAG_AUTO); missing ones
      are dropped with a log message

    Parameters
    ----------
    obj : array-like / table
        Detection catalog with required columns.
    trend_cols : list[str] or None
        Columns used to model smooth trends; set to None/[] to skip detrending.
    trend_scales : list[float] or None
        Per-dimension scaling for LOESS distances; must match trend_cols
        length. If None (default), scales are auto-computed as ~3x the
        standard deviation of each trend column.
    contamination : float or 'auto'
        Expected fraction of artefacts, passed to IsolationForest. With
        'auto' (default) the threshold follows the original paper and a
        data-dependent fraction is flagged even on clean catalogs; set an
        explicit value (e.g. 0.02) to control the rejection rate.
    return_features : bool
        If True, return the feature list (arrays + labels) without fitting.
    return_classifier : bool
        If True, return a callable that classifies new catalogs.
    random_state : int
        Random seed for IsolationForest.
    verbose : bool or callable
        Logging control; can be a print-like function.
    **kwargs :
        Extra arguments passed to `ApproxLoessRegressor` (e.g., k, robust_iters).

    Returns
    -------
    good : ndarray[bool] or callable
        Boolean mask of “good” detections, or a classifier callable if
        return_classifier is True.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    features = _get_catalog_features(obj)

    if not features:
        raise ValueError(
            "No usable feature columns found in the catalog "
            "(need FLUX_RADIUS, fwhm/FWHM_IMAGE, FLUX_MAX+FLUX_AUTO or peak+flux)"
        )

    if return_features:
        return features

    log(
        "Using isolation forest outlier detection over columns ({})".format(
            ", ".join([_[1] for _ in features])
        )
    )

    # LOESS neighbor count; popped here so it does not leak into other kwargs
    k = kwargs.pop('k', 20)

    # Exclude blends etc from the fit, as well as broken measurements;
    # FLAG_BOGUS itself is ignored so re-running the filter is idempotent
    idx = (obj['flags'] & FLAGS_UNRELIABLE_MASK) == 0
    for f in features:
        idx &= np.isfinite(f[0]) & (f[0] > 0)

    pos = None
    if trend_cols:
        # Drop trend columns missing from the catalog (keeping user-provided
        # scales aligned with the surviving columns)
        available = [_ for _ in trend_cols if _has_column(obj, _)]
        if trend_scales is not None and len(trend_scales) != len(trend_cols):
            raise ValueError("trend_scales length is inconsistent with trend_cols length")
        if len(available) < len(trend_cols):
            missing = [_ for _ in trend_cols if _ not in available]
            log("Trend columns not found in catalog: {}".format(", ".join(missing)))
            if trend_scales is not None:
                trend_scales = [s for c, s in zip(trend_cols, trend_scales) if c in available]
        trend_cols = available

    if trend_cols:
        log("Removing smooth trends in {} using approximate LOESS".format(", ".join(trend_cols)))

        if trend_scales is None:
            # Auto-compute per-dimension scales as ~3 sigma of each column
            trend_scales = []
            for col in trend_cols:
                scale = 3.0 * np.nanstd(np.asarray(obj[col], dtype=float))
                if not np.isfinite(scale) or scale <= 0:
                    scale = 1.0
                trend_scales.append(scale)
            log(
                "Auto trend scales: {}".format(
                    ", ".join([f"{c}={s:.3g}" for c, s in zip(trend_cols, trend_scales)])
                )
            )

        pos = np.column_stack([np.array(obj[_]) for _ in trend_cols])
        # Rows with non-finite trend columns cannot be detrended and would
        # get NaN features anyway; keep them out of the fits below
        idx &= np.all(np.isfinite(pos), axis=1)

    if np.sum(idx) < 3:
        raise ValueError(
            f"Too few clean detections ({np.sum(idx)}) to fit the outlier model"
        )

    if trend_cols:
        trend_models = []
        X = []

        for f in features:
            model = smoothing.ApproxLoessRegressor(k=k, scales=trend_scales, **kwargs)
            model.fit(pos[idx], f[0][idx])
            trend_models.append(model)
            X.append(np.array(f[0]) - model.predict(pos))

    else:
        trend_models = None
        X = [np.array(_[0]) for _ in features]

    X = np.column_stack(X)
    X[~np.isfinite(X)] = -100000  # Definitely outside of the good locus

    clf = IsolationForest(contamination=contamination, random_state=random_state).fit(X[idx])

    res = clf.predict(X)

    log(f"{np.sum(res > 0)} good, {np.sum(res < 0)} outliers")

    if return_classifier:

        def classifier(obj):
            features = _get_catalog_features(obj)

            if trend_cols:
                pos = np.column_stack([np.array(obj[_]) for _ in trend_cols])
                X = []

                for f, model in zip(features, trend_models):
                    X.append(np.array(f[0]) - model.predict(pos))
            else:
                X = [np.array(_[0]) for _ in features]

            X = np.column_stack(X)
            X[~np.isfinite(X)] = -100000  # Definitely outside of the good locus

            return clf.predict(X) > 0

        return classifier

    return res > 0


def filter_detections(
    obj,
    image=None,
    bg=None,
    err=None,
    mask=None,
    fwhm=None,
    method='auto',
    classifier='isolation',
    threshold=0.5,
    remove_trend=True,
    trend_cols=None,
    trend_scales=None,
    add_score=False,
    flag_bogus=False,
    verbose=True,
    **kwargs,
):
    """
    Filter detections using feature-based real-bogus classification.

    This is a convenience wrapper around realbogus_features.classify() that
    provides a simpler interface similar to filter_sextractor_detections().

    For full control over feature extraction and classification, use
    realbogus_features.classify() directly.

    Parameters
    ----------
    obj : astropy.table.Table
        Object catalog with 'x', 'y' columns. Modified in place when
        `add_score` or `flag_bogus` is set.
    image : ndarray, optional
        Science image. If provided, cutout features will be extracted.
    bg : ndarray or float, optional
        Background map or scalar.
    err : ndarray or float, optional
        Error/noise map or scalar.
    mask : ndarray, optional
        Boolean mask (True = masked).
    fwhm : float, optional
        Image FWHM. If None, estimated from catalog.
    method : str, optional
        Feature extraction method:
        - 'catalog': Catalog features only (no image needed)
        - 'cutout': Cutout features only
        - 'hybrid': Both catalog and cutout features
        - 'auto': 'hybrid' if image provided, else 'catalog'
    classifier : str, optional
        Classifier to use:
        - 'scoring': Rule-based scoring (no training needed)
        - 'isolation': IsolationForest (unsupervised, default)
    threshold : float, optional
        Score threshold for classification. Default: 0.5.
    remove_trend : bool, optional
        Remove spatial trends from features. Default: True.
    trend_cols : list of str, optional
        Columns for trend removal. Default: ['x', 'y'].
    trend_scales : list of float, optional
        Scales for trend removal. Default: auto-computed.
    add_score : bool, optional
        Add 'rb_score' column to the input catalog. Default: False.
    flag_bogus : bool, optional
        Set FLAG_BOGUS (0x4000) on bogus objects in the input catalog.
        Default: False.
    verbose : bool, optional
        Print progress.
    **kwargs
        Additional arguments passed to realbogus_features.classify().

    Returns
    -------
    good : ndarray[bool]
        Boolean mask of "good" (real) detections.

    See Also
    --------
    realbogus_features.classify : Full-featured classification function.
    filter_sextractor_detections : Original SExtractor-specific filter.

    Examples
    --------
    >>> # Simple catalog-only filtering (like original function)
    >>> good = filter_detections(obj, classifier='isolation')

    >>> # Cutout-based filtering with scoring (no training)
    >>> good = filter_detections(obj, image, classifier='scoring')

    >>> # Hybrid with trend removal
    >>> good = filter_detections(obj, image, method='hybrid',
    ...                          remove_trend=True, trend_cols=['x', 'y'])
    """
    from . import realbogus_features as rbf

    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Call the full classify function; the score is always needed to build
    # the output mask
    result = rbf.classify(
        obj,
        image=image,
        bg=bg,
        err=err,
        mask=mask,
        fwhm=fwhm,
        method=method,
        classifier=classifier,
        threshold=threshold,
        add_score=True,
        flag_bogus=flag_bogus,
        remove_trend=remove_trend,
        trend_cols=trend_cols,
        trend_scales=trend_scales,
        verbose=verbose,
        **kwargs,
    )

    good = np.asarray(result['rb_score'] >= threshold)

    # classify() works on a copy; propagate the requested columns back to
    # the caller's catalog so add_score/flag_bogus have a visible effect
    if add_score:
        obj['rb_score'] = result['rb_score']
    if flag_bogus:
        obj['flags'] = result['flags']

    log(f"{np.sum(good)} good, {np.sum(~good)} outliers")

    return good
