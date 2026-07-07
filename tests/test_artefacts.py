import numpy as np
import pytest

from stdpipe.artefacts import filter_sextractor_detections, filter_detections
from stdpipe.realbogus_features import FLAG_BOGUS


def _make_obj(n=50):
    rng = np.random.default_rng(0)
    flux_auto = rng.uniform(1000.0, 2000.0, size=n)
    flux_max = flux_auto * rng.uniform(0.2, 0.6, size=n)
    obj = {
        'FLUX_RADIUS': rng.uniform(1.0, 3.0, size=n),
        'fwhm': rng.uniform(2.0, 4.0, size=n),
        'FLUX_MAX': flux_max,
        'FLUX_AUTO': flux_auto,
        'flags': np.zeros(n, dtype=int),
        'x': rng.uniform(0.0, 2000.0, size=n),
        'y': rng.uniform(0.0, 2000.0, size=n),
        'MAG_AUTO': rng.uniform(15.0, 20.0, size=n),
    }
    return obj


def test_filter_sextractor_returns_features():
    obj = _make_obj(10)
    features = filter_sextractor_detections(obj, return_features=True)

    assert len(features) == 3
    assert [f[1] for f in features] == ['FLUX_RADIUS', 'FWHM', 'FLUX_MAX / FLUX_AUTO']


def test_filter_sextractor_classifier_matches_prediction():
    obj = _make_obj(60)
    # Inject a couple of problematic rows
    obj['FLUX_RADIUS'][0] = np.nan
    obj['FLUX_MAX'][1] = np.nan
    obj['flags'][2] = 1

    res = filter_sextractor_detections(
        obj, trend_cols=None, random_state=0, verbose=False
    )
    clf = filter_sextractor_detections(
        obj, trend_cols=None, return_classifier=True, random_state=0, verbose=False
    )

    res2 = clf(obj)
    assert res.shape == (len(obj['FLUX_AUTO']),)
    assert res2.shape == res.shape
    assert np.array_equal(res, res2)


def test_filter_sextractor_with_trend_cols():
    obj = _make_obj(80)

    res = filter_sextractor_detections(
        obj,
        trend_cols=['x', 'y', 'MAG_AUTO'],
        trend_scales=[1000, 1000, 2],
        random_state=1,
        verbose=False,
        k=15,
        robust_iters=0,
    )
    assert res.shape == (len(obj['FLUX_AUTO']),)


def test_filter_sextractor_auto_trend_scales():
    obj = _make_obj(80)

    # Default trend_scales=None should auto-compute per-column scales
    res = filter_sextractor_detections(
        obj,
        trend_cols=['x', 'y', 'MAG_AUTO'],
        random_state=1,
        verbose=False,
        k=15,
        robust_iters=0,
    )
    assert res.shape == (len(obj['FLUX_AUTO']),)


def test_filter_sextractor_drops_missing_trend_cols():
    obj = _make_obj(80)
    del obj['MAG_AUTO']

    # Default trend_cols include MAG_AUTO; it should be dropped gracefully
    res = filter_sextractor_detections(
        obj, random_state=1, verbose=False, k=15, robust_iters=0
    )
    assert res.shape == (len(obj['FLUX_AUTO']),)


def test_filter_sextractor_sep_columns():
    # SEP-style catalog: peak/flux instead of FLUX_MAX/FLUX_AUTO, no FLUX_RADIUS
    rng = np.random.default_rng(0)
    n = 60
    flux = rng.uniform(1000.0, 2000.0, size=n)
    obj = {
        'fwhm': rng.uniform(2.0, 4.0, size=n),
        'peak': flux * rng.uniform(0.2, 0.6, size=n),
        'flux': flux,
        'flags': np.zeros(n, dtype=int),
        'x': rng.uniform(0.0, 2000.0, size=n),
        'y': rng.uniform(0.0, 2000.0, size=n),
    }

    features = filter_sextractor_detections(obj, return_features=True)
    assert [f[1] for f in features] == ['FWHM', 'peak / flux']

    res = filter_sextractor_detections(
        obj, trend_cols=['x', 'y'], random_state=0, verbose=False, k=15, robust_iters=0
    )
    assert res.shape == (n,)


def test_filter_sextractor_contamination():
    obj = _make_obj(100)

    res = filter_sextractor_detections(
        obj, trend_cols=None, contamination=0.1, random_state=0, verbose=False
    )
    # Rejection rate should roughly follow the requested contamination
    assert 0 < np.sum(~res) <= 20


def test_filter_sextractor_too_few_clean_raises():
    obj = _make_obj(10)
    obj['flags'][:] = 1

    with pytest.raises(ValueError, match="Too few clean detections"):
        filter_sextractor_detections(obj, trend_cols=None, verbose=False)


def test_filter_detections_writes_back_score_and_flags():
    from astropy.table import Table

    rng = np.random.default_rng(0)
    n = 60
    flux = rng.uniform(1000.0, 10000.0, size=n)
    obj = Table({
        'x': rng.uniform(0.0, 500.0, size=n),
        'y': rng.uniform(0.0, 500.0, size=n),
        'flux': flux,
        'fluxerr': rng.uniform(10.0, 100.0, size=n),
        'fwhm': rng.normal(3.0, 0.3, size=n),
        'a': rng.uniform(1.5, 2.5, size=n),
        'b': rng.uniform(1.3, 2.0, size=n),
        'peak': flux / 10,
        'flags': np.zeros(n, dtype=np.int32),
    })

    good = filter_detections(
        obj, classifier='isolation', add_score=True, flag_bogus=True, verbose=False
    )

    assert good.shape == (n,)
    assert 'rb_score' in obj.colnames
    assert np.array_equal(good, np.asarray(obj['rb_score'] >= 0.5))
    assert np.array_equal(((obj['flags'] & FLAG_BOGUS) != 0), ~good)
