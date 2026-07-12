import copy
import pickle

import numpy as np
import pytest
from astropy.time import Time

from stdpipe import astrometry
from stdpipe.lcs import LCs


def _build_sample_lcs():
    # Two tight clusters separated by ~36 arcsec in RA
    c1_ra = 10.0 + np.array([0.0, 0.1, -0.1]) / 3600.0
    c1_dec = 10.0 + np.array([0.0, 0.05, -0.05]) / 3600.0
    c2_ra = 10.01 + np.array([0.0, 0.05]) / 3600.0
    c2_dec = 10.0 + np.array([0.0, -0.05]) / 3600.0

    ra = np.concatenate([c1_ra, c2_ra])
    dec = np.concatenate([c1_dec, c2_dec])
    flux = np.array([10.0, 11.0, 9.0, 20.0, 21.0])

    lcs = LCs()
    lcs.add(ra=ra, dec=dec, flux=flux)
    return lcs


def test_cluster_groups_points_and_analyze():
    np.random.seed(0)
    lcs = _build_sample_lcs()

    def analyze(obj, ids):
        return {'mean_flux': np.mean(obj.flux[ids])}

    lcs.cluster(sr=1 / 3600, min_length=2, verbose=False, analyze=analyze)

    assert len(lcs.lcs['ra']) == 2
    assert sorted(lcs.lcs['N'].tolist()) == [2, 3]
    assert 'mean_flux' in lcs.lcs
    assert len(lcs.lcs['mean_flux']) == 2


def test_cluster_respects_min_length():
    np.random.seed(0)
    lcs = _build_sample_lcs()

    lcs.cluster(sr=1 / 3600, min_length=4, verbose=False)

    assert len(lcs.lcs['ra']) == 0


def test_cluster_merges_when_radius_large():
    np.random.seed(0)
    lcs = _build_sample_lcs()

    lcs.cluster(sr=60 / 3600, min_length=1, verbose=False)

    assert len(lcs.lcs['ra']) == 1
    assert lcs.lcs['N'][0] == 5


def test_cluster_separates_bridged_overlapping_groups():
    sr = 1 / 3600
    ra0 = 0.0
    dec0 = 0.0
    sep = 1.8 / 3600
    bridge = 0.9 / 3600

    c1_ra = ra0 + np.array([-0.1, 0.0, 0.1]) / 3600
    c2_ra = ra0 + sep + np.array([-0.1, 0.0, 0.1]) / 3600
    c1_dec = dec0 + np.array([-0.05, 0.0, 0.05]) / 3600
    c2_dec = dec0 + np.array([0.05, 0.0, -0.05]) / 3600

    ra = np.concatenate([c1_ra, c2_ra, [ra0 + bridge]])
    dec = np.concatenate([c1_dec, c2_dec, [dec0]])

    np.random.seed(0)
    lcs = LCs()
    lcs.add(ra=ra, dec=dec)
    lcs.cluster(sr=sr, min_length=2, verbose=False)

    assert len(lcs.lcs['ra']) == 2
    # The bridge point falls within sr of both group centroids and ends up
    # claimed by each cluster.
    assert sorted(lcs.lcs['N'].tolist()) == [4, 4]


def test_cluster_gaussian_separation():
    sr = 2.0 / 3600
    sigma = 0.5 * sr
    n_per = 200
    ra0, dec0 = 100.0, 20.0
    sep = 2.0 * sr

    rng = np.random.default_rng(0)
    ra1 = ra0 + rng.normal(scale=sigma, size=n_per)
    dec1 = dec0 + rng.normal(scale=sigma, size=n_per)
    ra2 = ra0 + sep + rng.normal(scale=sigma, size=n_per)
    dec2 = dec0 + rng.normal(scale=sigma, size=n_per)

    ra = np.concatenate([ra1, ra2])
    dec = np.concatenate([dec1, dec2])

    np.random.seed(0)
    lcs = LCs()
    lcs.add(ra=ra, dec=dec)
    lcs.cluster(sr=sr, min_length=20, verbose=False)

    assert len(lcs.lcs['ra']) >= 2

    order = np.argsort(lcs.lcs['N'])[::-1]
    ra_centers = lcs.lcs['ra'][order[:2]]
    dec_centers = lcs.lcs['dec'][order[:2]]
    sep_meas = astrometry.spherical_distance(
        ra_centers[0], dec_centers[0], ra_centers[1], dec_centers[1]
    )
    assert sep_meas > 1.5 * sr
    assert sep_meas < 2.5 * sr


def test_add_mismatched_vector_lengths_raises():
    lcs = LCs()
    with pytest.raises(ValueError):
        lcs.add(ra=[1.0, 2.0], dec=[1.0, 2.0, 3.0])


def test_add_pads_omitted_and_new_keys():
    lcs = LCs()
    lcs.add(ra=[1.0, 2.0], flux=[10.0, 20.0], name=['a', 'b'])
    # Omitted keys get padded
    lcs.add(ra=[3.0])
    # New key gets backfilled for earlier rows
    lcs.add(ra=[4.0], mag=[15.0])

    # Float columns are padded with NaN and keep their numeric dtype
    assert len(lcs.ra) == 4
    assert len(lcs.flux) == 4
    assert lcs.flux.dtype.kind == 'f'
    assert list(lcs.flux[:2]) == [10.0, 20.0]
    assert np.isnan(lcs.flux[2]) and np.isnan(lcs.flux[3])
    assert len(lcs.mag) == 4
    assert lcs.mag.dtype.kind == 'f'
    assert np.all(np.isnan(lcs.mag[:3]))
    assert lcs.mag[3] == 15.0

    # Non-float columns are padded with None
    assert len(lcs.name) == 4
    assert list(lcs.name[:2]) == ['a', 'b']
    assert lcs.name[2] is None and lcs.name[3] is None


def test_time_column_with_missing_values():
    lcs = LCs()
    lcs.add(ra=[1.0, 2.0], dec=[0.0, 0.0], time=Time([60000.0, 60001.0], format='mjd'))
    # Time omitted: rows must become masked entries, not crash consolidation
    lcs.add(ra=[3.0], dec=[0.0])

    t = lcs.time
    assert isinstance(t, Time)
    assert len(t) == 3
    assert not np.any(t.mask[:2])
    assert t.mask[2]
    assert np.allclose(t.mjd[:2].filled(np.nan), [60000.0, 60001.0])


def test_cluster_deterministic_without_global_seed():
    # Two identical containers must produce identical centroids without
    # any global RNG seeding (jitter uses a fixed-seed local generator)
    res = []
    for _ in range(2):
        lcs = _build_sample_lcs()
        lcs.cluster(sr=1 / 3600, min_length=2, verbose=False)
        res.append((lcs.lcs['ra'].copy(), lcs.lcs['dec'].copy()))

    assert np.array_equal(res[0][0], res[1][0])
    assert np.array_equal(res[0][1], res[1][1])


def test_cluster_radius_is_angular():
    # Two points separated by 150 deg with sr=120 deg: the angle-as-chord
    # approximation would put them in one cluster (chord(150d)=1.93 < 2.09),
    # while the correct chord radius for 120 deg is 1.73
    lcs = LCs()
    lcs.add(ra=[0.0, 150.0], dec=[0.0, 0.0])
    lcs.cluster(sr=120.0, min_length=1, verbose=False)

    assert len(lcs.lcs['ra']) == 2


def test_verbose_callable_receives_progress():
    lcs = _build_sample_lcs()
    messages = []
    lcs.cluster(sr=1 / 3600, min_length=1, verbose=messages.append, N=2)

    assert any('lcs' in m for m in messages)


def test_add_after_cluster():
    np.random.seed(0)
    lcs = _build_sample_lcs()
    lcs.cluster(sr=1 / 3600, min_length=2, verbose=False)
    assert len(lcs.lcs['ra']) == 2

    # Append a third group and re-cluster
    lcs.add(
        ra=10.02 + np.array([0.0, 0.03]) / 3600,
        dec=10.0 + np.array([0.0, 0.03]) / 3600,
        flux=[5.0, 6.0],
    )
    lcs.cluster(sr=1 / 3600, min_length=2, verbose=False)
    assert len(lcs.lcs['ra']) == 3


def test_cluster_uses_requested_columns():
    np.random.seed(0)
    lcs = LCs()
    lcs.add(
        ra=[10.0, 10.0],
        dec=[10.0, 10.0],
        ra2=[50.0, 120.0],
        dec2=[-10.0, 30.0],
    )

    lcs.cluster(sr=1 / 3600, min_length=1, verbose=False)
    assert len(lcs.lcs['ra']) == 1

    # Re-clustering on different columns must rebuild the KDTree
    lcs.cluster(sr=1 / 3600, min_length=1, col_ra='ra2', col_dec='dec2', verbose=False)
    assert len(lcs.lcs['ra']) == 2


def test_cluster_missing_column_raises():
    lcs = LCs()
    lcs.add(ra=[1.0], dec=[2.0])
    with pytest.raises(KeyError):
        lcs.cluster(col_ra='nonexistent', verbose=False)


def test_copy_and_pickle():
    np.random.seed(0)
    lcs = _build_sample_lcs()

    lcs2 = copy.copy(lcs)
    assert np.allclose(lcs2.ra, lcs.ra)

    lcs3 = pickle.loads(pickle.dumps(lcs))
    assert np.allclose(lcs3.ra, lcs.ra)
    assert np.allclose(lcs3.flux, lcs.flux)


def test_analyze_returning_none():
    np.random.seed(0)
    lcs = _build_sample_lcs()

    def analyze(obj, ids):
        # Only report for larger clusters
        if len(ids) >= 3:
            return {'mean_flux': np.mean(obj.flux[ids])}
        return None

    lcs.cluster(sr=1 / 3600, min_length=2, verbose=False, analyze=analyze)
    assert len(lcs.lcs['ra']) == 2
    assert len(lcs.lcs['mean_flux']) == 1


def test_time_column_roundtrip():
    lcs = LCs()
    t1 = Time(['2020-01-01T00:00:00', '2020-01-02T00:00:00'])
    t2 = Time(['2020-01-03T00:00:00'])
    lcs.add(ra=[1.0, 2.0], dec=[0.0, 0.0], time=t1)
    lcs.add(ra=[3.0], dec=[0.0], time=t2)
    # Scalar Time gets broadcast too
    lcs.add(ra=[4.0], dec=[0.0], time=Time('2020-01-05T00:00:00'))

    assert isinstance(lcs.time, Time)
    assert lcs.time.format == 'isot'
    assert len(lcs.time) == 4
    assert np.allclose(
        lcs.time.mjd,
        np.concatenate([t1.mjd, t2.mjd, [Time('2020-01-05T00:00:00').mjd]]),
    )
