"""Tests for stdpipe.photometry_quality and the qf/fracflux/spread_model
columns added to stdpipe.photometry_psf.measure_objects_psf."""

import numpy as np
import pytest

from stdpipe import photometry_quality as pq


def _gaussian_stamp(size, fwhm, cx=None, cy=None):
    if cx is None:
        cx = (size - 1) / 2
    if cy is None:
        cy = (size - 1) / 2
    yy, xx = np.mgrid[:size, :size].astype('f4')
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    g = np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sigma ** 2).astype('f4')
    g /= g.sum()
    return g


@pytest.mark.unit
class TestPhotometryQualityHelpers:
    """Direct tests of the helper functions."""

    def test_qf_unmasked_is_one(self):
        psf = _gaussian_stamp(21, 3.0)
        psf_stack = np.stack([psf, psf, psf])
        weight = np.ones((3, 21, 21), dtype='f4')
        qf = pq.compute_qf(psf_stack, weight)
        assert np.allclose(qf, 1.0, atol=1e-5)

    def test_qf_drops_with_mask(self):
        psf = _gaussian_stamp(21, 3.0)
        psf_stack = np.stack([psf, psf])
        weight = np.ones((2, 21, 21), dtype='f4')
        # Mask out the half opposite the PSF center: top half (rows 0..9).
        weight[1, :10, :] = 0
        qf = pq.compute_qf(psf_stack, weight)
        assert qf[0] == pytest.approx(1.0, abs=1e-5)
        # Source 1 keeps the central row + bottom half — should retain >50% but <100%.
        assert 0.4 < qf[1] < 0.95

    def test_fracflux_isolated_is_one(self):
        psf = _gaussian_stamp(21, 3.0)
        data = psf.copy()
        # impsf == im for an isolated source: fracflux must be exactly 1 by construction
        psf_stack = psf[None]
        f = pq.compute_fracflux(data[None], data[None], psf_stack, np.ones((1, 21, 21), dtype='f4'))
        assert f[0] == pytest.approx(1.0)

    def test_fracflux_drops_with_neighbour_contamination(self):
        psf = _gaussian_stamp(21, 3.0)
        # Total data: this source + a neighbour offset by 4 px (~1.3 FWHM)
        neighbour = _gaussian_stamp(21, 3.0, cx=10 + 4, cy=10 + 4)
        data = psf + 0.5 * neighbour
        impsf_clean = psf.copy()  # neighbour cleanly removed
        f = pq.compute_fracflux(
            impsf_clean[None], data[None], psf[None], np.ones((1, 21, 21), dtype='f4')
        )
        # Some PSF-weighted flux at this position comes from the neighbour, so
        # the cleaned-stamp fraction is below 1.
        assert 0.5 < f[0] < 1.0

    def test_spread_model_zero_for_pure_psf(self):
        psf = _gaussian_stamp(31, 3.0)
        weight = np.ones((1, 31, 31), dtype='f4') * 100.0
        # data == PSF → spread_model must be ~0
        spread, dspread = pq.compute_spread_model(psf[None], psf[None], weight)
        assert abs(spread[0]) < 1e-4
        assert dspread[0] > 0  # non-zero uncertainty

    def test_spread_model_positive_for_extended_source(self):
        psf = _gaussian_stamp(31, 3.0)
        gal = _gaussian_stamp(31, 6.0)  # 2x wider than PSF
        weight = np.ones((1, 31, 31), dtype='f4') * 100.0
        spread, _ = pq.compute_spread_model(gal[None], psf[None], weight, fwhm=3.0)
        assert spread[0] > 0

    def test_spread_model_negative_for_subpsf_source(self):
        # An impulse-like (narrower than PSF) source should give spread_model < 0
        psf = _gaussian_stamp(31, 3.0)
        narrow = _gaussian_stamp(31, 1.5)
        weight = np.ones((1, 31, 31), dtype='f4') * 100.0
        spread, _ = pq.compute_spread_model(narrow[None], psf[None], weight, fwhm=3.0)
        assert spread[0] < 0

    def test_compute_psf_quality_returns_all_keys(self):
        psf = _gaussian_stamp(21, 3.0)
        data = psf.copy()
        weight = np.ones((1, 21, 21), dtype='f4')
        out = pq.compute_psf_quality(data[None], data[None], psf[None], weight)
        assert set(out.keys()) == {'qf', 'fracflux', 'spread_model', 'dspread_model'}
        for v in out.values():
            assert v.shape == (1,)


@pytest.mark.unit
class TestMeasureObjectsPsfQualityColumns:
    """Integration via measure_objects_psf."""

    def _make_test_image(self, sources, size=200, noise=5.0, seed=7):
        from stdpipe import photometry  # noqa
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[:size, :size].astype('f4')
        image = np.zeros((size, size), dtype='f4')
        for cx, cy, fwhm, flux in sources:
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            g = np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sigma ** 2)
            image += (g * flux / g.sum()).astype('f4')
        image += rng.normal(0, noise, (size, size)).astype('f4')
        return image

    def test_columns_are_added(self):
        from stdpipe import photometry, photometry_psf
        img = self._make_test_image([(50, 50, 3.0, 8e3), (150, 150, 3.0, 8e3)])
        obj = photometry.get_objects_sep(img, thresh=5.0, aper=3.0, fwhm=3.0)
        assert len(obj) >= 2
        result = photometry_psf.measure_objects_psf(obj, img, fwhm=3.0, verbose=False)
        for col in ('qf', 'fracflux', 'spread_model', 'dspread_model'):
            assert col in result.colnames

    def test_compute_quality_false_skips_columns(self):
        from stdpipe import photometry, photometry_psf
        img = self._make_test_image([(50, 50, 3.0, 8e3)])
        obj = photometry.get_objects_sep(img, thresh=5.0, aper=3.0, fwhm=3.0)
        result = photometry_psf.measure_objects_psf(
            obj, img, fwhm=3.0, compute_quality=False, verbose=False
        )
        for col in ('qf', 'fracflux', 'spread_model', 'dspread_model'):
            assert col not in result.colnames

    def test_isolated_sources_have_unit_fracflux(self):
        from stdpipe import photometry, photometry_psf
        img = self._make_test_image(
            [(50, 50, 3.0, 8e3), (150, 150, 3.0, 8e3), (100, 50, 3.0, 8e3)]
        )
        obj = photometry.get_objects_sep(img, thresh=5.0, aper=3.0, fwhm=3.0)
        result = photometry_psf.measure_objects_psf(obj, img, fwhm=3.0, verbose=False)
        good = np.isfinite(result['fracflux'])
        assert np.all(result['fracflux'][good] > 0.95)

    def test_blended_pair_has_subunit_fracflux(self):
        from stdpipe import photometry, photometry_psf
        # Pair separated by ~1.3 FWHM
        img = self._make_test_image(
            [(98, 100, 3.0, 8e3), (102, 100, 3.0, 8e3), (50, 150, 3.0, 8e3)]
        )
        obj = photometry.get_objects_sep(img, thresh=5.0, aper=3.0, fwhm=3.0)
        result = photometry_psf.measure_objects_psf(
            obj, img, fwhm=3.0, group_sources=True, verbose=False
        )
        # Find the blended sources (x near 100)
        blend_mask = np.abs(result['x'] - 100) < 10
        if np.sum(blend_mask) >= 2:
            assert np.all(result['fracflux'][blend_mask] < 0.97)

    def test_extended_source_has_positive_spread_model(self):
        from stdpipe import photometry, photometry_psf
        # One star + one extended source (FWHM 6 vs 3)
        img = self._make_test_image(
            [(50, 50, 3.0, 8e3), (150, 150, 6.0, 1.5e4)]
        )
        obj = photometry.get_objects_sep(img, thresh=5.0, aper=3.0, fwhm=3.0)
        result = photometry_psf.measure_objects_psf(obj, img, fwhm=3.0, verbose=False)
        # Star (x near 50) should be near zero, extended (x near 150) clearly larger.
        star = np.abs(result['x'] - 50) < 10
        ext = np.abs(result['x'] - 150) < 10
        if np.any(star) and np.any(ext):
            assert np.abs(result['spread_model'][star]).max() < 0.005
            assert result['spread_model'][ext].max() > result['spread_model'][star].max()

    def test_stampless_sources_get_nan(self):
        """Sources without a residual stamp (e.g. failed fits with NaN
        positions/fluxes) get NaN quality metrics, not the misleading zeros
        their empty stamps would produce."""
        import photutils.psf
        from astropy.table import Table
        from stdpipe import photometry_psf

        img = self._make_test_image([(50, 50, 3.0, 8e3)], size=100)
        sigma = 3.0 / (2 * np.sqrt(2 * np.log(2)))
        psf_model = photutils.psf.CircularGaussianSigmaPRF(sigma=sigma)
        phot_obj = photutils.psf.PSFPhotometry(
            psf_model=psf_model, fit_shape=15, aperture_radius=7.5
        )
        phot_obj(img, init_params=Table({'x': [50.0], 'y': [50.0]}))

        err = np.full_like(img, 5.0, dtype='f4')
        quality = photometry_psf._compute_psf_quality_columns(
            phot_obj,
            psf_model,
            img,
            err,
            None,
            np.array([50.0, np.nan]),
            np.array([50.0, np.nan]),
            np.array([8e3, np.nan]),
            15,
            3.0,
            lambda *a, **k: None,
        )
        assert quality is not None
        for key in ('qf', 'fracflux', 'spread_model', 'dspread_model'):
            assert np.isfinite(quality[key][0])
            assert np.isnan(quality[key][1])

    def test_position_dependent_mode_columns_are_nan(self):
        """Position-dependent PSF mode does not compute the metrics but keeps
        the output schema consistent by adding NaN columns."""
        from stdpipe import photometry, photometry_psf

        img = self._make_test_image([(50, 50, 3.0, 8e3), (150, 150, 3.0, 8e3)])
        obj = photometry.get_objects_sep(img, thresh=5.0, aper=3.0, fwhm=3.0)

        # Minimal degree-1 PSFEx-like dict (constant Gaussian term, zero
        # linear terms) to trigger the position-dependent path
        size = 25
        sigma = 3.0 / (2 * np.sqrt(2 * np.log(2)))
        yy, xx = np.mgrid[0:size, 0:size]
        data = np.zeros((3, size, size))
        data[0] = np.exp(-((xx - size // 2) ** 2 + (yy - size // 2) ** 2) / (2 * sigma**2))
        data[0] /= data[0].sum()
        psf_dict = {
            'data': data,
            'width': size,
            'height': size,
            'sampling': 1.0,
            'degree': 1,
            'ncoeffs': 3,
            'x0': 100.0,
            'y0': 100.0,
            'sx': 100.0,
            'sy': 100.0,
            'fwhm': 3.0,
            'type': 'psfex',
        }

        result = photometry_psf.measure_objects_psf(
            obj, img, psf=psf_dict, use_position_dependent_psf=True, verbose=False
        )
        for col in ('qf', 'fracflux', 'spread_model', 'dspread_model'):
            assert col in result.colnames
            assert np.all(np.isnan(result[col]))
