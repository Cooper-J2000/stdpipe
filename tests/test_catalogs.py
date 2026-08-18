"""
Unit tests for stdpipe.catalogs module.

Tests catalog querying and manipulation utilities.
"""

import pytest
import numpy as np
from astropy.table import Table

from stdpipe import catalogs


class TestCatalogDefinitions:
    """Test catalog configuration and definitions."""

    @pytest.mark.unit
    def test_catalogs_dict_exists(self):
        """Test that catalogs dictionary exists and is populated."""
        assert hasattr(catalogs, 'catalogs')
        assert isinstance(catalogs.catalogs, dict)
        assert len(catalogs.catalogs) > 0

    @pytest.mark.unit
    def test_catalog_ps1_definition(self):
        """Test Pan-STARRS catalog definition."""
        assert 'ps1' in catalogs.catalogs
        ps1 = catalogs.catalogs['ps1']

        assert 'vizier' in ps1
        assert 'name' in ps1
        assert ps1['name'] == 'PanSTARRS DR1'

    @pytest.mark.unit
    def test_catalog_gaia_definitions(self):
        """Test that various Gaia catalogs are defined."""
        gaia_catalogs = ['gaiadr2', 'gaiaedr3', 'gaiadr3syn']

        for cat in gaia_catalogs:
            assert cat in catalogs.catalogs
            assert 'vizier' in catalogs.catalogs[cat]
            assert 'Gaia' in catalogs.catalogs[cat]['name']

    @pytest.mark.unit
    def test_all_catalogs_have_required_fields(self):
        """Test that all catalog definitions have required fields."""
        for cat_name, cat_def in catalogs.catalogs.items():
            assert 'vizier' in cat_def, f"Catalog {cat_name} missing 'vizier'"
            assert 'name' in cat_def, f"Catalog {cat_name} missing 'name'"


class TestCatalogQuerying:
    """Test catalog query utilities (network-dependent)."""

    @pytest.mark.unit
    @pytest.mark.requires_network
    @pytest.mark.slow
    def test_get_cat_vizier_basic(self):
        """Test basic Vizier catalog query."""
        # Query a small region
        ra0 = 180.0
        dec0 = 45.0
        sr0 = 0.1  # 0.1 degree radius

        # Query Pan-STARRS
        cat = catalogs.get_cat_vizier(
            ra0, dec0, sr0,
            catalog='ps1',
            limit=10,
            verbose=False
        )

        # Should return a table
        assert isinstance(cat, Table)

        # Should have some standard columns
        # (exact column names may vary)
        assert len(cat.colnames) > 0

    @pytest.mark.unit
    @pytest.mark.requires_network
    @pytest.mark.slow
    def test_get_cat_vizier_with_filters(self):
        """Test Vizier query with column filters."""
        pytest.skip("Network-dependent test - implement as needed")

    @pytest.mark.unit
    @pytest.mark.requires_network
    @pytest.mark.slow
    def test_get_cat_vizier_different_catalogs(self):
        """Test querying different catalogs."""
        pytest.skip("Network-dependent test - implement as needed")

    @pytest.mark.unit
    def test_get_cat_vizier_invalid_catalog(self):
        """Test behavior with invalid catalog name."""
        # Should either raise error or return empty result
        # depending on implementation
        pytest.skip("Need to check error handling behavior")


class TestCatalogProcessing:
    """Test catalog processing and augmentation utilities."""

    @pytest.mark.unit
    def test_catalog_magnitude_conversions(self):
        """Test magnitude conversion utilities if they exist."""
        # The catalogs module augments some catalogs with
        # converted magnitudes (e.g., PS1 to Johnson-Cousins)
        # This would test those conversion functions
        pytest.skip("Magnitude conversion tests - implement based on actual functions")

    @pytest.mark.unit
    def test_catalog_coordinate_matching(self):
        """Test catalog cross-matching utilities."""
        # If there are cross-matching utilities in the module
        pytest.skip("Cross-matching tests - implement based on actual functions")


# ============================================================================
# Mock tests (don't require network)
# ============================================================================

class TestCatalogUtilitiesMocked:
    """Test catalog utilities with mocked data."""

    @pytest.mark.unit
    def test_catalog_result_processing(self):
        """Test processing of catalog query results."""
        # Create a mock catalog result
        mock_cat = Table()
        mock_cat['RAJ2000'] = [180.0, 180.1, 180.2]
        mock_cat['DEJ2000'] = [45.0, 45.1, 45.2]
        mock_cat['gmag'] = [15.0, 16.0, 14.5]
        mock_cat['rmag'] = [14.5, 15.5, 14.0]

        # Test any processing functions that might exist
        # (This is a placeholder - implement based on actual functions)
        assert len(mock_cat) == 3
        assert 'RAJ2000' in mock_cat.colnames


class TestCatalogAugmentation:
    """Test catalog augmentation functionality."""

    @pytest.mark.unit
    def test_detect_catalog_type_ps1(self):
        """Test auto-detection of Pan-STARRS catalogs."""
        cat = Table()
        cat['gmag'] = [15.0]
        cat['rmag'] = [14.5]
        cat['imag'] = [14.0]
        cat['zmag'] = [13.8]
        cat['ymag'] = [13.7]

        detected = catalogs._detect_catalog_type(cat)
        assert detected == 'ps1'

    @pytest.mark.unit
    def test_detect_catalog_type_gaiadr2(self):
        """Test auto-detection of Gaia DR2 catalogs."""
        cat = Table()
        cat['Gmag'] = [15.0]
        cat['BPmag'] = [15.5]
        cat['RPmag'] = [14.5]

        detected = catalogs._detect_catalog_type(cat)
        assert detected == 'gaiadr2'

    @pytest.mark.unit
    def test_detect_catalog_type_skymapper(self):
        """Test auto-detection of SkyMapper catalogs."""
        cat = Table()
        cat['gPSF'] = [15.0]
        cat['rPSF'] = [14.5]
        cat['iPSF'] = [14.0]
        cat['zPSF'] = [13.8]

        detected = catalogs._detect_catalog_type(cat)
        assert detected == 'skymapper'

    @pytest.mark.unit
    def test_detect_catalog_type_apass(self):
        """Test auto-detection of APASS catalogs."""
        cat = Table()
        cat['B_mag'] = [15.0]
        cat['V_mag'] = [14.5]
        cat['g_mag'] = [14.8]
        cat['r_mag'] = [14.3]
        cat['i_mag'] = [14.0]

        detected = catalogs._detect_catalog_type(cat)
        assert detected == 'apass'

    @pytest.mark.unit
    def test_detect_catalog_type_gaiadr3syn(self):
        """Test auto-detection of Gaia DR3 Synthetic catalogs."""
        cat = Table()
        cat['Fu'] = [1e-10]
        cat['Fg'] = [1e-9]
        cat['Fr'] = [1e-9]
        cat['Fi'] = [1e-9]
        cat['Fz'] = [1e-9]

        detected = catalogs._detect_catalog_type(cat)
        assert detected == 'gaiadr3syn'

    @pytest.mark.unit
    def test_detect_catalog_type_sdss(self):
        """Test auto-detection of SDSS catalogs."""
        cat = Table()
        cat['umag'] = [16.0]
        cat['gmag'] = [15.0]
        cat['rmag'] = [14.5]
        cat['imag'] = [14.0]
        cat['zmag'] = [13.8]
        # No ymag - this distinguishes from PS1

        detected = catalogs._detect_catalog_type(cat)
        assert detected == 'sdss'

    @pytest.mark.unit
    def test_detect_catalog_type_unknown(self):
        """Test that unknown catalogs return None."""
        cat = Table()
        cat['random_col1'] = [1.0]
        cat['random_col2'] = [2.0]

        detected = catalogs._detect_catalog_type(cat)
        assert detected is None

    @pytest.mark.unit
    def test_augment_ps1_basic(self):
        """Test PS1 catalog augmentation adds Johnson-Cousins magnitudes."""
        cat = Table()
        cat['gmag'] = np.array([15.0, 16.0])
        cat['rmag'] = np.array([14.5, 15.5])
        cat['imag'] = np.array([14.0, 15.0])
        cat['zmag'] = np.array([13.8, 14.8])
        cat['ymag'] = np.array([13.7, 14.7])
        cat['e_gmag'] = np.array([0.01, 0.02])
        cat['e_rmag'] = np.array([0.01, 0.02])
        cat['e_imag'] = np.array([0.01, 0.02])

        result = catalogs.augment_cat_bands(cat, catalog='ps1', verbose=False)

        # Should have added Johnson-Cousins magnitudes
        assert 'Bmag' in result.colnames
        assert 'Vmag' in result.colnames
        assert 'Rmag' in result.colnames
        assert 'Imag' in result.colnames

        # Should have added aliases
        assert 'B' in result.colnames
        assert 'V' in result.colnames
        assert 'R' in result.colnames
        assert 'I' in result.colnames

        # Should have added SDSS conversions
        assert 'g_SDSS' in result.colnames
        assert 'r_SDSS' in result.colnames

    @pytest.mark.unit
    def test_augment_gaiadr2(self):
        """Test Gaia DR2 catalog augmentation."""
        cat = Table()
        cat['Gmag'] = np.array([15.0, 16.0])
        cat['BPmag'] = np.array([15.5, 16.5])
        cat['RPmag'] = np.array([14.5, 15.5])
        cat['E_BR_RP_'] = np.array([1.2, 1.3])  # Column name as used in code
        cat['e_Gmag'] = np.array([0.01, 0.02])

        result = catalogs.augment_cat_bands(cat, catalog='gaiadr2', verbose=False)

        # Should have added Johnson-Cousins magnitudes
        assert 'Bmag' in result.colnames
        assert 'Vmag' in result.colnames
        assert 'Rmag' in result.colnames
        assert 'Imag' in result.colnames

        # Should have added PS1 conversions
        assert 'gmag' in result.colnames
        assert 'rmag' in result.colnames

    @pytest.mark.unit
    def test_augment_auto_detect(self):
        """Test that auto-detection works when catalog parameter is None."""
        cat = Table()
        cat['gmag'] = np.array([15.0])
        cat['rmag'] = np.array([14.5])
        cat['imag'] = np.array([14.0])
        cat['zmag'] = np.array([13.8])
        cat['ymag'] = np.array([13.7])
        cat['e_gmag'] = np.array([0.01])
        cat['e_rmag'] = np.array([0.01])
        cat['e_imag'] = np.array([0.01])

        # Don't specify catalog - should auto-detect as PS1
        result = catalogs.augment_cat_bands(cat, catalog=None, verbose=False)

        # Should have detected PS1 and augmented accordingly
        assert 'Bmag' in result.colnames
        assert 'Vmag' in result.colnames

    @pytest.mark.unit
    def test_augment_unknown_catalog(self):
        """Test that unknown catalog type is handled gracefully."""
        cat = Table()
        cat['random_mag'] = [15.0]

        # Unknown catalog should return catalog unchanged
        result = catalogs.augment_cat_bands(cat, catalog='unknown_catalog', verbose=False)

        # Should still return the catalog
        assert 'random_mag' in result.colnames
        # Should not crash

    @pytest.mark.unit
    def test_augment_preserves_original_columns(self):
        """Test that augmentation preserves original columns."""
        cat = Table()
        cat['RAJ2000'] = [180.0]
        cat['DEJ2000'] = [45.0]
        cat['gmag'] = [15.0]
        cat['rmag'] = [14.5]
        cat['imag'] = [14.0]
        cat['zmag'] = [13.8]
        cat['ymag'] = [13.7]

        orig_columns = set(cat.colnames)

        result = catalogs.augment_cat_bands(cat, catalog='ps1', verbose=False)

        # All original columns should still be present
        for col in orig_columns:
            assert col in result.colnames

    @pytest.mark.unit
    def test_augment_atlas_uses_ps1(self):
        """Test that ATLAS catalogs use PS1 augmentation."""
        cat = Table()
        cat['gmag'] = np.array([15.0])
        cat['rmag'] = np.array([14.5])
        cat['imag'] = np.array([14.0])
        cat['zmag'] = np.array([13.8])
        cat['ymag'] = np.array([13.7])
        cat['e_gmag'] = np.array([0.01])
        cat['e_rmag'] = np.array([0.01])
        cat['e_imag'] = np.array([0.01])

        # ATLAS should be normalized to PS1
        result = catalogs.augment_cat_bands(cat, catalog='atlas', verbose=False)

        # Should have PS1-style augmentation
        assert 'Bmag' in result.colnames
        assert 'Vmag' in result.colnames

    @pytest.mark.unit
    def test_augment_missing_columns_handled(self):
        """Test that missing columns are handled gracefully."""
        cat = Table()
        cat['gmag'] = [15.0]
        # Missing other required columns

        # Should not crash - helper functions have try/except
        result = catalogs.augment_cat_bands(cat, catalog='ps1', verbose=False)

        # Should still return catalog
        assert 'gmag' in result.colnames


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ============================================================================
# Pan-STARRS DR2 and Legacy Survey DR11 additions
# ============================================================================


class TestPS1DR2:
    """Test the explicit Pan-STARRS DR2 catalog entry."""

    @pytest.mark.unit
    def test_ps1dr2_definition(self):
        """ps1dr2 points to the Vizier DR2 table (II/389, not DR1's II/349)."""
        assert 'ps1dr2' in catalogs.catalogs
        assert catalogs.catalogs['ps1dr2']['vizier'] == 'II/389/ps1_dr2'
        assert catalogs.catalogs['ps1dr2']['name'] == 'PanSTARRS DR2'

    @pytest.mark.unit
    def test_ps1dr2_augmentation(self):
        """augment_cat_bands must treat ps1dr2 like ps1."""
        cat = Table()
        cat['gmag'] = [18.0]
        cat['rmag'] = [17.5]
        cat['imag'] = [17.2]
        cat['zmag'] = [17.0]
        cat['ymag'] = [16.8]
        cat['e_gmag'] = [0.01]
        cat['e_rmag'] = [0.01]
        cat['e_imag'] = [0.01]
        cat['e_zmag'] = [0.01]

        catalogs.augment_cat_bands(cat, catalog='ps1dr2')

        for col in ['Bmag', 'Vmag', 'Rmag', 'Imag', 'e_Bmag', 'g_SDSS', 'good']:
            assert col in cat.colnames


class TestLSDR11:
    """Test Legacy Survey DR11 catalogue retrieval from tractor bricks."""

    @pytest.fixture
    def bricks(self):
        t = Table()
        t['brickname'] = ['0000p000']
        t['ra'] = [0.05]
        t['dec'] = [0.05]
        t['survey'] = ['S']
        return t

    @pytest.fixture
    def brick_table(self):
        t = Table()
        t['brickname'] = ['0000p000'] * 5
        t['brickid'] = [1] * 5
        t['objid'] = [0, 1, 2, 3, 4]
        t['ra'] = [0.05, 0.055, 0.056, 0.057, 1.0]
        t['dec'] = [0.05, 0.05, 0.05, 0.05, 1.0]
        t['type'] = ['PSF'] * 5
        t['brick_primary'] = [True, True, False, True, True]
        t['flux_g'] = [100.0, 10.0, 100.0, 0.0, 100.0]
        t['flux_r'] = [100.0, 10.0, 100.0, 5.0, 100.0]
        t['flux_i'] = [100.0, 10.0, 100.0, 5.0, 100.0]
        t['flux_z'] = [100.0, 10.0, 100.0, 5.0, 100.0]
        t['flux_ivar_g'] = [1e4] * 5
        t['flux_ivar_r'] = [1e4] * 5
        t['flux_ivar_i'] = [1e4] * 5
        t['flux_ivar_z'] = [1e4] * 5
        return t

    def _query(self, bricks, brick_table, monkeypatch, tmp_path, **kwargs):
        monkeypatch.setattr(catalogs, '__lsdr11_bricks', bricks, raising=False)

        def fake_download(url, filename=None, overwrite=False, verbose=False):
            brick_table.write(filename, format='fits')
            return True

        monkeypatch.setattr(catalogs.utils, 'download', fake_download)

        return catalogs.get_cat_lsdr11(
            0.05, 0.05, 0.05, _cachedir=str(tmp_path), **kwargs
        )

    @pytest.mark.unit
    def test_basic_query(self, bricks, brick_table, monkeypatch, tmp_path):
        """brick_primary filtering, cone cut, flux->mag conversion."""
        cat = self._query(bricks, brick_table, monkeypatch, tmp_path)

        # 5 rows: one non-primary, one outside the circle
        assert len(cat) == 3
        assert 'RAJ2000' in cat.colnames and 'DEJ2000' in cat.colnames
        assert 'ra' not in cat.colnames

        # flux=100 nmgy -> mag 17.5; flux=10 nmgy -> mag 20.0
        assert np.isclose(cat['gmag'][0], 17.5)
        assert np.isclose(cat['rmag'][1], 20.0)
        # e_mag = 1.086 / (sqrt(ivar) * flux)
        assert np.isclose(cat['e_gmag'][0], 1.086 / (100.0 * 100.0))
        # zero/negative flux -> NaN magnitude
        assert np.isnan(cat['gmag'][2])

        # Augmented columns
        for col in ['Bmag', 'Vmag', 'Rmag', 'Imag', 'good']:
            assert col in cat.colnames

        assert cat.meta['name'] == 'Legacy Survey DR11'

    @pytest.mark.unit
    def test_brick_cache_reused(self, bricks, brick_table, monkeypatch, tmp_path):
        """Second call must not re-download the brick."""
        cat = self._query(bricks, brick_table, monkeypatch, tmp_path)
        assert len(cat) == 3
        assert len(list(tmp_path.glob('tractor-*.fits'))) == 1

        def failing_download(*args, **kwargs):
            raise AssertionError('download called despite cache')

        monkeypatch.setattr(catalogs.utils, 'download', failing_download)
        cat = self._query(bricks, brick_table, monkeypatch, tmp_path)
        assert len(cat) == 3

    @pytest.mark.unit
    def test_filters_and_limit(self, bricks, brick_table, monkeypatch, tmp_path):
        """Vizier-style filters and row limit are applied client-side."""
        cat = self._query(
            bricks, brick_table, monkeypatch, tmp_path, filters={'rmag': '<18'}
        )
        assert len(cat) == 1
        assert np.isclose(cat['rmag'][0], 17.5)

        cat = self._query(bricks, brick_table, monkeypatch, tmp_path, limit=2)
        assert len(cat) == 2

    @pytest.mark.unit
    def test_no_bricks(self, monkeypatch, tmp_path):
        """Region outside the DR11 footprint returns None."""
        empty = Table()
        empty['brickname'] = ['0000p000']
        empty['ra'] = [180.0]
        empty['dec'] = [-80.0]
        empty['survey'] = ['S']

        monkeypatch.setattr(catalogs, '__lsdr11_bricks', empty, raising=False)

        cat = catalogs.get_cat_lsdr11(0.05, 0.05, 0.01, _cachedir=str(tmp_path))
        assert cat is None

    @pytest.mark.unit
    def test_download_failure_other_region(self, bricks, brick_table, monkeypatch, tmp_path):
        """Failed download from one region falls back to the other."""
        monkeypatch.setattr(catalogs, '__lsdr11_bricks', bricks, raising=False)

        calls = []

        def flaky_download(url, filename=None, overwrite=False, verbose=False):
            calls.append(url)
            if '/dr11/south/' in url:
                return False
            brick_table.write(filename, format='fits')
            return True

        monkeypatch.setattr(catalogs.utils, 'download', flaky_download)

        cat = catalogs.get_cat_lsdr11(0.05, 0.05, 0.05, _cachedir=str(tmp_path))
        assert len(cat) == 3
        assert any('/dr11/south/' in u for u in calls)
        assert any('/dr11/north/' in u for u in calls)

    @pytest.mark.unit
    def test_north_brick_without_iband(self, bricks, monkeypatch, tmp_path):
        """Northern bricks lack flux_i entirely - magnitudes must become NaN."""
        t = Table()
        t['brickname'] = ['0000p000']
        t['brickid'] = [1]
        t['objid'] = [0]
        t['ra'] = [0.05]
        t['dec'] = [0.05]
        t['type'] = ['PSF']
        t['brick_primary'] = [True]
        t['flux_g'] = [100.0]
        t['flux_r'] = [100.0]
        t['flux_z'] = [100.0]
        t['flux_ivar_g'] = [1e4]
        t['flux_ivar_r'] = [1e4]
        t['flux_ivar_z'] = [1e4]
        # no flux_i / flux_ivar_i at all

        cat = self._query(bricks, t, monkeypatch, tmp_path)
        assert len(cat) == 1
        assert np.isclose(cat['gmag'][0], 17.5)
        assert np.isnan(cat['imag'][0])
        assert np.isnan(cat['e_imag'][0])

    @pytest.mark.unit
    def test_corner_bricks_skipped(self, monkeypatch, tmp_path):
        """Bricks whose box does not overlap the circle are not downloaded."""
        t = Table()
        t['brickname'] = ['0000p000', '0000p001']
        t['ra'] = [0.05, 0.25]   # second brick: center 0.2 deg away in RA
        t['dec'] = [0.05, 0.05]
        t['survey'] = ['S', 'S']
        monkeypatch.setattr(catalogs, '__lsdr11_bricks', t, raising=False)

        downloaded = []

        def fake_download(url, filename=None, overwrite=False, verbose=False):
            downloaded.append(url)
            bt = Table()
            bt['ra'] = [0.05]
            bt['dec'] = [0.05]
            bt['brick_primary'] = [True]
            bt['flux_g'] = [100.0]
            bt['flux_ivar_g'] = [1e4]
            bt.write(filename, format='fits')
            return True

        monkeypatch.setattr(catalogs.utils, 'download', fake_download)

        cat = catalogs.get_cat_lsdr11(0.05, 0.05, 0.05, _cachedir=str(tmp_path))
        assert len(cat) == 1
        # 0.2 deg - 0.125 (half brick) = 0.075 > 0.05: no overlap, not downloaded
        assert len(downloaded) == 1
        assert 'tractor-0000p000' in downloaded[0]
