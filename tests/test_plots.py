import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_context
import pytest

from astropy.visualization import ImageNormalize
from astropy.visualization.stretch import HistEqStretch

from stdpipe import plots


def test_imshow_respects_rcparams_origin_lower():
    image = np.arange(9).reshape(3, 3)

    with rc_context({'image.origin': 'lower'}):
        fig, ax = plt.subplots()
        try:
            img = plots.imshow(image, show_colorbar=False, show_axis=False, ax=ax)
            assert img.get_extent() == [-0.5, 2.5, -0.5, 2.5]
        finally:
            plt.close(fig)


def test_imshow_explicit_origin_overrides_rcparams():
    image = np.arange(9).reshape(3, 3)

    with rc_context({'image.origin': 'lower'}):
        fig, ax = plt.subplots()
        try:
            img = plots.imshow(
                image, show_colorbar=False, show_axis=False, ax=ax, origin='upper'
            )
            assert img.get_extent() == [-0.5, 2.5, 2.5, -0.5]
        finally:
            plt.close(fig)


def test_imshow_respects_xlim_ylim_extent():
    image = np.arange(100).reshape(10, 10)

    fig, ax = plt.subplots()
    try:
        img = plots.imshow(
            image, show_colorbar=False, show_axis=False, ax=ax, xlim=(2, 5), ylim=(3, 7)
        )
        assert img.get_extent() == [1.5, 5.5, 7.5, 2.5]
    finally:
        plt.close(fig)


def test_imshow_downscale_preserves_extent():
    image = np.arange(100).reshape(10, 10)

    fig, ax = plt.subplots()
    try:
        img = plots.imshow(
            image, show_colorbar=False, show_axis=False, ax=ax, max_plot_size=5
        )
        assert img.get_extent() == [-0.5, 9.5, 9.5, -0.5]
    finally:
        plt.close(fig)


def test_imshow_does_not_override_explicit_extent():
    image = np.arange(9).reshape(3, 3)
    extent = [1, 2, 3, 4]

    fig, ax = plt.subplots()
    try:
        img = plots.imshow(
            image,
            show_colorbar=False,
            show_axis=False,
            ax=ax,
            extent=extent,
            origin='lower',
        )
        assert img.get_extent() == extent
    finally:
        plt.close(fig)


def test_imshow_show_axis_toggle():
    image = np.arange(9).reshape(3, 3)

    fig, ax = plt.subplots()
    try:
        plots.imshow(image, show_colorbar=False, show_axis=False, ax=ax)
        assert not ax.axison
    finally:
        plt.close(fig)


def test_imshow_show_colorbar_toggle():
    image = np.arange(9).reshape(3, 3)

    fig, ax = plt.subplots()
    try:
        plots.imshow(image, show_colorbar=False, show_axis=False, ax=ax)
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_imshow_interpolation_heuristic():
    small = np.arange(100).reshape(10, 10)
    large = np.arange(160000).reshape(400, 400)

    fig, ax = plt.subplots()
    try:
        img_small = plots.imshow(small, show_colorbar=False, show_axis=False, ax=ax)
        assert img_small.get_interpolation() == 'nearest'
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        img_large = plots.imshow(large, show_colorbar=False, show_axis=False, ax=ax)
        assert img_large.get_interpolation() == 'bicubic'
    finally:
        plt.close(fig)


def test_imshow_qq_overrides_vmin_vmax():
    image = np.arange(9).reshape(3, 3)

    fig, ax = plt.subplots()
    try:
        img = plots.imshow(
            image,
            show_colorbar=False,
            show_axis=False,
            ax=ax,
            qq=[0, 100],
            vmin=2,
            vmax=3,
        )
        assert img.get_clim() == (0.0, 8.0)
    finally:
        plt.close(fig)


def test_imshow_histeq_sets_norm():
    image = np.arange(9).reshape(3, 3).astype(float)

    fig, ax = plt.subplots()
    try:
        img = plots.imshow(
            image,
            show_colorbar=False,
            show_axis=False,
            ax=ax,
            stretch='histeq',
            vmin=1,
            vmax=7,
        )
        assert isinstance(img.norm, ImageNormalize)
        assert isinstance(img.norm.stretch, HistEqStretch)
        assert img.norm.vmin == 1
        assert img.norm.vmax == 7
    finally:
        plt.close(fig)


@pytest.fixture
def rgb_image():
    """Three-color image with wildly different levels in every channel"""
    rng = np.random.default_rng(42)

    return np.stack(
        [rng.normal(level, 0.1 * level, (32, 32)) for level in [10, 100, 1000]],
        axis=-1,
    ).astype(np.float32)


def test_imshow_rgb_normalizes_channels_independently(rgb_image):
    fig, ax = plt.subplots()
    try:
        img = plots.imshow(rgb_image, show_axis=False, ax=ax)
        data = np.asarray(img.get_array())

        assert data.shape == rgb_image.shape
        # Matplotlib ignores the normalization for color data, so it has to be
        # applied to the data itself, and independently for every channel
        for i in range(3):
            assert data[..., i].min() == pytest.approx(0, abs=1e-6)
            assert data[..., i].max() == pytest.approx(1, abs=1e-6)
    finally:
        plt.close(fig)


def test_imshow_rgb_respects_cuts(rgb_image):
    fig, ax = plt.subplots()
    try:
        # Scalar cuts are broadcast onto all the channels, so only the brightest
        # one should reach the top of the 0..1 range
        img = plots.imshow(rgb_image, show_axis=False, ax=ax, vmin=0, vmax=1000)
        data = np.asarray(img.get_array())

        assert data[..., 0].max() < 0.1
        assert data[..., 1].max() < 0.5
        assert data[..., 2].max() == pytest.approx(1, abs=1e-6)

        # Per-channel cuts are applied as is
        img = plots.imshow(
            rgb_image, show_axis=False, ax=ax, vmin=[0, 0, 0], vmax=[20, 200, 2000]
        )
        data = np.asarray(img.get_array())

        for i in range(3):
            assert data[..., i].max() < 0.9
    finally:
        plt.close(fig)


def test_imshow_rgb_drops_colorbar(rgb_image):
    fig, ax = plt.subplots()
    try:
        plots.imshow(rgb_image, show_colorbar=True, show_axis=False, ax=ax)
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    'kwargs',
    [
        {},
        {'stretch': 'asinh'},
        {'stretch': 'histeq'},
        {'r0': 1},
        {'max_plot_size': 16},
        {'xlim': (4, 20), 'ylim': (4, 20)},
        {'qq': [1, 99]},
    ],
)
def test_imshow_rgb_options(rgb_image, kwargs):
    """All the processing options should work on color data, and keep it sane"""
    for mask in [None, np.zeros((32, 32), dtype=bool), np.zeros((32, 32, 3), dtype=bool)]:
        fig, ax = plt.subplots()
        try:
            img = plots.imshow(rgb_image, show_axis=False, ax=ax, mask=mask, **kwargs)
            data = np.asarray(img.get_array())

            assert data.shape[-1] == 3
            assert np.all(np.isfinite(data))
            assert data.min() >= 0 and data.max() <= 1
        finally:
            plt.close(fig)


def test_imshow_rgba_keeps_alpha(rgb_image):
    """Alpha channel should be passed through intact, and not normalized"""
    image = np.concatenate([rgb_image, np.full((32, 32, 1), 0.25, np.float32)], axis=-1)

    fig, ax = plt.subplots()
    try:
        img = plots.imshow(image, show_axis=False, ax=ax)
        data = np.asarray(img.get_array())

        assert data.shape[-1] == 4
        assert np.allclose(data[..., 3], 0.25)
    finally:
        plt.close(fig)


def test_imshow_rgb_integer_data():
    """Integer color data is 0..255 and should survive the round trip intact"""
    image = np.arange(256, dtype=np.uint8)[:, None, None].repeat(4, 1).repeat(3, 2)

    fig, ax = plt.subplots()
    try:
        img = plots.imshow(image, show_axis=False, ax=ax, vmin=0, vmax=255)
        data = np.asarray(img.get_array())

        assert np.allclose(data[:, 0, 0] * 255, np.arange(256), atol=1e-3)
    finally:
        plt.close(fig)


def test_imshow_rgb_non_finite():
    """Non-finite color data should not reach Matplotlib"""
    image = np.full((16, 16, 3), np.nan, dtype=np.float32)
    image[0, 0] = [1.0, 2.0, 3.0]

    fig, ax = plt.subplots()
    try:
        img = plots.imshow(image, show_axis=False, ax=ax)
        assert np.all(np.isfinite(np.asarray(img.get_array())))

        # Degenerate case with nothing to normalize at all
        img = plots.imshow(np.full((16, 16, 3), np.nan), show_axis=False, ax=ax)
        assert np.all(np.isfinite(np.asarray(img.get_array())))
    finally:
        plt.close(fig)


def test_imshow_matches_matplotlib_extent():
    image = np.arange(12).reshape(3, 4)

    with rc_context({'image.origin': 'upper'}):
        fig, ax = plt.subplots()
        try:
            mpl_img = ax.imshow(image)
            mpl_extent = mpl_img.get_extent()
        finally:
            plt.close(fig)


def test_imshow_matches_matplotlib_extent_lower_origin():
    image = np.arange(12).reshape(3, 4)

    with rc_context({'image.origin': 'lower'}):
        fig, ax = plt.subplots()
        try:
            mpl_img = ax.imshow(image)
            mpl_extent = mpl_img.get_extent()
        finally:
            plt.close(fig)

        fig, ax = plt.subplots()
        try:
            std_img = plots.imshow(image, show_colorbar=False, show_axis=False, ax=ax)
            assert std_img.get_extent() == mpl_extent
        finally:
            plt.close(fig)

        fig, ax = plt.subplots()
        try:
            std_img = plots.imshow(image, show_colorbar=False, show_axis=False, ax=ax)
            assert std_img.get_extent() == mpl_extent
        finally:
            plt.close(fig)


def test_adaptive_binned_map_raises_on_no_finite_points():
    x = np.array([np.nan, np.inf])
    y = np.array([0.0, 1.0])
    value = np.array([np.nan, np.inf])

    with pytest.raises(ValueError, match="No finite data points"):
        plots.adaptive_binned_map(x, y, value)


def test_adaptive_binned_map_handles_target_sn_with_zero_err():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    value = np.array([1.0, 2.0, 3.0, 4.0])
    err = np.zeros_like(value)

    fig, ax = plt.subplots()
    try:
        plots.adaptive_binned_map(
            x,
            y,
            value,
            target_sn=5,
            err=err,
            show_colorbar=False,
            show_axis=False,
            ax=ax,
        )
    finally:
        plt.close(fig)
