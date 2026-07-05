from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


def _tukey_bisquare(u: np.ndarray) -> np.ndarray:
    """
    Tukey bisquare weights for standardized residuals u = r / (c * s).
    Returns weights in [0,1].
    """
    w = np.zeros_like(u, dtype=float)
    m = np.abs(u) < 1.0
    t = 1.0 - u[m] ** 2
    w[m] = t**2
    return w


def _mad_sigma(x: np.ndarray) -> float:
    """Robust sigma estimate via MAD (consistent for normal)."""
    x = np.asarray(x)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad + 1e-12


@dataclass
class ApproxLoessRegressor:
    """
    Approximate LOESS/LOWESS via kNN local linear regression in D dimensions.

    Supports:
    - adaptive bandwidth (h = dist to k-th neighbor)
    - Gaussian kernel
    - robust IRLS using Tukey bisquare (weights computed once during fit
      via leave-one-out self-prediction at the training points)

    Samples with non-finite positions, values or weights are dropped during
    fit; queries with non-finite positions predict NaN.

    Typical use: model smooth trend y = f(x, y, mag) and subtract.
    """

    k: int = 300
    scales: tuple[float, ...] | None = None  # per-dimension scaling for distance metric
    kernel: str = "gaussian"  # currently only gaussian
    robust_iters: int = 2
    robust_c: float = 4.685  # Tukey tuning constant
    min_bandwidth: float = 1e-6
    ridge: float = 1e-10  # tiny Tikhonov for numerical stability
    leaf_size: int = 40
    n_jobs: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        """
        Fit stores training set and builds NN index.
        X: (N, D)
        y: (N,)
        sample_weight: optional base weights (e.g., inverse variance). Robust weights are applied on top.
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be (N, D)")
        if y.shape[0] != X.shape[0]:
            raise ValueError("y must have length N")

        D = X.shape[1]
        if self.scales is None:
            scales = np.ones(D, dtype=float)
        else:
            scales = np.asarray(self.scales, float)
            if len(scales) != D:
                raise ValueError(f"scales must have length D={D}")

        base_w = (
            np.ones_like(y)
            if sample_weight is None
            else np.asarray(sample_weight, float).reshape(-1)
        )
        if base_w.shape[0] != X.shape[0]:
            raise ValueError("sample_weight must have length N")
        if np.any(base_w < 0):
            raise ValueError("sample_weight must be non-negative")

        # Drop samples with non-finite positions, values or weights; the NN
        # index cannot handle them, and they carry no usable information
        good = np.all(np.isfinite(X), axis=1) & np.isfinite(y) & np.isfinite(base_w)
        if not np.all(good):
            X, y, base_w = X[good], y[good], base_w[good]
        if X.shape[0] == 0:
            raise ValueError("No finite samples to fit")

        self.X_ = X
        self.y_ = y
        self.base_w_ = base_w
        self.scales_ = scales
        self.Xs_ = self.X_ / self.scales_  # scaled for distance computations
        self.n_samples_ = self.X_.shape[0]
        self.k_ = min(self.k, self.n_samples_)

        self.nn_ = NearestNeighbors(
            n_neighbors=self.k_,
            algorithm="auto",
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
        )
        self.nn_.fit(self.Xs_)

        # Robust IRLS weights from leave-one-out self-prediction at the
        # training points, to downweight artifacts in the training set.
        # They depend on the training data only, so they are computed once
        # here rather than on every predict() call
        robust_w = np.ones_like(self.y_)
        for _ in range(self.robust_iters):
            yfit = self._predict_core(self.X_, self.Xs_, robust_w, exclude_self=True)
            resid = self.y_ - yfit
            # Residual scale from points actually contributing to the fit
            weighted = self.base_w_ > 0
            s = _mad_sigma(resid[weighted] if np.any(weighted) else resid)
            u = resid / (self.robust_c * s)
            robust_w = _tukey_bisquare(u)
            # Keep points whose self-prediction failed instead of treating
            # them as outliers
            robust_w[~np.isfinite(resid)] = 1.0
        self.robust_w_ = robust_w

        return self

    def predict(self, Xq: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """
        Predict yhat for query points Xq using local linear LOESS.

        Queries with non-finite coordinates return NaN. Robust weights for
        the training points were pre-computed during :meth:`fit`.
        """
        Xq = np.asarray(Xq, float)
        if Xq.ndim != 2:
            raise ValueError("Xq must be (M, D)")
        if Xq.shape[1] != self.X_.shape[1]:
            raise ValueError("Xq must have same D as training X")

        yhat = np.full(Xq.shape[0], np.nan, dtype=float)

        # Non-finite queries would crash the NN search; predict NaN for them
        good = np.all(np.isfinite(Xq), axis=1)
        if np.any(good):
            Xg = Xq[good]
            yhat[good] = self._predict_core(
                Xg, Xg / self.scales_, self.robust_w_, chunk=chunk
            )

        return yhat

    def _predict_core(
        self,
        Xq: np.ndarray,
        Xqs: np.ndarray,
        robust_w_train: np.ndarray,
        chunk: int = 4096,
        exclude_self: bool = False,
    ) -> np.ndarray:
        """
        Core prediction with fixed robust weights for training points.
        Uses local linear regression with weights = kernel(distance/h) * base_w * robust_w.
        If exclude_self is True, the nearest self-neighbor is removed where possible.
        """
        M, D = Xq.shape
        out = np.empty(M, float)

        # Query neighbors
        for start in range(0, M, chunk):
            end = min(M, start + chunk)
            k_base = self.k_
            k_query = k_base + 1 if exclude_self and self.n_samples_ > k_base else k_base
            dists, idxs = self.nn_.kneighbors(
                Xqs[start:end], n_neighbors=k_query, return_distance=True
            )  # (m, k_query)
            if exclude_self and k_query > k_base:
                # Drop one zero-distance (self) neighbour per row where present
                has_self = (dists[:, 0] == 0.0)[:, None]
                dists = np.where(has_self, dists[:, 1:], dists[:, :k_base])
                idxs = np.where(has_self, idxs[:, 1:], idxs[:, :k_base])
            # adaptive bandwidth per query point: h = distance to furthest neighbor
            h = np.maximum(dists[:, -1], self.min_bandwidth)  # (m,)
            # kernel weights
            if self.kernel != "gaussian":
                raise NotImplementedError("Only gaussian kernel is implemented")

            # Gaussian kernel with adaptive bandwidth:
            # w_kernel = exp(-0.5*(d/h)^2)
            z = dists / h[:, None]
            w_kernel = np.exp(-0.5 * z * z)
            if exclude_self and k_query == k_base:
                # k covers the whole training set; zero out one zero-distance
                # (self) neighbour per row, consistent with the branch above
                w_kernel[dists[:, 0] == 0.0, 0] = 0.0

            # Pull neighbor values
            Yn = self.y_[idxs]  # (m, k)
            # Combine weights: base * robust * kernel
            w = w_kernel * (self.base_w_[idxs] * robust_w_train[idxs])  # (m, k)

            # Where base/robust weights rejected every neighbour, fall back
            # to plain kernel weights instead of silently predicting 0 from
            # the ridge-only system; if even those are all zero, predict NaN
            rejected = ~(np.sum(w, axis=1) > 0)
            if np.any(rejected):
                w[rejected] = w_kernel[rejected]
            unusable = ~(np.sum(w, axis=1) > 0)

            # Local linear regression around each query:
            # y ≈ b0 + b1*(x-xq) + b2*(y-yq) + b3*(mag-magq) ...
            # Design matrix per query: A = [1, dX] with shape (k, 1+D)
            Xn = self.X_[idxs]  # (m, k, D)
            dX = Xn - Xq[start:end, None, :]  # (m, k, D)

            # Build weighted normal equations per query:
            # Beta = (A^T W A + ridge*I)^(-1) (A^T W y)
            # where A = [1, dX]
            m = end - start
            P = 1 + D
            # A: (m, k, P)
            A = np.empty((m, k_base, P), float)
            A[:, :, 0] = 1.0
            A[:, :, 1:] = dX

            # Compute ATA and ATy with vectorized einsum
            # Apply weights by multiplying rows of A and y by sqrt(w)
            sw = np.sqrt(np.maximum(w, 0.0))
            Aw = A * sw[:, :, None]  # (m, k, P)
            yw = Yn * sw  # (m, k)

            ATA = np.einsum("mkp,mkq->mpq", Aw, Aw)  # (m, P, P)
            ATy = np.einsum("mkp,mk->mp", Aw, yw)  # (m, P)

            # Ridge for numerical stability (especially in sparse regions);
            # ATA + ridge*I is symmetric positive definite, so the batched
            # solve cannot encounter an exactly singular system
            ATA[:, range(P), range(P)] += self.ridge

            # Batched solve of the per-query small linear systems.
            # We need b0 only (intercept) because dX=0 at query point.
            b0 = np.linalg.solve(ATA, ATy[:, :, None])[:, 0, 0]
            b0[unusable] = np.nan
            out[start:end] = b0

        return out


def _fit_loess_field_2d(x, y, dx, dy, scales, k, **kwargs):
    pos = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    model_dx = ApproxLoessRegressor(k=k, scales=scales, **kwargs)
    model_dx.fit(pos, np.asarray(dx, float))
    if dy is None:
        def predict(xq, yq):
            q = np.column_stack([np.asarray(xq, float), np.asarray(yq, float)])
            return model_dx.predict(q)
        return predict
    model_dy = ApproxLoessRegressor(k=k, scales=scales, **kwargs)
    model_dy.fit(pos, np.asarray(dy, float))
    def predict(xq, yq):
        q = np.column_stack([np.asarray(xq, float), np.asarray(yq, float)])
        return model_dx.predict(q), model_dy.predict(q)
    return predict


def _fill_grid_nearest(values, valid, x_centers, y_centers):
    filled = np.array(values, copy=True)
    if np.all(valid):
        return filled
    yy, xx = np.meshgrid(y_centers, x_centers, indexing="ij")
    pts_valid = np.c_[xx[valid], yy[valid]]
    pts_missing = np.c_[xx[~valid], yy[~valid]]
    tree = cKDTree(pts_valid)
    _, idx = tree.query(pts_missing, k=1)
    filled[~valid] = values[valid][idx]
    return filled


def _smooth_grid(values, weights, sigma):
    if sigma <= 0:
        return values
    num = gaussian_filter(values * weights, sigma=sigma, mode="nearest")
    den = gaussian_filter(weights, sigma=sigma, mode="nearest")
    out = np.array(values, copy=True)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def _fit_grid_one(x, y, vals, x_edges, y_edges, min_per_cell, smooth_sigma):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    vals = np.asarray(vals, float)

    # Non-finite positions would be silently binned into the last cell by
    # digitize+clip, and non-finite values would poison cell medians
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(vals)
    if not np.all(good):
        x, y, vals = x[good], y[good], vals[good]

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    ix = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(y, y_edges) - 1, 0, ny - 1)

    grid = np.full((ny, nx), np.nan, dtype=float)
    counts = np.zeros((ny, nx), dtype=float)
    threshold = max(1, int(min_per_cell))
    valid = None
    while True:
        grid.fill(np.nan)
        counts.fill(0.0)
        # Group sample indices by 1-D bin id, then per-bin nanmedian
        bin_id = iy * nx + ix
        order = np.argsort(bin_id, kind="stable")
        bin_sorted = bin_id[order]
        vals_sorted = vals[order]
        starts = np.searchsorted(bin_sorted, np.arange(ny * nx))
        ends = np.searchsorted(bin_sorted, np.arange(ny * nx) + 1)
        for b, (s, e) in enumerate(zip(starts, ends)):
            if e - s >= threshold:
                jy, jx = divmod(b, nx)
                grid[jy, jx] = np.nanmedian(vals_sorted[s:e])
                counts[jy, jx] = e - s
        valid = np.isfinite(grid) & (counts > 0)
        if np.any(valid) or threshold == 1:
            break
        threshold = max(1, threshold // 2)

    if not np.any(valid):
        raise RuntimeError("No grid cells have any samples")

    grid = _fill_grid_nearest(grid, valid, x_centers, y_centers)
    weights = np.where(valid, counts, 0.0)
    grid = _smooth_grid(grid, weights, smooth_sigma)
    interp = RegularGridInterpolator(
        (y_centers, x_centers), grid,
        bounds_error=False, fill_value=None,
    )
    return interp, grid, counts, threshold


def _fit_grid_field_2d(
    x, y, dx, dy, image_shape, grid_shape, min_per_cell, smooth_sigma,
):
    x = np.asarray(x, float); y = np.asarray(y, float)
    dx = np.asarray(dx, float)
    if grid_shape[0] < 2 or grid_shape[1] < 2:
        # RegularGridInterpolator needs at least 2 points per dimension
        raise ValueError("grid_shape dimensions must be at least 2")
    if image_shape is None:
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            raise ValueError("No finite sample positions")
        H = float(np.max(y[finite])) - float(np.min(y[finite]))
        W = float(np.max(x[finite])) - float(np.min(x[finite]))
        if W <= 0 or H <= 0:
            raise ValueError(
                "Degenerate x/y extent; provide image_shape explicitly"
            )
        x0, y0 = float(np.min(x[finite])), float(np.min(y[finite]))
        x_edges = np.linspace(x0, x0 + W, grid_shape[0] + 1)
        y_edges = np.linspace(y0, y0 + H, grid_shape[1] + 1)
    else:
        x_edges = np.linspace(0.0, image_shape[1], grid_shape[0] + 1)
        y_edges = np.linspace(0.0, image_shape[0], grid_shape[1] + 1)

    interp_dx, _, _, _ = _fit_grid_one(
        x, y, dx, x_edges, y_edges, min_per_cell, smooth_sigma,
    )
    if dy is None:
        def predict(xq, yq):
            pts = np.c_[np.asarray(yq, float), np.asarray(xq, float)]
            return np.asarray(interp_dx(pts), float)
        return predict
    interp_dy, _, _, _ = _fit_grid_one(
        x, y, np.asarray(dy, float), x_edges, y_edges, min_per_cell, smooth_sigma,
    )
    def predict(xq, yq):
        pts = np.c_[np.asarray(yq, float), np.asarray(xq, float)]
        return (np.asarray(interp_dx(pts), float),
                np.asarray(interp_dy(pts), float))
    return predict


def fit_vector_field_2d(
    x: np.ndarray,
    y: np.ndarray,
    dx: np.ndarray,
    dy: Optional[np.ndarray] = None,
    *,
    backend: str = "loess",
    scales: Optional[Tuple[float, float]] = None,
    k: int = 200,
    image_shape: Optional[Tuple[int, int]] = None,
    grid_shape: Tuple[int, int] = (12, 8),
    min_per_cell: int = 6,
    smooth_sigma: float = 1.0,
    **kwargs,
) -> Callable[[np.ndarray, np.ndarray],
              Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """Fit a smooth scalar or vector 2-D field to scattered samples.

    Reconstructs a 2-D field (e.g. astrometric (dx, dy) residuals) from
    per-source positions and per-source measurements and returns a callable
    that evaluates the smoothed field at arbitrary positions.

    Two backends are available with the same return interface:

    * ``backend="loess"`` (default) wraps :class:`ApproxLoessRegressor`.
      High-quality local-linear smoothing with adaptive bandwidth and
      optional robust IRLS. Best when prediction is needed at modest
      numbers of points (a few times the fit size).
    * ``backend="grid"`` bins the samples on a regular ``grid_shape`` grid,
      takes per-cell medians, fills empty cells from the nearest filled
      cell, optionally Gaussian-smooths in cell units, and returns a
      bilinear interpolator. ~600–1000× faster at prediction than LOESS,
      at the cost of cell-scale resolution and blockier output.

    Parameters
    ----------
    x, y : array-like, shape (N,)
        Sample positions.
    dx : array-like, shape (N,)
        Sample values, or first component of a vector field if ``dy`` is
        also provided.
    dy : array-like, shape (N,), optional
        Second component of a vector field. When given, the returned
        ``predict`` callable evaluates both components at once.
    backend : {"loess", "grid"}
        Smoothing backend.
    scales : (sx, sy) tuple, optional
        LOESS only. Per-axis distance scaling forwarded to
        :class:`ApproxLoessRegressor`. Defaults to ``(1.0, 1.0)``.
    k : int
        LOESS only. Neighbour count for each local linear fit.
    image_shape : (H, W) tuple, optional
        Grid only. Image shape used to lay out the grid edges. If omitted,
        the bounding box of the input ``(x, y)`` is used.
    grid_shape : (nx, ny) tuple
        Grid only. Number of cells in x and y.
    min_per_cell : int
        Grid only. Minimum sample count per cell required for a valid
        median; cells below the threshold are filled from the nearest
        valid neighbour. Threshold is halved automatically until at least
        one cell is valid.
    smooth_sigma : float
        Grid only. Gaussian smoothing sigma in cell units applied to the
        gridded medians (count-weighted).
    **kwargs :
        LOESS only. Additional keyword arguments forwarded to
        :class:`ApproxLoessRegressor` (``robust_iters``, ``robust_c``, ...).

    Returns
    -------
    predict : callable
        ``predict(xq, yq)`` returns a single ``ndarray`` for a scalar
        field, or a ``(dx_pred, dy_pred)`` tuple for a vector field.

    Notes
    -----
    Samples with non-finite positions or values are ignored by both
    backends, and queries at non-finite positions return NaN.
    """
    if backend == "loess":
        if scales is None:
            scales = (1.0, 1.0)
        return _fit_loess_field_2d(x, y, dx, dy, scales, k, **kwargs)
    elif backend == "grid":
        if kwargs:
            raise TypeError(
                f"unexpected kwargs for grid backend: {sorted(kwargs)}"
            )
        return _fit_grid_field_2d(
            x, y, dx, dy, image_shape, grid_shape, min_per_cell, smooth_sigma,
        )
    else:
        raise ValueError(f"unknown backend {backend!r}; use 'loess' or 'grid'")
