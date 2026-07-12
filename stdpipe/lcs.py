import sys
import numpy as np
from astropy.stats import mad_std
from astropy.time import Time

from scipy.spatial import cKDTree

from . import astrometry


# Fast-conversion registry for types that cannot be stored directly as
# ndarray chunks (e.g. astropy.time.Time).
#
# Each handler is a callable ``handler(val) -> (storage_values, finalizer)``:
#   * ``storage_values`` is an ndarray (or scalar) of plain numbers
#     (typically float64) that is stored in place of the original value.
#   * ``finalizer`` is called once during column consolidation as
#     ``finalizer(ndarray) -> typed_object`` to rebuild the original
#     container from the accumulated storage values.
#
# To add support for another such type, append a (predicate, handler)
# tuple to ``_FAST_TYPE_HANDLERS``.


def _time_handler(val):
    """Round-trip Time through MJD floats (~150 ms for 3.5 M rows)."""
    scale, fmt = val.scale, val.format

    def finalize(arr):
        # Rows padded with NaN (missing values) become masked Time entries,
        # as Time rejects non-finite plain doubles
        if np.any(np.isnan(arr)):
            arr = np.ma.masked_invalid(arr)
        out = Time(arr, format='mjd', scale=scale)
        if fmt != 'mjd':
            out.format = fmt
        return out

    return val.mjd, finalize


_FAST_TYPE_HANDLERS = [
    (lambda v: isinstance(v, Time), _time_handler),
]


def _fast_storage_handler(val):
    for predicate, handler in _FAST_TYPE_HANDLERS:
        if predicate(val):
            return handler
    return None


def _vector_length(val):
    """Length of a vector-like value, or None for scalars, strings and None."""
    if val is None or isinstance(val, str):
        return None
    try:
        return len(val)
    except TypeError:
        # Objects like scalar Time define __len__ but raise on scalars
        return None


class _PadChunk:
    """
    Placeholder for rows without a value for a given key.

    The actual fill value and dtype are decided lazily during column
    consolidation, once the dtype of the real chunks is known: NaN for
    floating-point columns (including fast-converted types like Time),
    None (object dtype) otherwise.
    """

    __slots__ = ('n',)

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class LCs:
    """
    Container for light-curve data vectors with spatial clustering utilities.

    Stores user-provided per-detection vectors (e.g., ra/dec/flux/time) and
    groups detections into spatial clusters using a KDTree radius search.
    Clustering returns per-cluster centroids and member indices in `self.lcs`.

    Notes
    -----
    - `add()` broadcasts scalars to the length of the vector inputs, which
      must all share the same length. Keys omitted from a call, and rows
      preceding the first appearance of a new key, are padded so that all
      stored vectors stay aligned: with NaN for floating-point columns
      (missing Time entries become masked), with None (object dtype)
      otherwise.
    - Data vectors are stored as per-key lists of ndarray chunks and
      consolidated lazily into single arrays on first attribute access.
    - `cluster()` refines centroids and can call an `analyze(self, ids)` callback
      per cluster.
    - Coordinate jitter is applied when building the KDTree to avoid degeneracy
      from repeated positions.
    - Clustering results are stored in `self.lcs` with keys:
      - `x`, `y`, `z`: centroid unit-vector coordinates.
      - `ra`, `dec`: centroid sky coordinates in degrees.
      - `N`: number of points per cluster.
      - `ids`: list of index arrays for member points in the container.
      - `kd`: KDTree built from centroid vectors for fast queries.
    """

    def __init__(self):
        # Storage for user-supplied data vectors: key -> list of ndarray chunks
        self._params = {}
        # Cache of consolidated (concatenated and finalized) columns
        self._columns = {}
        # Per-key finalizer callables for keys whose values were stored via
        # the fast-conversion path (see ``_FAST_TYPE_HANDLERS``). Applied
        # during consolidation to restore the original type.
        self._finalizers = {}
        # Total number of stored rows
        self._length = 0
        # Data version, and the (version, col_ra, col_dec) the KDTree was built from
        self._version = 0
        self._kd_key = None

        self.lcs = None
        self.kd = None

    def __getattr__(self, name):
        # Allows direct access to stored data vectors. Underscored names never
        # resolve here, so the `self._params` access below cannot recurse when
        # the instance dict is not yet populated (e.g. during unpickling).
        if name.startswith('_'):
            raise AttributeError(name)

        if name in self._params:
            return self._get_column(name)

        raise AttributeError(
            "'%s' object has no attribute '%s'" % (self.__class__.__name__, name)
        )

    def __dir__(self):
        # For auto-completion of stored data vectors names
        return (
            list(self.__dict__.keys())
            + list(self.__class__.__dict__.keys())
            + list(self._params.keys())
        )

    def _materialize_pads(self, name, chunks):
        """Convert _PadChunk placeholders to arrays of NaN or None."""
        real = [_ for _ in chunks if not isinstance(_, _PadChunk)]
        dtype = np.result_type(*real) if real else None

        if (dtype is not None and dtype.kind in 'fc') or (
            dtype is None and name in self._finalizers
        ):
            # Floating-point columns (including fast-converted types whose
            # storage is numeric) can hold NaN without dtype degradation
            fill = np.nan
            fill_dtype = dtype if dtype is not None else np.float64
        else:
            fill = None
            fill_dtype = object

        return [
            np.full(len(_), fill, dtype=fill_dtype) if isinstance(_, _PadChunk) else _
            for _ in chunks
        ]

    def _get_column(self, name):
        """Return the consolidated (concatenated and finalized) vector for a key."""
        if name not in self._columns:
            chunks = self._params[name]

            if any(isinstance(_, _PadChunk) for _ in chunks):
                chunks = self._materialize_pads(name, chunks)
                self._params[name] = chunks

            if not chunks:
                arr = np.array([])
            elif len(chunks) == 1:
                arr = np.asanyarray(chunks[0])
            else:
                arr = np.concatenate(chunks)
                # Keep the single consolidated chunk so repeated access is cheap
                self._params[name] = [arr]

            finalizer = self._finalizers.get(name)
            if finalizer is not None:
                arr = finalizer(arr)

            self._columns[name] = arr

        return self._columns[name]

    def add(self, **kwargs):
        """
        Add per-detection vectors to the container.

        Each keyword defines a stored vector. Scalars are broadcast to the
        length of the vector inputs, which must all share the same length
        (ValueError is raised otherwise). This method may be called repeatedly
        to append new chunks of measurements (e.g., per-image batches) to the
        existing vectors. Previously stored keys omitted from a call, as well
        as rows preceding the first appearance of a new key, are padded so
        that all stored vectors stay aligned - with NaN for floating-point
        columns (missing Time entries become masked), with None otherwise.

        Examples
        --------
        >>> lcs = LCs()
        >>> lcs.add(ra=[1, 2], dec=[3, 4], flux=10.0)
        """

        # All vector inputs must agree on the chunk length
        vec_lengths = {}
        for key, val in kwargs.items():
            n = _vector_length(val)
            if n is not None:
                vec_lengths[key] = n

        if len(set(vec_lengths.values())) > 1:
            raise ValueError(
                'Mismatched vector lengths in add(): '
                + ', '.join('%s has %d' % _ for _ in vec_lengths.items())
            )

        length = next(iter(vec_lengths.values()), 0)

        for key in set(kwargs) | set(self._params):
            if key not in self._params:
                # New key: backfill rows stored before it first appeared
                self._params[key] = (
                    [_PadChunk(self._length)] if self._length else []
                )

            val = kwargs.get(key)

            # Types like astropy.time.Time cannot be stored as plain ndarray
            # chunks; convert them to numeric storage values and remember the
            # finalizer that restores the original type during consolidation
            handler = _fast_storage_handler(val)
            if handler is not None:
                val, finalizer = handler(val)
                self._finalizers.setdefault(key, finalizer)

            if _vector_length(val) is not None:
                chunk = np.asanyarray(val)
            elif val is not None:
                # Broadcast scalar to the common vector length
                chunk = np.full(length, val)
            else:
                # Key omitted from this call, or explicit None
                chunk = _PadChunk(length)

            if len(chunk):
                self._params[key].append(chunk)

        self._length += length
        # Invalidate consolidated columns and KDTree built from previous data
        self._columns = {}
        self._version += 1

    def cluster(
        self,
        sr=1 / 3600,
        min_length=None,
        col_ra='ra',
        col_dec='dec',
        verbose=True,
        analyze=None,
        N=1000,
        max_refine_iter=1,
        rng=0,
    ):
        """
        Spatially cluster the data vectors using ra/dec values stored in `col_ra` and `col_dec`.

        Parameters
        ----------
        sr : float, optional
            Clustering radius in degrees.
        min_length : int or None, optional
            Minimum number of points required to keep a cluster.
        col_ra : str, optional
            Name of the RA column in stored vectors.
        col_dec : str, optional
            Name of the Dec column in stored vectors.
        verbose : bool or callable, optional
            Logging control, can be a print-like function.
        analyze : callable or None, optional
            Optional callback `analyze(self, ids)` called per accepted cluster.
            Any returned mapping entries are appended into `self.lcs` under their
            respective keys (one entry per cluster). The callback may also
            return None to skip reporting for a given cluster.
        N : int, optional
            Progress update interval in points.
        max_refine_iter : int, optional
            Maximum number of centroid refinement iterations (default 1).
        rng : int, numpy.random.Generator, or None, optional
            Seed or generator for the coordinate jitter used to break KDTree
            degeneracies from repeated positions. The default fixed seed makes
            clustering deterministic; pass None for non-deterministic jitter.

        Notes
        -----
        Clustering is greedy in storage order: every not-yet-masked point seeds
        a radius search, and all points within the refined cluster radius are
        excluded from seeding afterwards. Such points may still be claimed as
        members of later clusters, so nearby clusters can share points. For
        point distributions wider than `sr`, the exact set of clusters may
        depend on the ordering of the stored points.
        """

        log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

        if min_length is None:
            min_length = 0

        # Clustering radius as 3-D chord length corresponding to the angular radius
        sr0 = 2 * np.sin(np.deg2rad(sr) / 2)

        for col in (col_ra, col_dec):
            if col not in self._params:
                raise KeyError("Column '%s' is not stored in the container" % col)

        kd_key = (self._version, col_ra, col_dec)
        if self._kd_key != kd_key:
            log('Building positional KDTree')

            self._xarr, self._yarr, self._zarr = astrometry.radectoxyz(
                self._get_column(col_ra), self._get_column(col_dec)
            )
            # Add some additional jitter to coordinates, or KDTree may hang on repeating positions
            gen = np.random.default_rng(rng)
            self._xarr = gen.normal(self._xarr, 0.01 / 206265)
            self._yarr = gen.normal(self._yarr, 0.01 / 206265)
            self._zarr = gen.normal(self._zarr, 0.01 / 206265)
            self.kd = cKDTree(np.array([self._xarr, self._yarr, self._zarr]).T)
            self._kd_key = kd_key

        def refine_pos(x, y, z):
            """Returns mean position for a list of individual positions"""
            x1, y1, z1 = [np.mean(_) for _ in [x, y, z]]

            # Normalize back to unit sphere
            r = np.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
            x1, y1, z1 = [_ / r for _ in [x1, y1, z1]]

            return x1, y1, z1

        xarr = self._xarr
        yarr = self._yarr
        zarr = self._zarr
        kd = self.kd

        vmask = np.zeros(len(xarr), bool)

        self.lcs = {'x': [], 'y': [], 'z': [], 'N': [], 'ids': []}
        lcs_x = self.lcs['x']
        lcs_y = self.lcs['y']
        lcs_z = self.lcs['z']
        lcs_n = self.lcs['N']
        lcs_ids = self.lcs['ids']

        log(
            'Starting spatial clustering of %d points with %.1f arcsec radius'
            % (len(vmask), sr * 3600)
        )

        def process_seed(i):
            # Select points around seed position
            ids = kd.query_ball_point([xarr[i], yarr[i], zarr[i]], sr0)

            if len(ids) < min_length:
                vmask[ids] = True
                return

            x1, y1, z1 = refine_pos(xarr[ids], yarr[ids], zarr[ids])
            ids = kd.query_ball_point([x1, y1, z1], sr0)

            for _ in range(max_refine_iter - 1):
                x2, y2, z2 = refine_pos(xarr[ids], yarr[ids], zarr[ids])
                ids2 = kd.query_ball_point([x2, y2, z2], sr0)
                if set(ids2) == set(ids):
                    x1, y1, z1 = x2, y2, z2
                    break
                ids = ids2
                x1, y1, z1 = x2, y2, z2

            vmask[ids] = True  # Mask all points around mean position

            if len(ids) >= min_length:
                # Actual processing of points
                lcs_x.append(x1)
                lcs_y.append(y1)
                lcs_z.append(z1)
                lcs_n.append(len(ids))
                lcs_ids.append(ids)

                if analyze is not None and callable(analyze):
                    ares = analyze(self, ids)

                    if ares:
                        for _, __ in ares.items():
                            if _ not in self.lcs:
                                self.lcs[_] = []
                            self.lcs[_].append(__)

        for i in range(len(vmask)):
            if not vmask[i]:
                process_seed(i)

            if i % N == 0:
                if verbose is True:
                    # Interactive in-place progress for plain print mode
                    sys.stdout.write("\r %d points - %d lcs" % (i, len(self.lcs['x'])))
                    sys.stdout.flush()
                elif callable(verbose):
                    log("%d points - %d lcs" % (i, len(self.lcs['x'])))

        if verbose is True:
            sys.stdout.write("\n")
            sys.stdout.flush()

        for _ in self.lcs.keys():
            if isinstance(self.lcs[_], list) and _ not in ['ids']:
                self.lcs[_] = np.array(self.lcs[_])

        self.lcs['ra'], self.lcs['dec'] = astrometry.xyztoradec(
            [self.lcs['x'], self.lcs['y'], self.lcs['z']]
        )
        self.lcs['kd'] = cKDTree(np.array([self.lcs['x'], self.lcs['y'], self.lcs['z']]).T)

        log('%d spatial clusters isolated' % len(self.lcs['ra']))
