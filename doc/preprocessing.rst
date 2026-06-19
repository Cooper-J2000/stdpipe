Pre-processing the data
=======================

The codes in *STDPipe* expect as an input the *science-ready* images, cleaned as much as possible from instrumental signatures and imaging artefacts. In practice, it means that the image should be

-  bias and dark subtracted
-  flat-fielded.

Also, the artefacts such as saturated stars, bleeding charges, cosmic ray hits etc have to be masked.

All these tasks are outside of *STDPipe* per se, as they are highly instrument and site specific.
On the other hand, they may usually be performed easily using standard Python/NumPy/AstroPy routines and libraries like `Astro-SCRAPPY <https://github.com/astropy/astroscrappy>`_ etc.

E.g. to pre-process the raw image using pre-computed master dark and flat, mask some common problems, and cleanup the cosmic rays you may do something like that:

.. code-block:: python

   image = fits.getdata(filename).astype(np.double)
   header = fits.getheader(filename)

   dark = fits.getdata(darkname)
   flat = fits.getdata(flatname)

   image -= dark
   image *= np.median(flat)/flat

   saturation = header.get('SATURATE') or 50000 # Guess saturation level from FITS header

   mask = np.isnan(image) # mask NaNs in the input image
   mask |= image > saturation # mask saturated pixels
   mask |= flat < 0.5*np.median(flat) # mask underilluminated/vignetted regions

   from astropy.stats import mad_std
   mask |= dark > np.median(dark) + 10.0*mad_std(dark) # mask hotter pixels

   gain = header.get('GAIN') or 1.0 # Guess gain from FITS header

   import astroscrappy
   # mask cosmic rays using LACosmic algorithm
   cmask,cimage = astroscrappy.detect_cosmics(image, mask, gain=gain, verbose=True)
   mask |= cmask


We have a simple routine that implements these steps, :func:`stdpipe.pipeline.make_mask`, which is documented below.

.. autofunction:: stdpipe.pipeline.make_mask
   :noindex:


Removing fringes
----------------

Images taken in red and near-infrared bands with thinned CCDs often show *fringes* - a pattern of curved, wave-like ripples caused by interference of night-sky emission lines inside the detector. The pattern is additive and varies smoothly across the field on scales of tens of pixels, sitting between the size of the stars (a few pixels) and of the large-scale sky background (hundreds of pixels). As such, it is too fine to be captured by the usual grid-based background estimators, and too coarse and irregular to be removed by frequency-domain (Fourier) filtering, as real fringes are typically curved and non-stationary rather than strictly periodic.

*STDPipe* provides :func:`stdpipe.fringe_removal.remove_fringes` that models the fringe pattern directly in image space. It masks the stars and other significant sources, estimates the smooth intermediate-scale residual by mask-aware Gaussian smoothing (so the fringe structure is transparently *inpainted* under the masked positions), and refines this estimate over a few iterations. The resulting fringe map is then subtracted from the image, while the large-scale sky background is left intact.

.. code-block:: python

   from stdpipe import fringe_removal

   # Remove the fringes, automatically selecting the smoothing scale
   cleaned = fringe_removal.remove_fringes(image, mask=mask)

   # Also return the model that was subtracted, e.g. for inspection
   cleaned, fringe_model = fringe_removal.remove_fringes(
       image, mask=mask, get_fringe_model=True)

The smoothing kernel size is the main parameter. By default (``scale='auto'``) it is selected automatically from the measured fringe-to-noise ratio: strong, sharp fringes get a small kernel that follows them closely, while faint, broad fringes get a larger one that averages the pixel noise down instead of leaking it into the model. You may set ``scale`` to a number to fix it manually, and use ``noise_frac`` to trade model cleanliness against fringe-tracking fidelity when using the automatic mode.

.. code-block:: python

   # Fix the smoothing scale manually (pixels)
   cleaned = fringe_removal.remove_fringes(image, mask=mask, scale=10)

   # Make the automatic selection less aggressive (cleaner model, smaller kernel)
   cleaned = fringe_removal.remove_fringes(image, mask=mask, noise_frac=0.4)

Bright stars carry azimuthally symmetric structures - extended halos, ghost reflection rings, and bowls left by the coarse background estimation - that are locally indistinguishable from fringes. By default (``halo_sn=20``) these are modelled with radial profiles around bright stars and removed together with the fringes; pass ``halo_sn=None`` to disable this and keep the faint stellar wings in the image.

.. attention::
   The fringe model is an *additive* correction estimated from the science image itself. It assumes the fringes are faint compared to the stars and vary on scales well above the stellar size. For fields dominated by extended low-surface-brightness sources (large galaxies, nebulosity), part of their flux may be absorbed into the fringe model - mask such regions explicitly via the ``mask`` argument in that case.

The standalone script ``examples/fringe_removal_example.py`` runs the routine on a FITS image and visualises the original image, the reconstructed fringe pattern and the corrected result side by side, exposing all parameters as command-line options - useful for inspecting the correction and tuning it for a given instrument.

.. autofunction:: stdpipe.fringe_removal.remove_fringes
   :noindex:


Stacking the images
-------------------

You may want to stack/coadd or mosaic some images before processing them. While there are dedicated large-scape packages like `Montage <http://montage.ipac.caltech.edu>`_ that handle it *properly*, it still may be done manually with relatively little efforts using e.g. Python `reproject <https://github.com/astropy/reproject>`_ package.

You may check `simple example notebook <https://github.com/karpov-sv/stdpipe/blob/master/notebooks/image_stacking.ipynb>`_
that shows how to do it.

.. attention::
   The stacking modify the statistical properties of resulting image! The reasons are both averaging (or especially median averaging!) of the images that modify effective gain value (typically increasing it by the factor equal to number of averaged images), and pixel interpolation when re-projecting the images onto the same pixel grid.

*STDPipe* provides two methods for image reprojection:


Lanczos reprojection (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default method :func:`stdpipe.reproject.reproject_lanczos` is a pure Python implementation of Lanczos interpolation with two features borrowed from SWarp that are critical for photometric accuracy:

- **Automatic oversampling** - when output pixels are larger than input pixels (downscaling), the interpolation is evaluated at multiple sub-pixel positions and averaged. This prevents aliasing of undersampled stars.
- **Jacobian flux conservation** - the output is multiplied by the pixel area ratio (Jacobian determinant) so that total flux is conserved. Set ``conserve_flux=False`` to preserve surface brightness instead.

It also supports reprojection of integer flag/mask images (``is_flags=True``) using nearest-neighbor resampling and bitwise AND combining, matching SWarp's behavior.

No external binaries are required.

.. code-block:: python

   from stdpipe.reproject import reproject_lanczos

   # Reproject and coadd a list of FITS files
   coadd = reproject_lanczos(filenames, wcs=target_wcs, shape=(1024, 1024))

   # Reproject from (image, WCS) tuples
   coadd = reproject_lanczos([(image1, wcs1), (image2, wcs2)],
                wcs=target_wcs, shape=(1024, 1024))

   # Preserve surface brightness instead of total flux
   coadd = reproject_lanczos([(image, wcs_in)], wcs=wcs_out,
                shape=(512, 512), conserve_flux=False)

   # Reproject integer flag/mask image
   mask = reproject_lanczos([(flags, wcs_in)], wcs=wcs_out,
                shape=(512, 512), is_flags=True)

.. autofunction:: stdpipe.reproject.reproject_lanczos
   :noindex:


SWarp reprojection
^^^^^^^^^^^^^^^^^^

Alternatively, :func:`stdpipe.reproject.reproject_swarp` wraps the `SWarp <https://github.com/astromatic/swarp>`_ external binary. It is implemented to resemble the calling conventions of the `reproject <https://github.com/astropy/reproject>`_ package - i.e. allows directly stacking image files without loading them to memory first. Requires SWarp to be installed.

.. autofunction:: stdpipe.reproject.reproject_swarp
   :noindex:
