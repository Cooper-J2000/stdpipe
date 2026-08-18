#!/usr/bin/env python3

import numpy as np

from astropy.io import fits
from astropy.table import Table, vstack

bricks_dr11_south = Table(
    fits.getdata(
        'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/survey-bricks-dr11-south.fits.gz'
    )
)

bricks_dr11_north = Table(
    fits.getdata(
        'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/north/survey-bricks-dr11-north.fits.gz'
    )
)

s1 = bricks_dr11_south[['brickname', 'ra', 'dec']]
s1['survey'] = 'S'

s2 = bricks_dr11_north[['brickname', 'ra', 'dec']]
s2['survey'] = 'N'

short = vstack([s1, s2])

# Bricks in the north/south overlap region appear in both tables - keep the
# south (DECam) one, which also has i-band coverage
_, idx = np.unique(short['brickname'], return_index=True)
short = short[np.sort(idx)]

short.write('legacysurvey_dr11_bricks.fits', format='fits', overwrite=True)
