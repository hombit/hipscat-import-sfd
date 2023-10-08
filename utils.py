from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


def produce_ids(x: np.ndarray[int], y: np.ndarray[int]) -> np.ndarray:
    # Max number of digits in y:
    d = int(np.log10(y.max())) + 1
    return x * 10**d + y


def get_data_coords_ids(filename: Path) -> tuple[np.ndarray, SkyCoord, tuple[np.ndarray, np.ndarray]]:
    hdu, = fits.open(filename, memmap=False)
    wcs = WCS(hdu.header)
    data = hdu.data

    # Produce pixel coordinates
    pixel_grid = np.meshgrid(
        np.arange(data.shape[0]),
        np.arange(data.shape[1]),
    )
    coord = wcs.pixel_to_world(*pixel_grid)

    return data, coord, tuple(pixel_grid)


def get_hemisphere(dir: Path, hemisphere: Literal["n", "s"]) -> pd.DataFrame:
    filename = dir / f"SFD_dust_4096_{hemisphere}gp.fits.gz"
    b_sign = 1 if hemisphere == "n" else -1

    data, coord, pixel_grid = get_data_coords_ids(filename)
    # And pixel IDs
    ids = produce_ids(*pixel_grid)

    # Filter out pixels belong to the other hemisphere
    mask = b_sign * coord.b > 0
    data, coord, ids = data[mask], coord[mask], ids[mask]

    # Convert to equatorial coordinates
    eq = coord.transform_to("icrs")
    # Swap bytes to get the correct endianness
    data = data.newbyteorder().byteswap(inplace=True)

    df = pd.DataFrame(dict(
        id=ids.ravel(),
        ra_deg=eq.ra.deg.ravel(),
        dec_deg=eq.dec.deg.ravel(),
        ebv=data.ravel(),
    ))
    return df
