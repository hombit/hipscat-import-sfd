# SFD map for HiPSCat and LSDB

This repository contains the SFD map tooling for HiPSCat and LSDB.

### Data exploration

`find_largest_gradient.ipynb` looks for the largest gradient between two points of the SFD map.
We use this to estimate the largest healpix order we can use to have interpolation errors to be smaller than the SFD map errors of 16%.
These 16% corresponds to the order 13, but we use the order 14 to be on the safe side.

### Get SFD map of the fixed healpix order

`interpolate_sfd.ipynb` interpolates SFD map into the healpix map of the order 14, using center of each tile as the interpolation point.
The result is saved into intermediate parquet files.

### Import the map into HiPSCat

`import_sfd_as_catalog.ipynb` uses `hipscat-import` to create a HiPSCat catalog.

### Use the map in LSDB

`join_stars_and_sfd.ipynb` loads the SFD map and the test LSDB point source catalog and joins them together.
It implements the merging algorithm through the cross-matching interface of LSDB.

## WIP

### Multiorder HiPSCat

`multiorder_map.ipynb`, `min_max_mean_state.py` and `multiorder.py` implement generation of the SFD map for HiPSCat with multiple orders.
Currently, it works extremely slow, because it merges the highest order map from bottom up with a lot of single tile work.
Python overhead makes it super-slow.