# Data provenance and redistribution boundaries

This package distributes processed point-aligned CSV inputs, not the raw archives. The aligned tables cover the western-U.S. benchmark used in the paper and rebuttal experiments.

| Input family | Upstream source/product | Use in the processed tables | Redistribution note |
|---|---|---|---|
| Occurrence labels and hard negatives | USGS Mineral Resources Data System (MRDS) plus the project porphyry-copper occurrence list | Positive labels, hard-negative candidates, coordinates, state grouping | Only the derived benchmark rows are included. Consult the upstream USGS terms for raw records. |
| Geochemistry | USGS geochemical release and reanalyzed NURE-HSSR data | Point-neighborhood geochemical predictors | Raw databases and dictionaries are not duplicated. Derived, aligned numeric columns are included. |
| Geophysics and geology | NOAA/NCEI gravity products and USGS CMMI gravity, fault, and geology derivatives | Gravity, structural, lithologic, and terrain predictors | Raw rasters/vector products are not included. |
| Modern climate | TerraClimate 1991-2020 climatology | Climate predictors and residualization adjusters | Raw NetCDF grids are not included. |
| Multispectral surface observations | ASTER VNIR/SWIR/TIR scenes | Basic bands and 132 engineered spectral features | Scene files are not included; point-aligned derived features are included. |
| Land cover | Project-aligned land-cover product used with ASTER | Surface-cover predictors | Raw raster tiles are not included. |
| Historical temperature | WNATA reconstruction (King et al., 2024 data files) | 1700-1850 JJA maximum-temperature anomaly summaries | Raw reconstruction files are not included. |
| Historical precipitation | NASPA 2020 reconstruction | 1700-1850 cool-season precipitation summaries | Raw reconstruction files are not included. |

Primary public landing pages used by the project include:

- MRDS: <https://mrdata.usgs.gov/mrds/>
- TerraClimate: <https://www.climatologylab.org/terraclimate.html>
- ASTER Level 1T: <https://lpdaac.usgs.gov/products/ast_l1tv003/>
- NOAA paleoclimate archive: <https://www.ncei.noaa.gov/products/paleoclimatology>

For a formal release, the repository owners should add the exact upstream dataset versions, access dates, and applicable licenses/terms from the final manuscript data appendix. This package does not override upstream data licenses and does not introduce a project software license.
