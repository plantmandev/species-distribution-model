# Study of Butterfly Ranges in North America: A Species Distribution Model Approach

This repository contains scripts for procurement, cleaning, and processing of research-grade occurrences of North American butterflies. The aim is to visualize ranges, understand phenology, and estimate year-to-year range shifts.

## Environmental Variables

- **Climate**: Temperature, precipitation, seasonality (WorldClim bioclimatic variables)
- **Elevation**: Altitude and topographic variation
- **Land cover/Vegetation**: NDVI, habitat types (proxy for host plant availability)
- **Temporal**: Seasonal flight periods, inter-annual variation
- **Geographic coordinates**: Latitude, longitude for spatial structure

## Methods for Improving Model Accuracy

### Sampling Bias Correction
- **Spatial thinning**: One observation per grid cell to reduce clustering
- **Target-group background sampling**: Use other butterfly species as pseudo-absences to control observer bias
- **Sampling effort layer**: Account for uneven geographic coverage

### Modeling Approach
- **Random Forest**: Handles non-linear relationships, variable interactions
- **Ensemble modeling**: Combine multiple algorithms for robust predictions
- **Spatial block cross-validation**: Prevents overfitting from spatial autocorrelation
- **Variable selection**: Remove correlated predictors (r > 0.7)

## Tools & Libraries

- **pygbif**: GBIF/iNaturalist occurrence data acquisition
- **geopandas**: Geospatial data processing
- **rasterio**: Environmental raster handling
- **scikit-learn**: Machine learning (Random Forest, model validation)
- **QGIS**: Spatial visualization and cartography
- **pandas/numpy**: Data manipulation

## Workflow

1. Data procurement from GBIF (includes iNaturalist research-grade observations)
2. Spatial filtering and bias correction
3. Environmental variable extraction and correlation analysis
4. Species Distribution Modeling with cross-validation
5. Temporal stratification by season/year
6. Range prediction and polygon generation
7. Visualization in QGIS


## Constraints

The main data source for this project is research-grade occurrence observations from [GBIF.org](https://www.gbif.org/). These occurrences are presence-only data points, which suffer from **sample selection bias** more strongly than other data types suitable for this project such as presence-absence data (Elith et al., 2011). Additionally, **prevalence is not identifiable from presence-only data** (Elith et al., 2011). This is extremely significant, as it ultimately limits the scope of this analysis. **This species distribution model therefore calculates relative habitat suitability, not absolute probability of presence**.

This model was made with these caveats in mind, and efforts were made in the determination and configuration of species distribution model to best account for these constraints. 



# Sources

- [A statistical explanation of MaxEnt for ecologists](https://onlinelibrary.wiley.com/doi/10.1111/j.1472-4642.2010.00725.x) - Jane Elith, et al
- [Ensamble forecasting of species distributions](https://www.sciencedirect.com/science/article/abs/pii/S016953470600303X) - Miguel Araujo, et al
- [Mapping species distributions with MaxEnt using a geographically biased sample of presence data: A performance assessment of methods for correcting sampling bias](https://pmc.ncbi.nlm.nih.gov/articles/PMC4018261/) - Yoan Fourcade, et al
- [Sample selection bias and presence-only distribution models:
implications for background and pseudo-absence data](https://esajournals.onlinelibrary.wiley.com/doi/10.1890/07-2153.1) - Steven Phillips, et al