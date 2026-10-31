# Data requirements for fire-occurrence prediction

This document describes the data needed to predict whether a satellite-detectable
fire will occur in a location during a future time window. It is a data plan,
not a claim that the current FRP regression model predicts future fires.

## Work completed so far

- The repository contains two NASA FIRMS VIIRS ZIP archives: NOAA-20 and
  Suomi NPP. Each includes archive and near-real-time fire detections. The
  current cleaning script combines their CSVs, keeps records within the
  configured India bounds, removes invalid FRP rows and exact duplicates,
  and writes `data/firms_india_raw.csv`.
- An XGBoost regressor was trained on all 4,713,018 cleaned detections, using
  an 80/20 random split: 3,770,414 training rows and 942,604 test rows. Training
  requested CUDA on the RTX 4060.
- The FRP model uses location, brightness, brightness temperature, scan/track,
  confidence, day/night, satellite, and cyclic month/pass-time features. Its
  held-out random-split results were MAE 0.259 on `log1p(FRP)`, MAE 2.433 MW,
  and R-squared 0.746 on `log1p(FRP)`. The median baseline had MAE 4.107 MW.
  These results describe estimation of FRP for already detected fires, not
  advance prediction of fire occurrence. Brightness-related features are
  closely related to the FRP target, and the random split may be optimistic.
- The project also has an ESP32 sensor sketch and a Python serial packet
  parser with tests. These sensor readings are not included in the FRP model.
- The existing Open-Meteo sample contains hourly relative humidity for one
  location only. No India-wide weather join, occurrence classifier, future
  fire labels, vegetation/fuel dataset, or terrain dataset has been built yet.

## Define the prediction target

Start with a fixed target such as:

> At the end of each day, estimate whether a new fire will be detected in each
> 0.1-degree grid cell during the following 24 hours.

A 0.1-degree cell is roughly 11 km north-to-south in India; its east-to-west
width varies by latitude. The cell size, forecast horizon, and meaning of
"new fire" should be finalized before assembling training rows. A 72-hour
horizon can be evaluated separately.

Each training row should represent one grid cell and forecast issue time. Its
features must only use information available at or before that issue time.
Do not include the future detection's FRP, brightness, confidence, or other
observation values as input features.

## Required data

### 1. Historical fire detections: labels

Use the two FIRMS VIIRS archives already in `data/raw/` as positive evidence.
Retain at least location, acquisition date/time, satellite, confidence, and
FRP. Deduplicate observations from the two satellites and repeated detections,
using a documented space-and-time rule.

For each cell and issue time:

- Label positive if a qualifying fire detection occurs in the forecast window.
- Select negative examples from cells/times with no detection in that window.
- Exclude or flag periods/areas with poor observation coverage where possible.

No FIRMS detection does not prove that no fire occurred: clouds, satellite
overpass timing, and sensor limits can hide events. The resulting label should
therefore be described as "FIRMS detection in the horizon" unless it is checked
against an independent ground-truth source.

### 2. Historical weather: predictors

Get time-matched historical weather across the study area, not only the
single-location sample in `data/samples/`. Useful variables include:

- temperature and relative humidity
- wind speed and direction
- precipitation over recent periods (for example, 1, 3, 7, and 30 days)
- where available, soil moisture or vapor-pressure deficit

Use the weather values known before each issue time. Archive/reanalysis data
can support retrospective training; a deployed forecast must use weather
forecasts that would actually have been available at prediction time.

Open-Meteo Historical/Archive data is one possible source. Cache responses,
record the requested location, time zone, variables, and retrieval date, and
check its coverage and usage terms before downloading at India scale.

### 3. Location and fuel context

Join static or slowly changing context to each grid cell:

- vegetation/land-cover type and forest cover
- elevation and slope
- seasonal vegetation or fuel-dryness indicators, if available

Possible sources include ESA WorldCover for land cover, a digital elevation
model for terrain, and MODIS or comparable satellite vegetation products.
Record each source's product/version, spatial resolution, date range, and
license. These are recommended additions; weather and FIRMS labels are the
minimum starting point.

### 4. Past fire activity

Prior FIRMS detections can be useful predictors, for example counts in the
same cell and neighboring cells during the previous 1, 7, and 30 days. Compute
them strictly from detections before the issue time. Exclude the current
prediction window to prevent target leakage.

## Dataset row and split design

Build a table with one row per `(cell_id, issue_time)`, containing:

- `cell_id`, cell center coordinates, `issue_time`, and forecast horizon
- future detection label
- past-only weather, vegetation/fuel, terrain, and fire-history features
- observation-coverage or quality flags used to qualify the label

Keep all rows from a forecast window and nearby duplicate detections together
when splitting. For a credible evaluation:

1. Train on earlier years and test on a later untouched period.
2. Also evaluate on held-out geographic regions if the intended model must
   generalize spatially.
3. Fit preprocessing and tune thresholds using training/validation data only.
4. Report precision-recall, recall, precision, false alarms per area/time, and
   calibration; occurrence is a rare-event task, so accuracy alone is
   misleading.
5. Compare against simple seasonal and recent-fire-history baselines.

Do not use a random row split as the sole evaluation because adjacent cells and
times can share nearly identical weather and fire episodes.

## Scope of the existing repository data

The two FIRMS ZIP files provide satellite detections that can contribute
positive labels and historical-fire features. The Open-Meteo sample currently
contains humidity for only one location and is not sufficient for
India-wide weather features. No ground-truth absence labels, broad weather
coverage, vegetation/fuel layers, or terrain layers are currently assembled.
The existing FRP model estimates intensity from an observed fire point; it is
not an occurrence-forecast model.

## Suggested build order

1. Finalize the grid, issue cadence, horizon, and label definition.
2. Create and validate deduplicated FIRMS event labels.
3. Build a small regional/time-limited weather join and cache it.
4. Add past-only weather and fire-history features; verify there is no
   look-ahead leakage.
5. Train a baseline occurrence classifier and evaluate on a later time period.
6. Add vegetation and terrain covariates, then compare against the baseline.
7. Keep FRP/severity prediction as a separate task with its own target and
   evaluation.

This is a research/educational workflow, not an operational warning system.
