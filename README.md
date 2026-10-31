# Early Wildfire Detection and Analysis using ML

Educational prototype. Indian satellite fire points (NASA FIRMS VIIRS) joined with weather (Open-Meteo) to estimate fire intensity, shown in a Streamlit dashboard and fed live by an ESP32 sensor unit. **Not an official forecast and not a safety device.**

> **FRP (Fire Radiative Power, MW) is a satellite intensity proxy. It is not burned area in hectares.**

## Status

| Part | State |
|---|---|
| FIRMS cleaning (`src/data_prep.py`) | done: 4,097,001 vegetation fire points |
| Weather join (`src/weather_join.py`) | done: 1,910,632 fire points from 2024-25 across the 180 busiest 1 degree cells |
| Weather model (`src/train_weather_model.py`) | done: results below |
| Dashboard (`src/app.py`) | done: runs offline with no sensor |
| ESP32 sketch and Wokwi project (`hardware/`) | sketch compiles and runs on wokwi.com; dashboard link tested against a stand-in server, not yet against live Wokwi; not yet flashed to a real board |
| Slides (`docs/EL_Phase1_Wildfire.pptx`) | needs the final model name and an SDG slide |

## Results

Trained on 879,112 fires from 2024 and tested on 1,031,520 fires from 2025. The model was chosen on held-out 2024 dates, so 2025 stayed unseen.

| Measure on 2025 | Median guess | Final model (XGBoost, monotonic weather) |
|---|---|---|
| Typical error (MAE) | 4.72 MW | 4.01 MW |
| R2 on log FRP | -0.01 | 0.46 |

| Predicted band | 2025 fires | Share that were large (15 MW or more) |
|---|---|---|
| LOW (below 4.4 MW) | 523,921 | 1.1 % |
| MEDIUM (4.4-6.9 MW) | 343,903 | 6.0 % |
| HIGH (6.9 MW and above) | 163,696 | 27.2 % |

Bands are relative: a fire in the HIGH band was about 25 times as likely to be large as one in the LOW band.

**Weather adds little once place, season and time of day are known.** Weather alone reaches R2 0.17, place, month and day or night alone reach 0.46, and all of them together also reach 0.46. Day or night accounts for 86 % of the model's feature importance, and daytime detections have a median of 4.7 MW against 1.3 MW at night. So temperature moves the predicted FRP only slightly: from 15 to 45 degrees C at Similipal in April it goes from 3.3 to 4.3 MW. Humidity has no effect there above about 22 %, but very dry air matters: at 36 degrees C the estimate goes from 3.8 MW (LOW) at 25 % to 7.4 MW (HIGH) at 5 %. That is what the data shows, and the model is not tuned to look more sensitive.

The dashboard therefore also shows the **Fosberg Fire Weather Index**, a published formula from temperature, humidity and wind. It is not machine learning, and it reacts directly to the sensor. Tested the same way on 2025 fires, 4.1 % of fires in its LOW band were large, against 12.4 % in its HIGH band.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
python -m pytest tests -q
streamlit run src/app.py
```

The trained model and a 30,000-row sample are in the repo, so the dashboard needs no download. The first load can take up to a minute.

To rebuild everything from the FIRMS zips:

```bash
python src/data_prep.py
python src/weather_join.py fetch --cells 180   # about 2 hours, resumes if stopped
python src/weather_join.py join
python src/train_weather_model.py
```

`src/train_model.py` is a separate satellite-only model that uses brightness and other values measured during a detection. It answers a different question and the dashboard does not use it. `models/wildfire_model_metrics.json` predates the vegetation-fire filter, so retrain before quoting it.

## Hardware and Wokwi

See [hardware/README.md](hardware/README.md) for Wokwi, wiring, flashing and calibration. The same sketch runs in Wokwi and on the real ESP32. The DHT22 replaces the temperature and humidity sliders, and the smoke sensor drives the alert only, because the model is not trained on smoke.

## SDGs

See [docs/SDG_alignment.md](docs/SDG_alignment.md).

## Repository layout

```
data/raw/        FIRMS VIIRS zips (tracked); extracted CSVs are git-ignored
data/            india_fires_weather.csv (30,000-row sample); the full join and cache are local only
src/             app.py  predict.py  features.py  weather_join.py  train_weather_model.py
                 data_prep.py  train_model.py  model_io.py  serial_source.py
models/          wildfire_weather_model.joblib and its metrics
hardware/        ESP32 sketch, Wokwi project and wiring guide
tests/           pytest tests
evidence/        screenshots, plots, photos for the report
docs/            slides, SDG alignment, fire occurrence data notes
```

## Data

| File | Content |
|---|---|
| `data/raw/DL_FIRE_J1V-C2_834163.zip` | VIIRS NOAA-20 (J1), 375 m. `fire_archive` 2,370,155 rows + `fire_nrt` 32,132 rows |
| `data/raw/DL_FIRE_SV-C2_834164.zip` | VIIRS Suomi NPP (SV), 375 m. `fire_archive` 2,281,841 rows + `fire_nrt` 29,259 rows |
| `data/samples/open-meteo-21.97N78.98E696m.csv` | Hourly relative humidity for one point (21.97 N, 78.98 E), 2023 onward |

Verified facts about the archives: 2023-01-01 to 2026-07-31, already inside India's box (lat 8.0-35.1, lon 68.5-97.2). Columns: `latitude, longitude, brightness, scan, track, acq_date, acq_time, satellite, instrument, confidence, version, bright_t31, frp, daynight, type`. `fire_nrt` continues the series after July 2026.

- **Confidence is a letter for VIIRS**: `n` nominal (~79%), `l` low (~18%), `h` high (~2.5%). Map to categories, or drop `l` for cleaner training.
- **FRP is heavily skewed**: median about 3.5 MW, 90th percentile about 11 MW, 99th about 47 MW. Train on `log1p(frp)`.
- `acq_time` is UTC as `HHMM`. India is UTC+5:30.
- Source: NASA FIRMS (https://firms.modaps.eosdis.nasa.gov/). Weather: Open-Meteo Historical API (https://open-meteo.com/), non-commercial free use.

## Figures

Numbers 1-3 and 4 are already made; 5-10 are produced as the work progresses. Save images in `evidence/`.

| Fig | Where | What it shows | Caption |
|---|---|---|---|
| 1 | Slide 3 | Concept image: a VIIRS satellite scanning India with fire hotspots | A satellite detects fire hotspots over India; each carries an FRP value. |
| 2 | Slide 8 | Architecture. Top lane: offline ML pipeline (FIRMS data, clean, weather join, XGBoost, evaluate, saved model). Middle lane: ESP32 with DHT22 and MQ-135, USB serial packet, Python reader. Bottom lane: dashboard with fire map, predicted FRP and risk band, live sensor panel, slider fallback | System architecture: ML pipeline, ESP32 sensor unit and Streamlit dashboard. |
| 3 | Slide 9 | Dashboard mockup (illustrative values): sliders and live ESP32 panel on the left, India fire map in the centre, predicted FRP, risk band and feature importance on the right, sample fire points table at the bottom | Dashboard layout. |
| 4 | README | Wiring diagram: DHT22 to GPIO4, MQ-135 AO through a 10k/20k divider to GPIO34, USB to PC | ESP32 sensor wiring. |
| 5 | `evidence/fig5_frp_hist.png` | Histogram of FRP on a log axis, showing the long right tail (median ~3.5 MW, p99 ~47 MW) | Distribution of satellite FRP over India. |
| 6 | `evidence/fig6_monthly.png` | Fire count by month, showing the dry-season peak | Seasonality of detections. |
| 7 | `evidence/fig7_pred_vs_actual.png` | Predicted vs actual FRP on the test set with the y = x line | Model accuracy on held-out data. |
| 8 | `evidence/fig8_importance.png` | Model feature importance | Which inputs matter most. |
| 9 | `evidence/fig9_map.png` | Real dashboard map: fire points coloured by FRP band | Fire detections across India. |
| 10 | `evidence/fig10_rig.jpg` | Photo of the ESP32 rig next to the dashboard showing live values | Hardware integrated with the dashboard. |

## Limitations to state in the report

- Satellites pass a few times a day: no detection does not mean no fire, and clouds block detections.
- Weather comes from one point per 1 degree cell, and only the 180 busiest cells are covered (about 90 % of 2024-25 detections).
- FRP depends on fuel and fire size as well as weather, so weather-only R2 is modest (0.17).
- The model estimates how intense a fire would be if one burns. It does not predict whether a fire will start.
- Risk bands are relative to this dataset, not an official danger rating.
- The ESP32 smoke value is a relative index from a low-cost sensor.

## References

Breiman (2001), *Machine Learning* 45(1); Giglio, Schroeder & Justice (2016), *Remote Sensing of Environment* 178; Schroeder et al. (2014), *Remote Sensing of Environment* 143; Wooster et al. (2005), *J. Geophys. Res.* 110, D24311; Jain et al. (2020), *Environmental Reviews* 28(4). Fosberg (1978), *Weather in wildland fire management: the fire weather index*, Conference on Sierra Nevada Meteorology. Verify each citation before submitting.
