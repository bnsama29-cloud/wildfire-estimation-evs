# UN Sustainable Development Goals alignment

This is an educational prototype. It supports the goals below through awareness, open data and low-cost monitoring. It does not measure progress on them, and it is not an official forecast or a safety system.

| SDG | Target | How the project contributes | Evidence in this repo |
|---|---|---|---|
| **13 Climate Action** | 13.1 Resilience to climate-related hazards | Shows how heat, dryness and wind relate to fire intensity across India | Model and metrics in `models/` |
| **13 Climate Action** | 13.3 Education and awareness on climate | An interactive dashboard makes satellite and weather data easy to explore | `src/app.py` |
| **15 Life on Land** | 15.2 Sustainable management of forests | Uses vegetation-fire detections to show where and when intense fires occur, including forest sites | `src/data_prep.py`, fire map |
| **11 Sustainable Cities and Communities** | 11.5 Reduce losses from disasters | A low-cost sensor unit raises a local smoke alert, showing how communities near forests could get early warning | `hardware/` |
| **3 Good Health and Well-being** | 3.9 Illnesses from air pollution | The smoke reading demonstrates simple local air-quality awareness | MQ-135 live panel |

## How to present it

- **Primary goals:** SDG 13 and SDG 15. The project addresses these most directly.
- **Supporting goals:** SDG 11 and SDG 3, through the sensor unit.
- **Keep the claims modest.** Say "supports" or "contributes to". Do not claim it prevents fires or protects people.
