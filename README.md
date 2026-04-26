# Egg Delivery Solver

Dash application for planning and optimizing salmon egg allocation from production
batches to customer orders. The model combines biological delivery windows,
operational constraints and a PuLP-based linear solver so allocation decisions can
move from manual spreadsheet work to a reproducible planning workflow.

## Current Status

The repository contains a working technical prototype that is close to an early
production decision-support tool. The main remaining work is domain validation:
results should be aligned against Odin's or another external forecasting model,
and the team should make a final decision on whether week-level or day-level
delivery validation is the operational default.

The latest project context from December 2025 points in the same direction:
core app functionality is implemented, while harmonization with the external
Excel/hatch-calculator workflow remains the main calibration step before final
business adoption.

## What The App Does

- Allocates customer orders to egg batches with linear programming through PuLP.
- Supports hard constraints for site, broodstock group and organic requirements.
- Applies soft preferences for preferred site or group.
- Supports week-level and day-level delivery-window validation.
- Builds sales windows from biological stripping and temperature inputs.
- Supports both table-based grading and formula-based hatch-calculator growth logic.
- Accepts precomputed Excel batch windows with `SalesStartWeek` and `SaleStopWeek`.
- Uploads complete scenario workbooks or order-only files.
- Exports allocation results, possible batch matches, generated batches and input data.
- Renders a lightweight planning overview for large scenarios instead of a heavy Gantt chart.

## Repository Layout

```text
egg_delivery_solver/
├── README.md
└── new_version_for_meeting/
    ├── app.py                    # Dash UI and callbacks
    ├── config.py                 # Embedded example data and model constants
    ├── logic.py                  # Parsing, batch generation, feasibility and solver logic
    ├── requirements.txt          # Python dependencies
    ├── IMPLEMENTATION_PLAN.md    # Notes on hatch-calculator alignment work
    ├── assets/
    │   └── app.css               # App styling
    ├── tests/
    │   └── test_logic.py         # Unit/regression tests
    └── example_data/             # Local/sample input generators and scenario files
```

## Input Formats

The app supports two `Fiskegrupper` formats.

### Biological Batch Inputs

Use this format when the app should calculate production windows from stripping
periods and temperature assumptions.

Required columns include:

- `Site`
- `Site_Broodst_Season`
- `StrippingStartDate`
- `StrippingStopDate`
- `MinTemp_C`
- `MaxTemp_C`
- `Gain-eggs`
- `Shield-eggs`
- `Organic`

### Precomputed Batch Windows

Use this format when an external model has already calculated batch and sales
windows.

Required columns:

- `Site`
- `Site_Broodst_Season`
- `BatchID`
- `Produksjonvolum`
- `SalesStartWeek`
- `SaleStopWeek`

`SalesStartWeek` and `SaleStopWeek` use ISO-style `YYYYWW` values. The app maps
the start week to Monday and the stop week to Sunday.

### Orders

The minimum practical order columns are:

- `OrderNr`
- `DeliveryDate`
- `Volume`

Optional columns such as `Product`, `RequireOrganic`, `LockedSite`,
`LockedGroup`, `PreferredSite` and `PreferredGroup` are filled with safe defaults
when missing.

## Local Development

```bash
cd new_version_for_meeting
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:8051
```

The app also respects Render's `PORT` environment variable:

```python
port = int(os.environ.get("PORT", 8051))
app.run(debug=False, host="0.0.0.0", port=port)
```

## Testing

Run the logic test suite from `new_version_for_meeting`:

```bash
python -m unittest tests.test_logic
```

The tests cover:

- interpolation from the grading table
- formula-based hatch-calculator growth logic
- week-vs-day delivery window behavior
- hard constraints for site, group and organic status
- precomputed Excel sales-week windows
- capacity handling in allocation
- planning overview aggregation and unallocated-order exceptions

## Render Deployment

Create a Render Web Service from this repository.

Recommended settings:

- Branch: `main`
- Root directory: `new_version_for_meeting`
- Build command:

```bash
pip install -r requirements.txt
```

- Start command:

```bash
python app.py
```

After deploy, test the default scenario first, then upload the large Excel
scenario and confirm that `Planoversikt` renders and Excel export still works.

## Notes For Reviewers

This is a planning model, not only a dashboard. The most important review points
are therefore the input interpretation, delivery-window logic, constraint
handling and agreement with the external reference model. UI changes should be
evaluated by whether they make large allocation runs easier to inspect, not by
whether they preserve the previous Gantt-style visualization.
