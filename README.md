# Egg Delivery Solver

A Dash web app for optimizing salmon egg allocation from production batches to customer orders.

## Quick Start

```bash
cd new_version_for_meeting
pip install -r requirements.txt
python app.py
```

Open http://localhost:8051

## What It Does

- Allocates fish egg orders to weekly production batches
- Uses linear programming (PuLP) to optimize allocation
- Prioritizes fresher eggs (FIFO-like)
- Visualizes delivery windows with interactive Gantt chart

## Project Structure

```
egg_delivery_solver/
├── README.md
└── new_version_for_meeting/
    ├── app.py              # Main application
    └── requirements.txt    # Dependencies
```

## Requirements

- Python 3.8+
- dash, dash-bootstrap-components, pandas, numpy, plotly, pulp

## Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m "Add feature"`)
4. Push (`git push origin feature/your-feature`)
5. Open a Pull Request
