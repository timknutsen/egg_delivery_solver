# Egg Delivery Solver


## Overview

The **Egg Delivery Solver** is a Dash-based web application designed to optimize the allocation of salmon roe (eggs) from broodstock groups to customer orders. The app allows users to input orders and available roe, set optimization constraints and methods, and visualize the allocation results through interactive tables and graphs. It addresses common challenges in roe allocation, such as matching product types, respecting delivery date constraints, and prioritizing orders.

The application implements two allocation methods:
- **Binary Allocation**: Assigns an order entirely to one broodstock group or not at all, using the PuLP linear programming library.
- **Partial Allocation**: Uses a greedy algorithm to allocate eggs from multiple broodstock groups to fulfill orders partially if needed.

Key features include:
- Interactive Dash dashboard with editable tables for orders and roe inventory.
- Configurable optimization settings (constraints, allocation method, order priority).
- Visualizations for broodstock utilization and allocation timeline.
- Support for adding/deleting orders and roe entries dynamically.

## Repository Structure

```
egg_delivery_solver/
│
├── app_v2.py           # Main Dash application file
├── optimization.py     # Optimization logic for binary and partial allocation
└── README.md           # Project documentation
```

## Prerequisites

Ensure you have the following installed:
- **Python 3.8+**
- **pip** (Python package manager)

### Required Python Packages

Install the required packages using the following command:

```bash
pip install dash pandas plotly pulp
```

- `dash`: For building the web application.
- `pandas`: For data manipulation.
- `plotly`: For interactive visualizations.
- `pulp`: For linear programming in binary allocation.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/egg_delivery_solver.git
   cd egg_delivery_solver
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If a `requirements.txt` file doesn't exist, create one with the following content:
   ```
   dash
   pandas
   plotly
   pulp
   ```
   Then run the command above.

3. **Run the Application**:
   ```bash
   python app_v2.py
   ```
   The app will start on `http://localhost:10000` (or the port specified by the `PORT` environment variable).

## Usage

1. **Access the Dashboard**:
   Open your web browser and navigate to `http://localhost:10000`.

2. **Explore the Interface**:
   - **Inventory Analytics**: View total inventory, orders, and balance.
   - **Customer Orders**: Add, edit, or delete orders (e.g., Order ID, Customer, Eggs Ordered, Product, Delivery Date).
   - **Available Roe**: Add, edit, or delete roe inventory (e.g., Broodstock Group, Produced Eggs, Location, Product, Dates).
   - **Optimization Settings**:
     - **Constraints**: Select constraints like Product Match, Date Constraints, or NST Priority.
     - **Allocation Method**: Choose between Binary (full allocation to one group) or Partial (split across groups).
     - **Order Priority**: Prioritize orders Chronologically (by delivery date) or to Maximize Allocation (by egg quantity).
     - **Run Solver**: Click to execute the allocation.
   - **Allocation Results**: View the allocation table and any unfulfilled orders.
   - **Visualization**: See broodstock utilization and a timeline of allocations.

3. **Example Workflow**:
   - Add a new order via the "Customer Orders" table or the "Add Order" button.
   - Adjust the optimization settings (e.g., enable Product Match and Date Constraints, select Binary allocation).
   - Click "Run Solver" to allocate eggs.
   - Review the results in the "Allocation Results" table and visualizations.

## Optimization Settings

The app supports the following optimization settings:
- **Constraints**:
  - **Product Match**: Ensures the product type of the roe matches the ordered product.
  - **Date Constraints**: Ensures the delivery date falls within the roe’s availability period.
  - **NST Priority**: Placeholder for prioritizing specific customers (not implemented).
- **Allocation Method**:
  - **Binary**: Uses PuLP to allocate an order entirely to one broodstock group or not at all.
  - **Partial**: Uses a greedy algorithm to allocate eggs from multiple groups if needed.
- **Order Priority**:
  - **Chronological**: Processes orders based on delivery date.
  - **Maximize Allocation**: Prioritizes orders with larger egg quantities.

## Data Structure

### Orders Data
- **OrderID**: Unique identifier for the order.
- **CustomerID**: Customer name (e.g., "Mowi ASA").
- **OrderedEggs**: Number of eggs ordered.
- **Product**: Product type (e.g., "Shield", "Gain (Premium)").
- **DeliveryDate**: Desired delivery date (format: YYYY-MM-DD).

### Roe Data
- **BroodstockGroup**: Group name (e.g., "Natural (November)").
- **ProducedEggs**: Number of eggs available.
- **Location**: Facility location (e.g., "Hemne").
- **Product**: Product type (e.g., "Shield").
- **StartSaleDate**: Date when eggs become available (format: YYYY-MM-DD).
- **ExpireDate**: Date when eggs expire (format: YYYY-MM-DD).
- **GroupID**: Unique identifier for each roe group (auto-assigned).

## Known Issues

- **NST Priority**: Currently a placeholder; selecting this constraint has no effect.
- **Input Validation**: Limited validation on user inputs in editable tables (e.g., non-numeric egg quantities or invalid dates may cause errors).
- **Performance with Large Data**: Binary allocation using PuLP may be slow with large datasets due to the 10-second solver time limit.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please ensure your code follows PEP 8 style guidelines and includes appropriate comments.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


