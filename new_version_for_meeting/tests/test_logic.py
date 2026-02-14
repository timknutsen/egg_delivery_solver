import unittest
import importlib.util

import pandas as pd

if importlib.util.find_spec("pulp") is None:
    raise unittest.SkipTest("Skipping logic tests: dependency 'pulp' is not installed.")

from logic import (
    build_feasibility_set,
    calculate_formula_days,
    calculate_grading_days,
    generate_weekly_batches,
    preprocess_data,
    run_allocation,
)


class LogicTests(unittest.TestCase):
    def test_calculate_grading_days_interpolates_between_table_values(self):
        # 80%-days: temp 3 => 112, temp 4 => 93, midpoint should be 102.5
        self.assertAlmostEqual(calculate_grading_days(3.5, "80"), 102.5)

    def test_calculate_formula_days_is_monotonic(self):
        # Høyere temperatur skal gi færre dager.
        cold = calculate_formula_days(2.0, "80")
        warm = calculate_formula_days(6.0, "80")
        self.assertGreater(cold, warm)

    def test_generate_weekly_batches_creates_expected_batch_windows(self):
        fish_groups = pd.DataFrame(
            [
                {
                    "Site": "TestSite",
                    "Site_Broodst_Season": "GroupA",
                    "StrippingStartDate": "2024-09-01",
                    "StrippingStopDate": "2024-09-15",
                    "MinTemp_C": 2.0,
                    "MaxTemp_C": 6.0,
                    "Gain-eggs": 1000.0,
                    "Shield-eggs": 500.0,
                    "Organic": True,
                }
            ]
        )

        batches = generate_weekly_batches(fish_groups)
        self.assertEqual(len(batches), 2)
        self.assertTrue((batches["MaturationEnd"] <= batches["ProductionEnd"]).all())
        self.assertAlmostEqual(float(batches["GainCapacity"].sum()), 1000.0, places=6)
        self.assertAlmostEqual(float(batches["ShieldCapacity"].sum()), 500.0, places=6)

    def test_generate_weekly_batches_supports_formula_model(self):
        fish_groups = pd.DataFrame(
            [
                {
                    "Site": "TestSite",
                    "Site_Broodst_Season": "GroupFormula",
                    "StrippingStartDate": "2024-09-01",
                    "StrippingStopDate": "2024-09-15",
                    "MinTemp_C": 2.0,
                    "MaxTemp_C": 6.0,
                    "Gain-eggs": 1000.0,
                    "Shield-eggs": 500.0,
                    "Organic": False,
                }
            ]
        )

        batches = generate_weekly_batches(fish_groups, growth_model="formula")
        self.assertEqual(len(batches), 2)
        self.assertTrue((batches["MaturationEnd"] <= batches["ProductionEnd"]).all())
        self.assertTrue(batches["CalcInfo"].str.contains("Model:formula").all())

    def test_build_feasibility_set_respects_window_and_hard_constraints(self):
        orders = pd.DataFrame(
            [
                {
                    "OrderNr": 1,
                    "Customer": "A",
                    "DeliveryDate": "2025-01-05",
                    "Product": "Gain",
                    "Volume": 100.0,
                    "MinTemp_C": 2.0,
                    "MaxTemp_C": 6.0,
                    "RequireOrganic": True,
                    "LockedSite": None,
                    "LockedGroup": None,
                    "PreferredSite": None,
                    "PreferredGroup": None,
                },
                {
                    "OrderNr": 2,
                    "Customer": "B",
                    "DeliveryDate": "2025-01-05",
                    "Product": "Gain",
                    "Volume": 100.0,
                    "MinTemp_C": 2.0,
                    "MaxTemp_C": 6.0,
                    "RequireOrganic": False,
                    "LockedSite": "WrongSite",
                    "LockedGroup": None,
                    "PreferredSite": None,
                    "PreferredGroup": None,
                },
            ]
        )
        orders, _ = preprocess_data(
            orders,
            pd.DataFrame(
                columns=[
                    "Site",
                    "Site_Broodst_Season",
                    "StrippingStartDate",
                    "StrippingStopDate",
                    "MinTemp_C",
                    "MaxTemp_C",
                    "Gain-eggs",
                    "Shield-eggs",
                    "Organic",
                ]
            ),
        )

        batches = pd.DataFrame(
            [
                {
                    "BatchID": "B1",
                    "Group": "G1",
                    "Site": "Site1",
                    "StripDate": pd.Timestamp("2024-11-01"),
                    "MaturationEnd": pd.Timestamp("2025-01-01"),
                    "ProductionEnd": pd.Timestamp("2025-01-10"),
                    "GainCapacity": 1000.0,
                    "ShieldCapacity": 500.0,
                    "Organic": True,
                }
            ]
        )

        feasible = build_feasibility_set(orders, batches, window_mode="week")
        self.assertEqual(set(feasible["OrderNr"].tolist()), {1})

    def test_week_mode_accepts_same_week_even_if_day_window_misses(self):
        orders = pd.DataFrame(
            [
                {
                    "OrderNr": 3,
                    "Customer": "WeeklyCase",
                    "DeliveryDate": "2025-01-08",  # onsdag
                    "Product": "Gain",
                    "Volume": 100.0,
                    "MinTemp_C": 2.0,
                    "MaxTemp_C": 6.0,
                    "RequireOrganic": False,
                    "LockedSite": None,
                    "LockedGroup": None,
                    "PreferredSite": None,
                    "PreferredGroup": None,
                }
            ]
        )
        orders, _ = preprocess_data(
            orders,
            pd.DataFrame(
                columns=[
                    "Site",
                    "Site_Broodst_Season",
                    "StrippingStartDate",
                    "StrippingStopDate",
                    "MinTemp_C",
                    "MaxTemp_C",
                    "Gain-eggs",
                    "Shield-eggs",
                    "Organic",
                ]
            ),
        )

        batches = pd.DataFrame(
            [
                {
                    "BatchID": "B2",
                    "Group": "G2",
                    "Site": "Site2",
                    "StripDate": pd.Timestamp("2024-11-01"),
                    "MaturationEnd": pd.Timestamp("2025-01-01"),
                    "ProductionEnd": pd.Timestamp("2025-01-07"),  # tirsdag
                    "GainCapacity": 1000.0,
                    "ShieldCapacity": 500.0,
                    "Organic": False,
                }
            ]
        )

        feasible_day = build_feasibility_set(orders, batches, window_mode="day")
        feasible_week = build_feasibility_set(orders, batches, window_mode="week")

        self.assertEqual(len(feasible_day), 0)
        self.assertEqual(len(feasible_week), 1)

    def test_run_allocation_end_to_end_allocates_single_order(self):
        fish_groups = pd.DataFrame(
            [
                {
                    "Site": "Site1",
                    "Site_Broodst_Season": "GroupX",
                    "StrippingStartDate": "2024-10-07",
                    "StrippingStopDate": "2024-10-07",
                    "MinTemp_C": 2.0,
                    "MaxTemp_C": 8.0,
                    "Gain-eggs": 10000.0,
                    "Shield-eggs": 10000.0,
                    "Organic": False,
                }
            ]
        )
        # With strip date 2024-10-07 and MaxTemp 8 -> MaturationEnd approx 2024-11-26
        orders = pd.DataFrame(
            [
                {
                    "OrderNr": 11,
                    "Customer": "TestCust",
                    "DeliveryDate": "2024-11-27",
                    "Product": "Gain",
                    "Volume": 100.0,
                    "MinTemp_C": 2.0,
                    "MaxTemp_C": 6.0,
                    "RequireOrganic": False,
                    "LockedSite": None,
                    "LockedGroup": None,
                    "PreferredSite": None,
                    "PreferredGroup": None,
                }
            ]
        )

        out = run_allocation(fish_groups, orders)
        self.assertIn("results", out)
        self.assertEqual(len(out["results"]), 1)
        self.assertNotEqual(out["results"].iloc[0]["BatchID"], "IKKE TILDELT")


if __name__ == "__main__":
    unittest.main()
