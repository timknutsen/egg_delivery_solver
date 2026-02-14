# Implementation Plan: Align With Hatch Calculator Model

## Goal
Align allocation results with the external Excel/hatch calculator workflow while preserving a safe migration path.

## Scope
1. Keep current `week` delivery window mode as default.
2. Add selectable growth model:
   - `table` (current `GRADING_TABLE`)
   - `formula` (hatch calculator formula from email)
3. Add regression comparison against `allokeringsresultater (14).xlsx`.

## Work Breakdown
1. Implement formula-based day estimation in `logic.py`.
2. Add model selector parameter to batch generation and allocation pipeline.
3. Add UI selector in `app.py` for growth model.
4. Add tests:
   - Unit tests for formula conversion output stability.
   - Regression test for week-vs-day acceptance remains correct.
   - Comparison test on a fixture dataset with tolerated date delta.
5. Export metadata:
   - Include chosen `window_mode` and `growth_model` in result workbook.

## Commit Plan
1. `feat(logic): add hatch formula growth model option`
2. `test(logic): add formula and regression coverage`
3. `feat(app): add growth model selector and wire into callbacks`
4. `chore(export): include run configuration in output workbook`

## Acceptance Criteria
1. App supports both `table` and `formula` growth models.
2. `week` mode remains default and tested.
3. Regression suite passes in `conda` env `dash`.
4. Comparison report shows reduced start/stop deltas versus Excel reference.
