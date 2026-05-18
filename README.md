# AI Government Test Task

This repository contains a completed solution for the test_006_government assignment.

## Contents

- solution.ipynb — main notebook with the problem setup, model, code, results, and conclusions
- solution.py — script version of the same approach
- results.json — saved Monte Carlo outputs used in the report
- visual_report.html — compact visual summary of the final comparison
- test_006_government_TZ.md — task statement in markdown form

## Approach

The core state variable is the law imbalance:

`x(t) = D(t) - R(t)`

Then the approval dynamics become:

`W(t + 1) = W(t) + x(t) * rho(t)`

Inside every forecast window, the decision problem is solved with dynamic programming over the next N days. Three variants are included:

1. Block strategy: recompute once per forecast block.
2. Cautious strategy: the same logic with a hard cap on imbalance.
3. Daily refresh strategy: recompute every day using the rolling exact forecast.

## Reported Parameters

- `N = 10`
- `T = 100`
- `W(0) = 1000`
- `penalty(T) = 100`

## Quick Results

- Block strategy: mean `1097.135`, std `14.303`
- Cautious strategy: mean `1056.765`, std `4.835`
- Daily refresh strategy: mean `1159.903`, std `36.267`
- In the reported Monte Carlo runs, early removal was not observed

## Notes

The notebook is the primary deliverable for review. The HTML file is only a compact visual companion and does not replace the notebook.
