# 190623_LifeCycleWithOneShockOnConsumption

> update: June 2019

This demo is for my blog on [this page](https://clpr.github.io/pages/blogs/190622_The_household_life_cycle_problem_with_a_continuous_medical_expenditure_shock.html).
Please read this blog first.


This demo shows how to use standard solver to solve a life-cycle model with **one** Markov-style shock on consumption.
It consists of:
|Demo|Description|Time-costing?|Borrowing constraint?|
|----|-----------|-------------|---------------------|
|Part 1|Analytical solution | No | Cannot handle it |
|Part 2|Deterministic DP (the shock's initial distribution is stationary) | Yes | Can handle it|
|Part 3|Stochastic DP (the shock's initial distribution is non-stationary) | Yes | Can handle it|

Please read `demo.ipynb` for the best reading experience.
If it does not work, you may read `demo.html`. The two are the same.

