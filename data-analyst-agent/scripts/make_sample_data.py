"""Generates a small synthetic customer dataset (seeded, so it is reproducible)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 600
plan = rng.choice(["basic", "pro", "enterprise"], n, p=[.5, .35, .15])
base = {"basic": 20, "pro": 60, "enterprise": 180}
spend = np.array([base[p] for p in plan]) * rng.normal(1, .15, n)
tickets = rng.poisson(2, n)
tenure = rng.integers(1, 48, n)
p_cancel = np.clip(.08 + .06 * tickets - .002 * tenure, .02, .8)
status = np.where(rng.random(n) < p_cancel, "cancelled", np.where(rng.random(n) < .05, "paused", "active"))
signup = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
spend[rng.choice(n, 6, replace=False)] *= 6  # a few planted outliers
df = pd.DataFrame({
    "customer_id": np.arange(1, n + 1), "signup_date": signup.strftime("%Y-%m-%d"),
    "plan": plan, "region": rng.choice(["AMER", "EU", "APAC", "LATAM"], n, p=[.4, .3, .2, .1]),
    "monthly_spend": spend.round(2), "tenure_months": tenure, "support_tickets": tickets, "status": status,
})
df.to_csv("sample_data/customers.csv", index=False)
print("wrote sample_data/customers.csv", df.shape)
