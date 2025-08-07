---
title: "From Compliance to Dashboard: Building a Basel III Risk Platform with Streamlit and PostgreSQL"
description: "An end-to-end risk analytics platform simulating Basel III metrics with PostgreSQL, SQLAlchemy, and Streamlit."
author: "Thomas Martins"
date: 2025-08-06
layout: post
categories: ["Risk Management"]
tags: [sqlalchemy, basel-iii, streamlit, sql, risk-management, data-engineering]
---


## Introduction

This project presents a complete, end-to-end **Basel III risk analytics platform**, designed to simulate how financial institutions report regulatory metrics under supervisory frameworks. The application demonstrates a scalable and interpretable risk data pipeline, from database schema design to interactive dashboard, all built with open-source tools.

[View the live dashboard](https://basel-risk-pipeline-tmartins.streamlit.app/)  
[View the GitHub repository](https://github.com/thomasmartins/basel-risk-pipeline)

---

## Step 1: Database Schema Design

The backend database is structured to simulate real-world financial institutions, with data necessary to report indicators in order to comply with constraints imposed by Basel III regulations. PostgreSQL was selected for its easy to use, widespread adoption, and compatibility with cloud hosting.

Key tables include:

- `balance_sheet`: balance sheet items e.g. assets, liabilities, Tier 1 capital, separated by time stamps and scenario (when applicable)
- `cashflows`: cashflows separated by product and counterparty type, sorted by maturity buckets and ASF/RSF and HQLA type (when applicable)
- `irrbb`: cashflows sorted by tenor bucket for IRRBB calculation, with PV01 and rate sensitivities, plus scenarios
- `rwa`: expositions separated by asset class and approach (IRB or STD), with amount, risk weights and capital requirements
- `scenarios`: scenario identifiers and descriptive metadata

This structure provides a normalized view of banking exposures suitable for risk aggregation and supervisory reporting.

Here’s a simplified schema definition for the core tables:

```sql
-- Table: balance_sheet
CREATE TABLE balance_sheet (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    item VARCHAR(50) NOT NULL,         -- e.g., CET1, Tier1, Total Capital, Assets
    amount NUMERIC(18,2) NOT NULL,
    scenario_id INTEGER REFERENCES scenarios(id) ON DELETE SET NULL
);

-- Table: cashflows
CREATE TABLE cashflows (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    product VARCHAR(50) NOT NULL,       -- e.g., loan, deposit, bond
    counterparty VARCHAR(50) NOT NULL,  -- retail, wholesale, interbank
    maturity_date DATE,
    bucket VARCHAR(20),                 -- e.g., 7d, 30d, 90d
    amount NUMERIC(18,2) NOT NULL,
    direction VARCHAR(10) CHECK (direction IN ('inflow', 'outflow')) NOT NULL,
    hqlatype VARCHAR(20) CHECK (hqlatype IN ('Level1', 'Level2A', 'Level2B', 'None')) NOT NULL,
    asf_factor NUMERIC(5,2) DEFAULT 0,  -- Available Stable Funding factor (NSFR)
    rsf_factor NUMERIC(5,2) DEFAULT 0,  -- Required Stable Funding factor (NSFR)
    scenario_id INTEGER REFERENCES scenarios(id) ON DELETE SET NULL
);

-- Table: irrbb
CREATE TABLE irrbb (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    instrument VARCHAR(50) NOT NULL,     -- bond, loan, deposit, derivative
    cashflow NUMERIC(18,2) NOT NULL,     -- Amount of cashflow
    maturity_date DATE NOT NULL,         -- Maturity of cashflow
    tenor_bucket VARCHAR(20),            -- e.g., 0-1y, 1-3y, etc.
    pv01 NUMERIC(10,6) NOT NULL,         -- PV01 for this instrument
    rate_sensitivity NUMERIC(10,6),      -- Delta cashflow per 1bp shift
    scenario_id INTEGER REFERENCES scenarios(id) ON DELETE SET NULL
);

-- Table: rwa
CREATE TABLE rwa (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    exposure_id VARCHAR(50) NOT NULL,
    asset_class VARCHAR(50) NOT NULL,       
    approach VARCHAR(20) CHECK (approach IN ('STD', 'IRB')) NOT NULL,
    amount NUMERIC(18,2) NOT NULL,           
    risk_weight NUMERIC(5,2) NOT NULL,       
    rwa_amount NUMERIC(18,2) NOT NULL,       -- Inserted by app: amount * risk_weight
    capital_requirement NUMERIC(18,2) NOT NULL, -- Inserted by app: rwa_amount * 0.08
    scenario_id INTEGER REFERENCES scenarios(id) ON DELETE SET NULL
);
```

These tables are then populated with simulated data.

---

## Step 2: Integration with Python using SQLAlchemy

The application uses **SQLAlchemy** as a lightweight interface for executing raw SQL queries within a safe and modular framework. 

The query layer, implemented in `src/queries.py`, decouples data access from metric logic, allowing each computation module to remain testable and composable.

Here’s an example from `src/queries.py` showing how balance sheet entries are fetched for a selected scenario:

```python
# src/queries.py

import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine(
    f"postgresql://{user}:{password}@{host}:{port}/{database}"

def get_balance_sheet(scenario_id=None):
    """
    Fetch balance sheet items with optional scenario filter.
    """
    query = """
    SELECT * FROM balance_sheet
    WHERE (:scenario IS NULL OR scenario_id = :scenario)
    """
    df = pd.read_sql(
        text(query),
        con=engine,
        params={'scenario': scenario_id}
    )
    return df
```

This approach combines the readability of parameterized SQL with the connection safety and flexibility of SQLAlchemy’s `engine`. 

---

## Step 3: Risk Metric Computation Layer

The logic for each Basel III metric is written in  `src/compute.py`. Each function pulls filtered data via query interfaces and applies vectorized transformations with `pandas` and `numpy`.

### Liquidity Risk
- **LCR**: HQLA segmentation, inflow/outflow caps when mandated
- **NSFR**: ASF and RSF weighting with scenario-specific factor lookup

### Capital Adequacy
- **CET1 Ratio**: Core capital / risk-weighted assets
- **Total Capital Ratio**: Broader capital base / RWA

### Interest Rate Risk (IRRBB)
- **PV01**: Present value sensitivity by tenor bucket
- **∆EVE** and **∆NII**: Rate scenario shocks applied to the term structure

Each metric is computed dynamically depending on the scenario selected in the dashboard, such as liquidity constraints or rate shocks.

---

## Step 4: Dashboard Front-End (Streamlit)

The application is served through a multi-page **Streamlit dashboard**, with dedicated views for:

- **Home**: Snapshot of most important KPIs
- **Liquidity**: LCR and NSFR with stress overlays and regulatory requirements
- **IRRBB**: Interest rate sensitivity and ∆EVE/∆NII charts
- **RWA and Capital**: CET1 and Total Capital breakdowns, with requirements
- **Stress Testing**: Interactive tools for stress test scenarios

Interactive features include:
- Scenario selection toggle
- Filters for exposure class or reporting date
- Dynamic charts powered by `plotly`

Below is an example from the IRRBB page showing a ∆EVE bar chart under parallel shock scenarios:

```python
# dashboard/IRRBB.py

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.subheader("∆EVE Under EBA IRRBB Shock Scenarios")

df_eve = compute.calculate_eve_eba_scenarios(scenario_id=scenario_id)

fig = px.bar(
    df_eve,
    x='Scenario',
    y='Delta EVE',
    text_auto='.2f',
    color='Delta EVE',
    color_continuous_scale='RdYlGn',
    title=f"∆EVE Across EBA IRRBB Shocks – Scenario: {scenario_label}"
)

fig.update_layout(
    yaxis_title="Delta EVE (EUR)",
    xaxis_title="Scenario",
    height=400
)

st.plotly_chart(fig, use_container_width=True)
```

---

## Step 5: Deployment Pipeline

The stack is hosted using entirely free-tier services:

- **PostgreSQL**: Provisioned on [Railway](https://railway.app)
- **Dashboard**: Deployed on [Streamlit Cloud](https://streamlit.io/cloud)
- **Credential Management**: Secure `.streamlit/secrets.toml` setup for local and remote environments

This approach makes the solution reproducible and cost-effective for personal portfolios or internal tools.

---

## Try the App

👉 [Live Dashboard](https://basel-risk-pipeline-tmartins.streamlit.app/)  
👉 [GitHub Repo](https://github.com/thomasmartins/basel-risk-pipeline)

---

## About Me

I’m Thomas Martins, a quantitative risk analyst with a background in statistics and macroeconomics. I use my spare time to build technically rigorous projects that bridge financial regulation and open-source engineering.

More at [thomasmartins.github.io](https://thomasmartins.github.io)