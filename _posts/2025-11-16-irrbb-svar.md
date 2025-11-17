---
title: "Bayesian SVAR analysis for macroeconomic and IRRBB forecasting with Python"
description: "A research project that links Bayesian macroeconomic modeling with Basel III IRRBB scenarios and NII/EVE impacts."
author: "Thomas Martins"
layout: single
author_profile: true
date: 2025-11-16
permalink: /posts/irrbb-svar/
tags: 
  - IRRBB
  - Basel III
  - SVAR
  - BVAR
  - Risk Management
  - PyMC
keywords: [irrbb, basel-iii, svar, bvar, risk-management, pymc]
categories: ["Risk Management", "Bayesian Analysis"]
---


Interest rate risk in the banking book (IRRBB) is one of the fundamental components of Basel III regulations. Supervisors require banks to evaluate the impact of standardized interest-rate shocks on **Net Interest Income (NII)** and **Economic Value of Equity (EVE)**. These shocks represent a series of pre-specified movements to the yield curve.

But real-world interest rate changes are **not arbitrary shifts**.  
They occur in a macro-financial system where:

- GDP and inflation respond to policy,
- deficits and debt interact with long-term rates,
- the yield curve embeds expectations about future monetary actions.

So a natural question emerges:

> **What if we embed the Basel IRRBB scenarios inside a full macroeconomic model?**  
> *What if we allow GDP, inflation, debt, and the policy rate to react consistently to these scenarios?*

This project builds exactly that end-to-end pipeline using modern Bayesian tools in Python.

👉 **Full technical notebook:**   
https://thomasmartins.github.io/IRRBB_SVAR_forecasting/notebooks/MAIN.html

---

# 1. Overview of the pipeline

The project integrates four components:

1. **Bayesian VAR (BVAR)** to capture macro-financial dynamics  
2. **SVAR identification using sign restrictions** to simulate macroeconomic scenarios
3. **Basel III IRRBB yield-curve scenarios** mapped into level/slope constraints  
4. **NII and EVE calculations** under both regulatory and structural shocks

Everything is estimated with **PyMC + BlackJAX**, producing full posterior uncertainty.

The codebase is modular (in `/src`), and the notebook walks through the full system from raw data to final NII/EVE estimates.

---

# 2. Data: the macro-financial panel

The dataset includes:

- Real GDP growth (q/q annualised)  
- Inflation (annualised)  
- Government deficit (% GDP, sign-corrected)  
- Debt-to-GDP  
- ECB policy rate  
- Yield curve **level**, **slope**, **curvature** factors  

These variables represent the minimal set needed to link:

- macroeconomic variables
- monetary and fiscal policy   
- term structure movements  

into a multivariate system.

---

# 3. Bayesian VAR (BVAR)

We estimate a VAR with:

- **Minnesota-style priors** (shrinkage) on coefficients  
- **HMC/NUTS sampling** via PyMC and BlackJAX  
- Full posterior distributions for all parameters  

This gives a flexible, stable, and regularized representation of macro-financial dynamics suitable for conditional forecasts.

---

# 4. SVAR identification with sign restrictions

Causality is hard to disentangle in reduced-form VAR shocks.  
This is why we impose **sign restrictions** on impulse responses to analyze macroeconomic scenarios.

### Monetary tightening shock

- Policy rate ↑ on impact  
- Yield curve flattens  
- Inflation ↓ (lagged)  
- GDP ↓ (lagged)

### Fiscal expansion shock

- Deficit increases  
- Level of the curve ↑ (moderately)  
- GDP ↑ (lagged)  
- Inflation ↑ (lagged)  
- Debt ↑ over the medium run  

Posterior-accepted structural shocks generate clean IRFs: they show the expected disinflation and flattening after a monetary hike, and the expected demand-side effects after a fiscal expansion (increased GDP and inflation).

---

# 5. Basel III IRRBB scenarios

The ECB prescribes six regulatory yield-curve shocks:

- **Parallel up / parallel down**  
- **Steepener / flattener**  
- **Short-end up / short-end down**

These are mapped into **factor constraints** on the:

- level  
- slope  
- curvature  

of the yield curve.

For example, "Parallel up": the level increases by 200 bps  

This allows the Basel scenarios to be imposed as **future path constraints** inside the BVAR.

---

# 6. Conditional forecasts

To simulate scenarios, I use **Waggoner–Zha conditioning**:

- Start with the unconditional BVAR forecast distribution of future macro variables.
- Impose **linear constraints** on future level/slope values.
- Compute the **conditional mean + covariance** for the entire system.

This gives *model-consistent macro reactions* to Basel-style yield shifts.

A key insight from the results:

> **Basel scenarios barely move GDP or inflation**  
> — exactly as expected, because these are *not* macroeconomic scenarios, but regulatory curve shocks.

Structural SVAR shocks, however, show clear macro dynamics.

---

# 7. NII and EVE: linking macro-finance to bank risk

Finally, I translate the conditional yield-curve paths into IRRBB metrics:

### **NII (Net Interest Income)**  
A flow measure: how earnings change over 1 year due to repricing.

### **EVE (Economic Value of Equity)**  
A stock measure: present value of the banking book, approximated with durations.

The implementation is stylised but realistic:

- Repricing buckets for NII
- Pass-through from market rates to customer rates
- Duration-based EVE impacts
- CET1 normalization

Some of the insights we get:

- Parallel-up increases NII, decreases EVE  
- Steepener/Flattener produce similar EVE due to the simplified duration structure of our model balance sheet
- Monetary tightening slightly reduces NII and increases EVE

---

# 8. Full notebook and repository

- 📒 **Jupyter notebook:**  
  https://thomasmartins.github.io/IRRBB_SVAR_forecasting/notebooks/MAIN.html

- 📦 **GitHub repo:**  
  https://github.com/thomasmartins/IRRBB_SVAR_forecasting

---

# 9. Final thoughts

This approach allows us to analyze IRRBB in a macroeconomic context, connecting curve movements to policy scenarios and macroeconomic variables.

This project demonstrates how Bayesian econometrics, SVAR identification, and IRRBB regulatory analytics can be combined into a coherent workflow.
