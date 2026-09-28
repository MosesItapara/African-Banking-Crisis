---
title: African Banking Crisis API
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
short_description: Banking-crisis risk forecasts for 13 African countries
---

# African Banking Crisis API

Forecasts the probability of a banking crisis from a country's macro indicators.
Source code, pipeline and MLOps setup: https://github.com/MosesItapara/African-Banking-Crisis

## Endpoints

- `GET /health`: status and model version
- `GET /countries`: supported countries and the next year each can predict
- `POST /predict`: JSON with `cc3`, `year` and the raw indicators