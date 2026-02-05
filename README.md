# TechCluster Customer Engagement Ranker

Weekly job to pull customer usage data from Yobi, compute an engagement score/rank/tier, and write results into HubSpot.

The full implementation details live in `Customer Ranker Update/spec.md`.

## GitHub Actions schedule

This repo includes a scheduled workflow at `.github/workflows/weekly_customer_engagement_ranker.yml`.

- **Schedule**: Sunday 11:05 UTC
- **Note**: GitHub cron is UTC-only; this will drift by 1 hour vs ET during DST (accepted for v1).
- **Manual runs**: supported via **Run workflow** (`workflow_dispatch`).

## Required secrets

Set these in GitHub repo settings → **Secrets and variables** → **Actions**:

- `HUBSPOT_ACCESS_TOKEN`
- `YOBI_EMAIL`
- `YOBI_PASSWORD`
- `SLACK_WEBHOOK_URL` (optional for v1)

## Local run (placeholder)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/customer_engagement_ranker.py
```

