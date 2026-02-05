## Customer Engagement Ranker (Yobi → HubSpot)

### Overview
We will build a weekly batch job that pulls customer usage metrics from Yobi and writes **three** engagement properties onto the corresponding **HubSpot Company** record:

- `customer_engagement_score` (integer)
- `customer_engagement_rank` (string, e.g. `"26 out of 150"`)
- `customer_engagement_tier` (string label: `"Top 30%"`, `"Middle 40%"`, `"Bottom 30%"`)

Customers are ranked weekly by a weighted score computed from:

- **New patient appointments booked** (weight 4)
- **Total appointments booked** (weight 3)
- **Tasks created** (weight 2)
- **Total calls** (weight 1)

The job runs **weekly** for the **Sun–Sat** week in a **single fixed timezone**: **America/New_York**. It is scheduled for **Sunday 06:00 ET** to reduce risk of end-of-day data latency.

This spec assumes we can use the existing integration patterns in `/Users/mdilatush/myAgents/SimpleAgent/` (Python clients in `lib/yobi.py` and `lib/hubspot.py`) and adapt them into this standalone job.

---

### Goals
- **Weekly** engagement scoring and ranking across all active paying customers.
- Write score/rank/tier to HubSpot Company properties reliably and deterministically.
- Provide a weekly **Slack/email summary**: top 10, bottom 10, biggest movers, anomalies.
- **Fail-fast** behavior: if any critical step fails, do **not** write partial results to HubSpot.

### Non-goals (for v1)
- Real-time updates or continuous streaming.
- Customer-facing UI.
- Historical time series in HubSpot (we will only write “current week’s” values unless we add a datastore).
- Perfect “attribution” if Yobi cannot provide clean per-tenant weekly breakdowns; we’ll implement best-available approach with explicit validation.

---

### Key decisions already made
- **Week definition**: Sun–Sat in a single fixed timezone.
- **Timezone**: `America/New_York`.
- **Schedule**: Sunday 06:00 ET.
- **Customer universe**: derived from HubSpot **Contacts** with `lifecyclestage=customer`, then mapped to their associated **Company** records (and any additional exclusions defined below).
- **Yobi ↔ HubSpot join**: do **not** rely on HubSpot storing Yobi tenant IDs. Instead, map HubSpot companies to Yobi tenants using deterministic matching (domain/email-domain/name) + explicit overrides (see Identity Resolution).
- **Yobi activity scope**: customer user activity only (exclude our internal staff/service accounts where applicable).
- **Rank format**: human-readable string like `"26 out of 150"`.
- **Tie-breaking**: break ties deterministically by metric priority: new patients > total appointments > tasks > calls, then stable by HubSpot company id.
- **Update semantics**: fail-fast; if a critical step fails, no HubSpot writes.
- **Summary**: weekly Slack/email includes top 10, bottom 10, movers vs prior week, anomalies.
- **Scale target**: <250 companies, <5 minutes runtime.

---

## Data sources & access

### HubSpot
**Auth**
- Use HubSpot Private App token via env var: `HUBSPOT_ACCESS_TOKEN`.

**Customer universe selection (contacts-first)**
- Primary filter: HubSpot **Contact** `lifecyclestage = "customer"`.
- For each matching contact, fetch associated **Company** records.
- Deduplicate companies (a company with multiple customer contacts is still a single ranked customer).
- Optional exclusion filters (recommended to add in v1 to avoid embarrassing rankings):
  - Exclude companies with an explicit boolean flag like `is_internal_test = true`
  - Exclude companies with a churned status property if present (e.g. `customer_status = "churned"`)
  - Exclude companies we cannot map to a Yobi tenant (treated as anomaly; see fail-fast nuance below)

**Properties to read**
- Contact: `lifecyclestage` (and optionally email for domain-based mapping).
- Company: `name`, `domain` (critical for mapping), plus:
  - `customer_engagement_rank` (for prior-week mover deltas; parse `"X out of N"` if present).
- `customer_engagement_score` (optional; for sanity checks).
- `customer_engagement_tier` (optional).

**Properties to write**
- `customer_engagement_score`
- `customer_engagement_rank`
- `customer_engagement_tier`

**Property types/constraints**
We selected “exist but unsure”. The job must validate property definitions at startup:
- `customer_engagement_score`: number (integer preferred)
- `customer_engagement_rank`: single-line text
- `customer_engagement_tier`: single-line text or enumeration (if enumeration, values must match exactly)

If property metadata is missing or incompatible, the job should fail-fast with an actionable error.

### Yobi
**Auth**
- Use env vars: `YOBI_EMAIL`, `YOBI_PASSWORD`.
- The existing client in `SimpleAgent/lib/yobi.py` logs in and then calls admin endpoints under `https://api.app.yib.io/api/v1`.

**Admin access scope**
Unknown whether the selected credentials can access cross-tenant admin endpoints for all customers. This must be verified in the Spike section.

**Metrics needed per tenant for the week**
- total calls
- tasks created
- total appointments booked
- new patient appointments booked

We will prefer **admin/statistics endpoints** that return per-tenant aggregates for a period, or per-day trend data that we can sum over the week window.

---

## Identity resolution (HubSpot customer → Yobi tenant)

Because HubSpot does not store Yobi tenant IDs for all customers, we must build a robust mapping strategy.

### Inputs
- HubSpot Company:
  - `domain` (preferred)
  - `name`
- HubSpot Contact:
  - `email` (for email-domain fallback when Company domain is missing)
- Yobi tenant list:
  - via admin endpoint (see `SimpleAgent/lib/yobi.py`): `/admin/dental-tenants/list`
  - expected fields include `tenant_id`, `tenant_name` and possibly website/domain (to be verified)

### Matching algorithm (deterministic, layered)
For each HubSpot Company (deduped from `lifecyclestage=customer` contacts):
1. **Overrides first**: if company id appears in `TENANT_MAPPING_OVERRIDES_JSON`, use that tenant id.
2. **Exact domain match** (preferred):
   - Normalize HubSpot company domain (lowercase, strip `www.`, strip scheme).
   - Match against any domain/website field available from Yobi tenant list (if present).
3. **Email-domain match**:
   - If company domain missing, infer from associated customer contacts’ email domains (exclude generic domains like gmail/yahoo/outlook).
   - Attempt match as in step (2).
4. **Name match** (fallback):
   - Normalize names (lowercase, remove punctuation, collapse whitespace, strip legal suffixes like LLC/Inc/PLLC).
   - Match against Yobi `tenant_name`.
5. If multiple Yobi tenants match, disambiguate deterministically:
   - Prefer exact domain match over name match.
   - Prefer the tenant with the highest weekly activity for the window (calls+tasks+appts) if we can compute cheaply; otherwise choose lowest `tenant_id` and flag anomaly.

### Output
- `company_id -> tenant_id` mapping table for this run.
- A list of unmapped companies (anomalies).

### Operational notes
- We should expect unmapped companies early on. The overrides file becomes the “escape hatch” and should be maintained over time.
- The job should never guess silently; any name-match mapping should be logged as “low confidence”.

---

## Weekly window definition
Let:
- \(TZ = \) `America/New_York`
- The job runs at Sunday 06:00 in \(TZ\).
- The “scored week” is the most recently completed Sun–Sat window.

Compute:
- `week_end` = the most recent Saturday at 23:59:59.999 in \(TZ\)
- `week_start` = the preceding Sunday at 00:00:00.000 in \(TZ\)

All Yobi metrics are aggregated over \([week_start, week_end]\) inclusive, in \(TZ\).

---

## Metric extraction (Yobi)

### Preferred approach (if Yobi supports week periods directly)
If Yobi admin endpoints accept a `period` that can represent the prior full week (e.g. `"lastWeek"` or equivalent), use that period for:
- calls
- tasks created
- appointments

### Fallback approach (if Yobi only supports “last 7 days” or “30 days”)
If Yobi only supports periods like `7days` / `30days` but returns **trend data by day**, then:
- Request a period that guarantees we can cover \([week_start, week_end]\) (e.g. `30days`)
- Sum only daily buckets whose date falls inside \([week_start, week_end]\) in \(TZ\)

If an endpoint returns only a single aggregate number (no breakdown) and cannot be constrained to the exact week window, the job must fail-fast until we implement a correct method.

### Mapping Yobi response fields to our metrics
Exact response keys must be confirmed by the Spike. Tentative mapping based on existing code:

- **Calls**:
  - Endpoint candidates:
    - `/admin/pms/stats/calls` (per-tenant with `tenant_id`)
    - `/admin/pms/stats/calls-by-tenant` (all tenants)
  - Candidate field: `stats.total_calls`

- **Tasks created**:
  - Endpoint candidate:
    - `/admin/dental-tenants/tasks/created`
  - Candidate field: `results.total.created` (or similar)

- **Appointments**:
  - Endpoint candidates:
    - `/admin/pms/stats/appointments` (per location) and `/admin/pms/stats/appointments-by-location`
    - tenant → location mapping may be required via `/admin/pms/locations`
  - Candidate fields:
    - `stats.total_appointments` (total)
    - **New patient appointments**: key TBD (may be `new_patients`, `new_patient_appointments`, or derived from other fields). This is a required field; if not available, the job must fail-fast.

### “Customer user only” filtering
If Yobi admin endpoints already measure tenant activity that excludes our internal staff, no extra filtering is needed.
If internal activity can leak into metrics, we will implement an exclusion mechanism:
- Maintain an allowlist/denylist of internal Yobi user IDs or email domains to exclude (configurable).
- Apply filtering only if the underlying endpoints provide a breakdown attributable to user (otherwise this remains a documented limitation).

---

## Scoring, ranking, and tiering

### Score
For each company \(c\):

\[
score(c) =
4 \cdot newPatientsBooked(c) +
3 \cdot totalAppointmentsBooked(c) +
2 \cdot tasksCreated(c) +
1 \cdot totalCalls(c)
\]

All component metrics are integers for the weekly window, and `customer_engagement_score` is stored as an integer.

### Rank
- Compute scores for all included companies.
- Sort descending by:
  1. `newPatientsBooked`
  2. `totalAppointmentsBooked`
  3. `tasksCreated`
  4. `totalCalls`
  5. stable tie-breaker: HubSpot company id (ascending) to guarantee deterministic output
- Assign rank as 1..N after sorting.

Store `customer_engagement_rank` as: `"{rank} out of {N}"`.

### Tier
Tier is based on rank position:
- `Top 30%`: ranks \(1..ceil(0.30 \cdot N)\)
- `Bottom 30%`: ranks \( (N - ceil(0.30 \cdot N) + 1)..N \)
- `Middle 40%`: all remaining ranks

Store `customer_engagement_tier` as one of:
- `"Top 30%"`
- `"Middle 40%"`
- `"Bottom 30%"`

Edge cases:
- If \(N < 3\), tiers still apply via the `ceil` formula; this will bias toward Top/Bottom. That’s acceptable for v1.

---

## HubSpot write behavior (fail-fast)

### Fail-fast rule
The job should not write anything to HubSpot unless **all** of these are true:
- HubSpot auth works and required properties are valid.
- Yobi auth works and required endpoints/fields are available.
- We computed metrics for all companies we intend to update (i.e., companies that mapped to Yobi tenants).
- We successfully generated a complete update set for N companies.

If any of the above fails, exit non-zero and send an alert (Slack/email).

**Important nuance (mapping gaps)**: because tenant IDs are not stored for all customers, some HubSpot companies may be **unmappable** in v1. Those companies are excluded from ranking for that week and are reported as anomalies. We still update HubSpot for the mapped set, but we do not clear or overwrite fields for unmapped companies.

### Update payload
For each company:
- `customer_engagement_score`: integer
- `customer_engagement_rank`: `"X out of N"`
- `customer_engagement_tier`: one of the 3 labels

### Batching / rate limits
Use batch endpoints where possible (HubSpot supports batch update for CRM objects). If not implemented initially:
- Update sequentially with a conservative rate (sleep/backoff on 429).
- Still compute all updates first, then apply writes (two-phase behavior).

---

## Weekly Slack/email summary
We will send a weekly summary after a successful HubSpot update:

- **Top 10**: company name, rank, score, tier
- **Bottom 10**: same
- **Biggest movers**: compare prior week rank vs new rank (delta)
  - Prior rank is read from existing `customer_engagement_rank` before writing new values (parse leading integer).
  - If prior rank is missing/unparseable, mark as “new/unknown”.
- **Anomalies**:
  - Unmappable HubSpot company → Yobi tenant (no domain/email-domain/name match; or ambiguous match)
  - Yobi metrics missing for a tenant (if we choose not to hard-fail)
  - Property type mismatches / blocked writes
  - Any API warnings/retries encountered

Slack implementation option:
- Incoming webhook via env var `SLACK_WEBHOOK_URL`, or
- Slack bot token with chat.postMessage (if needed later)

---

## Configuration (env vars)

### Required
- `HUBSPOT_ACCESS_TOKEN`
- `YOBI_EMAIL`
- `YOBI_PASSWORD`

### Recommended
- `JOB_TIMEZONE=America/New_York`
- `JOB_RUN_CRON=0 6 * * 0` (Sunday 06:00, scheduler-dependent)
- `SLACK_WEBHOOK_URL`

### Optional
- `HUBSPOT_COMPANY_EXCLUDE_FILTERS_JSON` (explicit exclusion rules; v1 can hardcode)
- `INTERNAL_YOBI_USER_EMAIL_DOMAINS` (comma-separated)
- `TENANT_MAPPING_OVERRIDES_JSON` (JSON string mapping HubSpot company ids/domains to Yobi tenant ids)
- `GENERIC_EMAIL_DOMAINS` (override list; defaults include gmail/yahoo/outlook/etc.)

---

## Execution & scheduling (GitHub Actions)

We will execute the weekly job via a **scheduled GitHub Actions workflow**.

### Workflow
- Location: `.github/workflows/weekly_customer_engagement_ranker.yml`
- Triggers:
  - `schedule` (weekly)
  - `workflow_dispatch` (manual run)

### Timezone note (UTC-only cron)
GitHub Actions cron schedules are **UTC-only**. For v1 we accept DST drift and schedule a single UTC time:
- Sunday **11:05 UTC** (06:05 ET in winter; 07:05 ET in summer)

If we later need “exactly Sunday 06:00 ET year-round”, we should switch to an hourly schedule with an ET gate inside the job.

### Secrets
Store credentials in GitHub repo **Actions secrets**:
- `HUBSPOT_ACCESS_TOKEN`
- `YOBI_EMAIL`
- `YOBI_PASSWORD`
- `SLACK_WEBHOOK_URL` (optional)

---

## Implementation plan (v1)

### Components
- **Extractor**: fetch HubSpot customer contacts → associated companies, then fetch required Yobi stats for the week.
- **Scorer**: compute metrics → score → rank → tier.
- **Writer**: validate HubSpot properties, perform two-phase writes (compute then write), fail-fast on any write error.
- **Notifier**: send Slack/email summary and anomalies report.

### Data flow
1. Load config + validate required env vars.
2. HubSpot: query contacts where `lifecyclestage=customer` and collect associated companies.
3. HubSpot: read needed company properties (`name`, `domain`, prior rank fields).
4. Yobi: fetch tenant list; build company→tenant mapping via Identity Resolution.
5. Compute score/rank/tier deterministically.
6. Preflight:
   - validate N > 0
   - validate no missing required stats for mapped tenants
   - validate HubSpot property metadata/types
7. Write updates to HubSpot (batch if possible).
8. Send Slack/email summary.

---

## Spike / verification (required before build is “done”)
Two unknowns must be verified with a short spike script before finalizing the implementation:

1. **Yobi admin access + weekly field availability**
   - With the intended `YOBI_EMAIL`/`YOBI_PASSWORD`, verify we can:
     - call a cross-tenant stats endpoint (or reliably iterate tenant ids)
     - obtain required fields for:
       - total calls
       - tasks created
       - total appointments booked
       - **new patient appointments booked**
     - constrain to the exact Sun–Sat week window (directly or via trends aggregation)
   - Outcome: confirm the final endpoint list + response-field mapping, or adjust approach.

2. **Company→tenant mapping feasibility**
   - Verify Yobi tenant list includes enough identifying data (tenant name and ideally website/domain).
   - Verify HubSpot companies for `lifecyclestage=customer` have `domain` populated at a high rate, or contacts have usable email domains.
   - Run mapping for a small sample and measure:
     - coverage (% mapped)
     - ambiguity rate (% multiple matches)
     - confidence breakdown (domain vs email-domain vs name match)
   - Outcome: confirm the matching algorithm + define initial `TENANT_MAPPING_OVERRIDES_JSON`.

If either verification fails, the job must not ship until we revise the approach (e.g., use different Yobi credentials or a different Yobi reporting endpoint).

---

## Testing plan
- **Unit tests**:
  - week window computation for multiple dates (including DST transitions)
  - tier boundary math for various N
  - parsing `"X out of N"` for prior rank
  - deterministic sorting / tie-breaks
- **Integration tests (manual / staged)**:
  - run against a small allowlist of companies (5–10) and verify HubSpot updates
  - verify Slack summary formatting
- **Safety checks**:
  - dry-run mode: compute and print summary without writing to HubSpot
  - idempotency: rerunning the job for the same week yields same ranks (given same underlying data)

---

## Rollout
- Phase 0: Spike verification (Yobi admin fields + company→tenant mapping feasibility).
- Phase 1: Dry-run in production credentials, no HubSpot writes; validate counts and ranking plausibility.
- Phase 2: Limited write to an allowlist of companies.
- Phase 3: Full rollout + scheduler enabled.
