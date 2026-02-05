"""
Weekly Customer Engagement Ranker (Yobi → HubSpot)

This script:
- Builds the HubSpot "customer universe" from Contacts where lifecyclestage=customer
- Dedupe to Companies
- Maps each HubSpot Company to a Yobi tenant (domain/name match + overrides)
- Pulls Yobi weekly usage metrics
- Computes score/rank/tier and updates HubSpot Company properties

Modes:
- Dry run (default): compute + print summary; no HubSpot writes
- Write mode: updates HubSpot for an explicit allowlist of company IDs

Spec: `Customer Ranker Update/spec.md`
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HUBSPOT_API_BASE = "https://api.hubapi.com"
YOBI_API_BASE_DEFAULT = "https://api.app.yib.io/api/v1"


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_csv(name: str) -> List[str]:
    v = os.environ.get(name, "").strip()
    if not v:
        return []
    return [p.strip() for p in v.split(",") if p.strip()]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Config:
    universe: str  # "hubspot" or "yobi"
    hubspot_access_token: str
    yobi_email: str
    yobi_password: str
    yobi_base_url: str

    timezone_name: str
    dry_run: bool
    full_write: bool
    max_companies: int
    max_contact_pages: int
    company_ids_allowlist: List[str]

    active_calls_period: str
    active_calls_threshold: int

    slack_webhook_url: Optional[str]
    tenant_mapping_overrides: Dict[str, int]  # hubspot company id -> yobi tenant id

    @classmethod
    def from_env(cls) -> "Config":
        universe = (os.environ.get("UNIVERSE") or "hubspot").strip().lower() or "hubspot"
        if universe not in ("hubspot", "yobi"):
            universe = "hubspot"
        hubspot_access_token = os.environ.get("HUBSPOT_ACCESS_TOKEN", "").strip()
        yobi_email = os.environ.get("YOBI_EMAIL", "").strip()
        yobi_password = os.environ.get("YOBI_PASSWORD", "").strip()
        yobi_base_url = os.environ.get("YOBI_BASE_URL", "").strip() or YOBI_API_BASE_DEFAULT

        timezone_name = os.environ.get("JOB_TIMEZONE", "America/New_York").strip() or "America/New_York"
        dry_run = _env_bool("DRY_RUN", True)
        full_write = _env_bool("FULL_WRITE", False)
        max_companies = _env_int("MAX_COMPANIES", 250)
        max_contact_pages = _env_int("MAX_CONTACT_PAGES", 50)
        company_ids_allowlist = _env_csv("COMPANY_IDS")
        slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "").strip() or None

        active_calls_period = (os.environ.get("ACTIVE_CALLS_PERIOD") or "30days").strip() or "30days"
        active_calls_threshold = _env_int("ACTIVE_CALLS_THRESHOLD", 30)

        overrides_raw = os.environ.get("TENANT_MAPPING_OVERRIDES_JSON", "").strip()
        tenant_mapping_overrides: Dict[str, int] = {}
        if overrides_raw:
            try:
                parsed = json.loads(overrides_raw)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        if k is None:
                            continue
                        try:
                            tenant_mapping_overrides[str(k)] = int(v)
                        except Exception:
                            continue
            except json.JSONDecodeError:
                # Treat as unset; we keep going in dry run.
                tenant_mapping_overrides = {}

        return cls(
            universe=universe,
            hubspot_access_token=hubspot_access_token,
            yobi_email=yobi_email,
            yobi_password=yobi_password,
            yobi_base_url=yobi_base_url,
            timezone_name=timezone_name,
            dry_run=dry_run,
            full_write=full_write,
            max_companies=max_companies,
            max_contact_pages=max_contact_pages,
            company_ids_allowlist=company_ids_allowlist,
            active_calls_period=active_calls_period,
            active_calls_threshold=active_calls_threshold,
            slack_webhook_url=slack_webhook_url,
            tenant_mapping_overrides=tenant_mapping_overrides,
        )


# ---------------------------------------------------------------------------
# HubSpot client (minimal)
# ---------------------------------------------------------------------------


class HubSpotClient:
    def __init__(self, access_token: str, timeout_s: int = 30):
        if not access_token:
            raise ValueError("Missing HUBSPOT_ACCESS_TOKEN")
        self._timeout_s = timeout_s
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }
        )

    def _request(self, method: str, path: str, *, params: Dict[str, Any] | None = None, json_body: Any = None) -> Any:
        url = f"{HUBSPOT_API_BASE}{path}"
        resp = self._session.request(method, url, params=params, json=json_body, timeout=self._timeout_s)
        if resp.status_code >= 400:
            raise RuntimeError(f"HubSpot {method} {path} failed: {resp.status_code} {resp.text[:500]}")
        if not resp.content:
            return {}
        return resp.json()

    def get_company_property(self, prop_name: str) -> Dict[str, Any]:
        return self._request("GET", f"/crm/v3/properties/companies/{prop_name}")

    def search_contacts_lifecyclestage_customer(
        self,
        *,
        limit_per_page: int = 100,
        max_pages: int = 50,
        properties: Sequence[str] = ("email", "lifecyclestage"),
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        after: Optional[str] = None
        for _ in range(max_pages):
            payload: Dict[str, Any] = {
                "filterGroups": [
                    {
                        "filters": [
                            {"propertyName": "lifecyclestage", "operator": "EQ", "value": "customer"},
                        ]
                    }
                ],
                "properties": list(properties),
                "limit": limit_per_page,
            }
            if after:
                payload["after"] = after
            data = self._request("POST", "/crm/v3/objects/contacts/search", json_body=payload)
            page_results = data.get("results", []) or []
            results.extend(page_results)
            paging = data.get("paging", {})
            next_after = paging.get("next", {}).get("after") if isinstance(paging, dict) else None
            if not next_after:
                break
            after = str(next_after)
        return results

    def get_contact_company_ids(self, contact_id: str) -> List[str]:
        data = self._request("GET", f"/crm/v3/objects/contacts/{contact_id}/associations/companies")
        ids = []
        for r in data.get("results", []) or []:
            cid = r.get("id")
            if cid is not None:
                ids.append(str(cid))
        return ids

    def batch_get_contact_company_ids(self, contact_ids: Sequence[str]) -> Dict[str, List[str]]:
        """
        Batch fetch contact -> associated company IDs using v4 Associations batch read.

        Endpoint: POST /crm/v4/associations/contacts/companies/batch/read
        Docs: https://developers.hubspot.com/docs/api-reference/crm-associations-v4/guide
        """
        out: Dict[str, List[str]] = {str(cid): [] for cid in contact_ids}
        for i in range(0, len(contact_ids), 1000):
            chunk = contact_ids[i : i + 1000]
            payload = {"inputs": [{"id": str(cid)} for cid in chunk]}
            data = self._request("POST", "/crm/v4/associations/contacts/companies/batch/read", json_body=payload)
            for r in data.get("results", []) or []:
                from_id = str((r.get("from") or {}).get("id") or "")
                to_list = r.get("to") or []
                if not from_id:
                    continue
                company_ids = []
                for t in to_list:
                    tid = t.get("toObjectId") or t.get("id")
                    if tid is not None:
                        company_ids.append(str(tid))
                out[from_id] = company_ids
        return out

    def batch_read_companies(self, company_ids: Sequence[str], properties: Sequence[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        # HubSpot batch read supports up to 100 inputs.
        for i in range(0, len(company_ids), 100):
            chunk = company_ids[i : i + 100]
            payload = {
                "properties": list(properties),
                "inputs": [{"id": cid} for cid in chunk],
            }
            data = self._request("POST", "/crm/v3/objects/companies/batch/read", json_body=payload)
            out.extend(data.get("results", []) or [])
        return out

    def batch_update_companies(self, updates: Sequence[Tuple[str, Dict[str, Any]]]) -> None:
        # updates: [(company_id, {prop: value})]
        for i in range(0, len(updates), 100):
            chunk = updates[i : i + 100]
            payload = {
                "inputs": [{"id": cid, "properties": props} for (cid, props) in chunk],
            }
            self._request("POST", "/crm/v3/objects/companies/batch/update", json_body=payload)


# ---------------------------------------------------------------------------
# Yobi client (minimal)
# ---------------------------------------------------------------------------


class YobiClient:
    def __init__(self, *, email: str, password: str, base_url: str = YOBI_API_BASE_DEFAULT, timeout_s: int = 30):
        if not email or not password:
            raise ValueError("Missing YOBI_EMAIL or YOBI_PASSWORD")
        self._email = email
        self._password = password
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._session = requests.Session()
        self._token: Optional[str] = None
        self._token_expiry_ms: Optional[int] = None

    def _headers(self, include_content_type: bool = True) -> Dict[str, str]:
        h = {"Accept": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        if include_content_type:
            h["Content-Type"] = "application/json"
        return h

    def login(self) -> None:
        url = f"{self._base_url}/users/login"
        resp = self._session.post(
            url,
            json={"user": self._email, "password": self._password},
            headers={"Content-Type": "application/json"},
            timeout=self._timeout_s,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Yobi login failed: {resp.status_code} {resp.text[:500]}")
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"Yobi login unsuccessful: {data.get('message')}")
        results = data.get("results", {}) or {}
        self._token = results.get("token")
        self._token_expiry_ms = results.get("tokenExpirationInstant")
        if not self._token:
            raise RuntimeError("Yobi login succeeded but no token returned")

    def _ensure_auth(self) -> None:
        if not self._token:
            self.login()
            return
        if self._token_expiry_ms:
            now_ms = int(_utc_now().timestamp() * 1000)
            # refresh if expiring in next 5 minutes
            if now_ms >= (int(self._token_expiry_ms) - 300_000):
                self.login()

    def _request(self, method: str, path: str, *, params: Dict[str, Any] | None = None, json_body: Any = None) -> Any:
        self._ensure_auth()
        url = f"{self._base_url}{path}"
        include_ct = method.upper() in ("POST", "PUT", "PATCH")
        resp = self._session.request(
            method.upper(),
            url,
            params=params,
            json=json_body,
            headers=self._headers(include_content_type=include_ct),
            timeout=self._timeout_s,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Yobi {method} {path} failed: {resp.status_code} {resp.text[:500]}")
        return resp.json() if resp.content else {}

    def list_dental_tenants(self) -> List[Dict[str, Any]]:
        data = self._request("GET", "/admin/dental-tenants/list")
        results = data.get("results", {})
        if isinstance(results, dict):
            return results.get("results", results.get("tenants", [])) or []
        if isinstance(results, list):
            return results
        return []

    def get_pms_locations(self) -> List[Dict[str, Any]]:
        data = self._request("GET", "/admin/pms/locations")
        results = data.get("results", {})
        if isinstance(results, dict):
            return results.get("results", results.get("locations", [])) or []
        if isinstance(results, list):
            return results
        return []

    def calls_by_tenant(self, *, period: str) -> Any:
        return self._request(
            "GET",
            "/admin/pms/stats/calls-by-tenant",
            params={"period": period, "include_non_integrated": "true"},
        )

    def tasks_created(self, *, period: str) -> Any:
        return self._request("GET", "/admin/dental-tenants/tasks/created", params={"period": period})

    def appointments_by_location(self, *, period: str) -> Any:
        return self._request("GET", "/admin/pms/stats/appointments-by-location", params={"period": period})


# ---------------------------------------------------------------------------
# Ranking logic
# ---------------------------------------------------------------------------


def _normalize_domain(domain: str) -> str:
    d = (domain or "").strip().lower()
    d = re.sub(r"^https?://", "", d)
    d = d.split("/")[0]
    d = re.sub(r"^www\.", "", d)
    return d


_LEGAL_SUFFIXES = (" llc", " inc", " inc.", " co", " co.", " ltd", " pllc", " pc", " p.c.", " corporation", " corp", " corp.")


def _normalize_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for suf in _LEGAL_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
    return s


def _token_set(s: str) -> Set[str]:
    return set([t for t in _normalize_name(s).split(" ") if t])


_MATCH_STOPWORDS: Set[str] = {
    # Generic dental/practice words that cause false-positive matches
    "dental",
    "dentistry",
    "dentist",
    "care",
    "family",
    "smile",
    "smiles",
    "clinic",
    "center",
    "centre",
    "group",
    "associates",
    "associate",
    "office",
    "offices",
    "health",
    "services",
    "service",
    "cosmetic",
    "implant",
    "implants",
    "emergency",
    "kids",
    "pediatric",
    "orthodontics",
    "orthodontic",
    "endodontics",
    "endodontic",
    "periodontics",
    "periodontic",
    "oral",
    "surgery",
    "specialists",
    "specialist",
    "and",
    "of",
    "the",
}


def _meaningful_tokens(name: str) -> Set[str]:
    """Tokens for matching, with generic stopwords removed."""
    return {t for t in _token_set(name) if t not in _MATCH_STOPWORDS}


@dataclass
class CompanyRow:
    hubspot_company_id: str
    name: str
    domain: str
    prev_rank_raw: str


@dataclass
class TenantMetrics:
    tenant_id: int
    tenant_name: str
    total_calls: int = 0
    tasks_created: int = 0
    total_appointments: int = 0
    new_patient_appointments: int = 0

    @property
    def score(self) -> int:
        return (
            4 * int(self.new_patient_appointments or 0)
            + 3 * int(self.total_appointments or 0)
            + 2 * int(self.tasks_created or 0)
            + 1 * int(self.total_calls or 0)
        )


def _parse_prev_rank(prev_rank_raw: str) -> Optional[int]:
    if not prev_rank_raw:
        return None
    m = re.match(r"^\s*(\d+)\s+out\s+of\s+\d+\s*$", prev_rank_raw, flags=re.I)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _tier_for_rank(rank: int, n: int) -> str:
    import math

    top_n = int(math.ceil(0.30 * n))
    bottom_n = int(math.ceil(0.30 * n))
    if rank <= top_n:
        return "Top 30%"
    if rank > (n - bottom_n):
        return "Bottom 30%"
    return "Middle 40%"


# ---------------------------------------------------------------------------
# Yobi response parsing helpers (defensive)
# ---------------------------------------------------------------------------


def _safe_int(v: Any) -> int:
    if v is None:
        return 0
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return int(v)
    try:
        s = str(v).strip()
        if not s:
            return 0
        return int(float(s))
    except Exception:
        return 0


def _index_tenants(tenants: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for t in tenants:
        tid = t.get("tenant_id") or t.get("id")
        if tid is None:
            continue
        try:
            out[int(tid)] = t
        except Exception:
            continue
    return out


def _pick_tenant_for_company(
    *,
    company: CompanyRow,
    tenants: List[Dict[str, Any]],
    overrides: Dict[str, int],
) -> Tuple[Optional[int], str]:
    # Returns (tenant_id, reason)
    if company.hubspot_company_id in overrides:
        return overrides[company.hubspot_company_id], "override(company_id)"

    c_domain = _normalize_domain(company.domain)
    c_name_norm = _normalize_name(company.name)
    c_tokens = _meaningful_tokens(company.name)

    # domain match if tenant has any domain-like field
    if c_domain:
        for t in tenants:
            for key in ("domain", "website", "site", "url"):
                tv = t.get(key)
                if not tv:
                    continue
                t_domain = _normalize_domain(str(tv))
                if t_domain and t_domain == c_domain:
                    tid = t.get("tenant_id") or t.get("id")
                    if tid is not None:
                        return int(tid), f"domain({key})"

    # exact normalized name match
    for t in tenants:
        tn = str(t.get("tenant_name") or t.get("name") or "").strip()
        if not tn:
            continue
        if _normalize_name(tn) == c_name_norm and c_name_norm:
            tid = t.get("tenant_id") or t.get("id")
            if tid is not None:
                return int(tid), "name(exact)"

    # token overlap best-effort
    scored: List[Tuple[float, int]] = []
    for t in tenants:
        tid = t.get("tenant_id") or t.get("id")
        if tid is None:
            continue
        tn = str(t.get("tenant_name") or t.get("name") or "").strip()
        if not tn:
            continue
        t_tokens = _meaningful_tokens(tn)
        if not t_tokens or not c_tokens:
            continue
        overlap = len(c_tokens & t_tokens)
        denom = max(len(c_tokens), len(t_tokens))
        score = overlap / float(denom)
        # Require stronger signal when we have enough meaningful tokens
        min_overlap = 1
        if len(c_tokens) >= 2 and len(t_tokens) >= 2:
            min_overlap = 2
        if overlap >= min_overlap and score >= 0.5:
            scored.append((score, int(tid)))

    if scored:
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][1], f"name(token_overlap={scored[0][0]:.2f})"

    return None, "unmapped"


def _parse_calls_by_tenant(data: Any) -> Dict[int, int]:
    """
    Returns tenant_id -> total_calls.
    Shape varies; we handle common patterns:
    - results.stats: [{tenant_id, total_calls, ...}]
    - results: { stats: { tenant_id: {total_calls} } } (rare)
    """
    results = data.get("results", data) if isinstance(data, dict) else data
    stats = None
    if isinstance(results, dict):
        stats = results.get("stats") or results.get("results") or results.get("tenants")
    if stats is None and isinstance(data, dict):
        stats = data.get("stats")

    out: Dict[int, int] = {}
    if isinstance(stats, list):
        for row in stats:
            if not isinstance(row, dict):
                continue
            tid = row.get("tenant_id") or row.get("tenantId") or row.get("id")
            if tid is None:
                continue
            out[int(tid)] = _safe_int(row.get("total_calls") or row.get("totalCalls") or row.get("calls") or 0)
    elif isinstance(stats, dict):
        for k, v in stats.items():
            try:
                tid = int(k)
            except Exception:
                continue
            if isinstance(v, dict):
                out[tid] = _safe_int(v.get("total_calls") or v.get("calls") or 0)
            else:
                out[tid] = _safe_int(v)
    return out


def _parse_tasks_created(data: Any) -> Dict[int, int]:
    """
    Returns tenant_id -> tasks_created.
    We handle:
    - results.by_tenant: [{tenant_id, total: {created: n}}] or [{tenant_id, created: n}]
    - results: { by_tenant: [...] }
    """
    results = data.get("results", data) if isinstance(data, dict) else data
    by_tenant = None
    if isinstance(results, dict):
        by_tenant = results.get("by_tenant") or results.get("byTenant") or results.get("tenants") or results.get("results")
    out: Dict[int, int] = {}
    if isinstance(by_tenant, list):
        for row in by_tenant:
            if not isinstance(row, dict):
                continue
            tid = row.get("tenant_id") or row.get("tenantId") or row.get("id")
            if tid is None:
                continue
            created = row.get("created")
            if created is None and isinstance(row.get("total"), dict):
                created = row["total"].get("created")
            if created is None and isinstance(row.get("counts"), dict):
                created = row["counts"].get("created")
            out[int(tid)] = _safe_int(created)
    return out


def _parse_appointments_by_location(data: Any) -> List[Dict[str, Any]]:
    """
    Returns list of rows; each row should include location_id and stats-ish fields.
    """
    results = data.get("results", data) if isinstance(data, dict) else data
    if isinstance(results, dict):
        # common nesting for this endpoint: results.locations
        return (
            results.get("locations")
            or results.get("stats")
            or results.get("results")
            or results.get("by_location")
            or results.get("byLocation")
            or []
        )
    if isinstance(results, list):
        return results
    return []


def _aggregate_appointments_by_tenant(
    *,
    appt_rows: List[Dict[str, Any]],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Returns (tenant_total_appointments, tenant_new_patient_appointments)
    For `/admin/pms/stats/appointments-by-location`, rows include tenant_id directly.
    """
    total: Dict[int, int] = {}
    newp: Dict[int, int] = {}

    for row in appt_rows:
        if not isinstance(row, dict):
            continue
        tid = row.get("tenant_id") or row.get("tenantId")
        if tid is None:
            continue
        try:
            tenant_id = int(tid)
        except Exception:
            continue

        # Endpoint provides counts directly
        stats = row.get("stats") if isinstance(row.get("stats"), dict) else row
        total_appts = _safe_int(
            stats.get("appointment_count")
            or stats.get("appointments")
            or stats.get("total_appointments")
            or stats.get("totalAppointments")
            or stats.get("total")
        )
        new_patients = _safe_int(stats.get("new_patients") or stats.get("newPatients") or stats.get("new_patient_appointments"))

        total[tenant_id] = total.get(tenant_id, 0) + total_appts
        newp[tenant_id] = newp.get(tenant_id, 0) + new_patients

    return total, newp


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _compute_week_period_string(now: datetime, tz_name: str) -> str:
    """
    Yobi supports a limited set of `period` values (e.g. yesterday, 7days, 30days, ...).
    We default to 7days unless overridden via env var `YOBI_PERIOD`.

    Note: this is not an exact Sun–Sat window; see spec for future improvement.
    """
    _ = now, tz_name
    return (os.environ.get("YOBI_PERIOD") or "7days").strip() or "7days"


def _load_companies_from_hubspot(cfg: Config, hubspot: HubSpotClient) -> List[CompanyRow]:
    if cfg.company_ids_allowlist:
        props = ("name", "domain", "customer_engagement_rank")
        raw = hubspot.batch_read_companies(cfg.company_ids_allowlist, props)
        rows: List[CompanyRow] = []
        for r in raw:
            props_d = r.get("properties", {}) or {}
            rows.append(
                CompanyRow(
                    hubspot_company_id=str(r.get("id")),
                    name=str(props_d.get("name") or ""),
                    domain=str(props_d.get("domain") or ""),
                    prev_rank_raw=str(props_d.get("customer_engagement_rank") or ""),
                )
            )
        return rows

    # contacts-first universe, stop once we have enough unique companies
    contact_results = hubspot.search_contacts_lifecyclestage_customer(
        limit_per_page=100,
        max_pages=cfg.max_contact_pages,
        properties=("email",),
    )

    contact_ids = [str(c.get("id")) for c in contact_results if c.get("id") is not None]
    assoc_map = hubspot.batch_get_contact_company_ids(contact_ids)

    company_ids: List[str] = []
    seen: Set[str] = set()
    for cid in contact_ids:
        for comp_id in assoc_map.get(cid, []):
            if comp_id in seen:
                continue
            seen.add(comp_id)
            company_ids.append(comp_id)
            if len(company_ids) >= cfg.max_companies:
                break
        if len(company_ids) >= cfg.max_companies:
            break

    props = ("name", "domain", "customer_engagement_rank")
    raw = hubspot.batch_read_companies(company_ids, props)
    rows: List[CompanyRow] = []
    for r in raw:
        props_d = r.get("properties", {}) or {}
        rows.append(
            CompanyRow(
                hubspot_company_id=str(r.get("id")),
                name=str(props_d.get("name") or ""),
                domain=str(props_d.get("domain") or ""),
                prev_rank_raw=str(props_d.get("customer_engagement_rank") or ""),
            )
        )
    return rows


def _validate_hubspot_properties(hubspot: HubSpotClient) -> None:
    # Fail fast if properties are missing (types are not strictly enforced here).
    for prop in ("customer_engagement_score", "customer_engagement_rank", "customer_engagement_tier"):
        hubspot.get_company_property(prop)


def _fetch_yobi_metrics(cfg: Config, yobi: YobiClient) -> Tuple[List[Dict[str, Any]], Dict[int, TenantMetrics], List[str]]:
    anomalies: List[str] = []
    tenants = yobi.list_dental_tenants()
    tenants_by_id = _index_tenants(tenants)

    now = _utc_now()
    period = _compute_week_period_string(now, cfg.timezone_name)

    # Calls
    calls_raw = yobi.calls_by_tenant(period=period)
    calls_by_tid = _parse_calls_by_tenant(calls_raw)

    # Tasks created
    tasks_raw = yobi.tasks_created(period=period)
    tasks_by_tid = _parse_tasks_created(tasks_raw)

    # Appointments
    appt_raw = yobi.appointments_by_location(period=period)
    appt_rows = _parse_appointments_by_location(appt_raw)
    appts_total_by_tid, appts_newp_by_tid = _aggregate_appointments_by_tenant(appt_rows=appt_rows)

    # Build metrics objects for all tenants (zeros if missing)
    metrics_by_tid: Dict[int, TenantMetrics] = {}
    for tid in sorted(tenants_by_id.keys()):
        t = tenants_by_id.get(tid, {})
        t_name = str(t.get("tenant_name") or t.get("tenantName") or t.get("name") or f"tenant_{tid}")
        metrics_by_tid[tid] = TenantMetrics(
            tenant_id=tid,
            tenant_name=t_name,
            total_calls=calls_by_tid.get(tid, 0),
            tasks_created=tasks_by_tid.get(tid, 0),
            total_appointments=appts_total_by_tid.get(tid, 0),
            new_patient_appointments=appts_newp_by_tid.get(tid, 0),
        )

    # If we see only rates but not counts for new patient appointments, highlight.
    if metrics_by_tid and all(m.new_patient_appointments == 0 for m in metrics_by_tid.values()):
        anomalies.append("new_patient_appointments appears to be 0 for all tenants (Yobi may not provide count fields via this endpoint).")

    return tenants, metrics_by_tid, anomalies


def _send_slack(webhook_url: str, text: str) -> None:
    try:
        resp = requests.post(webhook_url, json={"text": text}, timeout=15)
        if resp.status_code >= 400:
            print(f"WARNING: Slack webhook failed: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"WARNING: Slack webhook error: {e}")


def main() -> int:
    cfg = Config.from_env()

    required = ["YOBI_EMAIL", "YOBI_PASSWORD"]
    if cfg.universe == "hubspot":
        required = ["HUBSPOT_ACCESS_TOKEN", "YOBI_EMAIL", "YOBI_PASSWORD"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print("Missing required secrets:", ", ".join(missing))
        # In scheduled automation we want failures visible; but keep exit 0 for now until fully live.
        return 0

    yobi = YobiClient(email=cfg.yobi_email, password=cfg.yobi_password, base_url=cfg.yobi_base_url)

    print(
        "Config:"
        f" dry_run={cfg.dry_run}"
        f" full_write={cfg.full_write}"
        f" max_companies={cfg.max_companies}"
        f" max_contact_pages={cfg.max_contact_pages}"
        f" allowlist={len(cfg.company_ids_allowlist)}"
        f" tz={cfg.timezone_name}"
        f" active_calls={cfg.active_calls_threshold}+/{cfg.active_calls_period}"
        f" universe={cfg.universe}"
    )

    if cfg.universe == "yobi" and not cfg.dry_run:
        print("ERROR: UNIVERSE=yobi supports dry-run only (no HubSpot writes).")
        return 1

    # Fail fast on Yobi auth (avoids doing HubSpot work when creds are wrong)
    try:
        yobi.login()
    except Exception as e:
        print("ERROR: Yobi authentication failed. Check `YOBI_EMAIL` / `YOBI_PASSWORD` secrets.")
        print(f"Details: {e}")
        return 1

    # Fetch Yobi stats
    try:
        tenants, metrics_by_tid, yobi_anomalies = _fetch_yobi_metrics(cfg, yobi)
    except Exception as e:
        print("ERROR: Failed to fetch Yobi metrics.")
        print(f"Details: {e}")
        return 1
    print(f"Loaded {len(tenants)} Yobi tenants; metrics available for {len(metrics_by_tid)} tenants.")

    # Determine "active" tenants by call volume (previous 30 days by default)
    try:
        active_calls_raw = yobi.calls_by_tenant(period=cfg.active_calls_period)
        calls_active_window_by_tid = _parse_calls_by_tenant(active_calls_raw)
    except Exception as e:
        print("ERROR: Failed to fetch Yobi calls-by-tenant for active filter.")
        print(f"Details: {e}")
        return 1

    active_tenant_ids: Set[int] = {
        int(tid) for tid, calls in calls_active_window_by_tid.items() if _safe_int(calls) > cfg.active_calls_threshold
    }

    if cfg.universe == "yobi":
        # Rank all Yobi tenants (active only by default)
        active_metrics: List[TenantMetrics] = [m for tid, m in metrics_by_tid.items() if int(tid) in active_tenant_ids]
        inactive_count = len(metrics_by_tid) - len(active_metrics)
        print(
            f"Active filter (Yobi universe): active_tenants={len(active_metrics)} inactive_tenants={inactive_count} "
            f"(threshold>{cfg.active_calls_threshold} calls in {cfg.active_calls_period})"
        )

        active_metrics.sort(
            key=lambda m: (-m.new_patient_appointments, -m.total_appointments, -m.tasks_created, -m.total_calls, m.tenant_id)
        )

        n = len(active_metrics)
        if n == 0:
            print("No active tenants to rank.")
            return 0

        print("Preview (top 10 tenants):")
        for idx, m in enumerate(active_metrics[:10], start=1):
            print(
                f"{idx:>3} | score={m.score:<6} tier={_tier_for_rank(idx, n):<10} "
                f"calls={m.total_calls:<4} tasks={m.tasks_created:<4} appts={m.total_appointments:<4} newp={m.new_patient_appointments:<4} "
                f"| {m.tenant_name} (tenant_id={m.tenant_id})"
            )

        print("Preview (bottom 10 tenants):")
        start_rank = max(1, n - 9)
        for idx, m in enumerate(active_metrics[-10:], start=start_rank):
            print(f"{idx:>3} | score={m.score:<6} | {m.tenant_name} (tenant_id={m.tenant_id})")

        if yobi_anomalies:
            for a in yobi_anomalies:
                print("ANOMALY:", a)
        return 0

    # HubSpot universe below
    hubspot = HubSpotClient(cfg.hubspot_access_token)

    # Validate HubSpot properties exist
    _validate_hubspot_properties(hubspot)

    # Load target companies
    companies = _load_companies_from_hubspot(cfg, hubspot)
    companies = [c for c in companies if c.hubspot_company_id]  # safety
    print(f"Loaded {len(companies)} HubSpot companies to consider.")

    # Map companies to tenant_ids
    mapped: List[Tuple[CompanyRow, int, str]] = []
    unmapped: List[Tuple[CompanyRow, str]] = []
    for c in companies:
        tid, reason = _pick_tenant_for_company(company=c, tenants=tenants, overrides=cfg.tenant_mapping_overrides)
        if tid is None:
            unmapped.append((c, reason))
        else:
            mapped.append((c, tid, reason))

    print(f"Mapped {len(mapped)} companies to Yobi tenants; unmapped={len(unmapped)}.")
    if unmapped:
        print("Unmapped examples:")
        for c, reason in unmapped[:10]:
            print(f"- company_id={c.hubspot_company_id} name={c.name!r} domain={c.domain!r} reason={reason}")

    inactive_mapped: List[Tuple[CompanyRow, int]] = [(c, tid) for (c, tid, _) in mapped if int(tid) not in active_tenant_ids]
    active_mapped: List[Tuple[CompanyRow, int, str]] = [(c, tid, why) for (c, tid, why) in mapped if int(tid) in active_tenant_ids]
    print(
        f"Active filter: active_companies={len(active_mapped)} inactive_companies={len(inactive_mapped)} "
        f"(threshold>{cfg.active_calls_threshold} calls in {cfg.active_calls_period})"
    )

    # Helpful debug for allowlist runs: show mapping + active call counts
    if cfg.company_ids_allowlist:
        print("Allowlist debug (company -> tenant -> active calls):")
        for c, tid, why in mapped:
            calls_30 = _safe_int(calls_active_window_by_tid.get(int(tid), 0))
            active = "ACTIVE" if int(tid) in active_tenant_ids else "INACTIVE"
            tenant_name = metrics_by_tid.get(int(tid)).tenant_name if metrics_by_tid.get(int(tid)) else f"tenant_{tid}"
            print(
                f"- company_id={c.hubspot_company_id} name={c.name!r} domain={c.domain!r} "
                f"=> tenant_id={tid} tenant_name={tenant_name!r} match={why} calls={calls_30} {active}"
            )

    # Build scored list (only active + those with tenant metrics present)
    scored: List[Dict[str, Any]] = []
    missing_metrics = 0
    for c, tid, why in active_mapped:
        m = metrics_by_tid.get(int(tid))
        if not m:
            missing_metrics += 1
            continue
        scored.append(
            {
                "company": c,
                "tenant_id": int(tid),
                "map_reason": why,
                "metrics": m,
            }
        )
    if missing_metrics:
        print(f"WARNING: {missing_metrics} mapped companies had no metrics for their tenant_id.")

    if not scored:
        print("No companies could be scored (no mappings with metrics).")
        if yobi_anomalies:
            for a in yobi_anomalies:
                print("ANOMALY:", a)
        return 0

    # Deterministic sort per spec (prioritize metrics, then company_id)
    scored.sort(
        key=lambda r: (
            -r["metrics"].new_patient_appointments,
            -r["metrics"].total_appointments,
            -r["metrics"].tasks_created,
            -r["metrics"].total_calls,
            int(r["company"].hubspot_company_id),
        )
    )

    n = len(scored)
    updates: List[Tuple[str, Dict[str, Any]]] = []

    # Build updates + mover deltas
    movers: List[Tuple[int, str, int]] = []  # abs(delta), company_id, delta
    for idx, row in enumerate(scored, start=1):
        c: CompanyRow = row["company"]
        m: TenantMetrics = row["metrics"]
        prev = _parse_prev_rank(c.prev_rank_raw)
        delta = (prev - idx) if prev is not None else None  # positive means moved up
        if delta is not None:
            movers.append((abs(delta), c.hubspot_company_id, delta))

        tier = _tier_for_rank(idx, n)
        updates.append(
            (
                c.hubspot_company_id,
                {
                    "customer_engagement_score": str(m.score),  # HubSpot numeric props accept stringified numbers
                    "customer_engagement_rank": f"{idx} out of {n}",
                    "customer_engagement_tier": tier,
                },
            )
        )

    # Print preview
    print("Preview (top 10):")
    for idx, row in enumerate(scored[:10], start=1):
        c = row["company"]
        m = row["metrics"]
        print(
            f"{idx:>3} | score={m.score:<6} tier={_tier_for_rank(idx, n):<10} "
            f"calls={m.total_calls:<4} tasks={m.tasks_created:<4} appts={m.total_appointments:<4} newp={m.new_patient_appointments:<4} "
            f"| {c.name} (company_id={c.hubspot_company_id})"
        )

    print("Preview (bottom 5):")
    for idx, row in enumerate(scored[-5:], start=n - min(5, n) + 1):
        c = row["company"]
        m = row["metrics"]
        print(f"{idx:>3} | score={m.score:<6} | {c.name} (company_id={c.hubspot_company_id})")

    if yobi_anomalies:
        for a in yobi_anomalies:
            print("ANOMALY:", a)

    # Safety: write-mode requires allowlist unless explicitly full_write=true
    if not cfg.dry_run:
        if cfg.company_ids_allowlist:
            allow = set(cfg.company_ids_allowlist)
            updates = [u for u in updates if u[0] in allow]
            updated_ids = {cid for cid, _ in updates}
            missing_from_updates = sorted(list(allow - updated_ids))
            print(f"Write-mode enabled. Will update {len(updates)} allowlisted companies.")
            if missing_from_updates:
                print(
                    "WARNING: Some allowlisted company IDs are not eligible for update "
                    "(unmapped, inactive, or missing metrics)."
                )
                for cid in missing_from_updates[:20]:
                    print(f"- {cid}")
        else:
            if not cfg.full_write:
                raise RuntimeError("Write-mode requires COMPANY_IDS allowlist OR FULL_WRITE=true.")
            print(f"Write-mode enabled (FULL_WRITE). Will update {len(updates)} eligible companies.")
        hubspot.batch_update_companies(updates)
        print("HubSpot update complete.")

    # Slack summary (optional)
    if cfg.slack_webhook_url:
        top_lines = []
        for idx, row in enumerate(scored[:10], start=1):
            c = row["company"]
            m = row["metrics"]
            top_lines.append(f"{idx}. {c.name} — score {m.score}")
        bot_lines = []
        for idx, row in enumerate(scored[-10:], start=n - min(10, n) + 1):
            c = row["company"]
            m = row["metrics"]
            bot_lines.append(f"{idx}. {c.name} — score {m.score}")

        movers.sort(reverse=True)
        mover_lines = []
        for _, cid, d in movers[:10]:
            mover_lines.append(f"{cid}: {'+' if d > 0 else ''}{d}")

        msg_lines: List[str] = [
            f"*Customer engagement ranker* ({'dry run' if cfg.dry_run else 'write'})",
            f"Scored companies: {n} (unmapped: {len(unmapped)})",
            "",
            "*Top 10*",
            *top_lines,
            "",
            "*Bottom 10*",
            *bot_lines,
            "",
            "*Biggest movers* (rank delta)",
        ]
        msg_lines.extend(mover_lines if mover_lines else ["(no prior ranks)"])
        msg_lines.extend(["", "*Anomalies*"])
        msg_lines.extend(yobi_anomalies[:10] if yobi_anomalies else ["(none)"])

        msg = "\n".join(msg_lines)
        _send_slack(cfg.slack_webhook_url, msg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

