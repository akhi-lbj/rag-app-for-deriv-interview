# Platform Incident Severity and Escalation Policy

## 1. Incident Severity Definitions and SLAs
Incidents impacting platform availability or data integrity are categorized into four severity tiers:

- **P1 - Critical Outage**:
  - Full system outage, core trading engine unavailable, or severe data loss.
  - **Initial Response SLA**: **15 minutes**.
  - Escalation: Automatic paging of Executive Incident Commander and Primary On-Call Engineering Lead.
- **P2 - Major Degradation**:
  - Significant feature impairment or degraded API performance affecting >15% of active users.
  - **Initial Response SLA**: **30 minutes**.
- **P3 - Minor Incident**:
  - Non-critical issue with viable operational workarounds available.
  - **Initial Response SLA**: **4 hours**.
- **P4 - Low Priority**:
  - Cosmetic issues, minor administrative bugs, or non-blocking queries.
  - **Initial Response SLA**: **24 hours**.

## 2. On-Call Escalation Matrix
- On-call engineers receive pages via automated alerting (PagerDuty).
- If the Primary On-Call engineer fails to acknowledge the alert within **10 minutes**, the system automatically escalates to the Secondary On-Call engineer and Engineering Manager.

## 3. Post-Incident Review (PIR) Timeline
- A blameless Post-Incident Review (PIR) document and retrospective meeting are mandatory for all **P1 and P2 incidents**.
- The preliminary PIR report must be published internally within **72 hours** of incident resolution.
