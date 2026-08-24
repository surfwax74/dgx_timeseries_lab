# SYS-000 — Procedure Precedence and Conflict Resolution

**Document ID:** SYS-000
**Revision:** C
**Supersedes:** SYS-000 Rev B

## Purpose

Mission procedures occasionally give conflicting direction. This document
is the single authority for resolving those conflicts. Any procedure that
contradicts SYS-000 is subordinate to it.

## Precedence rules

Apply in order. The first rule that matches governs.

1. **Crew/asset safety overrides everything.** Not applicable to
   uncrewed platforms except where a deorbit or collision-avoidance
   burn is in progress.

2. **Thermal survival overrides power optimization.** If any thermal
   margin per TCS-105 is below **5 °C**, thermal procedures take
   precedence over EPS load-management procedures — including
   EPS-201. This is the most commonly missed rule.

3. **During eclipse, EPS-201 takes precedence over SYS-001** for load
   shedding order, *unless* rule 2 applies. Outside eclipse, SYS-001
   governs.

4. **Never shed a load that is required to maintain the next ground
   contact** if that contact is the last opportunity before a
   command-loss timer expires. See COM-220 for window definitions and
   the command-loss timer.

5. When two procedures of equal precedence conflict, the more recently
   revised document governs. Check the revision letter, not the
   document number.

## Required reporting

Any conflict resolved under rules 2 through 5 must be logged in the
anomaly record with the rule number cited. Ground will review.

## Related

- EPS-201 — EPS Undervoltage Response
- SYS-001 — Safe Mode Entry
- TCS-105 — Thermal Limits and Margins
- COM-220 — Ground Contact Windows
