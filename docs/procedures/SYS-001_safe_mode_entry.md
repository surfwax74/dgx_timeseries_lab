# SYS-001 — Safe Mode Entry

**Document ID:** SYS-001
**Revision:** D

## Purpose

Safe mode places the spacecraft in a minimum-power, sun-pointed,
thermally stable configuration pending ground intervention.

## Entry criteria

- Bus voltage below **24.0 V**, or
- EPS-201 load shedding exhausted without arresting decline, or
- Loss of attitude knowledge for more than 5 minutes, or
- Ground command

## Load shed order (safe mode)

Note this order **differs from EPS-201**. Safe mode prioritizes
preserving thermal control and attitude over communications:

1. **Comms transmitter** (15 W)
2. **Payload** (12 W)
3. **Reaction wheel desaturation** (6 W)
4. Secondary heaters are **retained** in safe mode — they are
   considered survival load, not sheddable load

The ordering difference versus EPS-201 is intentional and is resolved
by SYS-000 Rev C rule 3 according to eclipse state. Do not attempt to
reconcile the two procedures locally; apply the precedence rule.

## Recovery

Safe mode exit requires ground command. The spacecraft will hold
sun-pointed indefinitely. Expect reduced telemetry rate — see COM-220
for the safe-mode contact schedule, which differs from nominal.

## Related

- SYS-000 — Procedure Precedence
- EPS-201 — EPS Undervoltage Response
- TCS-105 — Thermal Limits and Margins
- COM-220 — Ground Contact Windows
