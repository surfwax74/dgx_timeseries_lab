# EPS-201 — EPS Bus Undervoltage Response

**Document ID:** EPS-201
**Revision:** F
**Applies to:** Primary 28 V bus

## Trigger conditions

Initiate this procedure when any of the following hold for more than
30 consecutive seconds:

- Primary bus voltage below **26.5 V**
- Bus voltage drop exceeding **0.5 V** within any 60 s window
- Battery state of charge below **40%** while in eclipse

## Response — load shed order

Shed in this order, pausing 60 s between steps to observe recovery:

1. **Payload** (nominal draw 12 W, peak 18 W)
2. **Secondary heaters** (nominal 8 W) — but see TCS-105 before
   shedding; heater removal has thermal consequences
3. **Comms transmitter** (nominal 15 W) — but see COM-220; do not shed
   if a required ground contact falls within the next 45 minutes
4. **Reaction wheel desaturation** (nominal 6 W, intermittent)

Do **not** shed the onboard computer or attitude determination sensors
under this procedure. If shedding through step 4 does not arrest the
voltage decline, escalate to SYS-001 Safe Mode Entry.

## Power budget note

Total sheddable load through step 4 is approximately **41 W** nominal.
See EPS-310 for the full budget and for the recovery-margin
calculation. Shedding the payload alone recovers roughly 12 W, which
historically arrests a decline of up to about 0.4 V.

## Precedence

This procedure conflicts with SYS-001 on shed ordering. **SYS-000
Rev C rule 3** governs: during eclipse EPS-201 ordering applies;
outside eclipse SYS-001 ordering applies. Rule 2 of SYS-000 overrides
both when thermal margin is below 5 °C.

## Related

- SYS-000 — Procedure Precedence
- SYS-001 — Safe Mode Entry
- EPS-310 — Power Budget and Constraints
- TCS-105 — Thermal Limits and Margins
- COM-220 — Ground Contact Windows
