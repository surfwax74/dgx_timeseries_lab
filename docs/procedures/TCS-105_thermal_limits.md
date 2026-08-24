# TCS-105 — Thermal Limits and Margins

**Document ID:** TCS-105
**Revision:** B

## Component limits

| Component | Operational min | Operational max | Survival min | Survival max |
|---|---:|---:|---:|---:|
| Battery pack | -5 °C | +40 °C | -20 °C | +55 °C |
| Solar panel X | -60 °C | +85 °C | -80 °C | +110 °C |
| Solar panel Y | -60 °C | +85 °C | -80 °C | +110 °C |
| Payload detector | -30 °C | +25 °C | -40 °C | +45 °C |
| Onboard computer | -20 °C | +60 °C | -35 °C | +75 °C |

## Margin definition

**Thermal margin** is the distance from the current reading to the
nearest *operational* limit — not the survival limit. A component at
+37 °C against a +40 °C operational max has a margin of **3 °C**.

Margin below **5 °C** on any component triggers SYS-000 Rev C rule 2,
which elevates thermal procedures above EPS load management. This is
the rule most often missed during undervoltage events, because the
operator is focused on the power problem.

## Heater dependency

Secondary heaters maintain the battery pack and payload detector above
their operational minimums during eclipse. Eclipse duration on the
nominal orbit is approximately **35 minutes**.

Shedding secondary heaters during eclipse produces a battery-pack
cooling rate of roughly **0.4 °C per minute**. From a typical eclipse
entry temperature of +8 °C, the pack reaches its -5 °C operational
minimum in approximately **32 minutes** — marginally inside a single
eclipse. Do not shed heaters during eclipse without computing the
remaining eclipse duration first.

## Related

- SYS-000 — Procedure Precedence
- EPS-201 — EPS Undervoltage Response
- EPS-310 — Power Budget and Constraints
