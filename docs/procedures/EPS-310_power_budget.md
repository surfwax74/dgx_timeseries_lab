# EPS-310 — Power Budget and Constraints

**Document ID:** EPS-310
**Revision:** A

## Nominal load budget

| Load | Nominal | Peak | Sheddable |
|---|---:|---:|:---:|
| Onboard computer | 9 W | 11 W | No |
| Attitude sensors | 5 W | 5 W | No |
| Secondary heaters | 8 W | 14 W | Conditional (TCS-105) |
| Comms transmitter | 15 W | 22 W | Conditional (COM-220) |
| Payload | 12 W | 18 W | Yes |
| Reaction wheel desat | 6 W | 9 W | Yes |
| **Total nominal** | **55 W** | — | — |

## Generation

- Solar array output, full sun: **72 W** end-of-life
- Solar array output, eclipse: **0 W**
- Battery capacity: 30 Ah at 28 V nominal

## Eclipse energy balance

Nominal eclipse draw of 55 W over a 35-minute eclipse consumes
approximately **32 Wh**, which is about **3.8%** of a full battery.
A healthy pack enters eclipse above 80% state of charge and exits
above 75%.

## Recovery margin calculation

To arrest a voltage decline, shed load must exceed the deficit. Compute:

```
deficit_W = (nominal_draw_W) - (available_generation_W)
required_shed_W = deficit_W * 1.2      # 20% margin
```

Then select shed steps from EPS-201 whose cumulative wattage meets or
exceeds `required_shed_W`. Shedding the payload alone yields 12 W;
payload plus reaction-wheel desaturation yields 18 W; adding the comms
transmitter yields 33 W.

Note the 1.2 margin factor. An operator who computes a bare deficit and
sheds exactly that much will under-shed and the decline will continue.

## Related

- EPS-201 — EPS Undervoltage Response
- TCS-105 — Thermal Limits and Margins
- COM-220 — Ground Contact Windows
