# COM-220 — Ground Contact Windows

**Document ID:** COM-220
**Revision:** E

## Nominal contact schedule

The spacecraft has **four** ground contacts per day on the primary
station, each approximately **9 minutes** in duration. Contacts are
spaced irregularly because of the sun-synchronous ground track.

## Command-loss timer

The onboard command-loss timer is set to **72 hours**. If no valid
ground command is received within that window, the spacecraft
autonomously enters safe mode per SYS-001 and reconfigures to the
omnidirectional antenna.

## Transmitter shed constraint

The comms transmitter must **not** be shed if doing so would cause a
required ground contact to be missed, where "required" means:

- It is the final contact before the command-loss timer expires, **or**
- Ground has flagged the contact as mandatory for an uplink

Per SYS-000 Rev C rule 4, this constraint overrides the EPS-201 shed
ordering. A transmitter shed that costs a non-required contact is
acceptable; one that costs a required contact is not.

## Safe-mode contact schedule

In safe mode the spacecraft transmits on the omnidirectional antenna at
reduced rate. Effective contact duration drops to approximately
**4 minutes** and the usable downlink volume falls by roughly 85%.
Plan recovery commanding accordingly — a safe-mode recovery typically
requires two or three contacts rather than one.

## Related

- SYS-000 — Procedure Precedence
- SYS-001 — Safe Mode Entry
- EPS-201 — EPS Undervoltage Response
