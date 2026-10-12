# Snapshot recognition procedure
procedure_id: P-TIMING
version: 1
effective_date: 2025-09-01
status: active

## P-TIMING-1: Recognition dates
For this fictional control, System A includes an exposure from its booking date.
System B includes the exposure from its settlement date. Both snapshots include
eligible dates on or before the snapshot date. A transaction booked by the cutoff
but settling after it can therefore appear only in A.

## P-TIMING-2: Required timing evidence
A timing explanation requires the A record, the missing B population, and a
current settlement-event record confirming PENDING at the cutoff with the same
source ID, amount and scheduled settlement date. Dates in A alone are insufficient.
Report the signed A-minus-B difference. Explain it as timing without changing
records or claiming the future settlement has already occurred.
