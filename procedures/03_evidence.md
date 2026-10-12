# Evidence and disposition procedure
procedure_id: P-EVIDENCE
version: 1
effective_date: 2025-09-01
status: active

## P-EVIDENCE-1: Evidence states
Record observable facts and current source references. Distinguish missing evidence
from contradictory evidence. If an essential lookup is absent or unavailable,
escalate with the missing item identified; do not invent it.

## P-EVIDENCE-2: Amounts and reconciliation
Calculate balances in code. balance_difference_inr is A total minus B total.
unresolved_a_inr and unresolved_b_inr are the amounts of source rows still
unexplained after supported matches or timing explanations. Never substitute a
zero net balance difference for successful record-level reconciliation.
disposition is EXPLAINED or ESCALATE. EXPLAINED requires zero unresolved exposure
on both sides and current support for every proposed explanation.

## P-EVIDENCE-3: Human action
The prototype explains and recommends. It does not edit source transactions.
A human-review pause records the reviewer's decision and next action. Resuming
the workflow cannot silently convert an unresolved case to an explained case.
