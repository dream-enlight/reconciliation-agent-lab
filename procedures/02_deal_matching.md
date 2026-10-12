# Economic deal matching procedure
procedure_id: P-DEAL
version: 1
effective_date: 2025-09-01
status: active

## P-DEAL-1: Alternate identifiers
Different source identifiers can represent the same exposure. To reconcile them,
retrieve current Deal Register links for both source system/identifier pairs.
Both links must confirm the same economic_deal_id, and the amounts must agree.
The link date must equal the case snapshot date. Cite both link IDs and source rows.

## P-DEAL-2: Misleading similarities
Matching customer names, amounts, dates or total balances can identify candidates
but cannot establish deal identity. If authoritative links confirm different deals,
the proposed same-exposure match is invalid. Preserve both unmatched exposures
and escalate to a human to establish the actual population issue.
