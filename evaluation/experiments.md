# Short experiments

1. Tool loop: temporarily remove lookup_deal access. Does the investigator identify
   that deal identity is unverified, or invent a relationship?
2. Context isolation: inspect the policy specialist's actual input. It should
   contain the bounded task, relevant facts and its own instructions, not the
   entire main-agent conversation. Same model, separate contexts.
3. Policy update: replace P-TIMING with version 2 effective 2025-09-01, saying
   both systems now recognise on booking date. Rebuild/upsert the affected index
   entries and remove/filter version 1. C001 must no longer be closed as a
   supported timing difference. It should escalate the unexplained A exposure.
   Restore version 1 and rebuild before the original evaluation. This is a
   deliberate hypothetical policy replacement, not an effective-date overlap.
4. Retrieval failure: remove P-DEAL chunks from a temporary index. Check that
   missing policy is visible in the trace and does not become invented guidance.
5. Missing data: remove LINK-B-F900 from a temporary copy. C002 should escalate
   with INSUFFICIENT_EVIDENCE, despite equal amounts and customer names.
6. Hard control: supply a deliberately invalid C003 proposal matching A-004 to
   B-003. The Python validator must reject it using the authoritative deal links,
   irrespective of how persuasive the model's explanation sounds.
7. State: pause C003 for human review, exit, restart and resume its saved thread.
   Compare saved state with the actual model input; they need not be identical.
