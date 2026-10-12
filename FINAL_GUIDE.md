# Finish the Reconciliation Agent Lab

Put the new files beside your existing lessons, data/ and procedures/. Keep your
existing virtual environment and .env. No new packages are required if lesson 10
already ran successfully. All commands below use python3.

## Run the final checks and evaluation

From the existing lab folder, with its virtual environment active:

```bash
python3 test_final_workflow.py
python3 11_optimized_workflow.py evaluate --run final-1 --model nvidia/nemotron-3-ultra-550b-a55b:free
```

The first command runs offline tests, with simulated model replies. It makes no
API requests, needs no key, and does not modify your fixtures.

The second command runs the three real model cases and checks their results. It
uses the fixed model slug from the observed successful run, with your own OpenRouter key. This pins
the model instead of letting openrouter/free choose different models per request.
There are at most six model requests for the three cases, each capped at 4,096
output tokens. Model availability and free-service quotas can change; usage is recorded in the reports.

If you want to retain your .env model setting instead, omit --model. A free-router
run is useful as a smoke test, but model switching makes comparisons less controlled.

Read the three short explanations and the evaluation summary. PASS means the
implemented source/rule checks and expected structured case checks passed. It
does not prove every prose sentence or citation is supported.

To record your review after inspecting the results:

```bash
python3 11_optimized_workflow.py resume --run final-1 --decision approve
```

Only validated proposals with unchanged sources can be approved. This records
your decision; it does not change transactions. To reject the group instead:

```bash
python3 11_optimized_workflow.py resume --run final-1 --decision reject
```

Status and resume make no model requests:

```bash
python3 11_optimized_workflow.py status --run final-1
```

For a new evaluation, choose a new name (e.g. --run final-2). Existing saved runs
are never silently restarted. For a small repeatability check, add --repeat 3:
nine case runs, at most eighteen requests, with separately recorded results.
One successful run per case is a smoke test, not a production reliability claim.

## What you should see

```text
PREPARE C001: A=120,000; B=60,000; A-B=+60,000 INR
MODEL CALL 1/2: structured proposal
  Served: ...; input=...; output=...; ...s
VALIDATION: PASS — implemented v1 checks
PAUSED: saved for human review...
RESULT C001: EXPLAINED / TIMING
```

If the proposal is invalid, the errors print once. The graph may make one targeted
repair call using the same prepared evidence. After a second invalid response it
stops at review as BLOCKED. Transport failures or output-limit failures stop rather
than automatically spending tokens on retries. An invalid proposal cannot be approved.

The expected case outcomes are C001: timing explained; C002: alternate identifiers
explained; C003: different deals escalated with 75,000 INR unresolved on each side.
The evaluator's answer key is read only after model execution, and is never placed
in prompts or used by the independent validator.

## Inspect context without repeated dumps

Default output shows the input structure, evidence counts, model usage, validation
errors and concise results. There are no tools in the optimized model requests.

For a no-cost preview of the actual prepared input:

```bash
python3 11_optimized_workflow.py start --case C001 --preview --show-context
```

For detailed context during a real run, add --show-context to start/evaluate. It
prints system instructions and rules once, case evidence once per case, and only
the added response/errors for repair. Reused material is identified instead of
being reprinted. It is still sent in the API request; the display does not change
token usage.

## Execution and context

1. main() selects the case/run and constructs the graph. Construction makes no
   model requests.
2. graph.stream() starts the preparation node. Python batch-reads source records,
   computes totals and obtains current events/links. It assigns no root-cause label.
3. The model node builds a system message (instructions and contract) and a user
   message (question, evidence and seven v1 rule sections). model.invoke() sends them.
4. Python parses the returned JSON. The validate node independently re-reads source
   files and checks amounts, match identities, timing evidence, record reuse,
   disposition, citation IDs and current procedure versions.
5. A conditional edge selects repair or review. Repair gets the same original
   messages plus the rejected response and precise errors; it does not retrieve again.
6. interrupt() saves the review pause in SQLite. A later command reconstructs the
   graph and resumes only review, with no model configured.
7. Python saves one JSON report per case and one summary for the evaluation group.

State is a dictionary of evidence, proposal, validation, attempt count and review
decision. It is not an append-only conversation. make_messages() explicitly selects
what reaches the model. Checkpointed state and audit details are not automatically
model context.

JSON is requested in the prompt; the provider is not required to enforce a JSON
schema. The local parser and validator enforce acceptance. This keeps the final
step portable across the models used in the lab; malformed output can still occur.

All seven short rules are supplied directly. The earlier RAG and specialist lessons
remain useful demonstrations; the final small workflow does not need their search
or delegation loops. The validator implements the exact supplied v1 procedures.
If procedure text/version changes, it stops until that implementation is updated
and retested. It cannot automatically turn arbitrary new policy prose into code.

## Reports and evaluation limits

reports/<thread>.json contains the proposal, evidence snapshot, procedure versions,
source digest, validation errors, review decision and each visible response with
its validation and usage/timing (including a rejected first proposal). It is
written again when review is recorded. evaluation-<run>.json contains the expected
structured-case comparison and aggregate case metrics. No API key is recorded.

Metrics distinguish model calls, repairs, input/output/cached tokens and model-call
seconds. Token totals aggregate requests; they are not the context size of one call.
Model seconds are API-call time, not elapsed human-review time. Missing provider
usage is marked by usage_complete=false; do not treat missing usage as zero cost.

The historical lesson-10 C001 run used 18 model calls. The new design targets one
call per case, with a maximum of two. This is an architectural comparison, not a
controlled quality benchmark against that historical run: its free router served
multiple models. Use the same model, data and repeat count to compare quality fairly.

The offline tests cover all three valid structured proposals, false explained
dispositions, different/ambiguous deals, missing/wrong timing events, stale links,
duplicate use of a row, unknown references, malformed JSON, changed procedures,
source changes before approval, bounded repair, transport failure and review across
new Python processes. They validate the implementation, not live model reasoning.

## Files to understand

11_optimized_workflow.py owns the LangGraph execution, API requests, concise log,
review/checkpoint commands, reports and post-run evaluation.

reconciliation_core.py owns preparation, the JSON contract/parser and the independent
source/rule validator. It has no model dependencies and never reads the answer key.

test_final_workflow.py supplies simulated responses to exercise real workflow,
checkpoint and validation code without using API credits.

## Project scope and honest claims

This is a synthetic learning prototype. You implemented API integration, tool
calling, LangChain agents, document embeddings/Chroma retrieval, grounded answers,
LangGraph routing/persistence/review, isolated subagent delegation, structured output,
independent validation and evaluation. Earlier lessons preserve the experiments.

Keep live evaluation results and review decisions before claiming model accuracy.
This is not a deployed banking control, production benchmark or reproduction of
JPMC infrastructure. Your original work's prompt/evaluation contribution remains
distinct from this new technical implementation.
