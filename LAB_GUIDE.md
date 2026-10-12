# Reconciliation Agent Lab

Build a small, inspectable Python agent in one focused day. This is an independent
learning prototype with fictional data and procedures, not a reconstruction of
an employer's system. Amounts are integer INR; one snapshot; no FX or materiality.

## What you will build

An investigator that reads case records, calculates differences in Python,
checks authoritative deal links, retrieves procedure passages from a local
vector store, and proposes a cited explanation or escalation. Later it delegates
policy interpretation to one specialist and runs inside an explicit LangGraph
workflow with validation and human review.

## The three cases

- C001: a booking/settlement timing difference with current supporting evidence.
- C002: different source identifiers, confirmed to belong to the same deal.
- C003: same customer and amount, but confirmed different deals; escalate.

Cases C002 and C003 have equal aggregate balances. Equal totals are deliberately
insufficient to establish record-level reconciliation.

## Build sequence and time budget

1. Setup (30 minutes): Python virtual environment, model access, one successful
   request. Default: local Python in your existing editor, one provider for
   generation and embeddings where supported. Keep credentials in environment
   variables. Choose and record tested package/model versions during setup.
2. Visible tool loop (45 minutes): load C001; register read_case and
   calculate_balances; execute model-requested functions and return results.
   Inspect messages, call IDs, arguments and tool results. Use a short supplied
   policy directly in context for this first baseline. Limit model/tool rounds.
3. LangChain (45 minutes): implement the same investigator with create_agent.
   Add lookup_deal. Compare its trace with the handwritten loop. Preserve the
   earlier version so the framework's contribution remains visible.
4. RAG (75 minutes): split procedures by Markdown section; embed them; persist
   chunks, source IDs, versions and active-status metadata in local Chroma. Use cosine similarity and
   retrieve the top three active chunks. Expose search_procedures as a tool.
   Remove the full policy from the investigator prompt. Inspect the actual
   retrieved text and scores. Similarity selects candidate passages; citations
   and applicability checks remain necessary.
5. Main agent + one specialist (45 minutes): wrap a policy specialist as an
   ask_policy tool. Give it its own instructions, a bounded task brief, relevant
   facts and only search_procedures. Return a compact interpretation plus source
   references. Log its input separately to verify context isolation.
6. LangGraph (75 minutes): implement load_case -> investigate -> validate.
   Route supported proposals to reporting and unresolved/invalid proposals to
   a human-review interrupt, then report. Store state with a local SQLite
   checkpointer and demonstrate resume using the same thread ID. Resumption
   reads saved state; it does not retrain the model. Validation must load
   authoritative records itself, not trust factual claims in model prose.
7. Evaluation and explanation (45 minutes): run all three cases, compare
   structured fields with evaluator-only expectations, record failures, make
   one targeted improvement and rerun. Test changed policy and a missing lookup.

Core work: about six hours. Allow one to two further hours for questions,
environment issues and debugging. This is a guided estimate, not a guarantee.

## Planned tools

- read_case(case_id): reads that case's task, source rows and current settlement
  events for the relevant source IDs.
- calculate_balances(case_id): code returns A total, B total and signed A-B.
- lookup_deal(source_system, source_id): returns a current deal-link record.
- search_procedures(query): returns applicable chunks with IDs and versions.
- ask_policy(task_brief, relevant_facts): invokes the separately scoped specialist.

Use one main agent and one specialist, sequential delegation, console execution
and local storage. Keep the first day's scope bounded. The tools fetch raw facts;
they must not return the answer key or encode a case-ID-to-root-cause lookup.

## Output contract

Use schema/result_contract.json as a human-readable field specification. Return
one structured proposal per case, then generate one report and an evidence log.
The evidence log records observable requests, arguments, returned data, source
citations, validator results and human decisions. Do not ask for private
chain-of-thought. A brief evidence-based explanation is sufficient.

The answer key in evaluation/expected_results.json is evaluator-only. Never load
it into prompts, retrieval, tool outputs or a specialist's context. The fixture
checker checks the supplied case pack; it is not an agent implementation or an
agent evaluation.

## Completion criteria

- You can trace one raw tool request to execution and the next model input.
- LangChain runs that same task with real tool calls.
- RAG retrieves cited passages from a persistent index; the full procedures are
  absent from the answering prompt.
- The policy specialist receives a separate, bounded context and returns evidence.
- LangGraph performs a code-defined branch and pause/resume with saved state.
- All three cases meet their expected disposition and evidence requirements.
- You alter one record or policy, rerun, and explain the changed behaviour.

Passing three cases demonstrates the prototype's intended behaviours, not broad
production accuracy. Further capability would require a larger evaluation set.

## First step

Unzip this folder into your working directory and run:

    python3 check_fixtures.py

Then perform setup and build only the visible tool loop. Add each capability
after you can explain the preceding trace.

## Official references

- https://docs.langchain.com/oss/python/langchain/overview
- https://docs.langchain.com/oss/python/langgraph/overview
- https://docs.langchain.com/oss/python/langchain/multi-agent/subagents
- https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
- https://docs.langchain.com/oss/python/langgraph/interrupts
- https://developers.openai.com/api/docs/quickstart
