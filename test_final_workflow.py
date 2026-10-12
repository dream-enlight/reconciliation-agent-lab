"""Offline tests: real graph/checkpoints/validator; simulated model responses.

python3 test_final_workflow.py
No API key, model requests, embeddings or changes to your fixtures are needed.
Passing these tests does NOT evaluate live model behaviour.
"""

from contextlib import redirect_stdout
from copy import deepcopy
from importlib import import_module
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from langchain_core.messages import AIMessage

import reconciliation_core as core
workflow = import_module("11_optimized_workflow")
PROJECT = Path(__file__).resolve().parent


def controlled_proposal(case_id):
    # This helper belongs ONLY to the offline tests, not the model workflow.
    expected = next(e for e in core.read_json(PROJECT, "evaluation/expected_results.json") if e["case_id"] == case_id)
    fields = ("case_id", "disposition", "root_cause", "balance_difference_inr", "unresolved_a_inr", "unresolved_b_inr",
              "proposed_matches", "timing_explanations")
    proposal = {key: deepcopy(expected[key]) for key in fields}
    proposal.update(explanation="Controlled offline proposal; this text does not establish live model quality.",
                    evidence_ids=list(expected["required_evidence_ids"]),
                    procedure_refs=[{"section_id": section, "version": 1}
                                    for section in sorted(set(expected["required_procedure_sections"]) | {"P-EVIDENCE-2"})],
                    next_action="NONE" if expected["disposition"] == "EXPLAINED" else
                    "Escalate the unmatched A/B records and contradictory deal links for population investigation.")
    return proposal


class SimulatedModel:
    def __init__(self, transform=None, audit=None, raise_error=False):
        self.requests = []
        self.transform = transform
        self.audit = audit
        self.raise_error = raise_error

    def invoke(self, messages):
        self.requests.append(messages)
        if self.audit:
            with self.audit.open("a") as handle:
                handle.write(json.dumps({"event": "model_call", "messages": [m.content for m in messages]}) + "\n")
        if self.raise_error:
            raise TimeoutError("controlled failure")
        payload = json.loads(messages[1].content)
        result = controlled_proposal(payload["evidence_and_rules"]["case"]["case_id"])
        if self.transform:
            result = self.transform(result, len(self.requests))
        return AIMessage(content=json.dumps(result),
                         response_metadata={"model_name": "offline-simulated", "finish_reason": "stop"},
                         usage_metadata={"input_tokens": 100, "output_tokens": 200, "total_tokens": 300})


def configure(root):
    workflow.ROOT = root
    workflow.DB_PATH = root / "checkpoints" / "lesson11.sqlite"
    workflow.REPORTS = root / "reports"


class FinalWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="reconciliation_final_test_")
        self.root = Path(self.temp.name)
        for folder in ("data", "procedures", "evaluation"):
            shutil.copytree(PROJECT / folder, self.root / folder)
        configure(self.root)
        self.output = io.StringIO()
        self.capture = redirect_stdout(self.output)
        self.capture.__enter__()

    def tearDown(self):
        self.capture.__exit__(None, None, None)
        configure(PROJECT)
        self.temp.cleanup()

    def validate(self, case_id, proposal):
        bundle = core.prepare_case(self.root, case_id)
        return core.validate_proposal(self.root, case_id, proposal, core.digest(bundle))

    def codes(self, result):
        return {e["code"] for e in result["errors"]}

    def edit_json(self, relative_path, transform):
        path = self.root / relative_path
        data = json.loads(path.read_text())
        path.write_text(json.dumps(transform(data)))

    def test_correct_record_level_proposals_pass_all_three_cases(self):
        for case_id in workflow.CASES:
            with self.subTest(case_id=case_id):
                result = self.validate(case_id, controlled_proposal(case_id))
                self.assertTrue(result["passed"], result)

    def test_equal_balances_do_not_hide_unmatched_exposures(self):
        proposal = controlled_proposal("C003")
        proposal.update(disposition="EXPLAINED", unresolved_a_inr=0, unresolved_b_inr=0, next_action="NONE")
        result = self.validate("C003", proposal)
        self.assertFalse(result["passed"])
        self.assertTrue({"AMOUNT", "DISPOSITION"} <= self.codes(result))

    def test_different_deals_cannot_be_matched(self):
        proposal = controlled_proposal("C003")
        proposal.update(root_cause="ALTERNATE_ID", disposition="EXPLAINED", unresolved_a_inr=0, unresolved_b_inr=0,
                        next_action="NONE", proposed_matches=[{"a_record_id": "A-004", "b_record_id": "B-003"}])
        self.assertIn("DIFFERENT_DEALS", self.codes(self.validate("C003", proposal)))

    def test_missing_event_blocks_timing_but_allows_explicit_escalation(self):
        self.edit_json("data/settlement_events.json", lambda rows: [])
        bad = controlled_proposal("C001")
        self.assertIn("MISSING_TIMING_EVIDENCE", self.codes(self.validate("C001", bad)))
        bad.update(disposition="ESCALATE", root_cause="INSUFFICIENT_EVIDENCE", unresolved_a_inr=60000,
                   timing_explanations=[], next_action="Obtain the missing cutoff settlement event for L101.")
        bad["evidence_ids"].remove("EV-101")
        bad["procedure_refs"].append({"section_id": "P-EVIDENCE-1", "version": 1})
        self.assertTrue(self.validate("C001", bad)["passed"])

    def test_event_amount_and_status_must_support_timing(self):
        def change(rows):
            rows[0]["amount_inr"] = 50000
            rows[0]["status"] = "SETTLED"
            return rows
        self.edit_json("data/settlement_events.json", change)
        self.assertIn("EVENT_MISMATCH", self.codes(self.validate("C001", controlled_proposal("C001"))))

    def test_old_links_are_not_current_evidence(self):
        def change(rows):
            for row in rows:
                if row["source_id"] == "F900":
                    row["as_of"] = "2025-09-29"
            return rows
        self.edit_json("data/deal_links.json", change)
        self.assertIn("UNSUPPORTED_ALTERNATE_ID", self.codes(self.validate("C002", controlled_proposal("C002"))))

    def test_ambiguous_links_block_even_same_id_matches(self):
        def change(rows):
            extra = deepcopy(rows[0])
            extra.update(link_id="LINK-CONTRADICTORY", economic_deal_id="OTHER-DEAL")
            return rows + [extra]
        self.edit_json("data/deal_links.json", change)
        self.assertIn("AMBIGUOUS_DEAL", self.codes(self.validate("C001", controlled_proposal("C001"))))

    def test_snapshot_populations_cannot_include_ineligible_rows(self):
        def change(rows):
            rows[0]["settlement_date"] = "2025-10-02"
            return rows
        self.edit_json("data/system_b.json", change)
        with self.assertRaises(core.SourceError):
            core.prepare_case(self.root, "C001")

    def test_rows_cannot_be_explained_twice(self):
        proposal = controlled_proposal("C001")
        proposal["timing_explanations"] *= 2
        self.assertIn("DOUBLE_USE", self.codes(self.validate("C001", proposal)))

    def test_unknown_evidence_and_rule_versions_are_rejected(self):
        proposal = controlled_proposal("C002")
        proposal["evidence_ids"].append("MADE-UP-EVENT")
        proposal["procedure_refs"][0]["version"] = 99
        self.assertTrue({"UNKNOWN_EVIDENCE", "UNKNOWN_RULE"} <= self.codes(self.validate("C002", proposal)))

    def test_json_duplicate_keys_and_non_json_numbers_are_rejected(self):
        for text in ('{"case_id":"C001","case_id":"C003"}', '{"amount":NaN}', '[]', 'hello {"case_id":"C001"}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                core.parse_proposal(text)

    def test_boolean_amounts_and_wrong_case_are_rejected(self):
        proposal = controlled_proposal("C001")
        proposal["unresolved_a_inr"] = False
        self.assertIn("SHAPE", self.codes(self.validate("C001", proposal)))
        proposal = controlled_proposal("C001")
        proposal["case_id"] = "C003"
        self.assertIn("CASE_ID", self.codes(self.validate("C001", proposal)))

    def test_changed_procedures_require_validator_update(self):
        path = self.root / "procedures/01_timing.md"
        path.write_text(path.read_text() + "\nAn unreviewed new rule.\n")
        with self.assertRaises(core.SourceError):
            core.prepare_case(self.root, "C001")

    def test_answer_key_is_not_a_runtime_input(self):
        proposal = controlled_proposal("C001")
        before = core.prepare_case(self.root, "C001")
        self.edit_json("evaluation/expected_results.json", lambda rows: [{**r, "balance_difference_inr": 999999} for r in rows])
        after = core.prepare_case(self.root, "C001")
        self.assertEqual(before, after)
        self.assertTrue(self.validate("C001", proposal)["passed"])
        messages = workflow.make_messages({"bundle": before})
        self.assertNotIn("required_evidence_ids", messages[1].content)
        self.assertNotIn("expected_results", messages[1].content)

    def test_one_request_path_and_bounded_repair_reuse_inputs(self):
        def wrong_once(result, number):
            if number == 1:
                result["balance_difference_inr"] = 123
            return result
        model = SimulatedModel(transform=wrong_once)
        display = workflow.ContextDisplay(True)
        state = workflow.run_case("C001", "repair-test", model=model, display=display)
        self.assertEqual(state["attempts"], 2)
        self.assertTrue(state["validation"]["passed"])
        self.assertEqual(model.requests[0][1].content, model.requests[1][1].content)
        self.assertEqual(len(model.requests[1]), 4)
        report = json.loads((workflow.REPORTS / "repair-test.json").read_text())
        self.assertFalse(report["model_calls"][0]["validation"]["passed"])
        self.assertTrue(report["model_calls"][1]["validation"]["passed"])
        self.assertIn('"balance_difference_inr": 123', report["model_calls"][0]["visible_response"])
        log = self.output.getvalue()
        self.assertEqual(log.count("SYSTEM INPUT (shown once)"), 1)
        self.assertEqual(log.count("REFERENCE RULES (shown once)"), 1)
        self.assertIn("not reprinted", log)

    def test_two_invalid_responses_stop_and_cannot_be_approved(self):
        def always_wrong(result, number):
            result["balance_difference_inr"] = 123
            return result
        model = SimulatedModel(transform=always_wrong)
        state = workflow.run_case("C001", "invalid-test", model=model)
        self.assertEqual(len(model.requests), 2)
        self.assertFalse(state["validation"]["passed"])
        with self.assertRaises(ValueError):
            workflow.run_case("C001", "invalid-test", "resume", "approve")
        rejected = workflow.run_case("C001", "invalid-test", "resume", "reject")
        self.assertEqual(rejected["review_decision"], "reject")

    def test_oversized_explanation_is_repaired_without_preparing_again(self):
        def long_once(result, number):
            if number == 1:
                result["explanation"] = "word " * 201
            return result
        model = SimulatedModel(transform=long_once)
        state = workflow.run_case("C001", "length-test", model=model)
        self.assertTrue(state["validation"]["passed"])
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(self.output.getvalue().count("PREPARE C001"), 1)
        self.assertIn("EXPLANATION_LENGTH", model.requests[1][-1].content)
        self.assertEqual(model.requests[0][1].content, model.requests[1][1].content)

    def test_transport_failure_does_not_trigger_blind_retry(self):
        model = SimulatedModel(raise_error=True)
        state = workflow.run_case("C001", "timeout-test", model=model)
        self.assertEqual(len(model.requests), 1)
        self.assertFalse(state["validation"]["passed"])

    def test_changed_sources_block_approval_without_new_model_calls(self):
        model = SimulatedModel()
        workflow.run_case("C001", "stale-test", model=model)
        self.edit_json("data/settlement_events.json", lambda rows: [])
        with self.assertRaises(ValueError):
            workflow.run_case("C001", "stale-test", "resume", "approve")
        self.assertEqual(len(model.requests), 1)
        state = workflow.run_case("C001", "stale-test", "status")
        self.assertEqual(state["review_decision"], "pending")

    def test_cross_process_evaluation_status_and_review(self):
        audit = self.root / "audit.jsonl"

        def run(*commands):
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--driver",
                                     str(self.root), str(audit), *commands], capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout

        preview = run("start", "--case", "C001", "--preview")
        self.assertIn("No API request", preview)
        self.assertFalse(audit.exists())
        self.assertFalse(workflow.DB_PATH.exists())
        sources_before = {str(p): p.read_bytes() for p in (self.root / "data").glob("*.json")}
        summary = run("evaluate", "--run", "offline-1")
        self.assertIn("Structured case checks: 3/3", summary)
        self.assertNotIn("SYSTEM INPUT", summary)
        self.assertLess(len(summary.splitlines()), 100)
        report = json.loads(workflow.evaluation_path("offline-1").read_text())
        self.assertTrue(all(r["metrics"]["model_calls"] == 1 for r in report["cases"]))
        before = audit.read_text()
        run("status", "--run", "offline-1")
        resumed = run("resume", "--run", "offline-1", "--decision", "approve")
        self.assertIn("Human review: approve", resumed)
        self.assertEqual(audit.read_text(), before)
        self.assertEqual(sources_before, {str(p): p.read_bytes() for p in (self.root / "data").glob("*.json")})
        self.assertTrue(all(r["review_decision"] == "approve" for r in json.loads(workflow.evaluation_path("offline-1").read_text())["cases"]))
        for row in report["cases"]:
            case_report = json.loads((workflow.REPORTS / f"{row['thread_id']}.json").read_text())
            self.assertIn("evidence_snapshot", case_report)
            self.assertEqual(case_report["review_decision"], "approve")
            self.assertEqual(case_report["model_calls"][0]["model_served"], "offline-simulated")


def driver():
    root, audit = Path(sys.argv[2]), Path(sys.argv[3])
    configure(root)
    def make_model(requested_model=None):
        with audit.open("a") as handle:
            handle.write(json.dumps({"event": "model_constructed"}) + "\n")
        return SimulatedModel(audit=audit)
    workflow.make_model = make_model
    sys.argv = ["11_optimized_workflow.py", *sys.argv[4:]]
    workflow.main()


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "--driver":
        driver()
    else:
        unittest.main(verbosity=2)
