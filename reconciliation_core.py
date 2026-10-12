"""Source preparation and independent checks for the fictional v1 control.

Standard library only. This module never calls a model or reads the answer key.
The validator implements the supplied seven v1 rules, not arbitrary policies.
"""

from datetime import date
import hashlib
import json
import re

SUPPORTED_RULE_DIGEST = "b295014a3a1eec08eeb0e407fe66e57de70c4de08ae100a1578c9d79c6965f40"
REQUIRED_SECTIONS = {
    "P-TIMING-1", "P-TIMING-2", "P-DEAL-1", "P-DEAL-2",
    "P-EVIDENCE-1", "P-EVIDENCE-2", "P-EVIDENCE-3",
}
CONTRACT = {
    "case_id": "string: supplied case ID",
    "disposition": "EXPLAINED or ESCALATE",
    "root_cause": "TIMING, ALTERNATE_ID, DIFFERENT_DEALS or INSUFFICIENT_EVIDENCE",
    "balance_difference_inr": "integer: A total minus B total",
    "unresolved_a_inr": "integer >= 0: A rows not covered by valid matches/timing",
    "unresolved_b_inr": "integer >= 0: B rows not covered by valid matches",
    "explanation": "non-empty string, at most 160 words; cite specific evidence/rules",
    "evidence_ids": "list of source record_id, event_id and link_id strings",
    "procedure_refs": "list of objects with exactly section_id (string) and version (integer)",
    "proposed_matches": "list of objects with exactly a_record_id and b_record_id strings",
    "timing_explanations": "list of objects with exactly a_record_id and event_id strings",
    "next_action": "non-empty string; NONE for EXPLAINED, a concrete human action for ESCALATE",
}


class SourceError(ValueError):
    pass


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(root, relative_path):
    return json.loads((root / relative_path).read_text(encoding="utf-8"))


def load_rules(root):
    rules = []
    for path in sorted((root / "procedures").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        header = text.split("\n## ", 1)[0]
        fields = dict(re.findall(r"^(procedure_id|version|effective_date|status):\s*(.+)$", header, re.M))
        if fields.get("status") != "active":
            continue
        title = text.splitlines()[0].removeprefix("# ")
        for section in re.split(r"^## ", text, flags=re.M)[1:]:
            heading, body = section.split("\n", 1)
            section_id = heading.split(":", 1)[0].strip()
            version = int(fields["version"])
            rules.append({"id": f"{section_id}:v{version}", "section_id": section_id,
                          "version": version, "effective_date": fields["effective_date"],
                          "source": path.name,
                          "text": f"{title} | version {version}\n{heading}\n{body.strip()}"})
    if len(rules) != 7 or {r["section_id"] for r in rules} != REQUIRED_SECTIONS:
        raise SourceError("Expected the seven supplied active procedure sections.")
    if any(r["version"] != 1 for r in rules) or digest(rules) != SUPPORTED_RULE_DIGEST:
        raise SourceError("Procedures changed. Update and retest the v1 validator before using new rules.")
    return rules


def prepare_case(root, case_id):
    """Batch-read source data; calculate numbers, without assigning a verdict."""
    cases = [c for c in read_json(root, "data/cases.json") if c["case_id"] == case_id]
    if len(cases) != 1:
        raise SourceError(f"Expected exactly one case definition for {case_id}.")
    case = cases[0]
    cutoff = case["snapshot_date"]
    date.fromisoformat(cutoff)
    a = [r for r in read_json(root, "data/system_a.json") if r["case_id"] == case_id]
    b = [r for r in read_json(root, "data/system_b.json") if r["case_id"] == case_id]
    ids = [r["record_id"] for r in a + b]
    if len(ids) != len(set(ids)):
        raise SourceError("Duplicate source record IDs; investigation blocked.")
    if case["currency"] != "INR" or any(type(r["amount_inr"]) is not int or r["amount_inr"] < 0 for r in a + b):
        raise SourceError("This fixture control supports non-negative integer INR exposures only.")
    for row in a + b:
        for field in ("booking_date", "settlement_date"):
            if date.fromisoformat(row[field]).isoformat() != row[field]:
                raise SourceError("Source dates must use YYYY-MM-DD.")
    if any(r["booking_date"] > cutoff for r in a) or any(r["settlement_date"] > cutoff for r in b):
        raise SourceError("A supplied source population contains a row ineligible at its snapshot.")
    events_all = read_json(root, "data/settlement_events.json")
    links_all = read_json(root, "data/deal_links.json")
    a_sources = sorted({r["source_id"] for r in a})
    pairs = sorted({("A", r["source_id"]) for r in a} | {("B", r["source_id"]) for r in b})
    events = [e for e in events_all if e["source_id"] in a_sources and e["as_of"] == cutoff]
    links = [l for l in links_all if (l["source_system"], l["source_id"]) in pairs
             and l["as_of"] == cutoff and l["authority"] == "Deal Register"]
    if len({e["event_id"] for e in events}) != len(events) or len({l["link_id"] for l in links}) != len(links):
        raise SourceError("Duplicate current evidence IDs; investigation blocked.")
    all_ids = ids + [e["event_id"] for e in events] + [l["link_id"] for l in links]
    if len(all_ids) != len(set(all_ids)):
        raise SourceError("Evidence IDs must be unique across records, events and links.")
    rules = load_rules(root)
    if any(r["effective_date"] > cutoff for r in rules):
        raise SourceError("A supplied rule is not effective at the case cutoff.")
    bundle = {
        "case": case,
        "population_scope": "Complete supplied A/B populations for this fictional case and snapshot.",
        "system_a_rows": a, "system_b_rows": b,
        "balances": {"system_a_total": sum(r["amount_inr"] for r in a),
                     "system_b_total": sum(r["amount_inr"] for r in b),
                     "difference_a_minus_b": sum(r["amount_inr"] for r in a) - sum(r["amount_inr"] for r in b)},
        "settlement_events": events, "deal_links": links,
        "lookup_statuses": {
            "settlement": [{"source_id": s, "as_of": cutoff,
                            "status": "FOUND" if any(e["source_id"] == s for e in events) else "NOT_FOUND"}
                           for s in a_sources],
            "deal": [{"source_system": system, "source_id": source, "as_of": cutoff,
                      "status": "FOUND" if any(l["source_system"] == system and l["source_id"] == source for l in links) else "NOT_FOUND"}
                     for system, source in pairs],
        },
        "rules": rules,
    }
    return bundle


def parse_proposal(text):
    """Accept one JSON object (optionally fenced), never an extracted fragment."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, flags=re.S)
        if not match:
            raise ValueError("Response must contain exactly one JSON object.")
        cleaned = match.group(1).strip()

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    result = json.loads(cleaned, object_pairs_hook=pairs, parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError("Proposal must be a JSON object.")
    return result


def validate_proposal(root, case_id, proposal, prepared_digest):
    """Re-read authoritative sources independently; do not consult expected_results."""
    errors = []

    def fail(code, field, message):
        errors.append({"code": code, "field": field, "message": message})

    fresh = prepare_case(root, case_id)
    if digest(fresh) != prepared_digest:
        fail("SOURCES_CHANGED", "evidence", "Sources changed after model input; start a fresh run.")
    result = {"passed": False, "errors": errors, "calculated": {}, "prose_review_required": True}
    if not isinstance(proposal, dict):
        fail("SHAPE", "proposal", "Expected one JSON object.")
        return result
    if set(proposal) != set(CONTRACT):
        fail("SHAPE", "proposal", f"Required fields: {', '.join(CONTRACT)}. No extra fields.")
    for name in ("case_id", "disposition", "root_cause", "explanation", "next_action"):
        if not isinstance(proposal.get(name), str) or not proposal[name].strip():
            fail("SHAPE", name, "Expected a non-empty string.")
    for name in ("balance_difference_inr", "unresolved_a_inr", "unresolved_b_inr"):
        if type(proposal.get(name)) is not int:
            fail("SHAPE", name, "Expected an integer, not a string, float or boolean.")
    if proposal.get("disposition") not in ("EXPLAINED", "ESCALATE"):
        fail("SHAPE", "disposition", "Use EXPLAINED or ESCALATE.")
    if proposal.get("root_cause") not in ("TIMING", "ALTERNATE_ID", "DIFFERENT_DEALS", "INSUFFICIENT_EVIDENCE"):
        fail("SHAPE", "root_cause", "Unsupported root-cause label.")
    if not isinstance(proposal.get("evidence_ids"), list) or any(not isinstance(x, str) for x in proposal.get("evidence_ids", [])):
        fail("SHAPE", "evidence_ids", "Expected a list of ID strings.")
    list_shapes = {"proposed_matches": {"a_record_id": str, "b_record_id": str},
                   "timing_explanations": {"a_record_id": str, "event_id": str},
                   "procedure_refs": {"section_id": str, "version": int}}
    for field, shape in list_shapes.items():
        entries = proposal.get(field)
        if not isinstance(entries, list) or any(not isinstance(x, dict) or set(x) != set(shape)
                                               or any(type(x.get(k)) is not kind for k, kind in shape.items()) for x in entries):
            fail("SHAPE", field, f"Expected a list of objects with fields {', '.join(shape)}.")
    if any(e["code"] == "SHAPE" for e in errors):
        return result

    if proposal["case_id"] != case_id:
        fail("CASE_ID", "case_id", "Proposal case does not match the requested case.")
    if len(proposal["explanation"].split()) > 160:
        fail("EXPLANATION_LENGTH", "explanation", "Shorten only the explanation to at most 160 words.")
    a = {r["record_id"]: r for r in fresh["system_a_rows"]}
    b = {r["record_id"]: r for r in fresh["system_b_rows"]}
    events = {e["event_id"]: e for e in fresh["settlement_events"]}
    links = {l["link_id"]: l for l in fresh["deal_links"]}
    known_ids = set(a) | set(b) | set(events) | set(links)
    cited = set(proposal["evidence_ids"])
    if len(cited) != len(proposal["evidence_ids"]):
        fail("DUPLICATE_ID", "evidence_ids", "Do not repeat evidence IDs.")
    if cited - known_ids:
        fail("UNKNOWN_EVIDENCE", "evidence_ids", f"Unknown/currently unavailable IDs: {sorted(cited - known_ids)}")
    refs = {(r["section_id"], r["version"]) for r in proposal["procedure_refs"]}
    if len(refs) != len(proposal["procedure_refs"]):
        fail("DUPLICATE_REF", "procedure_refs", "Do not repeat rule references.")
    available_refs = {(r["section_id"], r["version"]) for r in fresh["rules"]}
    if refs - available_refs:
        fail("UNKNOWN_RULE", "procedure_refs", "Every cited section/version must exist in the supplied rules.")

    def require_rule(section):
        if (section, 1) not in refs:
            fail("MISSING_RULE", "procedure_refs", f"Cite {section}:v1 for this conclusion.")

    def require_evidence(ids):
        missing = set(ids) - cited
        if missing:
            fail("MISSING_CITATION", "evidence_ids", f"Cite supporting IDs: {sorted(missing)}")

    def current_links(system, source_id):
        return [l for l in links.values() if l["source_system"] == system and l["source_id"] == source_id]

    def identity(system, source_id):
        found = current_links(system, source_id)
        deals = {l["economic_deal_id"] for l in found}
        return next(iter(deals)) if len(deals) == 1 else None

    used_a, used_b = set(), set()
    alternate_matches = 0
    valid_timing = 0
    for match in proposal["proposed_matches"]:
        ai, bi = match["a_record_id"], match["b_record_id"]
        if ai not in a or bi not in b:
            fail("UNKNOWN_MATCH", "proposed_matches", "Match must reference supplied A and B rows.")
            continue
        if ai in used_a or bi in used_b:
            fail("DOUBLE_USE", "proposed_matches", "A source row may be explained only once.")
            continue
        ra, rb = a[ai], b[bi]
        da, db = identity("A", ra["source_id"]), identity("B", rb["source_id"])
        if ((current_links("A", ra["source_id"]) and da is None)
                or (current_links("B", rb["source_id"]) and db is None)):
            fail("AMBIGUOUS_DEAL", "proposed_matches", f"Current deal links are contradictory for {ai}/{bi}.")
            continue
        if ra["amount_inr"] != rb["amount_inr"]:
            fail("MATCH_AMOUNT", "proposed_matches", f"Amounts differ for {ai}/{bi}.")
            continue
        if da is not None and db is not None and da != db:
            fail("DIFFERENT_DEALS", "proposed_matches", f"{ai}/{bi} link to different economic deals.")
            continue
        if ra["source_id"] != rb["source_id"]:
            require_rule("P-DEAL-1")
            if da is None or db is None or da != db:
                fail("UNSUPPORTED_ALTERNATE_ID", "proposed_matches", "Different IDs require unambiguous current links to the same deal.")
                continue
            alternate_matches += 1
            require_evidence([l["link_id"] for l in current_links("A", ra["source_id"]) + current_links("B", rb["source_id"])])
        require_evidence([ai, bi])
        used_a.add(ai)
        used_b.add(bi)

    cutoff = fresh["case"]["snapshot_date"]
    for timing in proposal["timing_explanations"]:
        ai, ei = timing["a_record_id"], timing["event_id"]
        if ai not in a or ei not in events:
            fail("MISSING_TIMING_EVIDENCE", "timing_explanations", "Timing requires a supplied A row and current settlement event.")
            continue
        if ai in used_a:
            fail("DOUBLE_USE", "timing_explanations", f"{ai} is already explained.")
            continue
        ra, ev = a[ai], events[ei]
        if not (ra["booking_date"] <= cutoff < ra["settlement_date"]):
            fail("TIMING_DATES", "timing_explanations", "Booking must be on/before cutoff; settlement must be after it.")
            continue
        if any(rb["source_id"] == ra["source_id"] for rb in b.values()):
            fail("PRESENT_IN_B", "timing_explanations", "The proposed A-only source is present in B.")
            continue
        if not (ev["source_id"] == ra["source_id"] and ev["amount_inr"] == ra["amount_inr"]
                and ev["scheduled_settlement_date"] == ra["settlement_date"]
                and ev["status"] == "PENDING" and ev["as_of"] == cutoff):
            fail("EVENT_MISMATCH", "timing_explanations", f"{ei} does not confirm the required timing evidence for {ai}.")
            continue
        require_rule("P-TIMING-1")
        require_rule("P-TIMING-2")
        require_evidence([ai, ei])
        used_a.add(ai)
        valid_timing += 1

    ua = sum(r["amount_inr"] for k, r in a.items() if k not in used_a)
    ub = sum(r["amount_inr"] for k, r in b.items() if k not in used_b)
    calculated = {"balance_difference_inr": fresh["balances"]["difference_a_minus_b"],
                  "unresolved_a_inr": ua, "unresolved_b_inr": ub}
    result["calculated"] = calculated
    for field, actual in calculated.items():
        if proposal[field] != actual:
            fail("AMOUNT", field, f"Source calculation is {actual}, not {proposal[field]}.")
    require_rule("P-EVIDENCE-2")
    required_disposition = "EXPLAINED" if ua == ub == 0 else "ESCALATE"
    if proposal["disposition"] != required_disposition:
        fail("DISPOSITION", "disposition", f"Unresolved A/B are {ua}/{ub}; use {required_disposition}.")
    if required_disposition == "EXPLAINED" and proposal["next_action"] != "NONE":
        fail("NEXT_ACTION", "next_action", "Use NONE for an explained case awaiting review.")
    if required_disposition == "ESCALATE" and proposal["next_action"].upper() == "NONE":
        fail("NEXT_ACTION", "next_action", "Identify the human investigation needed for unresolved exposure.")

    conflicts = []
    for ai, ra in a.items():
        for bi, rb in b.items():
            da, db = identity("A", ra["source_id"]), identity("B", rb["source_id"])
            if ai not in used_a and bi not in used_b and ra["amount_inr"] == rb["amount_inr"] and ra["customer"] == rb["customer"] and da and db and da != db:
                conflicts.append((ai, bi, ra, rb))
    cause = proposal["root_cause"]
    if cause == "TIMING" and not valid_timing:
        fail("ROOT_CAUSE", "root_cause", "TIMING requires at least one supported timing explanation.")
    elif cause == "ALTERNATE_ID" and not alternate_matches:
        fail("ROOT_CAUSE", "root_cause", "ALTERNATE_ID requires a supported match between different source IDs.")
    elif cause == "DIFFERENT_DEALS":
        require_rule("P-DEAL-2")
        if not conflicts or required_disposition != "ESCALATE":
            fail("ROOT_CAUSE", "root_cause", "DIFFERENT_DEALS requires current conflicting deal links for unresolved candidates.")
        else:
            for ai, bi, ra, rb in conflicts:
                require_evidence([ai, bi] + [l["link_id"] for l in current_links("A", ra["source_id"]) + current_links("B", rb["source_id"])])
    elif cause == "INSUFFICIENT_EVIDENCE":
        require_rule("P-EVIDENCE-1")
        missing = any(identity("A", r["source_id"]) is None for k, r in a.items() if k not in used_a)
        missing |= any(identity("B", r["source_id"]) is None for k, r in b.items() if k not in used_b)
        missing |= any(r["settlement_date"] > cutoff and not any(e["source_id"] == r["source_id"] for e in events.values())
                       for k, r in a.items() if k not in used_a)
        if required_disposition != "ESCALATE" or not missing:
            fail("ROOT_CAUSE", "root_cause", "INSUFFICIENT_EVIDENCE requires unresolved exposure and a missing/ambiguous essential lookup.")
    result["passed"] = not errors
    return result
