"""Check the fictional inputs and answer-key consistency; no model calls."""
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT / name).read_text())

def main():
    cases = read('data/cases.json')
    a = read('data/system_a.json')
    b = read('data/system_b.json')
    events = {x['event_id']:x for x in read('data/settlement_events.json')}
    links = {(x['source_system'],x['source_id']):x for x in read('data/deal_links.json')}
    expected = {x['case_id']:x for x in read('evaluation/expected_results.json')}
    ids = [x['record_id'] for x in a+b]
    assert len(ids) == len(set(ids)), 'Duplicate record IDs'
    assert {x['case_id'] for x in cases} == set(expected)
    for case in cases:
        cid = case['case_id']; cutoff = date.fromisoformat(case['snapshot_date'])
        aa = {x['record_id']:x for x in a if x['case_id']==cid}
        bb = {x['record_id']:x for x in b if x['case_id']==cid}
        exp = expected[cid]
        delta = sum(x['amount_inr'] for x in aa.values()) - sum(x['amount_inr'] for x in bb.values())
        assert delta == exp['balance_difference_inr']
        used_a = set(); used_b = set()
        for match in exp['proposed_matches']:
            ra=aa[match['a_record_id']]; rb=bb[match['b_record_id']]
            la=links['A',ra['source_id']]; lb=links['B',rb['source_id']]
            assert la['economic_deal_id']==lb['economic_deal_id']
            assert la['as_of']==lb['as_of']==case['snapshot_date']
            assert ra['amount_inr']==rb['amount_inr']
            assert ra['record_id'] not in used_a and rb['record_id'] not in used_b
            used_a.add(ra['record_id']); used_b.add(rb['record_id'])
        for timing in exp['timing_explanations']:
            ra=aa[timing['a_record_id']]; ev=events[timing['event_id']]
            assert date.fromisoformat(ra['booking_date']) <= cutoff < date.fromisoformat(ra['settlement_date'])
            assert ev['source_id']==ra['source_id'] and ev['amount_inr']==ra['amount_inr']
            assert ev['scheduled_settlement_date']==ra['settlement_date']
            assert ev['status']=='PENDING' and ev['as_of']==case['snapshot_date']
            assert not any(x['source_id']==ra['source_id'] for x in bb.values())
            assert ra['record_id'] not in used_a
            used_a.add(ra['record_id'])
        ua=sum(x['amount_inr'] for k,x in aa.items() if k not in used_a)
        ub=sum(x['amount_inr'] for k,x in bb.items() if k not in used_b)
        assert (ua,ub)==(exp['unresolved_a_inr'],exp['unresolved_b_inr'])
        assert exp['disposition']==('EXPLAINED' if ua==ub==0 else 'ESCALATE')
        if exp['root_cause']=='DIFFERENT_DEALS':
            ra=next(iter(aa.values())); rb=next(iter(bb.values()))
            assert links['A',ra['source_id']]['economic_deal_id']!=links['B',rb['source_id']]['economic_deal_id']
        print(f'{cid}: inputs and expected evidence consistent')
    print('3 cases checked. Agent behaviour has not been evaluated yet.')

if __name__ == '__main__':
    main()
