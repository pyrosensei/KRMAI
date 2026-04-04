#!/usr/bin/env python3
"""Run a 200-question student-style benchmark against KRMAI chat API.

Evaluation logic:
- In-domain questions pass when:
  1) expected keywords appear in answer (>= min_required)
  2) at least one source is returned
- Out-of-domain questions pass when guardrail/refusal language appears.

Outputs:
- evaluation/student_200_report.json
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import requests

API_URL = "http://localhost:8000/chat"
TIMEOUT_SECS = 120

OUT_OF_SCOPE_HINTS = [
    "outside",
    "outside krmu",
    "outside krmu's knowledge base",
    "don't have reliable information",
    "not available",
    "cannot answer",
    "not related",
    "no information",
]


@dataclass
class EvalCase:
    question: str
    category: str
    expect: str = "in_domain"  # in_domain | out_of_domain
    keywords: List[str] | None = None
    min_required: int = 1


@dataclass
class CaseResult:
    question: str
    category: str
    expect: str
    passed: bool
    status_code: int
    latency_sec: float
    found_keywords: List[str]
    missing_keywords: List[str]
    sources_count: int
    reason: str
    answer_preview: str


def make_in_case(question: str, category: str, keywords: List[str], min_required: int = 1) -> EvalCase:
    return EvalCase(
        question=question,
        category=category,
        expect="in_domain",
        keywords=keywords,
        min_required=min_required,
    )


def make_out_case(question: str, category: str = "Outside / Guardrail") -> EvalCase:
    return EvalCase(
        question=question,
        category=category,
        expect="out_of_domain",
        keywords=[],
        min_required=1,
    )


def build_cases() -> List[EvalCase]:
    cases: List[EvalCase] = []

    admission_kw = ["admission", "apply", "application", "eligibility", "interview", "enroll", "offer"]
    fee_kw = ["fee", "fees", "payment", "scholarship", "financial", "loan", "refund"]
    placement_kw = ["placement", "package", "recruit", "company", "career", "lpa", "cdc"]
    campus_kw = ["campus", "hostel", "bus", "route", "facility", "library", "tour"]
    academic_kw = ["academic", "exam", "policy", "ph.d", "regulation", "calendar", "code", "attendance"]
    life_kw = ["club", "societies", "events", "welfare", "committee", "grievance", "student"]

    admissions_questions = [
        "How can I apply for admission at KRMU?",
        "What is the eligibility for BTech admission at KRMU?",
        "Is there an entrance exam for admission?",
        "What is KREE in the admission process?",
        "What documents are required for admission?",
        "What is the application fee for admission?",
        "When does the admission process usually start?",
        "Can international students apply at KRMU?",
        "Can I apply for multiple programmes at once?",
        "Is there a counselling process before final admission?",
        "How will I know if I am selected?",
        "What are the admission office hours?",
        "Where can I apply for UG, PG, and PhD programmes?",
        "Does KRMU offer diploma courses?",
        "What is the PhD admission process?",
        "Does KRMU offer part-time PhD options?",
        "Is personal interview mandatory for PhD admission?",
        "How many steps are there in the KRMU admission journey?",
        "What does the admission offer letter include?",
        "Can I book a campus visit before applying?",
        "Whom should I contact for admission support?",
        "Is there an admission FAQ page for common doubts?",
        "Can I get admission after the regular joining date?",
        "Is lateral entry available in technical programmes?",
        "What is the difference between registration and enrollment at KRMU?",
    ]
    for q in admissions_questions:
        cases.append(make_in_case(q, "Admissions", admission_kw, 1))

    fees_questions = [
        "What is the fee structure for BTech CSE?",
        "How much are the application and registration fees?",
        "What is the annual hostel fee?",
        "Is hostel security refundable and how much is it?",
        "What payment options are available at KRMU?",
        "What should I do if fee payment fails?",
        "How can I check my payment status?",
        "Does KRMU have a refund policy for fees?",
        "Can the fee increase in subsequent years?",
        "What financial assistance options are available?",
        "Does KRMU help students with education loans?",
        "What are scholarship criteria for UG students?",
        "How much scholarship is offered for 90% in Class 12?",
        "How are scholarship installments released?",
        "Does KRMU provide sports scholarships?",
        "Can a student use multiple scholarships at the same time?",
        "Is scholarship first-come-first-served for first 100 students?",
        "When is scholarship document verification done?",
        "Is attendance required to continue scholarship benefits?",
        "What are M.Pharm programme fees at KRMU?",
        "What is the per-semester fee format for international admissions?",
        "Who is the helpdesk contact for payment-related issues?",
        "How do existing students pay semester fees?",
        "When should the first installment be paid after selection?",
        "Are there any additional charges beyond tuition fee?",
    ]
    for q in fees_questions:
        cases.append(make_in_case(q, "Fees / Scholarships", fee_kw, 1))

    placement_questions = [
        "What is the highest package at KRMU?",
        "What is the average placement package?",
        "Who are the top placed students at KRMU?",
        "How many recruiters visit KRMU campus?",
        "Which prominent companies recruit from KRMU?",
        "What are the stages in KRMU placement process?",
        "What is a pre-placement talk?",
        "Does KRMU conduct mock tests for placements?",
        "What placement training does CDC provide?",
        "What is the role of Career Development Center?",
        "How often do companies visit for campus drives?",
        "Does KRMU have an online placement portal?",
        "Do students get resume and interview preparation support?",
        "Can students withdraw midway from a recruitment process?",
        "Is there a placement policy after getting two offers?",
        "What is the campus-to-corporate initiative?",
        "Does KRMU support internship opportunities?",
        "Are aptitude and technical trainings available?",
        "What communication skill training is provided for placements?",
        "Do students receive maths aptitude preparation?",
        "What value-added courses help employability?",
        "Are there restrictions on appearing in many placement drives?",
        "What documents should I carry on recruiter visit day?",
        "Is dress code mandatory during campus recruitment?",
        "How is the waiting list treated in placement outcomes?",
    ]
    for q in placement_questions:
        cases.append(make_in_case(q, "Placements", placement_kw, 1))

    campus_questions = [
        "What bus routes are available for KRMU students?",
        "Are bus timings available for major pickup points?",
        "What hostel facilities are available at KRMU?",
        "How much are hostel charges and deposits?",
        "What is the size of KRMU campus?",
        "What library facilities are available on campus?",
        "What sports facilities are available at KRMU?",
        "How many labs are available for students?",
        "Are there conference facilities on campus?",
        "Is canteen or dining available on campus?",
        "What campus infrastructure highlights should students know?",
        "Is there a virtual campus tour option?",
        "How can I schedule a campus tour?",
        "Where exactly is KRMU located?",
        "Does KRMU provide transport route information?",
        "What anti-ragging support exists in hostels and campus?",
        "Do hostels exist for both male and female students?",
        "How far is KRMU from Delhi airport?",
        "Are smart classrooms available at KRMU?",
        "Is KRMU campus eco-friendly?",
        "Can international students get hostel accommodation?",
        "What are the major campus life highlights?",
        "What student facilities are available besides academics?",
        "Do hostel facilities include common rooms and support services?",
        "What should new students know about campus access and movement?",
    ]
    for q in campus_questions:
        cases.append(make_in_case(q, "Campus / Hostel / Transport", campus_kw, 1))

    academics_questions = [
        "Where can I check the academic calendar?",
        "How does the examination process work at KRMU?",
        "What is the attendance policy at KRMU?",
        "What does the code of conduct include?",
        "What is KRMU anti-ragging policy?",
        "How can students submit grievances?",
        "What is the role of Internal Complaints Committee?",
        "Who is the ombudsperson at KRMU?",
        "What does academic affairs office handle?",
        "What pedagogy approach does KRMU follow?",
        "What kind of international collaborations does KRMU have?",
        "What is industry connect at KRMU?",
        "What is Dean's Honor List?",
        "What research centers are available at KRMU?",
        "Where can I access PhD regulations?",
        "What is included in PhD submission checklist?",
        "Where can I see list of eligible PhD supervisors?",
        "Where can I check list of awarded PhDs?",
        "Where can I find PhD entrance exam syllabus?",
        "What is in PhD programme course handbook?",
        "What are recent PhD admission calls?",
        "Who can apply as full-time, part-time, or sponsored scholar?",
        "How many research publications are highlighted by KRMU?",
        "What are high-end teaching and research labs at KRMU?",
        "What is Academic Bank of Credits in KRMU context?",
    ]
    for q in academics_questions:
        cases.append(make_in_case(q, "Academics / Policies / PhD", academic_kw, 1))

    life_questions = [
        "What clubs and societies can students join at KRMU?",
        "What support does student welfare provide?",
        "What is Youth Red Cross Committee at KRMU?",
        "What does community connect initiative do?",
        "How can students track KRMU news and events?",
        "What is KRMU Times?",
        "Where can I see KRMU print coverage?",
        "Where can I access KRMU video gallery?",
        "What kind of conferences does KRMU host?",
        "How active is alumni engagement at KRMU?",
        "Are there career counselling events for students?",
        "What cultural activities are available on campus?",
        "What student-led initiatives happen at KRMU?",
        "How can I join extracurricular clubs?",
        "What committee supports student grievance and discipline issues?",
        "How does KRMU encourage social impact activities?",
        "Are there orientation programmes for freshers?",
        "What is campus quest event about?",
        "How can students showcase projects and innovation?",
        "Is there support for communication and personality development?",
        "How does KRMU promote leadership among students?",
        "What are opportunities for peer learning at KRMU?",
        "How can students stay updated with upcoming activities?",
        "Are there student support bodies beyond academics?",
        "Where can students raise concerns if unresolved at department level?",
    ]
    for q in life_questions:
        cases.append(make_in_case(q, "Student Life / Support", life_kw, 1))

    mixed_cases = [
        ("Tell me about bus routes and highest placement package.", ["bus", "route", "package", "placement"], 2),
        ("Explain BTech CSE fee structure and scholarship options.", ["fee", "scholarship", "btech"], 2),
        ("I need hostel charges and payment procedure details.", ["hostel", "payment", "fee"], 2),
        ("Give me admission steps and required documents.", ["admission", "application", "documents"], 2),
        ("Share PhD admission process and interview requirement.", ["ph.d", "admission", "interview"], 2),
        ("What are placement training modules and mock tests?", ["placement", "training", "mock"], 2),
        ("Tell me about anti-ragging policy and grievance mechanism.", ["ragging", "policy", "grievance"], 2),
        ("Summarize campus facilities and student clubs.", ["campus", "facilit", "club"], 2),
        ("I want fee refund policy and helpdesk contact.", ["refund", "fee", "help"], 2),
        ("What are international admission highlights and hostel options?", ["international", "admission", "hostel"], 2),
        ("Explain recruiters, top package, and placement process.", ["recruit", "package", "placement"], 2),
        ("Give me scholarship criteria and attendance condition.", ["scholarship", "criteria", "attendance"], 2),
        ("How do I book campus tour and where is university located?", ["campus", "tour", "located"], 2),
        ("What is dean honor list and academic calendar timing?", ["dean", "honor", "calendar"], 2),
        ("Compare admission office support with payment support.", ["admission", "payment", "support"], 2),
        ("Tell me CDC role and interview readiness support.", ["cdc", "career", "interview"], 2),
        ("Need details on bus transport and hostel life.", ["bus", "transport", "hostel"], 2),
        ("PhD regulations and submission checklist please.", ["ph.d", "regulation", "checklist"], 2),
        ("What clubs, events, and student welfare options are there?", ["club", "event", "welfare"], 2),
        ("I need placement portal info and recruiter list.", ["placement", "portal", "recruit"], 2),
        ("Can you give fee structure, scholarship, and financial aid summary?", ["fee", "scholarship", "financial"], 2),
        ("Explain admission offer, enrollment, and first fee payment.", ["admission", "enroll", "fee"], 2),
        ("How does KRMU support research and industry collaboration?", ["research", "industry", "collaboration"], 2),
        ("Tell me anti-ragging policy plus ICC and ombudsperson.", ["ragging", "icc", "ombud"], 2),
        ("Share campus facilities, sports, and library highlights.", ["campus", "sports", "library"], 2),
    ]
    for q, kw, mn in mixed_cases:
        cases.append(make_in_case(q, "Multi-topic", kw, mn))

    out_questions = [
        "What is the weather on Mars today?",
        "Who will win the next IPL season?",
        "Should I buy Bitcoin this week?",
        "Give me today's NIFTY intraday strategy.",
        "Write a Python quicksort implementation with complexity proof.",
        "Recommend best Korean drama to binge this weekend.",
        "What is the cure for migraine at home?",
        "Predict the next US presidential election result.",
        "Can you translate this sentence to Spanish?",
        "Plan a 7-day Europe budget itinerary for me.",
        "Who is better Messi or Ronaldo with full stats?",
        "What are guaranteed stocks to double in one month?",
        "Write a breakup apology letter for my girlfriend.",
        "How to crack Google coding interview in 30 days?",
        "Best gaming laptop under 1 lakh right now?",
        "Tell me calories in a keto diet for fat loss.",
        "How to file US taxes as a non-resident?",
        "Can you explain quantum field theory basics?",
        "Give me astrology prediction for my career.",
        "Who is the richest person in the world right now?",
        "Suggest safest way to invest 50 lakh in mutual funds.",
        "What are top tourist places in Japan in spring?",
        "Can you solve this LeetCode hard graph problem?",
        "What is latest transfer news in European football?",
        "Should I switch from Android to iPhone 18 this year?",
    ]
    for q in out_questions:
        cases.append(make_out_case(q))

    if len(cases) != 200:
        raise RuntimeError(f"Expected 200 cases, got {len(cases)}")
    return cases


def evaluate_case(case: EvalCase) -> CaseResult:
    payload = {"message": case.question}
    started = time.time()

    try:
        resp = requests.post(API_URL, json=payload, timeout=TIMEOUT_SECS)
        latency = time.time() - started
    except Exception as exc:
        return CaseResult(
            question=case.question,
            category=case.category,
            expect=case.expect,
            passed=False,
            status_code=0,
            latency_sec=round(time.time() - started, 3),
            found_keywords=[],
            missing_keywords=list(case.keywords or []),
            sources_count=0,
            reason=f"request_error: {exc}",
            answer_preview="",
        )

    if resp.status_code != 200:
        return CaseResult(
            question=case.question,
            category=case.category,
            expect=case.expect,
            passed=False,
            status_code=resp.status_code,
            latency_sec=round(latency, 3),
            found_keywords=[],
            missing_keywords=list(case.keywords or []),
            sources_count=0,
            reason=f"http_{resp.status_code}",
            answer_preview=resp.text[:280],
        )

    data = resp.json()
    answer = (data.get("answer") or "").strip()
    answer_l = answer.lower()
    sources = data.get("sources") or []
    source_count = len(sources)

    if case.expect == "out_of_domain":
        guardrail_ok = any(h in answer_l for h in OUT_OF_SCOPE_HINTS)
        return CaseResult(
            question=case.question,
            category=case.category,
            expect=case.expect,
            passed=guardrail_ok,
            status_code=200,
            latency_sec=round(latency, 3),
            found_keywords=[],
            missing_keywords=[],
            sources_count=source_count,
            reason="ok" if guardrail_ok else "missing_guardrail_language",
            answer_preview=answer[:280],
        )

    keywords = case.keywords or []
    found = [kw for kw in keywords if kw.lower() in answer_l]
    missing = [kw for kw in keywords if kw.lower() not in answer_l]
    keyword_ok = len(found) >= case.min_required
    source_ok = source_count > 0
    passed = keyword_ok and source_ok

    if not keyword_ok and not source_ok:
        reason = "missing_keywords_and_sources"
    elif not keyword_ok:
        reason = "missing_keywords"
    elif not source_ok:
        reason = "missing_sources"
    else:
        reason = "ok"

    return CaseResult(
        question=case.question,
        category=case.category,
        expect=case.expect,
        passed=passed,
        status_code=200,
        latency_sec=round(latency, 3),
        found_keywords=found,
        missing_keywords=missing,
        sources_count=source_count,
        reason=reason,
        answer_preview=answer[:280],
    )


def summarize(results: List[CaseResult]) -> Dict:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    acc = (passed / total * 100.0) if total else 0.0
    latencies = [r.latency_sec for r in results]

    by_category: Dict[str, Dict] = {}
    for r in results:
        bucket = by_category.setdefault(
            r.category,
            {"total": 0, "passed": 0, "failed": 0, "accuracy": 0.0},
        )
        bucket["total"] += 1
        if r.passed:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1

    for category, bucket in by_category.items():
        _ = category
        bucket["accuracy"] = round(bucket["passed"] / bucket["total"] * 100.0, 2)

    failed_samples = [
        {
            "category": r.category,
            "question": r.question,
            "reason": r.reason,
            "answer_preview": r.answer_preview,
        }
        for r in results
        if not r.passed
    ][:20]

    return {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "accuracy_percent": round(acc, 2),
            "avg_latency_sec": round(statistics.mean(latencies), 3) if latencies else None,
            "p95_latency_sec": round(statistics.quantiles(latencies, n=20)[18], 3) if len(latencies) >= 20 else None,
        },
        "by_category": by_category,
        "failed_samples": failed_samples,
        "results": [asdict(r) for r in results],
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    cases = build_cases()

    results: List[CaseResult] = []
    for idx, case in enumerate(cases, start=1):
        result = evaluate_case(case)
        results.append(result)
        if idx % 20 == 0:
            passed_so_far = sum(1 for r in results if r.passed)
            print(f"Progress: {idx}/200 | pass={passed_so_far} | fail={idx - passed_so_far}")

    report = summarize(results)
    report_path = root / "student_200_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n===== 200-Question Benchmark Complete =====")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
