"""Local eval gate — reads the golden dataset from CSV, runs the live agent
per example, scores with a Gemini judge, and writes eval_report_{provider}.json.

This sidesteps the LangSmith dataset API (which has flapping auth in this env).
The judge stays on gemini-2.0-flash for BOTH provider runs so the Gemini-vs-Claude
comparison is apples-to-apples; any Gemini-favoring bias makes the gate conservative.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import google.generativeai as genai

import config

genai.configure(api_key=config.GEMINI_API_KEY)
JUDGE_MODEL = config.GEMINI_MODEL_FLASH

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "eval_dataset_golden.csv")

CORRECTNESS_PROMPT = """\
You are evaluating an AI assistant called Momo. Given the user's message, Momo's response, and the correctness criteria below, determine if the response PASSES or FAILS.

User message: {input}
Momo's response: {output}
Criteria: {criteria}

A response PASSES if it meets ALL the criteria. It FAILS if it violates ANY criterion.
Respond with ONLY one of: "PASS" or "FAIL" followed by a brief reason.
"""

HALLUCINATION_PROMPT = """\
You are evaluating Momo, which has tools (Gmail, Calendar, Tasks, Knowledge Graph, Jira).
Momo MUST only reference data from tool calls; never fabricate emails/meetings/tasks/people.
Input: {input}
Output: {output}
Return "no hallucination" if every claim traces to tool results, the user's message, or is generic commentary. Return "hallucination" otherwise.
Respond with ONLY one of: "no hallucination" or "hallucination"
"""

QUALITY_PROMPT = """\
Evaluate Momo's response quality (casual, helpful, accurate, scannable).
User message: {input}
Momo's response: {output}
Score 1-5 (5 excellent, 1 bad). Respond with ONLY a number from 1 to 5.
"""


def _judge(prompt: str, **kwargs) -> str:
    model = genai.GenerativeModel(model_name=JUDGE_MODEL)
    resp = model.generate_content(prompt.format(**kwargs))
    return resp.text.strip()


def _load_examples():
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def _score_example(user_message, response, criteria):
    correctness_raw = _judge(CORRECTNESS_PROMPT, input=user_message, output=response, criteria=criteria)
    correctness = 1 if correctness_raw.upper().startswith("PASS") else 0

    halluc_raw = _judge(HALLUCINATION_PROMPT, input=user_message, output=response).lower()
    hallucination_clean = 1 if "no hallucination" in halluc_raw else 0

    quality_raw = _judge(QUALITY_PROMPT, input=user_message, output=response)
    digits = "".join(c for c in quality_raw if c.isdigit())
    quality = int(digits[0]) if digits else 3

    return {
        "correctness": correctness,
        "hallucination_clean": hallucination_clean,
        "response_quality": quality,
    }


def run(provider: str, limit=None):
    from agent import run_agent_loop

    examples = _load_examples()
    if limit:
        examples = examples[:limit]

    results = []
    for i, ex in enumerate(examples, 1):
        user_message = ex["user_message"]
        criteria = ex.get("correctness_criteria", "")
        try:
            response, _ = run_agent_loop(user_message, [])
        except Exception as exc:
            response = f"[agent error: {exc}]"
        scores = _score_example(user_message, response, criteria)
        results.append({
            "user_message": user_message,
            "category": ex.get("category", ""),
            "response": response,
            **scores,
        })
        print(f"[{i}/{len(examples)}] {ex.get('category','')}: "
              f"correct={scores['correctness']} clean={scores['hallucination_clean']} q={scores['response_quality']}")

    n = len(results)
    aggregate = {
        "n": n,
        "correctness_rate": sum(r["correctness"] for r in results) / n if n else 0,
        "hallucination_clean_rate": sum(r["hallucination_clean"] for r in results) / n if n else 0,
        "avg_quality": sum(r["response_quality"] for r in results) / n if n else 0,
    }
    report = {"provider": provider, "judge": JUDGE_MODEL, "aggregate": aggregate, "examples": results}
    out = f"eval_report_{provider}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")
    print(f"  correctness: {aggregate['correctness_rate']*100:.0f}%  "
          f"clean: {aggregate['hallucination_clean_rate']*100:.0f}%  "
          f"quality: {aggregate['avg_quality']:.2f}/5")
    return report


def compare(baseline_path, candidate_path, margin=0.1):
    with open(baseline_path) as f:
        base = json.load(f)
    with open(candidate_path) as f:
        cand = json.load(f)
    b, c = base["aggregate"], cand["aggregate"]
    print("\n=== A/B GATE ===")
    print(f"{'metric':<26}{'baseline':>10}{'candidate':>12}")
    for k in ("correctness_rate", "hallucination_clean_rate", "avg_quality"):
        print(f"{k:<26}{b[k]:>10.3f}{c[k]:>12.3f}")

    by_msg = {e["user_message"]: e for e in base["examples"]}
    regressions = []
    for e in cand["examples"]:
        bm = by_msg.get(e["user_message"])
        if not bm:
            continue
        bq, cq = bm["response_quality"] / 5.0, e["response_quality"] / 5.0
        if cq < bq - margin or e["correctness"] < bm["correctness"]:
            regressions.append((e["user_message"], bm, e))

    agg_ok = (c["correctness_rate"] >= b["correctness_rate"]
              and c["hallucination_clean_rate"] >= b["hallucination_clean_rate"]
              and c["avg_quality"] >= b["avg_quality"] - margin * 5)
    passed = agg_ok and not regressions
    print(f"\naggregate_ok={agg_ok} regressions={len(regressions)}")
    for msg, bm, e in regressions:
        print(f"  REGRESSED: {msg[:50]} (q {bm['response_quality']}->{e['response_quality']}, "
              f"correct {bm['correctness']}->{e['correctness']})")
    print(f"\nGATE: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["gemini", "claude"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--margin", type=float, default=0.1)
    args = parser.parse_args()

    if args.compare:
        ok = compare(args.compare[0], args.compare[1], margin=args.margin)
        sys.exit(0 if ok else 1)
    elif args.provider:
        run(args.provider, limit=args.limit)
    else:
        parser.error("provide --provider or --compare")
