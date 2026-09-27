"""Run an LLM over a ViHoRec annotation sheet via OpenRouter.

Two models act as two independent annotators, exactly like two people would:
each sees one item at a time, with no similarity score, no pipeline decision and
no other annotator's label. Disagreements go to the existing adjudication step
in ``scripts/annotation_packets.py``, so Cohen's kappa between the two models is
computed by the same code that would score two humans.

**These are model labels, not human labels.** They must be described that way in
the manuscript, in ``annotation/GUIDELINES.md`` §3 and in the response letter.
Every output row records which model produced it, and ``llm_provenance.json``
records which annotator slot each model filled, so the distinction survives even
if the CSVs are read on their own. The reviewer asked for a validated ground
truth; LLM labels adjudicated by a person are defensible when disclosed, and
indefensible when passed off as manual annotation.

Runs are resumable. Every answer is appended to a JSONL cache keyed by item id,
so an interrupted run costs nothing to restart and a re-run with the same model
issues no requests at all. Delete the cache file to force a fresh pass.

Three annotators are better than two here. With binary labels an odd panel
always has a majority, so a gold label exists without a tie-break, and Fleiss'
kappa over three genuine raters is meaningful in a way the withdrawn
criteria-based statistic never was. Non-unanimous items are still routed to a
human adjudicator, because a 2-1 split among models is exactly where correlated
model error shows up.

Usage:
    # one annotator per model
    python annotate.py --task er_pairs --model openai/gpt-5-mini     --as-annotator 1
    python annotate.py --task er_pairs --model openai/gpt-4.1-mini   --as-annotator 2
    python annotate.py --task er_pairs --model z-ai/glm-5.3-flash    --as-annotator 3

    # the smaller hotel-entry task
    python annotate.py --task qc_hotels --model openai/gpt-5-mini    --as-annotator 1
    python annotate.py --task qc_hotels --model openai/gpt-4.1-mini  --as-annotator 2
    python annotate.py --task qc_hotels --model z-ai/glm-5.3-flash   --as-annotator 3

    # try five items first to check the prompt and the cost
    python annotate.py --task er_pairs --model z-ai/glm-5.3-flash --limit 5 --dry-run

    # what would be sent for a given model
    python annotate.py --show-profile --model z-ai/glm-5.3-flash

Then merge and score with the existing tooling:
    cd ..  &&  python annotation_packets.py merge
              python er_evaluate.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C  # noqa: E402
from openrouter_client import AuthError, OpenRouterClient, OpenRouterError  # noqa: E402

CACHE_DIR = C.OUT_ANNOTATION / "llm_cache"
PACKET_DIR = C.OUT_ANNOTATION / "packets"
PROVENANCE = C.OUT_ANNOTATION / "llm_provenance.json"
MAX_ANNOTATORS = 5

# Per-family request settings. Vendors differ in ways that fail quietly rather
# than loudly, so the differences are declared here instead of being discovered
# on a 426-item bill.
#
#   openai/gpt-5*      reasoning model; rejects `temperature`; efforts low/medium/high.
#   z-ai/glm-5.3*      reasoning is always on and cannot be disabled. Efforts are
#                      low/high/max and default to `max`; anything else, including
#                      `medium`, silently resolves to `max` and costs several times
#                      more. Z.ai tunes the chat template for temperature 1.0 and
#                      top_p 0.95, so determinism comes from `seed`, not from
#                      forcing temperature to 0.
#   everything else    plain sampling model; temperature 0 for reproducibility.
DEFAULT_PROFILE = {
    "temperature": 0.0,
    "top_p": None,
    "reasoning": False,
    "efforts": (),
    "default_effort": None,
}
MODEL_PROFILES: tuple[tuple[str, dict], ...] = (
    (r"^openai/gpt-5", {
        "temperature": None,
        "top_p": None,
        "reasoning": True,
        "efforts": ("low", "medium", "high"),
        "default_effort": "low",
    }),
    (r"^z-ai/glm-5\.3", {
        "temperature": 1.0,
        "top_p": 0.95,
        "reasoning": True,
        "efforts": ("low", "high", "max"),
        "default_effort": "low",
    }),
)


def model_profile(model: str) -> dict:
    for pattern, profile in MODEL_PROFILES:
        if re.search(pattern, model, re.IGNORECASE):
            return {**DEFAULT_PROFILE, **profile}
    return dict(DEFAULT_PROFILE)


def resolve_request_params(model: str, args: argparse.Namespace) -> dict:
    """Merge the model profile with explicit CLI overrides.

    Raises SystemExit on a reasoning effort the model does not accept. GLM
    resolves an unknown effort to `max` without complaining, which turns a typo
    into a silent multiple of the expected cost, so it is rejected up front.
    """
    profile = model_profile(model)

    effort = None
    if profile["reasoning"]:
        effort = args.reasoning_effort or profile["default_effort"]
        if effort not in profile["efforts"]:
            raise SystemExit(
                f"--reasoning-effort '{effort}' is not valid for {model}.\n"
                f"  Accepted: {', '.join(profile['efforts'])}\n"
                "  Note: this model resolves unknown values to its most expensive "
                "level instead of erroring, so the run is stopped here."
            )
    elif args.reasoning_effort:
        print(f"note: {model} is not a reasoning model; --reasoning-effort ignored")

    temperature = profile["temperature"] if args.temperature is None else args.temperature
    top_p = profile["top_p"] if args.top_p is None else args.top_p
    return {
        "temperature": temperature,
        "top_p": top_p,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "reasoning_effort": effort,
    }

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "annotation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "integer", "enum": [0, 1]},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "reason": {"type": "string"},
            },
            "required": ["label", "confidence", "reason"],
            "additionalProperties": False,
        },
    },
}

# Mirrors annotation/GUIDELINES.md section 2.1. Keep the two in sync: if the
# decision rules change for humans, they must change for the models too.
ER_SYSTEM = """You annotate a Vietnamese hotel dataset. For each pair of listings, decide whether they refer to the SAME physical hotel.

Answer 1 (same hotel) when the names differ only by:
- diacritics or transliteration: "Phú Quốc" / "Phu Quoc"
- a property-type prefix or suffix: "Khách sạn X", "X Hotel", "Khu nghỉ dưỡng X", "X Resort"
- word order or punctuation: "Phan Thiet Ocean Dunes Resort" / "Khu nghỉ dưỡng Ocean Dunes Phan Thiết"
- an operator or brand spelling that resolves to the same property: "Mövenpick Resort Phan Thiet" / "Khu nghỉ dưỡng Movenpick Phan Thiết"

Answer 0 (different hotels) when any of these hold:
- the properties are in DIFFERENT cities, even under an identical brand: "Raon Hotel" Đà Nẵng vs "Raon Hotel" Quy Nhơn
- the names carry different tier or sub-brand markers of the same operator: "Mường Thanh Grand" vs "Mường Thanh Luxury"; "Majestic" vs "Majestic Premium"
- the names carry different branch numbers: "La Cactus Hotel" vs "La Cactus Hotel 2"
- the property types genuinely differ at the same brand: "Raon Hotel" vs "Raon Villa"; "Sea Links Beach" (hotel) vs "Sea Links Beach Villas"

A blank city means unknown, not a match. If the evidence is inconclusive, answer 0 and say so in the reason: a merge that cannot be confirmed must not be credited.

Judge each pair on its own. Do not try to balance how often you answer 1 or 0. Reply with JSON only."""

QC_SYSTEM = """You annotate a Vietnamese hotel catalogue. For each entry, decide whether it denotes exactly one real hotel at the stated location.

Answer 1 when the name identifies a single property that plausibly exists in the stated city.

Answer 0 when the entry is a chain-level or generic name that could cover several properties, when the location contradicts the name, or when the entry looks like a fragment of a name.

Reply with JSON only."""


def _er_prompt(row: pd.Series) -> str:
    return (
        "Listing A\n"
        f"  name: {row['name_a']}\n"
        f"  city: {row.get('city_a') or '(unknown)'}\n"
        f"  platforms: {row.get('platforms_a') or '(unknown)'}\n\n"
        "Listing B\n"
        f"  name: {row['name_b']}\n"
        f"  city: {row.get('city_b') or '(unknown)'}\n"
        f"  platforms: {row.get('platforms_b') or '(unknown)'}\n\n"
        "Do A and B refer to the same physical hotel?"
    )


def _qc_prompt(row: pd.Series) -> str:
    return (
        "Catalogue entry\n"
        f"  name: {row['name']}\n"
        f"  location: {row.get('location') or '(unknown)'}\n\n"
        "Does this entry denote exactly one real hotel at this location?"
    )


TASKS = {
    "er_pairs": {
        "source": "er_pairs_to_annotate.csv",
        "id_col": "pair_id",
        "show": ["pair_id", "name_a", "city_a", "platforms_a",
                 "name_b", "city_b", "platforms_b"],
        "system": ER_SYSTEM,
        "prompt": _er_prompt,
        "row_filter": None,
    },
    "qc_hotels": {
        "source": "qc_sample_to_annotate.csv",
        "id_col": "record_id",
        "show": ["record_id", "hotel_id", "name", "location"],
        "system": QC_SYSTEM,
        "prompt": _qc_prompt,
        "row_filter": lambda df: df["record_type"] == "hotel",
    },
}


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", model.lower()).strip("-")


def _parse_json(text: str) -> dict:
    """Read the model's JSON answer, tolerating markdown fences and prose."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"could not parse JSON from: {text[:200]}")


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in rec:
                out[str(rec["id"])] = rec
    return out


def _load_items(spec: dict, limit: int | None) -> pd.DataFrame:
    src = C.OUT_ANNOTATION / spec["source"]
    if not src.exists():
        raise SystemExit(
            f"Missing {src}.\n"
            "Build the sheets first:  python er_candidate_pairs.py  /  "
            "python annotation_agreement.py sample --n 250"
        )
    df = pd.read_csv(src)
    if spec["row_filter"] is not None:
        df = df[spec["row_filter"](df)]
    cols = [c for c in spec["show"] if c in df.columns]
    df = df[cols].copy()
    return df.head(limit) if limit else df


def run(args: argparse.Namespace) -> dict:
    spec = TASKS[args.task]
    items = _load_items(spec, args.limit)
    id_col = spec["id_col"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PACKET_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slug(args.model)
    cache_path = CACHE_DIR / f"{args.task}__{slug}.jsonl"
    cache = _load_cache(cache_path)

    todo = items[~items[id_col].astype(str).isin(cache)]
    print(
        f"task={args.task}  model={args.model}\n"
        f"  items={len(items)}  cached={len(items) - len(todo)}  to_request={len(todo)}"
    )

    params = resolve_request_params(args.model, args)

    if args.dry_run:
        if len(todo):
            preview = spec["prompt"](todo.iloc[0])
            print("\n--- system ---\n" + spec["system"])
            print("\n--- first user message ---\n" + preview)
        print("\n--- request parameters ---\n" + json.dumps(params, indent=2))
        print("\nDry run: no requests sent.")
        return {
            "dry_run": True,
            "items": int(len(items)),
            "to_request": int(len(todo)),
            "params": params,
        }

    client = OpenRouterClient(
        args.model,
        max_retries=args.max_retries,
        require_parameters=not args.no_require_parameters,
        **params,
    )

    write_lock = threading.Lock()
    failures: list[tuple[str, str]] = []
    done = 0

    def annotate_one(row: pd.Series) -> None:
        nonlocal done
        item_id = str(row[id_col])
        messages = [
            {"role": "system", "content": spec["system"]},
            {"role": "user", "content": spec["prompt"](row)},
        ]
        record = None
        last_exc = None
        # A malformed answer is usually a degenerate sample rather than a bad
        # prompt, so it is worth resampling. The seed must change: replaying the
        # same one reproduces the same broken output.
        for attempt in range(args.parse_retries + 1):
            try:
                result = client.chat(
                    messages,
                    response_format=RESPONSE_FORMAT,
                    seed=None if args.seed is None else args.seed + 1000 * attempt,
                )
                parsed = _parse_json(result.text)
                record = {
                    "id": item_id,
                    "label": int(parsed["label"]),
                    "confidence": parsed.get("confidence", ""),
                    "reason": str(parsed.get("reason", ""))[:500],
                    "model": result.model,
                    "requested_model": args.model,
                    "attempts": attempt + 1,
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                break
            except AuthError:
                raise
            except (OpenRouterError, ValueError, KeyError, TypeError) as exc:
                last_exc = exc

        if record is None:
            with write_lock:
                failures.append((item_id, str(last_exc)[:200]))
            return

        with write_lock:
            cache[item_id] = record
            with cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} done", flush=True)

    if len(todo):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(annotate_one, row) for _, row in todo.iterrows()]
            for fut in as_completed(futures):
                fut.result()  # re-raise AuthError so the run stops loudly

    out_path = _write_packet(items, spec, cache, slug, args)
    summary = {
        "task": args.task,
        "model": args.model,
        "items": int(len(items)),
        "labelled": int(sum(1 for i in items[id_col].astype(str) if i in cache)),
        "failed": len(failures),
        "cache": str(cache_path),
        "output": str(out_path),
        "usage": client.usage_summary(),
    }
    if failures:
        summary["failure_examples"] = failures[:5]
    if args.as_annotator:
        summary["annotator_slot"] = args.as_annotator
        _record_provenance(args, slug, summary, params)
    return summary


def _write_packet(items: pd.DataFrame, spec: dict, cache: dict,
                  slug: str, args: argparse.Namespace) -> Path:
    """Write the labels in the packet layout the merge step already reads."""
    id_col = spec["id_col"]
    out = items.copy()
    ids = out[id_col].astype(str)
    out["label"] = [
        cache[i]["label"] if i in cache else "" for i in ids
    ]
    out["notes"] = [
        f"[{cache[i]['requested_model']}|{cache[i]['confidence']}] {cache[i]['reason']}"
        if i in cache else ""
        for i in ids
    ]

    path = PACKET_DIR / f"{args.task}_llm_{slug}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")

    if args.as_annotator:
        # The merge step reads exactly this filename for annotator N.
        slot = PACKET_DIR / f"{args.task}_annotator{args.as_annotator}.csv"
        out.to_csv(slot, index=False, encoding="utf-8-sig")
        print(f"  also written as annotator {args.as_annotator}: {slot.name}")
    return path


def _record_provenance(args: argparse.Namespace, slug: str, summary: dict,
                       params: dict) -> None:
    """Keep a durable record of which model filled which annotator slot."""
    data = {}
    if PROVENANCE.exists():
        try:
            data = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data.setdefault(args.task, {})[f"annotator_{args.as_annotator}"] = {
        "kind": "llm",
        "model": args.model,
        "resolved_slug": slug,
        "items": summary["items"],
        "labelled": summary["labelled"],
        "params": params,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Model-generated labels. Disclose as LLM annotation, not manual annotation.",
    }
    PROVENANCE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", choices=sorted(TASKS),
                    help="required unless --show-profile is used")
    ap.add_argument("--model", default="openai/gpt-5-mini",
                    help="OpenRouter slug: openai/gpt-5-mini, openai/gpt-4.1-mini, "
                         "z-ai/glm-5.3-flash, ...")
    ap.add_argument("--as-annotator", type=int, default=None,
                    choices=range(1, MAX_ANNOTATORS + 1), metavar=f"1..{MAX_ANNOTATORS}",
                    help="also write packets/<task>_annotator<N>.csv for the merge step")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="annotate only the first N items")
    ap.add_argument("--temperature", type=float, default=None,
                    help="default comes from the model profile; see --show-profile")
    ap.add_argument("--top-p", type=float, default=None,
                    help="default comes from the model profile")
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED,
                    help="best-effort determinism; matters most where the vendor "
                         "recommends a high temperature")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--reasoning-effort", default=None,
                    help="valid values differ per model (gpt-5: low/medium/high; "
                         "glm-5.3: low/high/max); defaults to the cheapest level")
    ap.add_argument("--max-retries", type=int, default=5,
                    help="retries for transport and HTTP errors")
    ap.add_argument("--parse-retries", type=int, default=2,
                    help="resamples when the answer is not valid JSON, each with "
                         "a different seed")
    ap.add_argument("--no-require-parameters", action="store_true",
                    help="allow providers that ignore response_format; only for "
                         "models no strict-schema provider serves")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompt, parameters and request count, send nothing")
    ap.add_argument("--show-profile", action="store_true",
                    help="print the resolved request parameters for --model and exit")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.show_profile:
        print(json.dumps(
            {
                "model": args.model,
                "profile": model_profile(args.model),
                "resolved_params": resolve_request_params(args.model, args),
            },
            ensure_ascii=False, indent=2, default=list,
        ))
        return
    if not args.task:
        ap.error("--task is required (or use --show-profile)")

    try:
        summary = run(args)
    except AuthError as exc:
        raise SystemExit(f"Authentication failed.\n{exc}")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
