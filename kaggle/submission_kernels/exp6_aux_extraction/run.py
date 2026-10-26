"""Exp 6 step 1: report-derived auxiliary findings, Qwen3-14B AWQ on vLLM (T4x2).

Reads the competition train.csv inside Kaggle; report text never leaves the
kernel (internet is used only for `pip install vllm`). Gold studies (rows with
host labels) are skipped. Output: /kaggle/working/aux_findings.jsonl, one row per
study with the raw model text and the parsed yes/no/na dict. Written in chunks
so a partial run is still usable.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

VLLM_SPEC = "vllm==0.11.0"     # Python 3.13 wheels (Kaggle image); V1 engine, Triton attention on sm75 (T4)
LIMIT = int(os.environ.get("AUX_LIMIT", "0")) or None
CHUNK = 256
OUT = Path("/kaggle/working/aux_findings.jsonl")
ROOTS = (Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
         Path("/kaggle/input/rsna-knee-abnormality-detection"))
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# key -> definition shown to the model. Location / type detail beyond the 12 targets.
FINDINGS = {
    "pcl_tear": "posterior cruciate ligament tear (partial or complete)",
    "lcl_injury": "lateral collateral ligament / posterolateral corner sprain or tear",
    "acl_partial_tear": "ACL partial tear or high-grade sprain (not complete)",
    "acl_complete_tear": "ACL complete / full-thickness tear or rupture",
    "acl_reconstruction": "ACL graft / prior ACL reconstruction",
    "mcl_tear": "MCL partial or complete tear (grade II-III), not just periligamentous oedema",
    "medial_meniscus_posterior_horn_tear": "tear involving the posterior horn of the medial meniscus",
    "medial_meniscus_body_anterior_tear": "tear involving the body or anterior horn of the medial meniscus",
    "lateral_meniscus_posterior_horn_tear": "tear involving the posterior horn of the lateral meniscus",
    "lateral_meniscus_body_anterior_tear": "tear involving the body or anterior horn of the lateral meniscus",
    "meniscus_horizontal_or_degenerative_tear": "horizontal, oblique or degenerative meniscal tear (any meniscus)",
    "meniscus_complex_or_displaced_tear": "complex, radial, flap, bucket-handle or displaced meniscal tear (any meniscus)",
    "meniscal_root_tear": "meniscal root tear (any meniscus)",
    "meniscal_extrusion": "meniscal extrusion (any meniscus)",
    "meniscal_degeneration_no_tear": "intrasubstance meniscal degeneration / signal without a tear",
    "meniscal_cyst": "meniscal or parameniscal cyst",
    "cartilage_defect_medial": "cartilage loss / chondral defect / chondromalacia in the medial femorotibial compartment",
    "cartilage_defect_lateral": "cartilage loss / chondral defect / chondromalacia in the lateral femorotibial compartment",
    "cartilage_defect_patella": "cartilage loss / chondromalacia of the patella",
    "cartilage_defect_trochlea": "cartilage loss / chondral defect of the trochlea",
    "cartilage_full_thickness": "full-thickness or grade IV cartilage defect anywhere",
    "osteophytes": "osteophytes anywhere",
    "subchondral_cyst_or_sclerosis": "subchondral cysts or subchondral sclerosis anywhere",
    "bme_medial_femoral_condyle": "bone marrow oedema / bruise in the medial femoral condyle",
    "bme_lateral_femoral_condyle": "bone marrow oedema / bruise in the lateral femoral condyle",
    "bme_medial_tibial_plateau": "bone marrow oedema / bruise in the medial tibial plateau",
    "bme_lateral_tibial_plateau": "bone marrow oedema / bruise in the lateral tibial plateau",
    "bme_patella": "bone marrow oedema / bruise in the patella",
    "osteochondral_lesion_or_osteonecrosis": "osteochondral lesion, osteochondritis dissecans, osteonecrosis or insufficiency fracture",
    "patellar_tendon_abnormality": "patellar tendinosis, tendinopathy or tear",
    "quadriceps_tendon_abnormality": "quadriceps tendinosis, tendinopathy or tear",
    "popliteus_abnormality": "popliteus tendon or muscle strain, tendinosis or tear",
    "hoffa_fat_pad_oedema": "Hoffa (infrapatellar) fat pad oedema or impingement",
    "mpfl_retinaculum_injury": "MPFL or patellar retinaculum sprain or tear",
    "patellar_maltracking": "patellar tilt, lateral subluxation, patella alta or trochlear dysplasia",
    "bursitis": "any bursitis (pes anserine, prepatellar, infrapatellar, semimembranosus-MCL ...)",
    "iliotibial_band_abnormality": "iliotibial band oedema / friction syndrome",
    "loose_body": "intra-articular loose body",
    "plica": "synovial plica (thickened or symptomatic)",
    "postoperative_changes": "any prior knee surgery (meniscectomy, repair, hardware, reconstruction)",
}

SYSTEM = (
    "You extract structured findings from knee MRI reports written in any language. "
    "For each finding key answer exactly one of: \"yes\" (the report states it is present, "
    "including probable or suspected), \"no\" (the report states it is absent, normal or intact, "
    "including group statements such as 'cruciate ligaments intact' or 'menisci normal'), or "
    "\"na\" (the report does not address it). Do not infer from other findings. "
    "Reply with one JSON object only, using every key below exactly once.\n\nKeys:\n"
    + "\n".join(f"- {k}: {v}" for k, v in FINDINGS.items())
)


def install_vllm() -> None:
    try:
        import vllm  # noqa: F401
        return
    except ImportError:
        pass
    t = time.time()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", VLLM_SPEC, "transformers==4.57.1"], check=True)
    print(f"installed {VLLM_SPEC} in {time.time() - t:.0f}s", flush=True)


def locate_model() -> Path:
    for config in Path("/kaggle/input").rglob("config.json"):
        s = str(config).lower()
        if "qwen" in s and "14b" in s and "awq" in s:
            return config.parent
    raise FileNotFoundError("Attach the Kaggle model qwen-lm/qwen-3/transformers/14b-awq")


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in FINDINGS:
        v = str(raw.get(k, "na")).strip().lower()
        out[k] = v if v in ("yes", "no", "na") else "na"
    return out


def main() -> None:
    import csv
    csv.field_size_limit(10 ** 8)
    root = next((p for p in ROOTS if (p / "train.csv").is_file()), None)
    if root is None:
        raise FileNotFoundError("Competition train.csv is not attached")
    rows = list(csv.DictReader(open(root / "train.csv", newline="", encoding="utf-8")))
    rows = [r for r in rows if not any((r.get(c) or "").strip() for c in TARGETS) and r["Report"].strip()]
    rows = rows[:LIMIT] if LIMIT else rows
    print(f"{len(rows)} report-only studies (gold excluded)", flush=True)

    install_vllm()
    from vllm import LLM, SamplingParams
    model = locate_model()
    print(f"model {model}", flush=True)
    t = time.time()
    llm = LLM(model=str(model), quantization="awq", dtype="half", tensor_parallel_size=2,
              max_model_len=6144, gpu_memory_utilization=0.85, max_num_seqs=64, seed=0)
    print(f"engine up in {time.time() - t:.0f}s", flush=True)
    params = SamplingParams(temperature=0.0, max_tokens=900)
    tok = llm.get_tokenizer()

    def prompt(report: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Report:\n{report}"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    done = set()
    if OUT.exists():
        done = {json.loads(l)["StudyInstanceUID"] for l in OUT.open()}
    todo = [r for r in rows if r["StudyInstanceUID"] not in done]
    # a 16-report smoke batch first (timing + sample in the log), then full chunks
    chunks = [todo[:16]] + [todo[i:i + CHUNK] for i in range(16, len(todo), CHUNK)]
    t0, n, bad = time.time(), 0, 0
    for k, batch in enumerate(c for c in chunks if c):
        outs = llm.generate([prompt(r["Report"]) for r in batch], params, use_tqdm=False)
        with OUT.open("a") as f:
            for r, o in zip(batch, outs):
                text = o.outputs[0].text
                parsed = parse(text)
                bad += parsed is None
                f.write(json.dumps({"StudyInstanceUID": r["StudyInstanceUID"], "parsed": parsed,
                                    "text": text, "n_out_tokens": len(o.outputs[0].token_ids)}) + "\n")
        n += len(batch)
        el = time.time() - t0
        print(f"{n}/{len(todo)} done, {el / 60:.1f} min, {n / el * 60:.1f} reports/min, parse failures {bad}", flush=True)
        if k == 0:
            print("SAMPLE", json.dumps(parse(outs[0].outputs[0].text)), flush=True)
    print(f"AUX_DONE {n} studies, parse failures {bad}", flush=True)


if __name__ == "__main__":
    main()
