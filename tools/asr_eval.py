#!/usr/bin/env python3
"""Score TTS checkpoints by transcribing what they say.

Validation loss cannot choose between these checkpoints. The 10-epoch run's
loss fell monotonically to the last epoch, and that epoch is the one that drops
words — loss is computed under teacher forcing and never observes free-running
generation, so the failure is invisible to it.

This measures the failure instead. Each checkpoint speaks the same sentences at
the same seeds, Breeze-ASR-26 transcribes the result, and the transcript is
compared with the text that was asked for. Breeze-ASR-26 suits the job because
it writes Mandarin Han characters rather than Taigi orthography, which is the
script the TTS is prompted in.

Two numbers come out, and the second matters more:

  cer            edit distance over the reference. Mixes mispronunciation with
                 omission, and carries the ASR's own error — measured at 0.260
                 mean on real tai8 audio with correct transcripts, so absolute
                 values mean nothing. Only differences between checkpoints on
                 identical inputs are worth reading.
  length_ratio   transcript length over reference length. A misheard word still
                 produces roughly the right number of characters; a dropped
                 clause does not. This is the dropout signal.

  python tools/asr_eval.py --models taigi_full_4ep_ep4,taigi_full_10ep_ep10
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

TTS = "http://localhost:8065"
ASR = "http://localhost:8068"

SENTENCES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "eval_sentences.txt")


def load_sentences(path: str) -> list[tuple[str, str]]:
    """(domain, text) pairs. Comments and blank lines are skipped."""
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        domain, text = line.split("\t", 1)
        out.append((domain.strip(), text.strip()))
    return out


# Five seeds, fixed: the same take at a different seed scored twice as well in
# the first run, so a model compared at one seed is being compared on luck.
SEEDS = [2, 7, 42, 123, 777]

PUNCT = re.compile(r"[\s，。！？、,.!?~\-—…「」『』:：;；()（）\"']")


def post_json(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def post_audio(url: str, wav: bytes, reference: str, timeout: int) -> dict:
    boundary = "----asrEval"
    body = b""
    for name, value in (("reference", reference),):
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                 f"{value}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; "
             f"filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
    body += wav + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def synth(text: str, character: str, seed: int, endpoint: str) -> bytes:
    request = urllib.request.Request(
        f"{TTS}{endpoint}", data=json.dumps(
            {"text": text, "character": character, "seed": seed}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=900) as response:
        return response.read()


def score(cer: float, ratio: float) -> float:
    """Distance from a perfect reading.

    CER and |length_ratio - 1| are both fractions of the reference length, so
    they add without weighting. CER catches "said the wrong thing"; the ratio
    term catches "said too little" or "rambled", which CER alone can miss when
    insertions and deletions cancel.
    """
    return cer + abs(ratio - 1.0)


def _rank_rows(groups: dict, label: str) -> str:
    ranked = []
    for key, values in groups.items():
        cers = [v["cer"] for v in values]
        ratios = [v["length_ratio"] for v in values]
        mean_cer = statistics.mean(cers)
        mean_ratio = statistics.mean(ratios)
        se = statistics.stdev(cers) / len(cers) ** 0.5 if len(cers) > 1 else 0.0
        ranked.append((score(mean_cer, mean_ratio), key, mean_cer, mean_ratio, se, len(cers)))
    ranked.sort()
    best = ranked[0][0]
    out = []
    for i, (sc, key, mean_cer, mean_ratio, se, n) in enumerate(ranked, 1):
        # Anything inside one standard error of the leader is not distinguishable
        # from it; marking them stops the table being read as a strict order.
        tie = ' class="tie"' if i > 1 and sc - best <= se else ""
        medal = "&#x1F947;&#x1F948;&#x1F949;"[i - 1] if i <= 3 else str(i)
        out.append(
            f'<tr{tie}><td class="rk">{medal}</td><td>{key}</td>'
            f'<td class="num">{sc:.3f}</td><td class="num">{mean_cer:.3f}</td>'
            f'<td class="num">{mean_ratio:.3f}</td><td class="num dim">&plusmn;{se:.3f}</td>'
            f'<td class="num dim">{n}</td></tr>')
    return (f'<h2>{label}</h2><table><tr><th>#</th><th>{"模型" if "×" not in label else "模型 × 種子"}</th>'
            '<th>綜合分數</th><th>平均 CER</th><th>平均長度比</th>'
            '<th>CER 標準誤</th><th>樣本</th></tr>' + "\n".join(out) + "</table>")


def write_report(directory: str, takes: list[dict]) -> None:
    """One page, two tabs: the ranking, and every take with its audio.

    The ranking is what gets looked at first, so it is the default tab; the
    detail is where the ranking's reasons live, sorted worst first because that
    is where the interesting failures are.
    """
    by_model = collections.defaultdict(list)
    by_pair = collections.defaultdict(list)
    by_domain = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in takes:
        by_model[t["model"]].append(t)
        by_pair[f"{t['model']}  ·  seed {t['seed']}"].append(t)
        by_domain[t.get("domain", "?")][t["model"]].append(t)

    rows = []
    for t in sorted(takes, key=lambda x: -x["cer"]):
        flag = ' class="bad"' if t["cer"] >= 0.4 else (
            ' class="warn"' if t["cer"] >= 0.2 else "")
        rows.append(
            f'<tr{flag}><td>{t["model"]}</td><td>{t["seed"]}</td>'
            f'<td class="num cer">{t["cer"]:.3f}</td>'
            f'<td class="num">{t["length_ratio"]:.2f}</td>'
            f'<td class="txt"><div class="ref">{t["reference"]}</div>'
            f'<div class="hyp">{t["text"]}</div></td>'
            f'<td><audio controls preload="none" src="{t["file"]}"></audio></td></tr>')

    html = """<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASR 評分結果</title><style>
body{margin:0;background:#0f1115;color:#e6e9ef;font:13px/1.6 -apple-system,"Noto Sans TC",sans-serif}
h1{font-size:15px;padding:14px 20px;margin:0}
h2{font-size:13px;color:#8b93a7;margin:22px 20px 8px;font-weight:600}
.tabs{display:flex;gap:4px;padding:0 20px;border-bottom:1px solid #262b36}
.tabs button{background:transparent;border:0;border-bottom:2px solid transparent;
 color:#8b93a7;font:inherit;font-weight:600;padding:9px 14px;cursor:pointer}
.tabs button.on{color:#6ea8fe;border-bottom-color:#6ea8fe}
.note{padding:10px 20px;color:#8b93a7;font-size:12px;max-width:940px}
.note b{color:#e6e9ef}
table{border-collapse:collapse;width:100%}
th,td{padding:8px 10px;border-bottom:1px solid #1d2230;vertical-align:top;text-align:left}
th{color:#8b93a7;font-weight:600;font-size:11px;position:sticky;top:0;background:#171a21}
.num{font-variant-numeric:tabular-nums} .dim{color:#8b93a7}
.rk{width:34px;text-align:center}
.cer{font-weight:600}
tr.bad .cer{color:#e06c75} tr.warn .cer{color:#e0a458}
tr.tie td{opacity:.62}
.ref{color:#8b93a7} .hyp{color:#e6e9ef}
.txt{max-width:520px} audio{height:32px;width:230px}
section{display:none} section.on{display:block}
</style></head><body>
<h1>ASR 評分結果 &mdash; Breeze-ASR-26 聽寫 TTS 的輸出，與輸入文字比對</h1>
<div class="tabs">
  <button id="t-rank" class="on" onclick="show('rank')">排行</button>
  <button id="t-detail" onclick="show('detail')">明細</button>
</div>

<section id="rank" class="on">
<div class="note">綜合分數 = <b>平均 CER + |平均長度比 &minus; 1|</b>。兩者都以參考字數為分母，
可以直接相加：CER 抓「唸錯」，長度比抓「唸太少或多唸」，後者是 CER 在增刪互相抵消時看不到的。
<b>分數越低越好。</b><br>
淡化的列表示它與第一名的差距小於 CER 的標準誤，分不出高下，不要當成嚴格排序。<br>
<b>官方中文對照組（zh_official_15）會排在所有表格最前面，這是預期中的假象，不是它贏了：</b>
它沒有微調過，純講標準華語；Breeze-ASR-26 把台語和華語都寫成華語漢字，
所以一顆完全不講台語、只講華語的模型天生 CER 最低。它存在的作用是當地板，
不是候選人 &mdash; 排名時應該只在其餘台語模型之間比較。</div>
""" + _rank_rows(by_model, "模型排行") + _rank_rows(by_pair, "模型 &times; 種子排行") + "".join(_rank_rows(g, f"分領域：{d}") for d, g in sorted(by_domain.items())) + """
</section>

<section id="detail">
<div class="note">依 CER 由高到低排序，最糟的在最前面。紅色 CER &ge; 0.4、黃色 &ge; 0.2。
上排是輸入文字，下排是聽寫結果。</div>
<table><tr><th>模型</th><th>seed</th><th>CER</th><th>長度比</th>
<th>參考 / 辨識</th><th>音檔</th></tr>
""" + "\n".join(rows) + """</table>
</section>

<script>
function show(which) {
  for (const id of ["rank", "detail"]) {
    document.getElementById(id).classList.toggle("on", id === which);
    document.getElementById("t-" + id).classList.toggle("on", id === which);
  }
}
</script>
</body></html>"""

    path = os.path.join(directory, "index.html")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)
    print(f"\n  {len(takes)} 個音檔與 {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True,
                        help="comma separated registry names, in the order to test")
    parser.add_argument("--character", default="hayley")
    parser.add_argument("--endpoint", default="/tts")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--sentences", default=SENTENCES_FILE)
    parser.add_argument("--out", default=None, help="write per take rows as jsonl")
    parser.add_argument("--save-dir", default=None,
                        help="keep the generated audio here and write an index.html "
                             "that plays each take beside its transcript")
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild index.html from an existing --out jsonl, "
                             "without regenerating any audio")
    args = parser.parse_args()

    sentences = load_sentences(args.sentences)
    position = {text: i for i, (_, text) in enumerate(sentences)}

    if args.report_only:
        # The audio is already on disk and the filename is derivable, because
        # every reference is one of the benchmark sentences.
        takes = []
        for line in open(args.out, encoding="utf-8"):
            row = json.loads(line)
            index = position.get(row["reference"], 0)
            takes.append({**row,
                          "file": f"{row['model']}__seed{row['seed']}__{index:02d}.wav"})
        write_report(args.save_dir, takes)
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    handle = open(args.out, "w", encoding="utf-8") if args.out else None
    summary = {}
    # The aggregate CER turned out to be mostly noise at 12 takes per model —
    # the spread between models was 0.118 against a typical standard error of
    # 0.058. What the transcripts show individually is more use than the mean,
    # and that is only readable next to the audio it came from.
    takes = []
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for model in models:
        print(f"\n=== {model} ===", flush=True)
        print("  切換模型 ...", flush=True)
        started = time.perf_counter()
        try:
            result = post_json(f"{TTS}/model/switch", {"name": model}, timeout=900)
        except urllib.error.HTTPError as exc:
            print(f"  [FAIL] 切換失敗: {exc.read()[:200]}")
            continue
        if result.get("status") != "ok":
            print(f"  [FAIL] {result}")
            continue
        print(f"  已切換（{result['seconds']}s）", flush=True)

        cers, ratios = [], []
        for domain, text in sentences:
            for seed in seeds:
                try:
                    wav = synth(text, args.character, seed, args.endpoint)
                    scored = post_audio(f"{ASR}/transcribe", wav, text, timeout=600)
                except Exception as exc:
                    print(f"  [skip] seed {seed}: {type(exc).__name__}: {exc}")
                    continue
                if "cer" not in scored:
                    print(f"  [skip] seed {seed}: {scored}")
                    continue
                cers.append(scored["cer"])
                ratios.append(scored["length_ratio"])
                if args.save_dir:
                    name = f"{model}__seed{seed}__{position[text]:02d}.wav"
                    with open(os.path.join(args.save_dir, name), "wb") as audio_file:
                        audio_file.write(wav)
                    takes.append({"file": name, "model": model, "seed": seed,
                                  "domain": domain, **scored})
                if handle:
                    handle.write(json.dumps(
                        {"model": model, "seed": seed, "domain": domain, **scored},
                        ensure_ascii=False) + "\n")
                flag = "  <-- 疑似漏字" if scored["length_ratio"] < 0.75 else ""
                print(f"    seed {seed:>3}  cer {scored['cer']:.3f}  "
                      f"長度比 {scored['length_ratio']:.3f}  "
                      f"{scored['audio_seconds']:>5.1f}s  {text[:14]}…{flag}", flush=True)

        if cers:
            summary[model] = {
                "cer": statistics.mean(cers),
                "ratio": statistics.mean(ratios),
                "low_ratio": sum(1 for r in ratios if r < 0.75),
                "n": len(cers),
                "minutes": round((time.perf_counter() - started) / 60, 1),
            }

    if handle:
        handle.close()
    if args.save_dir:
        write_report(args.save_dir, takes)

    print("\n" + "=" * 74)
    print(f"{'模型':<26}{'平均 CER':>10}{'平均長度比':>12}{'長度比<0.75':>12}{'樣本':>7}")
    print("-" * 74)
    for model, s in summary.items():
        print(f"{model:<26}{s['cer']:>10.3f}{s['ratio']:>12.3f}"
              f"{s['low_ratio']:>10}/{s['n']:<4}{s['n']:>5}")
    print("=" * 74)
    print("CER 的絕對值沒有意義：Breeze-ASR-26 在 tai8 真人台語上自身平均 CER 就有 0.260。")
    print("要看的是同一批句子與 seed 下，模型之間的差異，以及長度比是否掉下來。")


if __name__ == "__main__":
    main()
