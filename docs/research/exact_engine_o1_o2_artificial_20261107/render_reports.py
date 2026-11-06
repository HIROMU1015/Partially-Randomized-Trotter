"""Validate saved artificial raw records and render tables, without rerunning kernels."""
import hashlib
import json
from pathlib import Path
import statistics

from benchmark import summarize, METHODS, HERE


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def render():
    out = HERE / "raw_run01"
    summary = summarize(out)
    assert summary == json.loads((out / "SUMMARY.json").read_text())
    source = json.loads((out / "SOURCE.json").read_text())
    for path, sha in source["files_sha256"].items():
        assert hashlib.sha256((HERE / path).read_bytes()).hexdigest() == sha
    memory = lines(out / "memory.jsonl")
    assert len(memory) == 30
    for name in dict.fromkeys(r["fixture"] for r in memory):
        rows = [r for r in memory if r["fixture"] == name]
        assert {r["method"] for r in rows} == set(METHODS)
        assert len({r["threshold"] for r in rows}) == 1
    records = lines(out / "timings.jsonl")
    assert len(records) == 540
    assert all(r["iterations"] == 1 and r["wall_ns"] >= 0 and r["cpu_ns"] >= 0 for r in records)
    prep = lines(out / "preprocessing.jsonl")
    env = json.loads((out / "ENVIRONMENT.json").read_text())
    plan = json.loads((out / "PLAN.json").read_text())
    assert not env["scientific_modules_loaded"]
    text = ["# O1/O2：人工benchmark結果", "", "2026-11-07 JST、SERVER1。**人工入力限定。実H4・SCV/RPEの性能結果ではない。**", "",
            "## 測定条件", "",
            f"{env['host']}、{env['processor']}、Python {env['python'].split()[0]}、stdlibのみ。SOURCEとPython/fractions hash、affinity・負荷・memory pressureは [ENVIRONMENT](raw_run01/ENVIRONMENT.json) と [SOURCE](raw_run01/SOURCE.json) に保存。科学依存moduleのimportは0。共有環境へのinstall/変更なし。", "",
            "[PLAN](raw_run01/PLAN.json) と全振幅を含む [FIXTURES](raw_run01/FIXTURES.json) を測定前に保存した。手書き有理数10 fixtureを固定し、結果後の除外・置換なし。各列は同じ初期stateから12 gateを作用する。qubit数/行列構造/初期bit長はPLANを正本とする。人工Hadamard型は有理数係数で、実gate行列を再生成したものではない。", "",
            "3方式の全6順列を各fixture/categoryで一回ずつ測り、6標本を保存。unmeasured warmup各1回。入力生成、correctness、gc.collect、記録処理はtimer外。GCはkernel中も通常通り。単独processは比較条件の統制であり科学worker数の上限ではない。wall=perf_counter_ns、CPU=process_time_ns。", "",
            "denseは固定旧関数そのもの。O1は準備済み疎row＋毎gate index生成、O1+O2は同じ演算＋private index LRU。候補はFraction pair型の検査も毎回行い、旧denseよりvalidationが多い。dense fallbackでもこの追加検査は残る。", "",
            "coldは各人工gate列開始時に疎rowと空index cacheを新規に作る条件。OS/page/allocator/CPU cacheのcoldではない。warmは疎rowと利用supportのindex planをtimer外で作成する。全方式でstate evolution結果は毎回計算し直し、state/p/bit cacheはない。O3以降の処理は導入していない。", ""]
    headings = (("warm_contraction", "前処理なし・warm contraction"),
                ("cold_end_to_end", "前処理込み・cold gate列＋旧Born＋旧threshold"),
                ("warm_end_to_end", "warm gate列＋旧Born＋旧threshold"))
    for category, title in headings:
        text += ["## " + title, "", "時間は12 gate列1回のwall中央値 [最小, 最大]、単位ms。倍率は各方式の中央値の比。1未満は候補が遅い。", "",
                 "| Fixture | Dense ms | O1 ms | O1+O2 ms | Dense/O1 | Dense/(O1+O2) | O1/(O1+O2) |", "|---|---:|---:|---:|---:|---:|---:|"]
        for r in summary:
            if r["category"] != category:
                continue
            cells = [f"{r['methods'][m]['wall_median_ns']/1e6:.3f} [{r['methods'][m]['wall_min_ns']/1e6:.3f}, {r['methods'][m]['wall_max_ns']/1e6:.3f}]" for m in METHODS]
            text.append("| " + r["fixture"] + " | " + " | ".join(cells) + f" | {r['dense_over_O1']:.3f} | {r['dense_over_O1O2']:.3f} | {r['O1_over_O1O2']:.3f} |")
        text.append("")
    text += ["## CPU time", "", "cold end-to-endのprocess CPU中央値、ms。並列worker時間和ではない。他categoryのCPU rawと中央値も保存している。", "",
             "| Fixture | Dense CPU ms | O1 CPU ms | O1+O2 CPU ms |", "|---|---:|---:|---:|"]
    for r in summary:
        if r["category"] == "cold_end_to_end":
            text.append("| " + r["fixture"] + " | " + " | ".join(f"{r['methods'][m]['cpu_median_ns']/1e6:.3f}" for m in METHODS) + " |")
    text += ["", "## 前処理とindex費用", "", "50反復batchで測った1回当たり時間の6標本中央値、µs。sparse rowはfixtureの全unique matrix、indexは疎経路の全unique supportの合計。mixedはmatrix4種・index3種。dense fallbackではindexを作らない。", "",
             "| Fixture | sparse row準備 µs | index新規作成 µs | warm lookup µs |", "|---|---:|---:|---:|"]
    for name in dict.fromkeys(r["fixture"] for r in summary):
        cells = []
        for category in ("sparse_rows", "index_build", "index_warm_lookup"):
            rows = [r for r in prep if r["fixture"] == name and r["category"] == category]
            cells.append(f"{statistics.median(r['wall_ns']/r['iterations']/1000 for r in rows):.3f}" if rows else "—")
        text.append("| " + name + " | " + " | ".join(cells) + " |")
    text += ["", "## Memory・cache", "", "各fixture/方式をfresh subprocessで一度実行。cold end-to-endのtracemalloc peak bytesで、Python object allocationだけを追跡し、fixture生成/importは除外する。前処理・cache・中間state・Bornは含む。native allocationや全worker物理memoryの測定ではない。", "",
             "| Fixture | Dense peak B | O1 peak B | O1+O2 peak B | O2 entries / indices / miss / hit |", "|---|---:|---:|---:|---|"]
    for name in dict.fromkeys(r["fixture"] for r in memory):
        rows = {r["method"]: r for r in memory if r["fixture"] == name}
        info = rows["O1+O2"]["cache"]
        cells = [str(rows[m]["python_peak_bytes"]) for m in METHODS]
        text.append("| " + name + " | " + " | ".join(cells) + f" | {info['entries']} / {info['stored_indices']} / {info['misses']} / {info['hits']} |")
    text += ["", "RSSはfresh子processの前後high-water（bytes）を [memory raw](raw_run01/memory.jsonl) に保存。前値はfixture/import終了後のhigh-waterで、差をkernelの物理memory増と扱わない。複数processのRSS和も求めていない。retained Python allocationとPIDも同rawにある。",
             f"今回の全30sampleの実行後RSS high-water範囲は {min(r['RSS_high_water_after_bytes'] for r in memory):,}–{max(r['RSS_high_water_after_bytes'] for r in memory):,} bytesで、方式間の物理memory差をこの指標では識別できなかった。", "",
             "O2 cacheは利用supportだけ遅延生成するLRU（default32 entry）。32はCPU/RAMの実行上限ではなくprototypeのcache容量。今回実使用は0–3 entryで、正しさテストでは容量1/2でeviction・再生成も検査した。index保存量は概ねentryごとO(state dimension)で、workerごとに複製すればその分増える。", "",
             "## 判断と限界", "",
             "疎/対角等の8 fixtureでは前処理込みでもO1がdenseより速く、O2の追加後にも中央値の低下を観測した。denseの2 fixtureは元kernelへfallbackし、検査・準備分の小さい悪化が残った。全fixtureを残しており、改善しない条件を除外していない。O2の小さい差は標本範囲と併せて読む。正式な統計的有意差、異なるhost/loadでの安定性は検証していない。", "",
             "memoryは全fixtureで候補のPython peakが増えた。cold end-to-endでの時間短縮と小さい追加allocationは同時に観測できたが、時間とbytesを単一尺度へ合算していない。多worker、より大きいstate、低再利用率、1 gateだけの用途、実科学のbit長/構造分布での有利性は未測定。", "",
             "実H4 gate/state/LAW/catalogは一切投入していない。実build・matrix hash・価格・RNG・manager・実worker schedulingは測っていない。従ってここでの倍率をH4全engineやPR/RPE総コストへ外挿できない。旧certificateの認証移転や科学的GOでもない。", "",
             "## 再現と保存値整合", "", "```bash", "python3 -B test_o1_o2.py", "python3 -B benchmark.py --run raw_reproduction", "python3 -B render_reports.py", "```", "",
             "このdirectoryをcwdにする。benchmarkはfresh出力を要求し、既存raw_run01を上書きしない。render_reportsは保存raw_run01から表を再生成するだけでkernelを再実行しない。SOURCE hash・540 timing・30 memory・方式間threshold・SUMMARY再導出を検査してから出力する。", "",
             "[timings raw](raw_run01/timings.jsonl)、[preprocessing raw](raw_run01/preprocessing.jsonl)、[SUMMARY](raw_run01/SUMMARY.json)、[correctness](EXACT_ENGINE_O1_O2_CORRECTNESS.md)、[REPORT](REPORT.md)。公開照合後mandatory STOP。", ""]
    (HERE / "EXACT_ENGINE_O1_O2_BENCHMARK.md").write_text("\n".join(text))
    write_check = dict(status="PASS", summary_matches_raw=True, source_hashes_match=True,
                       timing_records=len(records), memory_records=len(memory),
                       all_memory_thresholds_equal=True, scientific_modules_loaded=env["scientific_modules_loaded"])
    (HERE / "RAW_REPORT_CHECK.json").write_text(json.dumps(write_check, indent=2) + "\n")


if __name__ == "__main__":
    render()
