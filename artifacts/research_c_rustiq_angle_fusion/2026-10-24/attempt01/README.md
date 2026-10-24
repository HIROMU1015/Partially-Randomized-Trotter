# C4 B2限定科学実行packet（2026-10-24、attempt01）

状態：`C4_B2_CORRECTNESS_PASS_RESOURCES_RECORDED_MANDATORY_STOP`。SOURCE `0c9d0a480db95e4bbd57a1cda02f7fb24c2b2162`、契約SHA256 `f03839377158a44477a898fdc35af7d6a3b96bf953e86275915bf97324df035f`。1 attempt、retry/resume 0。

[結果・条件・解釈・限界](../../../../docs/research/research_c_angle_fusion_b2_execution_20261024.md)。

保存F1の4 mode・2粒子・occupied(0,1)、density-square合成H、L_D1・残差2fragment、T1/5 q2 δ1/10一点 r2 h1/20 K2。分子geometry/AO basis/DF rank非該当。Qiskit1.3.0 opt1 seed17 native rz,sx,x,cx、full measured X/Y。
全gateがPASSし、全4case・両axisの8 compiled指標でB2と保存R1が同値。単純融合の十分性を支持するが、Rustiq内部機構の一意同定、一般的優位、law期待値や認証shot・全round総資源ではない。C4の独立研究終了は正式採択していない。

- `approval.json`, `USER_AUTHORIZATION.txt`, `preflight.json`：今回の明示承認・実行前の照合。
- `preparation_contract.json`, `execution_binding.json`, `provenance.json`, `source_diff.patch`：SOURCE/input/比較条件。原本は公開準備commitに固定。
- `references/`, `circuits/`, `operators/`, `reports/`：再利用reference14 bundle、B2の回路・operator・絶対gate・元plan/fusion ledger。
- `paired_compiled_metrics.json`, `paired_metrics.csv`, `comparison_summary.json`：保存B0/B1/R1とB2のcase/axis対応。旧route再実行0。
- `raw_ir_diagnosis.json`：契約内の旧raw QPY8 decode、raw metric16。
- `journals/`, `execution.log`, `resources.json`, `classical_costs.json`：356 journal行、kernel/whole-build/cacheの時間、process RSS。matched R1時間は未取得。
- `summary.json`, `gate_summary.json`, `attempt_ledger.json`, `STOP.json`：実行状態・call・1 attempt。retry/resume禁止。
- `summarize_saved_results.py`：standard-libraryで保存JSON/reportを再照合するだけ。新scienceなし。
- `OUTPUT_HASH_MANIFEST.json`：全ファイルbytesのhash（自己除外）。

これはローカル実行証拠で、CIや外部独立再現ではない。旧packet・SOURCE・grant・STOPを上書きしない。公開照合後mandatory STOP、追加科学は別承認。
