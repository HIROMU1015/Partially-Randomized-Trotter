# checker v2保存データ再監査：別実行承認への引継ぎ

技術修正と29人工検査は完了。実H4再監査は未実行、旧NUMERICAL_ALLOWANCE失敗とUNVALIDATED_PRODUCER_ONLYを維持。実H4原因の確定や性能の再判定を今回の人工検査に帰属させない。

[修正内容と限界](qc_onebody_repartition_h_boundary_checker_v2_repair_20261015.md)、[固定契約](../../artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15/contract.json)、[source/input SHA](../../artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15/source_manifest.json)、[入力identity](../../artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15/input_identity.json)、[未発行承認target](../../artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15/authorization_target.json)、[実行前検査](../../artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15/preflight.json)。原scope・入力・保存branch/eligibility/targets/pricesは不変。旧attempt/source/result/認可を変更しない。

次に必要なのはユーザーの別明示承認：

> 固定checker v2によるH4保存h境界データの独立数値再監査（4572 cells、独立PF再構成最大4572、expm最大571500、新PF収集・eigensolver・RDM・selector・compile0）を、GitHubの固定source・契約どおり一回だけ実行することを承認します。retry0、独立監査結果公開・再取得後Mandatory STOPしてください。

この文は承認対象の例であり、ユーザー承認の発行ではない。契約/source_manifest/input_identityのSHAとlimitsへ別認可を結び付ける。旧認可は消費済みで流用しない。未発行targetをrunnerに渡しても拒否される。

認可後だけ次を起動する：

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /tmp/prt-qc-feasibility-source-v1-venv-20261013/bin/python \
  scripts/verify_qc_onebody_repartition_h_boundary_v2.py --authorization /path/to/new-issued-approval.json
```

全保存cellを一回検査し、Padé一致・保存幅自己整合・branch・derived records・refinement・query/cost・10%gateまで検証できた場合だけ新audit PASSとして別scopeで記録する。失敗ならcell/双方metrics/partialcountsを残して再試行せず停止する。どちらでも旧記録は保持し、新たな重要科学解釈/継続・終了・第二候補移行はGPTの別開始承認へ戻す。同じ設計の科学再レビューや追加数学探索を自動で始めない。
