# 固定H4一体再配分PF：h境界one-shot実行承認への引継ぎ

準備は完了。新しい分子性能結果はまだない。受領GPT科学レビューは限定継続・h境界準備を支持し、完了済み。詳細設計レビューの再実施は原則不要。今回の作業は新しい科学計算の承認ではなく、以下の固定scopeについてユーザーの別明示実行承認を待つ。

[設計・不変条件・未判定・限界](qc_onebody_repartition_h_boundary_preparation_v1_20261015.md)、[契約](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/contract.json)、[source manifest](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/source_manifest.json)、[未発行target](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/authorization_target.json)、[execution plan](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/execution_plan.json)。

- source `5e3341c12bd84f30f19072cefa590f4499c36c38` / scope `QC_ONEBODY_REPARTITION_H_BOUNDARY_V1_20261015`。固定H4linear1.00Å/STO-3G/保存DF rank12・全12generation順density、Nα=Nβ=2/36dim、v1 seal・X・state・frame・leading不変。PR L_D/delta変更なし。
- 比較h旧h0から1Ha^-1、全12methodに同じ368初期比較点。h/2等を含む736初期追跡点。最大2round×32区間の共通midpoint追加で432比較 / 864追跡点。
- 主比較・両target・両価格・予算分配・uniform10%gate維持。細分化・branch・数値・上端が不十分なら理由を保存しSTOP。新h・precision・係数・分子の救済なし。
- 新PF/Schur各5172、独立PF再構成5184/expm648000、method records5184/cost rows20736上限。新eig/RDM/leading/selector/optimizer/compile/T/ground準備/分子生成0、1worker1thread、RSS/output512MiB、log2MiB、wall/CPU capなし、producer/audit各attempt1、retry0。
- 新しい認可recordを作り、既存の未発行targetと旧認可は変更しない。source/入力/SHA/capsと別明示ユーザー指示を記録・commitしてclean worktreeから一回だけ実行する。承認前は--preflightのみ。
- 実行・失敗のどちらも固定sourceを修正・再試行せず保存する。収集完了なら独立checkerを一回、GitHub commit/push/re-fetch照合後Mandatory STOP。

将来の実行例（別認可が発行されるまで実行しない）：

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /tmp/prt-qc-feasibility-source-v1-venv-20261013/bin/python \
  scripts/run_qc_onebody_repartition_h_boundary_v1.py --execute \
  --authorization <new-issued-authorization.json>
```

caller logはauthorize前にworktreeをdirtyにしないよう/tmpへ保存し、終了後に結果資料へ移す。数値checkerは固定directory `artifacts/qc_onebody_repartition_h_boundary_v1/2026-10-15/attempt_1`にのみ適用し、再取得照合の際は数値checkerを再実行せずbyte/SHAを見る。

source/input/guardのpreflightと35人工テストを科学成功と呼ばない。新評価は同じdevelopment H4への追加でheld-outではない。h性能oracle・sampled continuation・静的価格proxyで、全QPE/RPE費用、情報測定、T/hardware利益、新規性は未確定。結果後の継続/終了/第二候補判断は重要GPTレビューの別開始承認事項。N1STOP/N2HOLD/N3CLOSED/THRIFT現template CLOSED、TrackA/B・C1–C4・global manifest既知FAIL・旧結果/認可/STOPを維持。
