# B2探索round：公開入口・SOURCE・再開

- B2公開成果・HANDOFF：`a4b4d0e1f2110e63ae9a1dc6a89e1699f95dc0eb`、
  `docs/research/rpe_method_optimizer_h4_20261116/HANDOFF.md`。
- 実行SOURCE：`a2e2ef50a108d6f34089a99bd00f11a7085dd9c9`。
  公開成果との差分は文書/予算0テンプレートで、runnerとoptimizerのcodeは同じ。
- 旧Skill：`eb14ca70e4de61e0fe1a8556f75571ee5e56fa18`。08b75526b用の記述は旧版の検証履歴である。
- 現在の接続検証：[REPORT](../../../../docs/research/rpe_cost_compare_round_skill_20261116/REPORT.md)。

## 実物の起動

CLIは`scripts/run_rpe_method_optimizer.py`、共通実装は`src/trottertracks/rpe_method_optimizer.py`。
実行SOURCEをcheckoutした自分の独立worktreeを使い、他担当の作業worktreeを変更しない。
Skillのbranchとprogramのbranchが別でも、それぞれのSOURCEと実行rootを記録して使う。
Sparse checkoutでは同じSOURCEの`configs/rpe_method_optimizer`も展開する。

確認runtime：
`/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python`。

H4の問題入力は`configs/rpe_method_optimizer/h4_df10_problem.json`。
[予算0のround例](h4-round-budget0.json)はB2公開テンプレート由来で、全4方式・各予算0。
利用者が指定した条件はproblem、方式/範囲/有限予算はroundに渡す。新たなLDやshot列を利用者に選ばせない。

program rootでの実行例（project環境activate済み）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B scripts/run_rpe_method_optimizer.py \
  --problem configs/rpe_method_optimizer/h4_df10_problem.json \
  --round /absolute/path/to/round.json --run-dir /absolute/path/to/run
```

新しいroundでは未使用のrun directoryとround_idを使う。round fileはB2 schema
`RPE_OPTIMIZER_SEARCH_ROUND_V1`に従う。

| field | 渡すもの |
|---|---|
| `round_id` | 探索1回分の識別子。同じroundの再開では保持 |
| `methods` | FULL_PF / TRUNCATED_PF / PR_PAULI / PR_DFから採択済み対象 |
| `max_new_q3_per_method` | 方式ごとの新規Q3**総上限**。予算0なら新規Q3を起動しない |
| `initial` | 既定all、または利用者が指定した保存結果の開始集合 |
| `method_overrides` | 採択済みの公式・幅・LD等の探索範囲。未指定なら既存problem設定を使う |
| `resume_from` | 必要な場合のみ既存ROUND_RESULT/RESULTのresume_recordsを取り込む |

`approved_by`はB2テンプレートの任意文字列であり、承認の証明/必須入力ではない。
利用者の明示指示が許可であり、Skillで証明書・承認ID・追加grantを要求しない。

## 同じroundの再開・予算追加

元のproblem/SOURCE、round file、round_id、run directoryを保持し、上の**同一コマンド**を再実行する。
`--resume`は存在しない。同じrun directoryの完了済み結果/実行中job/起動済み件数はprogramが管理する。
完了jobの再利用、実行中jobの二重投入防止、死んだjobのcollect/報告をSkillで再実装しない。

明示許可された予算追加がある場合のみ、同じround fileの方式別総上限を変更して同じコマンドで再開する。
既存の起動済み件数を差し引いた別枠を新roundとして作ったり、round_idを変えて予算をリセットしたりしない。
保存`ROUND.json`は初回の宣言を残すため、現在の上限/消費は最新ROUND_RESULTとINVOCATIONSも読む。
対象方式・initial・範囲・routeを変えて同じrun directoryへ押し込まず、実行契約の変更を利用者へ明示する。

`--resume-from <previous ROUND_RESULT.json>`は別の保存結果入力を読み込む追加機能であり、
同じroundのjob・予算を再開するために毎回必要な引数ではない。

## 出力・停止理由

round modeは`ROUND_RESULT.json.methods`が主結果。`best`（構成ID、X/Y、MSE区間、exact費用）、
`pending_request`（次候補/配分/費用/経路）、`stop_reason`、`held`、`resume_records`、`budget`を読む。
`PER_METHOD.csv`、`NEEDS.json`、`INVOCATIONS.jsonl`、`jobs/<method>/<job_id>`も同じrun directoryにある。

| 停止理由 | Skillでの扱い |
|---|---|
| NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET | 次候補あり・予算0。既評価集合の確認済み最安と未実行要求を返す |
| Q3_BUDGET_EXHAUSTED | 消費済みを含め予算到達。無断で上限を広げない |
| NO_IMPROVING_PROPOSAL | 探索則内に改善案なし。大域最適とは呼ばない |
| JOB_IN_PROGRESS | 実行中jobあり。重複投入/勝手な停止をしない |
| STALE_JOB_NEEDS_REVIEW | 死んだjob。自動retryや新runでの再投入をしない |
| NO_ROUTE_ADAPTER | 次候補の経路なし。旧runnerへの手動迂回をSkillで実装しない |
| UNKNOWN_RESULT | gate/保存等の不成立。MSE下界によるモデル不適合と区別する |

`INVOCATIONS`のlaunched、新規call、jobsのLAUNCH/OUTCOMEを合わせて実績を確認する。
予算0でも既存jobのcollect/保存結果取り込みはあり得る。新規Q3と同一視しない。

## 確認した機能と境界

この公開版は、既存routeによるQ3実行→検証/取込→次候補選択を有限予算内で行い、
同じroundの再開に対応する。旧版の「executor未実装」「checkpoint再開未実装」は現在の説明には使わない。
今回のSkill側検証は保存H4・新規Q3予算0の起動/再開であり、正予算の実科学Q3を実行した証拠ではない。
B2の人工executor受入は公開HANDOFFに記録されている。

新しいconfiguration/signal/価格の取得や、route viewの作成、未対応configurationの実行は別の不足である。
この版の方式menuは4方式で、Full-H pure RTEの自動比較までは確認していない。
価格SOURCEのMODEL参照・直接価格・条件付き閉包等の資格、条件付きQ3と物理的RMSE/十分Nを区別する。

報告専用modeも残る：`--problem ... --output <new directory>`、任意の`--ingest`、
`--resume-from`、`--dry-check-routes`。こちらのRESULT.json/人工FIXTUREをround結果と混ぜない。
探索の起動/同じround再開には`--round`と`--run-dir`を使う。
