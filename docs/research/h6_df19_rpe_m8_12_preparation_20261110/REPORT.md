# H6 DF19 RPE：残5ラウンドの一括実行準備

2026-11-10 JST。**m8..12の固定10候補を、既存統合pipelineで一度に評価するN専用runnerを準備した。追加人工・回帰7件PASS。実H6の新reference/signal/phase/radius/Nは未計算。** 利用者指示「科学計算はユーザーの明示的な実行許可後に開始する」に従い、科学実行は別途明示許可を待つ。

[PLAN.json](PLAN.json)が固定menu・科学条件・call範囲、[SOURCE_INDEX.json](SOURCE_INDEX.json)が20科学fileと保存参照、[TESTS.txt](TESTS.txt)が人工結果、[CHECKS.txt](CHECKS.txt)が静的保全確認。SOURCEをcommitで固定・公開照合した後、mandatory STOPする。承認証明書・grant等は追加しない。

## 固定対象・条件

| m | T | q_RPE | q_PF | PF candidate ID | PR candidate ID |
|---|---:|---:|---:|---|---|
| 8 | 25.6 | 256 | 16 | `26608f8bf53993c7095d` | `53797d9e5a2eeec31a13` |
| 9 | 51.2 | 512 | 32 | `408142c6d1cd1b104b72` | `4d34a6f3f136df9ad512` |
| 10 | 102.4 | 1024 | 64 | `20a160dafc0fa7d6bce5` | `e25c35e31bb6c840e252` |
| 11 | 204.8 | 2048 | 128 | `2d56c445820c4accd676` | `86665f416fa64ed7a1a1` |
| 12 | 409.6 | 4096 | 256 | `fb0cb7b53bbfc8080224` | `c9309f4a8219815577f5` |

各mでPF Full19対PR prefix7/R_outer16/K2、両方式4th m5_best・δ1.6。既存11 signed pieceの値・順序・負時間を保持する。PRのABS_TIME_PER_OUTER配分は`[1,2,1,2,1,2,1,2,1,2,1]`、**1 outer全体で16**。全event数はq_PF×16であり、q_RPEと同じ数になっても意味は異なる。候補の科学的成立性やPASSは未確認で、候補追加・差替えをしない。

Linear H6、隣接1.00Å、STO-3G、保存tol-only1e−8 DF19、weighted-Hermitian completion、12 system mode・Nα=Nβ=3・sector400、保存数値state、P=0。targetは保存DF-H同sector ground energy。保存H/state/snapshot、fragment順、scalar/tail identity、normalization、PR sampling lawを不変とする。元分子Hとの誤差や厳密ground同定を新たに保証しない。

εE=.00159360010199040 Ha、h=.1、m0..12の13round、PF/PR各αtotal=.05、αaxis=.05/26、βtotal/stat=.4/.2 rad、prior [−9.695399206821989,−3.1355322139814668] Ha、energy shiftなし、保存consecutive-dyadic decoderを転記した。prior/horizon/α/βは再計算・最適化しない。E_stateはreference診断だけに使い、prior/branch決定には使わない。有限時間のαaxis=.025を全roundへ流用しない。

## 再利用と追加範囲

基準は[統合pipeline][INTEGRATION] **`3b742fff26381514e12c07b51701276b7d1b26ec`**。`pf_pr_rpe_pipeline.py`と`h6_rpe_batch_provider.py`、旧D2 action/geometry/Hoeffding、m5 schedule、既存atomic save・resource選択をそのまま再利用し、これらのfileは変更していない。[m7完了結果][M7]のm0..7保存coverageと原resultを参照する。

追加は[`run_h6_df19_rpe_m8_12.py`](../../../scripts/run_h6_df19_rpe_m8_12.py)、[`check_h6_df19_rpe_m8_12.py`](../../../tests/check_h6_df19_rpe_m8_12.py)、固定PLANと本報告。専用runnerは既存`run_missing`へ保存N16行を渡し、`H6ActionBatch`を**1回だけ**呼ぶ。C providerは渡さず、価格・G計算器も呼ばない。旧D1/D2/m6/m7の固定guardを広げない。

20科学fileは、m7の保存16 file＋統合module2 file＋今回runner/PLANの2 file。実行時は使用fileとcommit blob、過去の科学hash、保存H/state/input3 file・tail contextを照合する。無関係なworktreeのHEAD/statusや他担当の進行は停止条件にしない。

## 承認後に行う一括call

| 対象 | 固定上限・処理 |
|---|---|
| 新reference | 上記5時点、expm_multiplyとexpmの2経路、計10 state action |
| finite mean | PF5＋PR5、計10候補 |
| PR内部対照 | exact-tail5候補、corrected/rawの独立列伝播 |
| phase/radius/normalization/十分N | 10候補、X/Y計20軸 |
| 保存m0..7 | 16候補Nをcopy/joinするだけ。再計算0 |
| 新SCF/DF/H/state/ground/eigensolve | 0 |
| sampling・実build/compile・C取得/予測/fit・N×C/全G・RPE成功率simulation・最適化 | 全て0 |

保存contextを一度だけ復元し、5時点のreferenceと共通半step cacheを共有する。候補jobは観測CPU/RAMに基づく既存並列度で処理する。科学menu10候補という数はCPU/worker上限ではない。既存のprocess内BLAS設定以外に共有設定を変更せず、他process/worktreeには干渉しない。

raw=corrected/Bの事後代入はしない。scalar/identity一回、finite K2、log B・attenuation・normalization margin、既存のstage依存roundoff/u・reference差・T×保存residualを保持する。数値allowanceはempiricalでありrigorous boundではない。未知actual implementation/law誤差やground同定誤差を0へ補完しない。

reference別経路差1e−10、norm/closure1e−9、正raw半径、βdet+βstat≤.4等の既存gateは不変。数値gate/phase/radius不成立はN=null・未資格として保存し、閾値変更・科学的retry・候補探索をしない。固定menu内の結果を記録してSTOPする。referenceやSOURCE/inputの不一致で停止した場合、未評価候補をPASSや0shot扱いにしない。

## 保存結果とA/Cへの接続

実行入口は、worktree rootで**明示実行許可後だけ**使用する。

```bash
../../h6-environments/20261011/venv/bin/python -B scripts/run_h6_df19_rpe_m8_12.py --output-dir docs/research/h6_df19_rpe_m8_12_results_20261110/run01
```

今回このコマンドは実行していない。fresh outputを要求し、旧runの上書き・resume・自動retryをしない。

出力は`result.json`、新10候補の個別JSON、`COVERAGE.json`、初期`progress.json`。provider単位のbatchが終わると新候補を保存する。`progress.json`は途中状態であり、最終判定はresultを参照する。計画call数と観測完了状態を分離し、reference途中STOP時に計画10 actionを完了数として表示しない。

`COVERAGE.json`は保存16行を不変で転記し、新10候補を加えた26行・13roundを表現する。各方式のqualified/unqualified roundを分け、科学的FAIL・未評価・SOURCE無効化を区別する。全13roundの条件付きNが揃った場合のflagと、**full_RPE_qualified=false／actual N未資格／G未取得**を別に保持する。

価格担当向けhandoffにはcandidate/T/q_RPE/q_PF/δ/formula/signed recipe/prefix/R/K/law、H/state/snapshot/preparation/contract、compiler/control/full-wrapper/P0、X/Y別N/α、normalization、科学SOURCE・資格状態を渡す。C/Gはnullで、Bや古典価格標本数を費用へ再乗算しない。歴史的binding.T=.8は保存referenceの時刻で、実発展時間はcandidate.T。A/Cの成果・SOURCE・worktreeを変更しない。

## 人工確認・現時点の資格・STOP

追加7件、全PASS（11.633秒）。固定10 menu/政策/配分の拒否、N専用のmetadata読取、13round coverageのPASS/FAIL/未評価/無効SOURCE区別、一括1呼出しと保存/fresh output、数値FAIL・reference例外保存とretry禁止、q16/q256の非可換2-level finite meanの既存spectral経路との一致、旧m7 guard保持を確認した。q256でも2-level人工系だけで、実H6のT409.6を計算した証拠ではない。旧moduleのDeprecationWarningは記録に残した。

統合時の人工/mock12件は既存記録として再利用し、数値kernelが不変のため同じ全面試験を反復していない。fresh processの`prepare()`はNumPy/SciPy/Qiskitもloadせず、CLIは`--help`だけ確認した。実m8..12のsignal/phase/Nと科学的適格性は未取得、正式結果directoryも作成していない。本担当自身の技術検査であり、独立科学レビューではない。

同じ統合branchを継続し、旧H6 N branch・固定SOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOPを保持した。公開remote SHA・主要成果物の全文一致を確認後**mandatory STOP**。

次は**ユーザー**：この固定5時点・10候補の一括科学実行を明示承認するか判断する。**Codex B**：承認後に同じ範囲を一括実行し、結果・13round coverage・A/C向けmetadataを公開照合後STOPする。価格取得、全G、追加候補、詳細科学レビューへ自動進行しない。

[INTEGRATION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b742fff26381514e12c07b51701276b7d1b26ec/docs/research/pf_pr_rpe_cost_integration_20261110/REPORT.md
[M7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/14c4f871fb9e8d4a23a6cc3244f5435ebc65a7ce/docs/research/h6_df19_rpe_m7_results_20261110/REPORT.md
