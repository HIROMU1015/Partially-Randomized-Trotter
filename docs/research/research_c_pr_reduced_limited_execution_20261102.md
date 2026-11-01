# 研究C：PR-only reduced限定実行・最初のtrialでstability未資格STOP

2026-11-02 JST。Repository: HIROMU1015/Partially-Randomized-Trotter。

## 1. 結果

ユーザーの実行許可後、科学SOURCE `b2e5ab4d5ff6986e0e991b7aa0d373ecdbf755c5` と[54-call固定契約](research_c_pr_reduced_limited_contract_unapproved_20261102.json)を変更せず、fresh run01へ一度実行した。**18/54 logical units COMPLETE、correctness FAIL0、retry/resume0**。結果は`REDUCED_GRADIENT_UNQUALIFIED` / `NO_ACCEPTED_U_UPDATE`。

初期U点は最新refinement pairの安定性・J一次条件・rank/nullspaceの数値資格を通った。BFGSが問い合わせた最初の未受理trialでは、J一次条件と構造guardは通ったが、**refinement間U勾配差7.3108942712e-6が固定基準1e-6を超えた**。その点で契約通り停止し、U更新受理は0。追加refinement、line search継続、別solver、旧初期値へのfallback、残り36 unitsの転用は行わない。

これはcorrectnessの例外FAILではなく、登録された数値admissionの未資格STOPである。`STOP.failed=false`は正常な証拠保存を表し、reduced資格PASSを意味しない。旧chart/physical整合FAILは今回評価した2点では発生しなかったが、**以前のtrial2には到達していない**ため、旧trajectory全体の問題解消とは主張しない。

[result原本](../../artifacts/research_c_pr_reduced_limited/2026-11-02/run01/result.json)／[optimization原本](../../artifacts/research_c_pr_reduced_limited/2026-11-02/run01/optimization.json)／[全trace](../../artifacts/research_c_pr_reduced_limited/2026-11-02/run01/optimization_trace.jsonl)／[summary](../../artifacts/research_c_pr_reduced_limited/2026-11-02/execution_summary.json)／[STOP](../../artifacts/research_c_pr_reduced_limited/2026-11-02/run01/STOP.json)。

## 2. 固定条件とpreflight

| 項目 | 実行条件 |
| --- | --- |
| 科学SOURCE | `b2e5ab4d5ff6986e0e991b7aa0d373ecdbf755c5`、SOURCE154無変更 |
| 契約SHA256 | `f0b8eba5dba70a687156d9cf3f8f3cb55ffd34eaf4f599ad72757e66c4f54c61` |
| SOURCE manifest SHA256 | `e261fe6a816cf8c4368a08d85598d6ea839ca7bbcf6b67265b44e6c255d92026` |
| 分子／表現 | 保存直鎖H4・隣接1.00Å・STO-3G、number CDF8、signed実対称J・実直交U |
| Split／物理規約 | 元frame順LD2、Nα=Nβ=2/sector36。今回はsector演算をしない |
| κ／scalar | 元g補正κ・scalarを保持、数値再評価なし |
| 初期化 | 元qualified J/U、eta=0。各distinct Uで元Jから開始し、同一点評価器で2 refinement |
| PR目的 | γ=.1、ε1e-8Ha、f*=2.1839632053132894e-9、M*=2.2093719370722074 |
| Inner | SciPy1.14.1 Newton-CG、maxiter1000、xtol1e-12、既存解析Hessian |
| Outer | 既存BFGS、maxiter2/gtol1e-7/norm2/xrtol0/c1=1e-4/c2=.9、最大8 distinct U点 |
| 数値条件 | J/U一次1e-5＋元margin、gradient atol1e-8/rtol1e-7、value atol1e-10/rtol1e-8、latest-pair U差1e-6、元rank/nullspace guard |
| Fresh output | `artifacts/research_c_pr_reduced_limited/2026-11-02/run01`、今回消費済み |

[preflight](../../artifacts/research_c_pr_reduced_limited/2026-11-02/execution_preflight.json)でSOURCE154/input4・既存fit依存lock/ffsim SOURCE・package versions、旧raw2401/旧SOURCE15組と未使用run01を確認した。元input SHA1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db、元initial SHA78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856は変更していない。既存226人工・回帰PASSの同一SOURCEを再利用し、共有環境の変更・追加installはしない。

承認はチャット許可に基づく。歴史的`unapproved` metadataは変更せず、承認証明書・承認原文保存・新grantは作成しない。

```bash
# 実行履歴。run01は消費済み、再起動しない。
PYTHONDONTWRITEBYTECODE=1 \
/tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python -B \
 scripts/run_research_c_pr_reduced_limited.py \
 --contract docs/research/research_c_pr_reduced_limited_contract_unapproved_20261102.json \
 --execute
```

## 3. 2点の数値資格と未受理trial

| 測定／判定 | 初期eta=0：point0 | 最初の未受理trial：point1 |
| --- | ---: | ---: |
| 最終refined objective | .012267644429815034 | 1.6466546474460215 |
| 独立最終J∞ | 7.6003919128e-8 | 1.1879262806e-7 |
| 最終inner／独立J gate | MET／MET | MET／MET |
| Partial physical U₂（両精度の大きい方） | 2.9549751184 | 22.4070321809 |
| U一次条件 | NOT_MET | NOT_MET |
| Latest-pair U勾配差 | 4.2369251510e-10 | **7.3108942712e-6** |
| 固定stability基準 | 1e-6：PASS | 1e-6：未資格 |
| Head rank/nullspace・tail injectivity | 数値guard PASS | 数値guard PASS |
| Block/refinement nonincrease | PASS | PASS |
| Admission | NUMERICAL_PILOT_ADMITTED | REDUCED_GRADIENT_UNQUALIFIED |

両点で初回・refine1・refine2のJ gateはすべてMET。構造は元規則のhead quotient rank19、universal nullity1を認めた数値判定であり、厳密rank enclosureではない。最終refinementのobjectiveは各点内で表示精度上同値だが、U勾配差が十分小さいとは限らなかった。

Point1のobjectiveは初期点より大きいが、これは**未受理のline-search問い合わせ点**である。受理更新による悪化や、BFGSがその点を採用したとは解釈しない。資格STOPにより後続の問い合わせはしておらず、その先の受理可否は不明である。

`accepted_labels=['0']`、`terminal_label='0'`、`stop_trial_label='1'`を分けて保存した。`terminal_kind='accepted'`は初期点が資格を通った意味で、U更新を1回した意味ではない。`diagnostic_terminal.npz`は初期Uにおける最終refined Jを保存し、point1のJ/Uと勾配は別のprecision-state/inner-raw/envelope NPZに保持する。

## 4. Inner raw状態とcorrectness

| U点／solve | Raw status / success | nit | nfev/njev/nhev |
| --- | --- | ---: | --- |
| 0 initial | 0 /true | 591 | 1065/1065/591 |
| 0 refine1 | 2 /false、precision loss | 1 | 2/2/2 |
| 0 refine2 | 0 /true | 1 | 2/2/1 |
| 1 initial | 0 /true | 297 | 494/494/297 |
| 1 refine1 | 0 /true | 1 | 2/2/1 |
| 1 refine2 | 2 /false、precision loss | 1 | 2/2/2 |

6 inner合計nit892、nfev/njev1567、nhev894。raw外の終端・点照合を含む実VG1585、Hessian894、独立adjoint12、same-array binary64診断6も保存した。Logical unitを1回のVG評価とは扱わない。

6 solveすべてでchart/physical-chain gradientは元allowance内、差max4.6475763102e-19。Value/独立stateの同一点微分も登録検査を通過した。Raw jacは元のpre-step/accepted/endpointいずれか一致のOR規約を満たす。初回solve0・初回solve1・refine1(point1)にはraw endpoint jacのFAIL観測が残るが、pre-stepで一致しており、合成correctness FAILではない。False観測やraw jacは上書きしていない。

Raw precision lossは2件残る。J一次METとprecision lossを独立に報告し、正常収束や真のgradient精度保証へ昇格しない。Latest-pairの未資格がraw precision lossだけで説明できるか、残るinner解精度・conditioning等のどれによるかは、この限定実行では原因確定していない。追加数値診断・閾値変更はしない。

Outer BFGSは1回起動したが、資格停止の例外によりSciPyが返る前に終了したため、**outer raw結果はnull**。架空のstatus/nitを付けず、受理更新0・distinct U点2・保存問い合わせ点を報告する。

## 5. 科学call・資源・未実施範囲

| Logical units | Attempts / COMPLETE |
| --- | ---: |
| saved_load / initial_identity / objective_setup | 各1 /1 |
| outer_driver | 1 /1 |
| inner_J_solve | 6 /6 |
| precision_state | 4 /4 |
| reduced_admission | 2 /2 |
| classification / export | 各1 /1 |
| **合計** | **18 /18、FAIL0** |

初回inner2回＋refinement4回、outer起動1回、受理U更新0、ffsim joint fitter/RC2継続/retry/resume0。上限54の残り36を次の実験へ流用しない。

実行前CPU affinity32、MemAvailable54864132KiB、CPU/memory pressure avg10=0を確認した。1本の依存付きtrajectoryに既存vectorized NumPy/SciPy/JAX/longdouble経路を使用し、小行列BLAS thread調整のみを適用。CPU/RAM/worker/wallの恣意的上限や共有venv/cache/OS変更を追加しない。独立armはないため外側を並列化していない。

保存resourceはwall5.2386763263秒、process CPU33.359541078秒、peak RSS449008KiB。Optimizer blockのwall5.1557018040秒/CPU28.953729857秒も保存。Process CPUはimport/preflight等を含み、wall比をspeedupとは解釈しない。

新しいsector operator DF norm・CAR・PF/RTE・ground・native build/compile・shots・総費用・最終CAは未実施。DF予算CA/8=.00019875Ha、PR性能、qualified tail比較はnullのまま。保存point1のpartial U derivativeをqualified envelope gradientと呼ばない。

## 6. 保全・公開・次の判断

新raw22ファイルにSOURCE/契約、全solveのraw/iteration trace、2点のprecision state/envelope、受理終点、結果・call・resources・STOPを保存。[postflight](../../artifacts/research_c_pr_reduced_limited/2026-11-02/execution_postflight.json)でSOURCE154/input4・旧raw2401/旧SOURCE・技術準備原本を保全確認し、研究概要・validation status・manifestへ今回の限定結果を追加、dated noteは追記のみとする。

次候補は**ユーザー／研究C**の判断後、**Codex**による保存ログ・人工問題でのrefinement stability未達の切り分け。新しい実H4再解析・追加refinement・outer継続・solver/数値条件変更には別の固定範囲と明示許可が必要である。RC2継続、Joint移行、paired実行、重要科学レビューへ自動進行しない。

実行branch `codex/research-c-pr-reduced-limited-execution-20261102`、独立worktree `/tmp/prt-c-pr-reduced-limited-execution-20261102`。Commit/push後、remote SHA/tree・公開報告/結果/SOURCE/契約/STOPと成果物の一致を照合し、**mandatory STOP**する。
