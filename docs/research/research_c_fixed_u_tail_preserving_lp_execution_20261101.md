# 研究C：固定U・tail保存型係数LPの限定実行結果

2026-11-01 JST。**LP1回実行、9/最大50 logical unitsで登録未解決分岐へ停止。結果UNRESOLVED_LP、retry/resume 0。**

## 1. 結論

保存直鎖H4・1.00 Å・STO-3G、通常number CDF rank8、Nα=Nβ=2／sector36、元frame順L_D=2、head0/1・tail2..7、partial ordered symmetric S2 h=0.1／s=1／r=128／K=0のPR-tail候補に、固定U・固定head・固定tail tensorのexact Pauli L1 LPを一度実行した。

HiGHSはsuccess=true／Optimalを返したが、独立診断でdual equality residual、primal-dual gap、objective arithmetic guardが登録基準を満たさなかった。**数値最適性は未資格**であり、新候補はexport・採用せず、tensor／CAR／PF-RTE／DF保持／attenuationの依存後続評価は行わなかった。

これは「固定Uの限定問題に改善余地がない」ことも「候補Aが科学的に失敗した」ことも示さない。SolverのOptimal表示を採用判断にしなかった結果で、数値問題の原因は未確定。候補B・追加LP・条件変更へ自動進行しない。

## 2. SOURCE・入力・実行

| 項目 | 固定値 |
| --- | --- |
| 科学SOURCE | 56a6acb8f7f48cd4792c560b64bc4f4aa270e0ad（無変更） |
| 実装commit | 22aedb27d70f93ca43687bb9120f7598dc0d4e9f |
| 契約SHA256 | bdee0d3c656e580fadbd36e4d54ae83c0c7671666ffbba21d3a9987aef8bdcff |
| SOURCE105 SHA256 | 8a9dbedd1a97702ff3a76a3939d98ee3d4447d8b0f7bf548e7ca752e692fdcfd |
| 元PR export SHA256 | 3fdb0cb77afc770294cbe7aebf47d8a4529e3567ef9e886895495f4490b7ee21 |
| 元input snapshot SHA256 | 1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db |
| Solver | SciPy1.14.1同梱HiGHS、highs-ds、presolve=True |
| Solver feasibility tolerance | primal／dual各1e-10、変更0 |
| LP | j60＋u216、全tensor等式256、不等式432、signed j |
| Fresh output | artifacts/research_c_fixed_u_tail_lp/2026-11-01/run01 |

[固定契約](research_c_fixed_u_tail_preserving_lp_contract_unapproved_20261101.json)と[準備報告](research_c_fixed_u_tail_preserving_lp_preparation_20261101.md)を維持し、チャットの明示科学許可後に独立branch `codex/research-c-fixed-u-tail-lp-execution-20261101`で既存runnerを一度実行した。歴史的unapproved metadata、SOURCE、allowance、原κ／scalar／frame順を変更していない。承認証明・承認原文保存・新grantは作成していない。

[実行前照合](../../artifacts/research_c_fixed_u_tail_lp/2026-11-01/execution_preflight.json)でSOURCE105／入力9／依存lock／NPY header／fresh output／実空き資源を確認した。準備時93人工PASS＋29 subtestsからSOURCE変更0のため、同じ回帰を不要に繰り返していない。元optimizerのUNQUALIFIEDは引き継ぎ、今回はCDF fitの再開ではない。

## 3. Solverと独立数値診断

[Solver原結果](../../artifacts/research_c_fixed_u_tail_lp/2026-11-01/run01/LP_solver.json)／[独立診断](../../artifacts/research_c_fixed_u_tail_lp/2026-11-01/run01/LP_diagnosis.json)。以下は実行済み診断JSONの値で、新しいLP／行列評価を追加していない。

| 項目 | 値／状態 |
| --- | --- |
| Solver success／status | true／0 |
| Message | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| LP solver nit | 479（CDF optimizer iterationではない） |
| Solver fun | 1.1204135568304 |
| 元候補exact λ [Ha] | 0.9545432846581517 |
| 返却点の診断λ [Ha] | 1.1204135568304006（未採用点） |
| 元λ−返却点λ [Ha] | −0.1658702721722488 |
| Primal epigraph objective | 1.1204135568304001 |
| Dual objective pilot | 1.1204136222550005（有効な下界とは認定しない） |
| Numerical optimality | false |
| Strict improvement | null／未判断 |

返却点の目的値は元の数値的feasible pointより大きい。これを正しい最適値や新DFの悪化結果とは扱わない。

| 登録診断 | 実測値 | 基準 | 判定 |
| --- | ---: | ---: | --- |
| 元j0のfull-row equality residual | 2.42861286636753e-17 | 1.0767046484579418e-10 | PASS |
| 返却点のfull-row equality residual | 2.0391403843245115e-11 | 同上 | PASS |
| Epigraph violation | 1.0766127572781059e-16 | 1e-9 | PASS |
| u非負違反 | 0 | 1e-9 | PASS |
| Dual equality residual | **7.870724498774343** | **1e-9** | **FAIL** |
| Dual box／inequality sign／u-dual違反 | 各0 | 1e-9 | PASS |
| Complementarity | 1.0766127572781059e-16 | 登録objective式、約2.12e-9 | PASS |
| Fun consistency／epigraph surplus | 2.2204e-16／4.4409e-16 | 同上 | PASS |
| Primal-dual gap | **−6.542460040392939e-8** | 絶対値が同基準以内 | **FAIL** |
| Dual dot absolute scale | 699224942.2418617 | 相殺guardの入力 | 診断値 |
| Objective arithmetic guard | **0.0010532779145245997** | 約2.12e-9 | **FAIL** |

Primal側の小さい残差だけでは採用しない。今回のdual equality不適格と相殺guard不適格のため、dual値をLP下界、gapを最適性保証、返却点λを資格付き最小値とは扱えない。

Conditioning記録は256×60のA、診断effective rank46、near-null columns14、cutoff1.8759480612571625e-13、retained condition99.84975897081932。元Uのorthogonality残差は4.440892098500626e-16でgate内だった。Retained conditionだけで、冗長／近従属等式を含むdual復元の安定性が保証されたとは言わない。SVD rankは数値診断で、exact nullspaceの証明ではない。Adapter側の制約削除・polar補正・U変更は0、HiGHS内部presolveの詳細を今回追加解析したわけではない。

## 4. STOPと未実行範囲

[最終result](../../artifacts/research_c_fixed_u_tail_lp/2026-11-01/run01/result.json)は`UNRESOLVED_LP`、診断は`UNRESOLVED_LP_DIAGNOSTICS`。登録した「LP未資格なら後続採用なし」の分岐で停止した。

| Logical unit | 実績 |
| --- | ---: |
| Evidence／input／export／保存reference読込 | 各1、計4 |
| Fixed input gate | 1 |
| LP assemble／solve／diagnose | 各1、計3 |
| 未解決結果のcomparison | 1 |
| **合計** | **9/最大50 COMPLETE** |

Call例外FAILは0、retry/resume0。[STOP](../../artifacts/research_c_fixed_u_tail_lp/2026-11-01/run01/STOP.json)の`failed=false`はPython/call例外なしという状態で、独立数値gateの全PASSや科学的成功を意味しない。診断の上記3項目は不適格である。残41 unitsはNOT_RUNで、未使用枠を別runや追加LPへ転用しない。LP1回枠は消費済み。

新candidate export／独立tensor保持／新旧model・law／head・identity／新CAR・一般K／DF保持／PF・補正RTE／attenuationはすべて**NOT_RUN**。未資格のraw LP xは保存したが、candidate.npz／candidate資格は出力していない。後続gateを未評価のままPASSと扱わない。

元DF予算PASSは旧候補の保存結果として不変。今回の点へDF／PF-RTE／ground／CA／shots／native価格を転記していない。新CDF fit・CDF optimizer iteration・RC2 LP・rank/U探索・追加LP・native build/compile・shots/W/総費用評価はすべて0。

## 5. 証拠・資源・保全

Run01 raw11件には[LP problem](../../artifacts/research_c_fixed_u_tail_lp/2026-11-01/run01/LP_problem.npz)（A/T/b/j0）、[LP生出力](../../artifacts/research_c_fixed_u_tail_lp/2026-11-01/run01/LP_raw.npz)（x/slack/con/equality・inequality・bound marginals/residual）、診断JSON、SOURCE、契約、固定入力gate、[call journal](../../artifacts/research_c_fixed_u_tail_lp/2026-11-01/run01/science_calls.jsonl)、STOP、資源を保存した。今回の報告整理ではNPZの追加decode・新しい数値診断を行っていない。

[資源原本](../../artifacts/research_c_fixed_u_tail_lp/2026-11-01/run01/resources.json)：wall0.234785837秒、CPU7.243747136秒、peak RSS78,496 KiB。実行前affinity32 CPU／空きRAM約56 GBを確認した。予定していたfull256 CAR12因子のworker計画は後続未実行のため起動していない。CPU/RAM/worker/wall／solver iteration/timeに人工上限を追加せず、共有環境・他ユーザーのjob・共有venv/cacheに変更はない。

[保全postflight](../../artifacts/research_c_fixed_u_tail_lp/2026-11-01/execution_postflight.json)でSOURCE105／入力9／旧native729／shots92／Stage B69／共通fit37／JSON修復7・旧SOURCE/contract/STOP、新raw11を照合する。SOURCE修正・retry・閾値緩和・自動fallbackは0。Commit/push後、remote SHA/treeと本文・SOURCE・主要rawを確認する。

## 6. 次に必要な最小限の作業

現時点で確定したのは、今回のHiGHS出力を独立診断で資格付けできなかったこと。Full-rowの近従属性、有限精度RHS、presolve、dual出力復元との整合が技術的な切り分け対象だが、**原因を確定したとは言わない**。大きなdual dotの相殺とdual stationarity不適格は、保存診断上の観測である。

次の候補は、Codexによる冗長・近従属等式と微小RHS丸めを含む人工LPの再現／diagnostic検証・実行準備。新しい実H4再解析や再LPはその作業と区別し、別SOURCE・条件・明示実行許可が必要。元科学runを編集・再起動せず、SOURCEやallowanceを事後変更しない。

ユーザー／研究Cが診断方針と次の固定範囲を判断する。数値最適性が未資格のため、改善不能という結論や候補Bへの移行根拠には使わない。重要科学レビューも今回実施していない。

**証拠公開・remote照合後mandatory STOP。追加LP・科学的再解析・候補B・fit・PF/RTE・shots/native/総資源へ自動進行しない。**
