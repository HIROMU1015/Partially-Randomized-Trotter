# 研究C：実H4 reduced-objective pilot・gradient stability未資格で停止

2026-11-02 JST。Repository: HIROMU1015/Partially-Randomized-Trotter。

## 1. 結果と結論の範囲

ユーザーの明示許可に基づき、固定SOURCE `e1582a5620e61caec52f0ee284c8cad8c0202580`、[1051-call契約](research_c_j_u_reduced_contract_unapproved_20261102.json)を変更せず、独立worktreeのfresh run01でPR-tail/RC2各1 trajectoryを実行した。**38/1051 logical units COMPLETE、correctness FAIL0、retry/resume0。両armは`REDUCED_GRADIENT_UNQUALIFIED`で停止した。**

PRは初期Uでのrefinement前後のU gradient差5.6058350e-6、RC2は最初の未受理trialで8.5692603e-6となり、固定stability allowance1e-6を満たさなかった。**受理されたU更新は両armとも0**で、J/U同時一次条件は未達。外側reduced optimizationの有効性・勾配reboundの回避を検証できるところまでは進まなかった。

一方、PRの共通head gaugeに関する登録rank/nullspace条件、全6 inner solveのJ一次条件、同一点の微分整合、終端codec/一般K-CAR表現、両診断終点のDF予算はPASSした。PRの追加nullspaceや内側iteration上限が今回の直接の停止要因だったとは報告しない。

[結果原本](../../artifacts/research_c_j_u_reduced/2026-11-02/run01/result.json)／[小集計と全point診断](../../artifacts/research_c_j_u_reduced/2026-11-02/execution_summary.json)／[STOP](../../artifacts/research_c_j_u_reduced/2026-11-02/run01/STOP.json)／[保全postflight](../../artifacts/research_c_j_u_reduced/2026-11-02/execution_postflight.json)。STOP.failed=falseはcorrectness例外がなかったという意味で、最適化収束・envelope資格PASSの意味ではない。

## 2. SOURCE・固定条件・実行

| 固定物 | 条件／identity |
| --- | --- |
| 科学SOURCE | `e1582a5620e61caec52f0ee284c8cad8c0202580`、133 filesを変更せず実行 |
| 契約SHA256 | `377db724f81353d47f7d38c7ece5cde3219cd8f53be24eabdfc5103503f06343` |
| SOURCE manifest SHA256 | `631824b62bb0ef0112670a3fe1a9ffdc8cf451cce738e8e686f91cd15bc86949` |
| 対象 | 保存直鎖H4、隣接1Å、STO-3G、Nα=Nβ2、sector36 |
| DF | 通常number CDF8、一般signed実対称J、実直交U、元順LD2/head0/1・tail2..7 |
| 初期値 | 同じ元qualified J/U/x0から独立2 arm。旧joint/分離型終点warm-startなし |
| 元入力SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| 元qualified初期値SHA256 | `78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856` |
| 目的 | PR smooth tail mass二乗γ=.1 / RC2全ordered J L2 α=.1、ε=1e-8Ha |
| Scales f* / M* / P0 | 2.1839632053132894e-9 / 2.2093719370722074 / 4.410884790420646 |
| 物理規約 | 元g補正κ、nuclear scalar一回、spin/frame順不変 |
| DF予算 | norm＋既存engineering guard≤CA/8=.00019875Ha |
| Outer | SciPy1.14.1 BFGS、64 iterations/128 distinct points/256 inner solvesを各arm上限とする固定政策 |
| Fresh output | `artifacts/research_c_j_u_reduced/2026-11-02/run01`、今回消費済み |

PFのδ・時間窓は今回の最適化に非適用で、PF/RTE/ground/quantum native/shots/総資源は評価していない。

[技術報告](research_c_j_u_coupled_reduced_objective_preparation_20261102.md)の186人工・回帰PASSは同一SOURCEの証拠として再利用し、不要な再テスト・再実装・詳細科学レビューを挟まなかった。[実行preflight](../../artifacts/research_c_j_u_reduced/2026-11-02/execution_preflight.json)で契約、SOURCE133/input25、両依存環境、旧raw1825、fresh outputと実資源を照合した。初期物理DF資格は固定の保存PASSを再利用し、workerで元J/U/x0のfidelity・objective scaleを確認した。追加の初期DF norm/一般K評価を未計上で繰り返していない。

ffsim0.0.83/NumPy1.26.4/SciPy1.14.1/JAX0.4.38の既存fit環境と、Qiskit1.3.0/OpenFermion1.6.1等の評価環境を分離した。契約の歴史的UNAPPROVED metadataはそのままで、承認証明書・原文保存・新grantは作成していない。

```bash
# 実行履歴。run01は消費済みで、このcommandを再起動しない。
PYTHONDONTWRITEBYTECODE=1 \
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
 scripts/run_research_c_j_u_reduced.py \
 --contract docs/research/research_c_j_u_reduced_contract_unapproved_20261102.json \
 --execute \
 --output /tmp/prt-c-j-u-reduced-execution-20261102/artifacts/research_c_j_u_reduced/2026-11-02/run01 \
 --fit-python /tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python
```

## 3. どこで停止したか

| Point | 独立J∞（refined） | 物理U₂ | Refinement U差norm | 構造条件 | 判定 |
| --- | ---: | ---: | ---: | --- | --- |
| PR 0、初期U | 1.3249031e-7、MET | 2.954975096 | 5.6058350e-6 | 共通gauge条件PASS | REDUCED_GRADIENT_UNQUALIFIED、outer未開始 |
| RC2 0、初期U | 6.2833529e-8、MET | .867029585 | 2.1699547e-7 | 正L2、augmented rank80/80 | NUMERICAL_PILOT_ADMITTED |
| RC2 1、未受理trial | 2.7930604e-7、MET | 79.26087532 | 8.5692603e-6 | 正L2、augmented rank80/80 | REDUCED_GRADIENT_UNQUALIFIED、line search途中停止 |

Stability値は登録定義どおりchart/physicalのrefinement前後の差normの大きい方。真のgradient誤差の厳密上界ではない。J一次閾値1e-5とmargin、stability allowance1e-6、value/gradient一致allowanceを事後変更していない。

PR初期点のA_headは期待rank19、共通gauge nullity1、最小retained singular .153064992511、null singular4.75607e-16、rank guard2.05795e-11。Tail T injectivityも登録数値条件を通った。これは有限精度SVDによる数値判定で、厳密rank enclosureではない。実H4のこの点で追加nullを検出した結果ではなく、**構造が通ってもinner refinementのstabilityが未資格となる**ことが確認された。

RC2のpoint1はF=1.2064220886500059、chart gradient₂=79.12292268495813で、既に数値stabilityの契約を満たさない。BFGSのtrialであり、受理点ではない。そのFがpoint0より大きいことを受理ステップの非増加違反とは扱わない。固定契約は未資格trialでarmを停止するため、その後のbacktrackingや別stepを試していない。

RC2の保存`qualification`は保持したpoint0のadmissionで、`stop_trial_qualification`はpoint1の不適格である。前者のPASSを、RC2 trajectory全体の資格と読み替えない。PRは最初の有限pairを未資格診断として保存し、RC2は最後の保持point0を診断として保存した。初期Jへの自動fallback、trial1の採用、joint J/ηへの切替は行っていない。

## 4. 終端の値・正しさ・物理DF予算

以下は両armとも初期UのJ solve/refinement後の**診断終点**であり、U最適化の収束解ではない。

| 保存測定 | PR-tail | RC2 |
| --- | ---: | ---: |
| 自己F | .012267644429814956 | .02919382875477165 |
| Exact tail λ / Ha | .7738355798429691 | 8.285829796166439 |
| J gradient∞ | 1.3249030831e-7、MET | 6.2833528609e-8、MET |
| 物理U gradient₂ | 2.9549750963、NOT_MET | .8670295846、NOT_MET |
| Physical classification | U_UNQUALIFIED | U_UNQUALIFIED |
| Tensor Frobenius | 1.1196306212e-8 | 3.9648981624e-9 |
| Tensor max-entry | 1.8078200881e-9 | 6.4213079298e-10 |
| Sector DF operator norm / Ha | 1.1064369850e-8 | 5.0203197503e-9 |
| 既存engineering guard / Ha | 3.2271714745e-9 | 3.1755775514e-9 |
| DF予算（norm＋guard≤.00019875Ha） | PASS | PASS |
| Sector general-K対raw CAR max-entry | 3.44169e-15 | 1.11022e-15 |
| 同じAD decode点でのJ-gradient差 | 1.22995e-7、PASS | 9.92024e-8、PASS |
| Codec decodeによるtensor Frobenius差 | 2.72445e-15 | 2.25950e-15 |

必要なobjective/gradient・codec・直交性・sector表現のcorrectness gateは全PASS。Newton raw jacは記録したpre-step/accepted/endpoint点で照合し、終点J一次条件は独立計算値を使った。Native parameter診断はffsim ADの実decode点で比較し、物理評価用の元candidate J/Uを変更していない。ここでのnativeはparameter診断で、量子回路build/compileではない。

DF guardは既存`sector_operation_scale_guard8_proposal_v1`というnumerical pilotで、厳密enclosureや最終CA証明ではない。`rank_assessment`のlegacy `CONVERGED`文字列はDF評価の返却statusであり、新optimizerの収束成功を意味しない。両一次資格・envelope適格性を満たしていないため、`qualified_tail_comparison=null`を維持した。

## 5. 内側solver・call・負担

| Inner solve | Solver raw結果 | 独立J∞ |
| --- | --- | ---: |
| PR point0 solve | Newton-CG success、status0、nit481/nfev785/njev785/nhev481 | 1.3643355e-6、MET |
| PR point0 refinement | Newton-CG success、status0、nit2/nfev2/njev2/nhev2 | 1.3249031e-7、MET |
| RC2 point0 solve / correction | 各gelsd、rank80、各linear solve1 | 7.7848076e-7 / 6.2833529e-8、MET |
| RC2 point1 solve / correction | 各gelsd、rank80、各linear solve1 | 1.1922420e-6 / 2.7930604e-7、MET |

PRの記録されたchart VG回数は788/5、Hessian回数は481/2。これらはsolver nfev/njevとは別である。RC2は全4 linear solvesで、LAPACKのnfev0をobjective評価0と解釈しない。独立state/envelope/終端評価も別logical unitsで行った。RefinementのVG総数等、raw metadataが返さないcountを架空の0で補完しない。

Outer driverはPR0回、RC2 1回。RC2は初期点とtrial1まで評価し、accepted U iteration0。登録の資格停止例外がSciPyの最終returnより先に発生したため、outer raw success/status/nit/nfev/njevは取得不能/nullである。停止理由とtrial/accepted記録は残し、solver正常収束や64 iteration消費を捏造していない。

| Logical call | 実績 |
| --- | ---: |
| Worker input/setup/元identity/保存history | 8 |
| Inner J solves、refinementも別計上 | 6 |
| Independent reduced states | 3 |
| Envelope/構造/stability admission | 3 |
| Outer BFGS driver（nested callとは別） | 1 |
| 終端独立診断/tensor/export | 6 |
| Parent snapshot/初期資格/context/physical/最終load/DF/表現/比較 | 11 |
| **合計** | **38/1051、38 COMPLETE、FAIL0** |

Parent11＋PR11＋RC2 16。各innerを個別journalへ計上し、driver1 callに隠していない。新paired trajectory2、各arm1、ffsim joint fitter0、追加fit/retry/resume0。残枠を継続や別アルゴリズムに転用しない。

| Process | Wall / 秒 | Process CPU / 秒 | Peak RSS / KiB |
| --- | ---: | ---: | ---: |
| Parent | 4.0474144295 | 1.617520415 | 289464 |
| PR worker | 2.2567148823 | 4.930158127 | 490992 |
| RC2 worker | 2.0871772347 | 4.649431822 | 489952 |

実行前はaffinity32 CPU、空きRAM56,171,925,504 bytes、CPU/memory pressure avg10=0。既存plannerの独立2 worker、process内tiny BLAS1、JAX/A,T/η cacheを使用し、恣意的なCPU/RAM/worker/wall上限を追加していない。短い時間は早期資格停止を含むため、旧収束過程に対する高速化率ではない。共有環境・他ユーザーjob・venvを変更していない。

## 6. 保存された旧結果との比較

以下は契約の比較callが読むJSON scalarで、旧配列をdecodeした新しいmetricではない。同じ保存直鎖H4/1Å/STO-3G、number CDF8/sector36/元順LD2、同じstrength/scalesで、各arm内だけを比較する。

| Arm / 保存方法 | 自己F | Exact tail λ / Ha | 一次資格 |
| --- | ---: | ---: | --- |
| PR 旧joint | .01866610400735 | .95454328465815 | 未資格 |
| PR 旧分離v1 | .01226717465191 | .77382076306254 | 未資格 |
| PR 修正版分離 | .01226573232259 | .77377527019109 | 未資格 |
| PR 今回保持point0 | .01226764442981 | .77383557984297 | J MET、U/envelope未資格 |
| RC2 旧joint | .02232445369552 | 8.54568712823010 | 未資格 |
| RC2 旧分離v1 | .02919345065645 | 8.28580975959188 | 未資格 |
| RC2 修正版分離 | .02919340235401 | 8.28581010345513 | 未資格 |
| RC2 今回保持point0 | .02919382875477 | 8.28582979616644 | J MET、U未資格、次trialのenvelope未資格 |

PRの今回F/λは旧jointより低いが、両分離型の保存値より改善していない。初期Uでの全J最適化による値であり、outer U更新の改善と認定しない。RC2の今回自己Fは旧jointや両分離型より高い。λはRC2の目的そのものでもない。各方法の古典budget・停止段階が異なり、十分最適化されたRC-DFに対する比較は未成立である。異なるPR/RC2のraw Fを直接順位付けしない。

## 7. Q1〜Q8・未解決事項

| 問い | 今回取得した事実と限界 |
| --- | --- |
| Q1 PR適用条件 | 初期点の共通head gauge数値条件はPASS。しかしrefinement stabilityはFAILで、全体のreduced-gradient admissionは未資格 |
| Q2 同時一次条件 | 両JはMET、両UはNOT_MET。両同時一次資格は未達 |
| Q3 U勾配rebound回避 | 受理U更新0のため検証できず。回避成功ともreduced方式一般の失敗とも言えない |
| Q4 PR F/λ改善 | 旧jointより低い診断値だが、旧分離/修正版分離を上回らず、qualified比較null |
| Q5 RC2の状態 | 初期J解はqualifiedでenvelope数値admissionも通るが、最初のtrialのstabilityで停止。終端F .02919382875、物理U₂ .867029585 |
| Q6 DF予算 | 保存した両診断終点はnorm＋既存guardでPASS。CA保証やU最適性を意味しない |
| Q7 制限要因 | 今回の直接の制限はinner refinementに対するU gradient stability。J一次未達・PR追加null・64/128上限ではない |
| Q8 次の判断材料 | 同じ微分点の整合は通り、内側J閾値だけでは外側stabilityを確保できなかった。より厳密/安定なinexact inner-oracleの技術検討、または局所joint J/η接続を次候補として選べるが、どちらも未採択 |

今回のstability差だけでは、真のinner解誤差、丸め・残差評価、conditioning等の寄与を分離できない。追加の実H4診断・cross-Hessian、精度/閾値変更、再solveを行って原因を断定していない。RC2で未資格となったtrialがその後backtrackingで回避できたかも未検証である。現在の停止規則を緩めて続行することはしない。

次担当は**ユーザー／研究C**：数値stabilityを扱う人工条件付きの技術検討と、joint J/η代替接続のどちらを優先するか判断する。**Codex**は新しい指示・必要な科学実行許可後にのみ新SOURCE/契約を準備・実行する。今回の残枠、旧run、旧grantを再利用しない。次の新アルゴリズムの採択、重要GPT科学レビュー、追加実H4計算は今回実施していない。

## 8. 保全・公開・STOP

実行branch `codex/research-c-j-u-reduced-execution-20261102`。SOURCE133/input25/契約・数値allowanceは不変、旧raw1825・旧SOURCE・旧契約/grant/ledger/marker/STOPを保全した。新raw54件にcall journal、生J/η/gradient、全qualification、診断終点、resources、STOPを保存した。研究概要・validation manifest・dated noteへこの未資格結果を追記し、過去の記録を上書きしない。

commit/push後にremote SHA/tree・公開本文・SOURCE/input/結果/STOPを照合し、**mandatory STOP**。追加fit、rank/strength/LD/初期値探索、joint fallback、PF/RTE/ground/shots/native量子回路/総資源、詳細科学レビューへ自動進行しない。
