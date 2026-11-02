# 研究C：F2-K1 Baseline固定trajectory・validated CAR接続結果

2026-10-27 JST。**7/7 logical calls COMPLETE、correctness FAIL0、retry/resume0。** 固定Baseline `baseline_00`の保存raw／compiled回路から、exact κH＋right-polar Qで定義した新理想CAR targetへの位相込みoperator2上界を取得した。両上界は登録allowance exact `1/100000000` 内である。

## SOURCE・契約・対象

- 科学実行SOURCE：`3742ec10ddc0df5faaaa2a6856a9cc60ea013eb9`。独立branch：`codex/research-c-F2-K1-execution-20261027`。
- Scope：`C_PR_AWARE_DF_FIXED_F2_K1_V1`。
- Target：`CAR_exact_kappa_H_polar_U_fixed_trajectory_v1`。
- [固定契約](research_c_pr_aware_df_F2_K1_contract_draft_20261027.json) SHA256：`bc7a63d6ba0dd3355c9ea8b4aa3231d6187a1cd5ee97f1c8e76d44d3185c4b63`。
- [SOURCE142 manifest](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-K1-preparation/SOURCE.json) SHA256：`de1fcf1b9bcaf227301687b921ced582c906658ad866370c5c9a362379cf6fd6`。
- [原本12ファイル](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-K1-run01/)／[要約](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2_K1_execution_summary.json)／[postflight](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2_K1_execution_postflight.json)。

保存直鎖H4・隣接1.00 Å・STO-3G、通常number CDF rank8 Baselineの一件のみ。Nα=Nβ2/sector36を含むfull8-mode/256、last ancilla/full512。元順LD2、partial symmetric S2 δ=.1/s4、h=1/40、両half-head=1/80、保存16 events/r16/K0。元g補正κ、J、constant、frame順、signed atoms、angle、head順序、tail identity phaseを保持した。一つのh-step trajectoryを対象とし、13roundの平均finite-RTE信号を新評価したものではない。Own-DF oracle／状態準備除外は既存比較条件であり、今回状態をロード・準備していない。

保存P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/qubits_initially_zero=false、既存Gaussian／declared-phase政策を維持し、新回路build/compileは0。Arb192／python-flint0.8.0／FLINT3.3.1、global phase fitなし。

科学SOURCE／runnerは準備commitから無変更で一度実行した。歴史的な`UNAPPROVED`という契約metadataは変更せず、今回のチャット許可に基づいて実行した。承認証明・承認原文保存・新grantは作成していない。

実行entry：`.venv/bin/python -B scripts/run_research_c_pr_aware_df_fixed_F2_K1.py --contract docs/research/research_c_pr_aware_df_F2_K1_contract_draft_20261027.json --execute`。`PYTHONDONTWRITEBYTECODE=1`とtask専用`MPLCONFIGDIR`を設定した。

## Gate・科学call

| Call／gate | 結果 |
|---|---|
| input_arrays | COMPLETE／PASS：固定h/g/cとJ/U、shape/dtype/finite、全8 Jのexact real symmetry |
| arm_replay | COMPLETE：元κの既存metadata再生。κHのnativeモデルへの書戻しなし |
| trajectory_load | COMPLETE／PASS：同一source・Baseline・保存16 signed K0 events、元順 |
| CAR_reference | COMPLETE／PASS：exact H/A identity、保存κ診断との一致、全8 Uのeta<1、κ/U bridge、全25 B0 blocks |
| qpy_decode | COMPLETE／PASS：同じbaseline_00 raw QPY／9 qubits |
| raw_CAR_certificate | COMPLETE／VALIDATED_WITHIN_FIXED_ALLOWANCE：full512全512入力列 |
| triangle_composition | COMPLETE／VALIDATED_FIXED_COMPILED_IDEAL_CAR：既存同一raw compiled/raw certificateを再利用 |

7/7 attempts、COMPLETE7、FAIL/NOT_RUN0。Compiledの保存triangle上界も同じ登録allowance以下であることを、取得済み有理数boundから確認した。Compiled回路の新decodeや再certificate計算はしていない。

旧F2の残3 callsを再利用していない。旧F2の4-call STOP、κ診断の6-call結果、旧B1/F1/C1のSOURCE/raw/STOPは不変。新fit／optimizer／sampling／build／compile／DF-PF-RTE再計算／C1再解析／shots／総資源／F3は0。

## Validated誤差上界と計上

以下のtrajectory operator誤差上界は無次元である。

V0=CAR(元κ,元U)、V1=CAR(exact κH,元U)、V2=CAR(exact κH,right-polar Q)。V0/V1を厳密unitaryと仮定せず、元κの非Hermiticityと元Uの非unitarityを別々に計上した。

| 項目 | operator2上界：上向き6桁表示 | 根拠 |
|---|---:|---|
| 保存raw／literal B0（V0） | **2.57295e-14** | 独立acb referenceと全512基底列。Raw/B0に距離2のcapなし |
| κ model bridge：V0→V1 | **2.07978e-17** | Exact κA entry-L1、full-Fock μ=8a、左右h/2、非unitary中間norm |
| U model bridge：V1→V2 | **9.53720e-16** | 元Uから唯一のright polar Qへの既存解析bridge |
| **Raw／K1 V2** | **2.67040e-14** | 上の3項を各1回 |
| 保存compiled／raw（旧B1） | **3.46947e-12** | 同一raw/hash・full512・位相・Arb192の旧certificate。再計算0 |
| **Compiled／K1 V2** | **3.49618e-12** | 保存compiled/raw＋今回raw/V2を各1回 |

いずれの表示値も保存exact有理数endpointをDecimal ROUND_CEILING／6有効桁で表示したもので、下向き丸めを上界として表示していない。正本は[raw_CAR.json](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-K1-run01/raw_CAR.json)、[compiled_CAR.json](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-K1-run01/compiled_CAR.json)、[K1_models.json](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-K1-run01/K1_models.json)のexact fractionsである。

Postflightで `εraw/V2=εraw/B0+εκ+εU`、`εcompiled/V2=εcompiled/raw+εraw/V2` の保存値が完全に一致することを確認した。Reference区間半径はacb column差に含まれ、raw評価radiusは各columnへ一度だけ含まれている。κ/U bridgeおよび旧compiled/rawも一度ずつで、二重加算なし。今回の値は2未満なので最終unitary endpointの距離2 capは結果を変えていない。

κHはexact有理数で定義してArbへ接続し、float64対称化で元κを書き換えていない。保存診断からのκ literal/H/A identityもPASSした。Uは元値のままで、Qを保存Uへ書き戻していない。κ bridgeはμ=8aと両half-headを用い、U-local積由来の中間norm上界を含む。[数学定義・証明](research_c_pr_aware_df_F2_K1_target_readiness_20261027.md)をそのまま使用した。

## Full-Fock・位相・比較identity

Referenceの25 spin-count blocksは全256 system basisを重複なく覆い、controlled embeddingは `diag(I256,V0)`。Raw比較は全512入力列と全512出力で、sector36だけの比較でも有限probeの外挿でもない。Scalar／vacuum／control相対位相を保持し、global phase fitは行っていない。

Raw SHA256：`dbd1bd9b250db3c33ea98a2ba346af010b9e90903564f0ced9e8f1b66e8ddbcf`。契約・保存QPY・新raw certificate・旧compiled/raw certificateはこの同じrawを参照する。Target V2、dimension512、phase_fit=false、full_operator_2、precision192のscopeを確認した。

旧F2 targetや元物理Hamiltonianへこの新上界を移送しない。元κのexact Hermiticity不成立という診断事実も変更しない。取得したのは、採用された**新K1 targetに対する固定一trajectoryの保証**である。

## 資源・保全

[Preflight](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2_K1_execution_preflight.json)でSOURCE142、保存input42、依存11 packages、Python3.11.1、FLINT runtime／FE_TONEAREST、旧protected inventory、fresh outputを照合PASS。準備時153 tests＋29 subtests PASSをSOURCE同一性とともに再利用し、同じ人工テストを重複実行していない。

CPU affinity32、空きRAM約56.6 GB、直近CPU/memory pressure0の条件で、既存resource planはblock25／column32 workersを選択した。Actual-resource process並列、process-local Arb context/cache、recentered評価をそのまま使用し、CPU/RAM/worker/wallの人工capを追加していない。共有venv/cache/OS/他jobは変更していない。

Runner wall **60.3244048692286秒**、parent peak RSS **462188 KiB**。CAR_reference 5.1387187112秒、raw_CAR_certificate 53.7723031947秒。これらは今回の記録であり、異なる旧jobとのspeedup比較ではない。瞬間ピークworker数・aggregate peak RSSは未採取で、25/32は実行が選択したplanである。資源observerは完了後の記録で自job残存0／process exit0を確認した。

実行前後でSOURCE・契約・input・旧F2／κ診断／B1/F1証拠のbytesが一致した。原本12ファイルの各SHA256をpostflightへ保存。Scientific SOURCE／旧raw／grant／ledger／marker／STOPは保全した。

## 科学的範囲・次担当・mandatory STOP

Baseline `baseline_00`の一trajectoryについて、raw／compiledと、exact κH＋polar Qの理想CAR trajectoryの位相込みoperator差をvalidatedに接続できた。旧F2のSTOPを解除したり、元κを厳密Hermitianへ変更したりした結果ではない。

元物理積分の厳密性、DF誤差の新保証、ProposedのCAR reference、他trajectory、全law native synthesis、PCG64 IID、CA達成、保証shots、cost support、母平均／総資源優位は未資格のまま。Stage C1の約83%総RZ削減は元の明示仮定付き点推定で、再解析・資格更新していない。

次は**ユーザー／研究C**が、Proposedにも同じK1定義を接続するF3の技術準備を依頼するか、別の最小検証を選ぶか判断する。**Codex**はその具体的依頼と必要な独立の科学実行許可後だけ対応する。本7-call枠の残作業・retry枠はない。

SOURCE・結果・call journal・STOPをcommit/pushし、remote SHA/Git tree/SOURCE/input/成果物照合後に**mandatory STOP**。F3、追加sampling、C1再解析、shots/総資源、別分子、詳細科学レビューへ自動進行しない。
