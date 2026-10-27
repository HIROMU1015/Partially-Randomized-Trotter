# 研究C：F3-K1 Proposed固定trajectory・validated CAR接続結果

2026-10-28 JST。**7/7 logical calls COMPLETE、correctness FAIL0、retry/resume0。** 保存Proposed `proposed_00`のraw／compiledから、exact κH＋Proposed right-polar Qで定義した理想CAR targetへの位相込みoperator2上界を取得した。両上界は固定allowance exact `1/100000000` 内である。

## SOURCE・契約・対象

- 科学実行SOURCE：`9b0f4d2b40159b77f28caf3f215ad22d0d075879`。独立branch：`codex/research-c-F3-K1-execution-20261028`。
- Scope：`C_PR_AWARE_DF_FIXED_F3_K1_V1`。Target：`CAR_exact_kappa_H_polar_U_fixed_trajectory_v1`。
- [固定契約](research_c_pr_aware_df_F3_K1_contract_draft_20261027.json) SHA256：`7c9d7abb3445de1941f68e05444545fa553b2887e6a48a25017dc83c3229a23a`。
- [SOURCE145 manifest](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-28/F3-K1-preparation/SOURCE.json) SHA256：`4711be5f75cb3e6965e87d51d443cf181f5fc5533859c7314790b02421ed95ce`。
- [原本12件](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-28/F3-K1-run01/)／[exact値を含む要約](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-28/F3_K1_execution_summary.json)／[postflight](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-28/F3_K1_execution_postflight.json)。

保存直鎖H4・隣接1.00 Å・STO-3G、通常number CDF rank8 Proposedの固定一trajectory。Nα=Nβ2／sector36を含むfull8-mode/256、最後のancilla/full512。元frame順0..7、LD2、partial ordered symmetric S2 δ=.1/s4/h=1/40、両half-head=1/80、r16/K0・保存16 signed events。元g補正κ、J、scalar、frame/event/head順、angle、tail identity phaseを保持した。Own-DF ground oracle／状態準備費用除外は既存条件であり、今回状態準備や13round平均finite-RTE評価はしていない。

保存P0 Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation1／qubits_initially_zero=false、Gaussian・declared phase・basis orderを維持。Arb192／python-flint0.8.0／FLINT3.3.1、global phase fitなし。科学SOURCE／runnerは準備commitから無変更で一度だけ実行した。チャットの明示許可に基づき、歴史的な契約`UNAPPROVED` metadataは変更せず、承認証明・承認原文保存・新grantは作成していない。

実行entry：`.venv/bin/python -B scripts/run_research_c_pr_aware_df_fixed_F3_K1.py --contract docs/research/research_c_pr_aware_df_F3_K1_contract_draft_20261027.json --execute`。独立worktree／task-local venv・cache、`PYTHONDONTWRITEBYTECODE=1`を使用した。

## Gate・科学call

| Call／gate | 結果 |
|---|---|
| input_arrays | COMPLETE／PASS：共通h/g/cとProposed J/Uのshape/dtype/finite、全8 Jのexact real symmetry |
| arm_replay | COMPLETE：Proposed metadata・元κを再生。κHのnativeモデルへの書戻しなし |
| trajectory_load | COMPLETE／PASS：Proposed source/seed/export・保存16 signed K0 eventsを元順で復元 |
| CAR_reference | COMPLETE／PASS：保存κ診断とのexact H/A一致、全8 Uのeta<1、Proposed固有κ/U bridge、25 B0 blocks |
| qpy_decode | COMPLETE／PASS：同一proposed_00 raw QPY／9 qubits |
| raw_CAR_certificate | COMPLETE／VALIDATED_WITHIN_FIXED_ALLOWANCE：full512全512入力列／全出力・位相込み |
| triangle_composition | COMPLETE／VALIDATED_FIXED_COMPILED_IDEAL_CAR：同一rawに対する既存Proposed F1 certificateのみ再利用 |

7 attempts／7 COMPLETE、FAIL/NOT_RUN0。SOURCE145・保存input41・依存11 packagesのpreflightがPASS。登録call内のJ・polar等の実数値gateもPASSした。Rawとcompiledの取得済みexact上界がともに固定allowance以下であることを照合した。新fit／optimizer／sampling／build-compile／DF-PF-RTE再計算／κ診断再実行／F1-F2再実行／C1再解析／shots／総費用は0。

## Validated上界と二つのmodel bridge

| 項目 | full operator2上界（6桁上向き丸め） |
|---|---:|
| raw／literal B0 | 3.33805E-14 |
| κ bridge：V0→V1 | 2.07978E-17 |
| U bridge：V1→V2 | 8.18918E-16 |
| raw／K1 target V2 | 3.42202E-14 |
| 既存Proposed F1 compiled／raw | 3.20927E-12 |
| compiled／K1 target V2 | 3.24349E-12 |

表示値はexact fractionから上向きに丸めたもので、raw値のexact fractionはJSONに保存した。通常のfloat丸め値を上界とは呼んでいない。下界・全512列・各raw評価半径も原本に残した。

V0=B0=CAR(元κ,Proposed元U)、V1=CAR(exact κH,元U)、V2=CAR(exact κH,Proposed right-polar Q)。Raw QPYから理想referenceを逆算せず、独立CAR algebraの25 spin-count blocksでfull256を被覆し、最後のancillaへ接続した。有限probeの最大値ではなく全512基底をArbで囲った固定operator評価である。

元κはexact Hermitianと仮定しない。κH=(κ+κ†)/2とκAをexact有理数で定義し、共通元h/gと`corrected_kappa`が一致する保存κ診断を再利用した。元κの書換え・診断の再計算はない。κAのentry実虚L1上界a、full-Fock粒子数上限8、μ=8a、b=exp(|h|μ/2)、Proposedのlocal U-factor product上界Lを使い、両half-headのtelescoping **εκ≤L(b²−1)** を計上した。共通aを使ってもLはProposed固有であり、Baselineのεκ値は代入していない。

U bridgeはProposed固有J/U/eventsから全frameのvalidated eta≥||U†U−I||2、eta<1を確認し、right-polar Qへ接続した。J/scalar/frame順・event/angle/signed supportを変更していない。

原本の有理数で **εraw/V2=εraw/B0＋εκ＋εU**、**εcompiled/V2=εF1_compiled/raw＋εraw/V2** を照合した。B0のacb uncertainty・raw column radius・各bridge・F1 certificateは各一度だけ計上する。非unitary B0に距離2のcapは適用していない。

## Proposed固有identityと再利用

| 保存入力 | SHA256 |
|---|---|
| 共通元input | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| Proposed J/U export | `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227` |
| Proposed trajectory | `7329febc4c26cdb4eae2294d3917152329ec1c9683d9bca29d12801562c8b93a` |
| Proposed raw QPY | `1c19b44c07ccf0f79b7e5f177b2488d82070f6a27f5eee4a259b188078add95c` |
| Proposed compiled QPY | `c735314757ccca29c469a7336d704d337f38e5c2a51ce9d0129495bb162d6dfa` |
| Proposed F1 compiled/raw | `417dc8545988ade7d43f6ff43a7cd5f6ebc3107c11cb2e1876b475d3eb8357e6` |

Proposed seed **12704117278500360798**、B2 sampling SOURCE digest `cdf79ed569127f5cb6c7ee017b76bce6422166a656bb3ae866de6e86917c45bf`、B2 execution SOURCE `a1cf0046967c206f414dbad568a75884ee076ebf`。F1科学SOURCE `e7d22a404442234662ef6d117c55dd9868217c9e`、SOURCE manifest SHA256 `0b10324fd59835dc371a65290dbd52e8909d9d85fbb44120421907f107f77701`、固定契約SHA256 `07d4dde2260179e84d828fa89f82bac783df5052c7c9dfa4b512e4010e073e07`を照合した。F1の7/7・3/3 PASS・retry/resume0、full512/operator2/Arb192/phase_fit=false・同一raw/compiled identityを再利用条件とした。

Baseline B1のcompiled/raw certificateやF2-K1のbridge/bound値をProposedへ転用していない。BaselineとProposedは別J/U・sampling law・seed・streamであり、同じindex0でもpaired trajectoryではない。共通shellはF1と同じ保存scope/identityを継承するだけで、再decode・新certificate生成はしない。

## 資源・保全・公開

[Preflight](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-28/F3_K1_execution_preflight.json)でCPU affinity32、空きRAM約56.4GB、CPU/memory pressure低値を確認し、既存block25／column32 process planを適用した。CPU/RAM/worker/wallの人工capはない。Process-local Arb/cache・recentered評価を再利用した。cgroup quotaファイルが当該mountで取得できない箇所はnullであり、無制限だと認定していない。

Runner wall **71.1893382081762秒**、parent peak RSS **461920 KiB**。CAR_reference **4.910217233933508秒**、raw_CAR_certificate **64.8728461349383秒**。[5秒ごとの自job monitor](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-28/F3_K1_resource_monitor.jsonl)は16 samplesで、観測max spawn workers32、aggregate RSS **9,138,660 KiB**。これは標本最大で連続的なpeakではなく、RSS合算は共有pageを重複計上し得る。速度向上率は測定していない。Process exit0・全worker回収を確認した。[CLI log](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-28/F3_K1_execution.log)と終了codeも保存した。

実行前後のSOURCE145／input41／保護inventory227ファイルは全てhash不変。元κ/U、旧F2 STOP、F2-K1/F1/κ診断/B2/C1等の旧raw・grant・ledger・marker・STOPを変更していない。新原本12ファイルのhashをpostflightへ保存した。人工検証は準備時の193 distinct tests＋29 subtests PASSをSOURCE同一性とともに再利用し、同じ試験を重複実行していない。科学SOURCEの差分は0で、今回追加は結果・運用資源記録・研究文書だけである。

本報告・SOURCE/raw/call journal/STOPをcommit/pushし、remote SHA・Git tree・SOURCE・保存input・成果物を照合する。照合の最終記録はチャットで報告する。

## 科学的範囲・次担当・mandatory STOP

Proposed `proposed_00`の固定一trajectoryについて、raw／compiledから採用K1 targetへvalidatedに接続できた。これによりBaseline/F2-K1とProposed/F3-K1の**個別の**固定trajectory証拠が揃った。元物理Hamiltonian／旧F2 targetへの保証移送や、両armの統計的精度優位は主張しない。

他trajectory、全law native synthesis、PCG64 IID、元物理積分/DF新保証、CA達成、保証shots、cost support、母平均費用／総資源優位は未資格/null。Stage C1の約83%総RZ削減は元の明示仮定付き点推定であり、今回その再解析・資格変更をしていない。詳細科学レビューも実施していない。

次は**ユーザー／研究C**が両armの固定証拠を確認し、全law/reference資格へ進むための最小の追加検証を選ぶ。**Codex**は具体的な次依頼と、その範囲で必要な科学実行・重要レビューの別明示許可後だけ対応する。本7-callは消費済みでretry/resume枠はない。公開照合後に**mandatory STOP**し、追加trajectory/sampling、F2/C1再解析、CA/shots/総資源、別分子へ自動進行しない。
