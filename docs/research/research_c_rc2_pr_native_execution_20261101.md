# 研究C：保存RC2／PR-tailのnative価格・数値同値性検証結果

2026-11-01 JST。**655/655 logical units COMPLETE、correctness FAIL 0、retry/resume 0。計算後mandatory STOP。**

## 1. 結果と範囲

保存直鎖H4・1.00 Å・STO-3G、通常number CDF rank8、Nα=Nβ=2／sector36、元frame順L_D=2、partial ordered symmetric S2 δ=h=0.1／s=1／r=128／K=0を対象に、RC2／PR-tail各32件の非paired・非adaptive trajectoryを一度だけ評価した。Own-DF ground oracle、状態準備費用除外、固定P0を維持した。測定単位は**完全な1 h-step controlled kernel**で、13 roundのcoherent反復やshots加重費用は評価しない。

64 kernelのraw／compiled／actual adjoint／X/Y、共通ancilla-only shell3件がすべて登録数値gateをPASSした。Finite-bit cohortの平均RZはRC2 **20,487**、PR-tail **13,246.375**、PR−RC2は**−7,240.625**。他の5費用指標も今回のcohortではPRの平均が小さい。

これは共通有限予算で得た**両optimizer未収束の固定終点**に対する観測である。最適化済みRC-DF一般との優位、母平均／ideal-law期待費用、因果的な改善機構、全law合成誤差、actual CA、総資源優位の認定ではない。

## 2. 固定SOURCE・入力・実行

| 対象 | Identity |
| --- | --- |
| 科学実行SOURCE | 14843a28f729cfe5f238d72019b060ee281c9a45（無変更） |
| 実装commit | 3bb00c1caddc30f21a292927b2d2effbacbf2164 |
| 元2-fit SOURCE | e2f7dfd44357510d54ab9c6c9c5032e0072ab376 |
| 保存Stage B SOURCE | d6b17b79461a39bd8ba2e6b7389a1f4fad4d74e6 |
| 保存shots SOURCE | e88fa599af7724f34e11a0f2eab959f92135cf13 |
| 契約SHA256 | 43d41fece2fa958627f908eaec352ee0c78dde8c6846d5a07f91e773cd53d9b9 |
| SOURCE manifest SHA256 | b17d0236f2ed03ca57ad0b84d314344ee1ed5e0a93aa6b987f84717a076a45b1 |
| RC2 export SHA256 | d6cb5d56dfc6a83090a6ad1bc1207fdaee62a14cf120542fcb5a7d116644a798 |
| PR export SHA256 | 3fdb0cb77afc770294cbe7aebf47d8a4529e3567ef9e886895495f4490b7ee21 |
| 元input snapshot SHA256 | 1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db |

[固定契約](research_c_rc2_pr_native_contract_unapproved_20261101.json)の歴史的UNAPPROVED表示は変更せず、チャットの明示実行許可に基づいてfresh run01を一度実行した。承認証明・承認原文保存・新grantは作成していない。[技術準備](research_c_rc2_pr_native_preparation_20261101.md)の人工・回帰105 PASSからSOURCE変更0のため、同じテストを不要に繰り返さずSOURCE130／入力26／依存pinsを再照合した。[実行前記録](../../artifacts/research_c_rc2_pr_native/2026-11-01/execution_preflight.json)。

独立worktree／branch `codex/research-c-rc2-pr-native-execution-20261101`で既存[entry](../../scripts/run_research_c_rc2_pr_native.py)を実行。契約の`policy`にあるnative_build=0等は元Stage Bのidentity照合用metadataを継承した値で、今回のnative実行範囲は新scope／native／outputs／call_limitsに従う。いずれの値も変更していない。

P0＝Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation_degree1／qubits_initially_zero=False。既存StrongBuilderのhead内相対Gaussian融合、tailの厳密basis_key共有・角度結合を同等に適用した。Head/tail境界融合・新規最適化・旧cohortのpool・旧価格の転用は0。

新fit／optimizer／rank探索／DF誤差再計算／ground eigensolve／PF・RTE・attenuation・shots再計算／総費用算定はすべて0。保存DF予算PASSとground/aliasを使用し、両optimizer UNQUALIFIEDを保持した。

## 3. 正しさの数値pilot

独立full256 CARのbase＋8 frameを各armの保存sector36 generatorsへ照合した。Signed event catalog／確率／順序／λ／identityと保存lawの照合もPASS。各trajectoryではfull512全成分についてcontrol=0 identity、control=1独立CAR/event evolutionを比較した。Global phase fitや個別Gaussianのsector射影で代用していない。

[全gate scalar要約](../../artifacts/research_c_rc2_pr_native/2026-11-01/checks_summary.json)。以下は32件内の既存診断値の最大で、新しいoperator計算ではない。

| 診断 | RC2 | PR-tail | 登録allowance |
| --- | ---: | ---: | ---: |
| full CAR／保存sector因子・和の最大成分差 | 8.882e-16 | 1.332e-15 | 1e-10 |
| tail law／full CAR最大成分差 | 1.066e-14 | 1.110e-15 | 1e-10 |
| raw／CAR最大成分差 | 1.995e-14 | 3.044e-14 | 1e-10 |
| compiled／CAR最大成分差 | 4.119e-12 | 2.148e-12 | 1e-10 |
| compiled／raw最大成分差 | 4.118e-12 | 2.147e-12 | 1e-10 |
| raw unitarity最大成分差 | 3.286e-14 | 6.084e-14 | 1e-10 |
| compiled unitarity最大成分差 | 1.450e-13 | 1.623e-13 | 1e-10 |
| actual raw adjoint／参照最大成分差 | 1.996e-14 | 2.966e-14 | 1e-10 |
| actual compiled adjoint／参照最大成分差 | 4.118e-12 | 2.154e-12 | 1e-10 |
| raw adjoint復帰identity最大成分差 | 3.242e-14 | 5.806e-14 | 1e-10 |
| compiled adjoint復帰identity最大成分差 | 1.289e-13 | 1.520e-13 | 1e-10 |
| raw X/Y複素signal誤差 | 1.293e-14 | 4.300e-14 | 2e-10 |
| compiled X/Y複素signal誤差 | 1.068e-13 | 1.360e-13 | 2e-10 |
| compiled X/Y state norm誤差 | 1.155e-13 | 1.318e-13 | 2e-10 |

Actual adjointは実際のraw／compiled回路から構築し、参照随伴との一致と元回路からの復帰を両方検証した。X/Yは保存own-DF ground stateに対する**単一trajectoryの複素signal**で、RTE平均signalや全roundのCA試験ではない。

Operator／unitarity allowance=1e-10、complex signal=2e-10、rtol=0、phase fitなしを実行前固定のまま使用した。これらはbinary64の有限cohort数値同値性であり、validated numerical enclosureや全law上界ではない。

## 4. Native価格

[result.json](../../artifacts/research_c_rc2_pr_native/2026-11-01/run01/result.json)、[64件metrics CSV](../../artifacts/research_c_rc2_pr_native/2026-11-01/native_metrics64.csv)、個別原本`samples/*_metrics.json`。平均±SEは既存runnerの記述的推定を引用した。同じsampleの反復でkernel費用を割り引いていない。

| P0 kernel指標 | RC2平均 ± SE | PR-tail平均 ± SE | PR−RC2平均 |
| --- | ---: | ---: | ---: |
| RZ count | 20487 ± 125.120 | 13246.375 ± 238.114 | −7240.625 |
| RZ depth | 6472.375 ± 37.141 | 3999.65625 ± 57.322 | −2472.71875 |
| CX count | 7558.25 ± 39.126 | 5344.875 ± 73.088 | −2213.375 |
| CX depth | 4482.15625 ± 20.324 | 3261.53125 ± 31.316 | −1220.625 |
| Total depth | 14192.9375 ± 81.644 | 8829 ± 125.614 | −5363.9375 |
| Circuit size | 41175.5 ± 251.925 | 26343.375 ± 478.885 | −14832.125 |

RZの観測min/maxはRC2 19018／21907、PR 10978／16432。32 trajectoryの観測範囲であってsupport下界・上界ではない。SEはfinite-bit lawの記述量で、同時保証区間・ideal-law保証・paired差の推定ではない。

外部head/X/Y shellは新しく3件を共通にbuild／compileし、ancilla-only構造・同じ9-qubit register・位相を確認した。旧shell価格は移送していない。

| P0 shell | RZ count | RZ depth | CX count/depth | Total depth | Circuit size | 測定数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| head | 2 | 2 | 0/0 | 3 | 3 | 0 |
| X | 2 | 2 | 0/0 | 4 | 4 | 1 |
| Y | 1 | 1 | 0/0 | 3 | 3 | 1 |

Readoutのtotal depth／sizeには測定を含み、quantum-only depthはX=3／Y=2。Shell compiled/raw最大成分差はhead/Xで1.665e-16、Yで5.551e-17。境界を跨ぐ新しい融合は仮定しない。

## 5. Call・資源・証拠

| Logical unit群 | 実績／上限 |
| --- | ---: |
| Evidence/input/CAR/model/law/sector/groundと18 factor照合 | 38/38 |
| Gaussian18＋相対Gaussian8 | 26/26 |
| shell build/compile/assessment/metrics | 12/12 |
| 64 task × sample/build/reference/raw/compile/compiled/adjoint/XY/metrics | 576/576 |
| arm校正2＋比較1 | 3/3 |
| **合計** | **655/655 COMPLETE** |

Call FAIL0、retry/resume0。Compileは64 kernel＋3 shell＝67、QPYは64×4＋3×2＝262。独立reference64件、イベント列・seed64件、gate・metrics原本を含むrun01 raw729件を保存した。[STOP](../../artifacts/research_c_rc2_pr_native/2026-11-01/run01/STOP.json)／[call journal](../../artifacts/research_c_rc2_pr_native/2026-11-01/run01/science_calls.jsonl)／[raw SOURCE](../../artifacts/research_c_rc2_pr_native/2026-11-01/run01/SOURCE.json)。QPY／referenceの追加decodeや再compileは行わない。

[実行資源](../../artifacts/research_c_rc2_pr_native/2026-11-01/run01/resources.json)：wall2442.233秒（約40分42秒）、CPU27023.938秒、peak RSS3,807,400 KiB。[resource plan](../../artifacts/research_c_rc2_pr_native/2026-11-01/run01/resource_plan.json)は実空き資源から32 workersを選び、並列時BLAS1で過剰多重化を調整した。Qiskit build/compile/actual-adjoint構築のprocess-local mutexを維持し、CAR/reference・Operator検証は並列。CPU/RAM/worker/wallの人工上限、共有venv/cache/global設定の変更はない。Lock待ち10992.985秒は複数threadの累積値でwall時間ではない。比較baselineを新規計算していないためspeedup factorは主張しない。

[保全・証拠inventory](../../artifacts/research_c_rc2_pr_native/2026-11-01/execution_postflight.json)でSOURCE130／保存入力26、旧shots raw92／Stage B raw69／共通fit raw37／JSON修復raw7と旧SOURCE群・STOPを照合する。新raw729件をhash固定し、commit/push後にremote SHA/tree・主要本文を確認する。Local numerical pilotを公開した証拠で、CI／外部独立再現ではない。

## 6. 未資格事項と次の研究判断

今回、有限64件の数値gateとnative価格が得られた。一方、whole-law synthesis bound、implemented-law距離、cost support、ideal-law期待価格、actual CA、保証shots、interrogation総費用、guaranteed advantageは引き続き**null**。元の両optimizer未収束も解消していない。

保存shots／Wの有効なideal／assumed laneへ今回の候補固有価格を接続する材料は揃ったが、**W×μ＋Sの計算は今回実施していない**。r=128の二重加算、未知error/supportを0で補完、sample最大値の全law上界扱い、cohort内価格差からの一般的・因果的結論は行わない。

次の担当はユーザー／研究C：明示仮定を保持した条件付き総資源接続の情報価値と、最小の次期評価範囲を判断する。Codexは具体的な依頼後に必要な技術準備を行い、新しい科学数値評価は別途明示許可後のみ実施する。重要科学レビューも今回実施していない。

**公開照合後mandatory STOP。新fit、全law上界追加計算、shots再算定、actual CA／総費用評価へ自動進行しない。**
