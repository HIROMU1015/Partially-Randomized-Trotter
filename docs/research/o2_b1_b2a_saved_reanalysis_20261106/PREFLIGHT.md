# O2 B2-a / B1 保存値再解析 preflight

2026-11-06 JST。状態：**PREFLIGHT_COMPLETE / AWAITING_EXPLICIT_SCIENTIFIC_EXECUTION_APPROVAL / MANDATORY_STOP**。

ユーザー添付指示§2・§8.1に基づく技術的準備。実保存値を用いる新規log B、normalization、G_CX、比・順位・統計は未評価。以下を実行承認の対象として固定する。新しいclaim、案X、正式研究statusの採択を含まない。

## 作業場所と原本保全

専用の独立GitHub clone、branch `codex/o2-b1-b2a-saved-reanalysis-20261106`、base `788ebad04e620055a80c75b1b3bef4a15712d4c2` を使用した。既存checkout、前回追補、科学SOURCE、結果、grant/ledger/marker/STOPを変更する操作は行っていない。preflight成果物は専用clone内にローカル保存している。§10のcommit/push・remote照合は科学的再解析承認後の工程である。

GitHub固定バイト98ファイルを取得・SHA256照合した。前回取得済み固定GitHubキャッシュも、前回SOURCE_INDEXのhashと一致するバイトに限り再利用した。各commit/path/blob/SHA256は `INPUT_INDEX.json` に収録した。Claude独立レビュー原本は現在のサーバーでread-only参照可能で、開始時hashとの一致を確認対象に含めた。

CPU128、確認時MemAvailable約981GB、load約89–112をread-onlyで確認した。今回は軽量のmetadata読取と人工テストであり、worker上限の新設、共有環境変更、他ユーザーのjob操作はない。承認後も処理開始時の空き資源を確認し、同一入力のcacheとバッチ読取を使う。新規installは行わず、既存Python3.12.3 / NumPy1.26.4をread-onlyで利用した。

## 取得できた固定資料と再利用SOURCE

直前追補commit `788ebad04e620055a80c75b1b3bef4a15712d4c2` のREPORT、MINIMAL_NEXT_VALIDATION、SOURCE_FORMULA_TRACE、METRIC_AVAILABILITY、SOURCE_INDEX、PRIOR_ART_MATRIX、VALIDATIONを取得した。O2台帳 `2779710a86d380bb484accc51d3dccba183370b8`、現行Pauli統合資料 `4591b8d75a2a6a55c74a51225f0c61648023c8ee` も参照した。

| role | 固定版 | 使用範囲 |
|---|---|---|
| Provider | `c31ef424175c02379eafebe8fc6c7548c1d9aabe` | `rte._paired_order_weight` / `finite_rte_distribution`、paired-Taylor定義 |
| Signal | `e417ffc9f52b13c18f5493193ce510d858664259` | `h46_m5_plan.weights/schedule`、`h46_new2_core.allocation` |
| Native | `27ebabdd2943a4de787ec8b976c38e06fda5a497` | 既存provenance・価格policyの版。native runnerは起動しない |
| Continuation | `ff323f1b09f96c0ce01955728832ff6b0f4c230f` | 継続実行provenance。最終report版と区別 |
| 長時間結果・最終会計 | `155529e20a8fac39e00786c40b57830b71203c4f` | `h46_pr_pilot_repair.canonical_tau`、`h246_saved_cost_evaluation.paired`、最終 `h46_long_time_report.cost_for` / 価格merge・binding |
| B3結果・最終binding | `ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed` | `h46_b3_report.confirm_account` のcohort選択規則。確認native SOURCEはprovenance内の `ae3d301e2318a3f9f1331c047c68d310771b6849` |

実行する純粋関数のファイルhashは `SOURCE_SPECS.json` に固定した。runner全体のimportを避け、必要な関数ASTだけを取り出す薄いadapterを準備した。独自normalizerは作成していない。保存meanだけの場合に限る固定Nの有理数加算はadapterに用意し、SEはNoneのままとする。

`h46_m5_plan.SYSTEMS` の古いH4 rank12は科学条件として使用しない。weights/schedule/allocationのみを使い、rankは実保存bindingのDF10/DF19に従う。allocationは各stageに最低1を予約し、残りを絶対係数比例の最大剰余法で配る元関数をそのまま使う。

既存 `test_finite_rte_phase_amplitude.py`、`test_h46_m5_plan.py`、`test_h46_long_time_accounting.py` 等の検査範囲を静的確認した。元suiteの一括起動は行っていない。今回の人工テストは元関数を直接用い、stageの正負・順序、K0/K2、最低step配分、整数/連続式、paired covariance、保存N不変、欠測/0を検査した。

## B2-a：固定対象

`B2A_FIXED_CANDIDATES.csv` に、結果参照とλ_R参照を含む432候補を固定した。新規normalization値・τ・r_iはまだ生成していない。

| 条件 | 固定範囲 |
|---|---|
| H / geometry / basis | Linear H4/H6、隣接1.0 Å、STO-3G、元の保存H/state/reference |
| DF | tol-only 1e-8、H4 DF10、H6 DF19、H6 weighted-Hermitian completionを維持 |
| family / split | B2：H4 prefix3、H6 prefix5。B3：two-body prefix0、one-bodyは決定論的に保持 |
| formula | S2 (`2nd`)、Yoshida4 (`4th`)、保存 `4th(m5_best)` |
| T | 0.8 / 2.0 |
| q | 1 / 8 |
| R_outer | 16 / 32 / 64 |
| K | 2 / 4 / 6 |
| 規模 | B2 216候補、B3 216候補。全432件に保存log B参照あり |

最初にR32/K4 sliceを評価し、次に固定済み残りを評価する。取得域の時間・qの両端とK/R感度を選んだ代表格子であり、全候補の再解析ではない。eligibility、既存G、近似残差に基づく候補の除外は行わない。T1.2/1.6、他prefix/q/Rへのnormalization一般化は今回の定量結果の範囲外とする。

B2の保存λ_Rは長時間carrier内の `track_a_h46_m5_signal_run02_results/2026-10-20/{H4,H6}_context_receipt.json` の該当prefix literalを使用する。receiptのSOURCEは `bec018c64e7a8458dbf13b1c0e9655d628292857` であり、現在のProvider/Signal roleとは区別して記録する。B3は `B3_correctness.json` のsystem別reconstruction literalを使用する。log Bからのλ逆推定は行わない。

`INPUT_BINDING_CHECKS.json` で、m5 readiness → 長時間provenance → B3 provenanceのH/state/snapshot/rank/compiler/control bindingがH4/H6双方で同一であることを確認した。flattened signal record自体にはper-candidate preparation hashがないため、λの対応は保存input binding・prefix・SOURCEの連鎖による。承認後のexact照合が失敗した場合、この連鎖を無条件に正しいとして進めない。

承認対象の演算は次の3水準である。δ=T/q、τ_i=λ_R w_i δ/r_i、R_outer=Σr_i。

- A：元finite-K分布による `q Σ r_i log β_K(τ_i)` と保存log Bの照合。
- B：元整数配分の `q λ_R² δ² Σ(w_i²/r_i)`。
- C：連続比例配分の `λ_R² T² (Σ|w_i|)²/(q R_outer)`。

A−保存値、B−A、B−Cを別列とし、signed係数、配分、min r_i=1のstage数、最大|τ_i|、K/R効果を記録する。K0は人工テストのみで、実データ対象に含めない。K0を後で追加する場合は別範囲とし、係数1/2をK≥2へ混ぜない。

## B2-a：精度と判定条件（実行前固定）

Python binary64、元SOURCEの `hypot` / `exp` / `lgamma` / `fsum` を使用する。signed timeの演算順は `T*w/q`、τは元canonical_tauに従う。保存JSON/CSVのdecimal literalを保持する。結果に合わせてSOURCE、λ、配分、許容差を変更しない。

1. **EXACT_REPRODUCED**：`abs(A−saved_log_B) <= 1e-10`。元長時間reportのlog_B検査と同じ絶対allowance。超過は **INCONSISTENT** とし、該当入力群をSTOPする。欠測・結合曖昧・非有限値は **UNRESOLVED**。
2. 近似の丸めguardを `η=max(1e-10, 256*u*(qR_outer)*max(1,|A|,|B|,|C|))`、`u=2^-52` と固定する。A/Bがη以下なら相対精度は **UNRESOLVED** とし、絶対残差のみ記録する。このguardをexact再現の閾値へ流用しない。
3. K≥2かつ最大|τ|≤1では、`S4=q Σ r_i τ_i^4` として `−η <= B−A <= (5/8) S4+η` を展開の整合検査に使う。通過を **APPROXIMATION_CONSISTENT** とする。最大|τ|>1はこのcertificateの適用外として **APPROXIMATION_LIMITED**。適用内で違反した場合は **INCONSISTENT** とし調査・STOPする。
4. 上記は近似の次数整合の判定であり、用途上十分なaccuracyの一律合格線ではない。相対残差 `(B−A)/A`、`(B−C)/A`、`(C−A)/A` と絶対残差、S4のboundを全行に表示する。約10%を受入閾値にしない。
5. Cauchyから `B−C >= −η` を検査する。負の超過は定義/配分不整合として **INCONSISTENT**。連続式の残差はfinite-τと整数配分の二つを併記し、一方の相殺で近似の妥当性を認定しない。small-τでも整数配分効果が残れば、その制限を記す。

3の保守的なscalar条件は既存normalizerの数学的性質を利用する。x=τ²≤1について、保持するorder0,2だけで `β_K≥sqrt(1+x)+x/2≥1+x−x²/8`。`log(1+y)≥y−y²/2` と既存paper上界 `β_K≤exp(x)` より `0≤x−logβ_K≤5x²/8`。これをq r_iで加重する。有限Kの比較を無限級数へ置換するものではない。新しい理論claimとして扱わない。

今回の判定はnormalizationモデルまでに限る。総費用G、bias/headroom、最適shot overheadへの因果寄与を推定しない。

## B1：固定比較・価格cohort・N

詳細candidate ID、保存N literal、元RZ参照、価格参照、cohortは `B1_FIXED_COMPARISONS.json` に凍結した。再選択・N再計算を行わない。承認後は各armの既存RZ会計の入力対応を照合してからCXを評価する。

| 対象 | 固定する比較 | CX入力と保証範囲 |
|---|---|---|
| DF長時間 | 40セル。保存best_configurationsのformula_group=ALL、PR対PF envelope。T0.8/1.2/1.6/2.0、元5精度 | 80armすべて同候補の両軸CX meanあり。PRは元EXPLORATION、PFは元決定論会計。点比較を主とする |
| DF B3 | 元40セル×B2/Full_PF/PF_envelopeの3基準。探索120比較と固定nominee確認120比較を分離 | 双方の両軸CX mean参照あり。B3はB3_EXPLORATION/B3_CONFIRMATION。基準側は元confirm_accountの確認overlay規則を保持。poolしない |
| Pauli | 現行H2/H4/H6各5精度のHybrid対純RTE、計15比較 | 全30armで同cohortのpaired CX rawがある。精度間で共有する価格標本を保持 |
| H8 Stage C | PR `7eb0a3d691b94089969b` CONFIRMATION n9対Stage B固定PF `1425ae6ed83fd9098b39` | Stage C trialsとStage B deterministic priceの両軸CX。T1.6、DF28、PR prefix8/q1/R32/K4/m5、PF prefix15/q1/m5。元fixed N |
| H4 DF–Pauli旧比較 | 同じ固定実装間の比較 | central系paired CXはあるがGenericPauli/DF側の同policy価格coverageは未確立。欠測armはDATA_UNAVAILABLE |
| P3 | 固定matched-partitionの保存Stage4会計 | H4 DF10、共有Gaussian state・partition A、q8/r4/K4、T0.8、仮想ρ0等の元条件を保持。既存CX会計を引用し、新規G_depthは作らない |

DFのPF endpointは元cost_forがcohortをDETERMINISTICと記す一方、価格ファイルの取得cohortがEXPLORATIONの場合がある。これは同じ決定論candidateのn1価格である。manifestにendpoint側cohortとprice_record_cohortの両方を保存した。PRでこの正規化やcohortの置換を行わない。

Pauliの具体的固定：

| 分子 | Hybrid | 純RTE | 固定cohortと標本数 |
|---|---|---|---|
| H2 | 全5精度 `PR_p0001_q1_r4_K4` | .05/.01は `PR_p0000_q1_r4_K2`、他3精度は `PR_p0000_q2_r2_K4` | confirmation、Hybrid96 / RTE384 |
| H4 | .05は `p036_q2_r8_K4`、他4精度は `p001_q8_r8_K4` | .05/.01は `p000_q8_r8_K2`、他3精度は `p000_q8_r8_K4` | confirmation、Hybrid48 / RTE384 |
| H6 | 全5精度 `PR_p0078_q8_r32_K2` | 全5精度 `H6_p000_q8_r64_K2` | confirmation、Hybrid64 / RTE48 |

旧latest15のH6 r8行を使わず、現行統合資料が指定するE6のr32行を固定した。H2/H4の確認後minimum、逐次延長、H6の探索nomineeと確認後minimumの一致をそのまま記録する。15精度行を15の独立実験として扱わない。

Pauliは各保存H/state/reference、JW、T0.8、S2、14/184/918 terms、元controlとfull Hadamard wrapper、状態準備除外/P0、仮想ρ0という比較scopeを保持する。H4旧DF–Pauli比較やP3の異なるstate/control/partitionへこの条件を流用しない。元compiler profile/axis/candidate fingerprintを保存参照で検査し、異なる条件を結合しない。価格は同じ保存compiled回路のCX欄を使用し、MODEL補完・RZ換算を行わない。条件が明記されないarmは推測せずUNRESOLVEDとしてSTOPする。

## B1：承認対象の演算と判定

`G_CX=N_cos*C_cos,CX + N_sin*C_sin,CX`。Nはmanifestの保存値を整数のまま固定する。既存G_RZの再現照合、新しいG_CX、比較差・比・方向を求める。RZ保存値照合はexact有理数がある場合はexactに行い、保存meanしかない場合は上記binary64 guardと元保存表現精度を明記する。照合不一致を許容するためにNやcohortを変えない。

- raw integer pricesはFractionでmean・paired covarianceを計算し、元 `paired` / `cost_for` を再利用する。精度間で同一価格をcacheする。
- 保存meanだけならそのdecimal literalの有理数として固定Nを掛け、保存平均に対する点比較と明記する。丸められたmeanの有理数表現を元trajectoryの厳密meanと呼ばない。
- CX SEが未保存でrawもない場合は **UNCERTAINTY_UNRESOLVED**。SE=0、RZ SEの転用、別cohortの代入を行わない。
- paired rawのある場合は同indexのcos/sin共分散を含める。n<2や必要な共分散の欠測では不確かさをUNKNOWNとする。保存された独立cohortの根拠を確認できる比較だけで差のSEを合成する。独立性が不明なら点方向と不確かさを分離する。
- 元の独立cohort根拠がある場合、差±1.96 SEを記述的な区間として表示できる。選抜・逐次延長・rare-order未観測・PRNG独立性未証明を含むため、母集団tail保証やwinner保証を付けない。zero observed varianceは観測事実として記し、母分散0と解釈しない。
- RZとCXの点差の符号が同じなら **DIRECTION_PRESERVED**、逆なら **DIRECTION_REVERSED**。uncertainty statusは別列とし、点方向と同時にUNCERTAINTY_UNRESOLVEDを表示できる。
- **POINT_DIFFERENCE_SMALL** は事前の実用同等性marginがないため、この範囲ではexact point tieに限る。近い比や区間が0を跨ぐ場合も数値を示すが、実用上の同等性は認定しない。区間が0を跨ぐなら順位の保証はUNCERTAINTY_UNRESOLVED。
- arm/metric/axis/Nの欠測は **DATA_UNAVAILABLE** として該当比較をSTOPする。数値0と欠測を区別する。

追加の新規指標はCXのみ。保存depth/RZ-depth/CX-depth/sizeは存在・定義を参考整理する。CX×shotは回路資源指標であり、runtime/FT T数/総物理資源へ換算しない。

## 未取得・未確認とSTOP

1. DF長時間/B3のCX paired統計を再現するraw coverageは今回の取得では確立していない。保存meanでの点比較を実施し、SEはUNKNOWNとする。元研究のRZ SEは保持する。
2. H4旧DF–Pauli比較のGenericPauli/DF側non-RZ価格coverageが不足している。新compileへ進まずDATA_UNAVAILABLEを報告する。
3. Pauli/H8のraw存在・field・candidate/cohort/index数は確認した。実CX価格の平均・分散・元RZ再現は未評価であり、承認後の検査対象である。価格archive内のQPYをdecodeしていない。
4. SOURCE/hash、H/state/reference、compiler/control/wrapper、candidate/cohort/axis、固定Nのいずれかが不一致なら該当入力群をSTOPし、未影響の固定比較のみ続ける。exact log B不再現・有限値異常は保全して報告する。
5. 追加sampling、RNG replay、build/compile/transpile、QPY decode、Hamiltonian/state/reference生成、候補再選択、N最適化、B2-b、bias/headroom fit、新指標、新詳細科学レビューが必要になった場合はその処理へ進まない。

**現在のmandatory STOP**：科学的再解析の明示承認を待つ。承認後は、この固定範囲の解析・検証・報告・機密検査・専用branchへの通常commit/push・remote SHA/本文/blob照合まで進め、再度mandatory STOPする。新しい承認証明書・grantを作らず、案X/MC1–MC4/論文化方針の判断をユーザーへ戻す。
