# B1/B2の最小の次検証案：未実行

2026-11-06 JST。以下は実行候補であり、今回の許可では開始しない。新しい科学的な比・平均・fit・N/C/G/SE・順位・normalization評価はすべて未実行。既存のmandatory STOPを解除しない。

## 推奨順序

1. MC2の二乗構造には直接先行があることを踏まえ、ユーザーが主張の差分と研究上の目的を選ぶ。
2. B2のnormalization部分だけを、固定保存candidateで検査する。実装と漸近近似の差を先に分ける。
3. B1をCX中心に、既存RZ nomineeを固定した感度比較として行う。metric別の候補再選択は別estimandとする。
4. Gの機構や最適shot overheadへの拡張は、2/3で必要性が残る場合に限り別範囲として扱う。

いずれも保存値再解析の明示承認が必要。compile・samplingの許可は含めない。

## B2-a：正規化の式と保存値

**目的**：固定finite-K定義の再現、整数配分、小τ補正を分離する。新しい理論の発見やGの因果説明を目的にしない。

固定入力候補：長時間carrier `155529e20a8fac39e00786c40b57830b71203c4f` の `same_q_prefix_R32_K4.csv`、candidate/endpoint/finite-mean保存record、scientific_provenance。B3 carrier `ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed` の `signed_schedules.json.gz`、matched controls、signal records、provenance。最初は同じH/state、T、prefix、q、R_outer、KのS2/Y4/m5対応行。B2とB3でλ_Rを共用しない。

再利用候補：provider `rte._paired_order_weight` / `finite_rte_distribution`、`h46_new2_core.allocation`、`h46_m5_plan.weights/schedule`、`h46_pr_pilot_repair.canonical_tau`。科学runner全体をimportせず、必要な純粋関数の依存を確認した専用adapterで使う。新しいnormalizerや別の比例配分を実装して元計算と同じと扱わない。

承認後に必要となる科学的演算：

- 保存λ_R・signed time・整数r_iから正確なfinite-K log Bを評価し、保存値と照合。
- 同じ整数r_iの二次leading式と照合。
- 連続比例配分のA²式と照合し、整数配分効果とfinite-τ残差を別に表示。
- K0を扱う場合だけleading係数1/2を別扱い。主取得域K2/4/6へ混ぜない。

λ_Rの原literalが公開input/recordに見つからない対象は、保存normalizationから逆推定せず、同サーバー上で参照許可のある固定metadataを回収するか、欠測としてSTOPする。別サーバーへの接続は許可しない。

判定条件：正確式と保存log Bの差は、保存精度、浮動小数点演算順、既存log_B marginの意味を確認して事前に定める。漸近式は最大|τ_i|と高次残差の大きさで評価し、Claudeの「約10%」を固定受入閾値にしない。近似が外れるなら適用域を狭める。元SOURCEや結果を変更して合わせない。

出力候補：candidate/cohortと入力literalを付けたexact/leading/continuousの比較表、残差の内訳、適用不能行。追加の科学的計算を要しないmetadata検査だけを技術確認とする。

## B2-b：Gと時間依存へ進める場合

既存の同条件log B整合はGの因果説明を与えない。q/prefix/R/K、PF bias、finite-K bias、headroom、N、C、価格選抜が同時に変わるからである。

候補は、保存 `C_N_G_decomposition.csv` とmatched controls、B3固定nominee、候補別endpoint/price/cohort。保存Nを固定するか、元のshot ruleで再評価するかを承認範囲に明示する。後者は新しいN計算であり、B2-aへ暗黙に含めない。

再利用候補：`d0_frozen_shot_rule_v1.u_aware_shots`、`run_pf_common_run.qualify_record`、`h46_long_time_report.cost_for`、`h246_saved_cost_evaluation.paired`、B3 reportのbinding。reportのcarrier版と実行版が異なるため、最終集計に対応する版を先に固定する。report全体の起動は選抜・再集計・描画を伴うため不要。

`B*²≈e`などの最適overheadは、bias/headroom固定、連続event数、線形C0+c n、Kや境界価格を固定した近似モデルの予測。保存最良点をそれに当てはめるだけでは機構確定にならない。fitや新しい最適値・予測値を作るなら独立の承認範囲とする。

結論が変わり得る条件：small-τ近似の適用不能、整数配分の支配、Cの非加法性、headroomの変化、metricや候補域の選択が観測差を主に支配する場合。結果は「整合的説明」「不整合」「未分離」に分け、一意な原因と認定しない。

## B1：保存副指標の感度

**目的**：RZで固定した主要比較の方向が、CX等でも保たれるかを調べる。異なるmetricで最良候補を探し直す作業は、この感度検査から分ける。

| 対象 | 最小入力・可能な範囲 | 不足／compile判断 |
|---|---|---|
| DF H4/H6長時間・B3 | 六指標の保存mean、candidate/cohort、endpointのN、必要時raw trial | 点比較は保存値で可能な対象がある。CX等のpaired SEはRZ統計を転用せず、同trajectoryのraw価格が必要 |
| Pauli H2/H4/H6 | 公開archive内のprice result、H4 trajectory_prices.csv、最新候補とcohort対応、保存N | 旧cohortの価格を最新選抜へ混ぜない。H6純RTE等の別extension側rawの追加回収が必要な場合あり |
| H8 Stage C | confirmation.jsonの各trial六指標と固定N、Stage B exploration.json.gzの固定PF IDの六指標 | 新しいPR meanを計算する段階は科学再解析。PFはStage Bの保存価格のまま。新PF確認samplingは不要 |
| H4 DF–Pauli固定実装 | conditional/generic系列とcentral系列の保存raw・統計、同case/axisのN | 比較task・control・wrapperを保持。最新各方式の最良候補や表現単独の因果比較に読み替えない |
| P3 Stage4 | Stage3/run04 prices/statistics＋Stage4 axes/accounting | RZ/CX/depthは既存会計も保存。既存結果の転記だけなら追加再解析不要。rz_depth/cx_depth/sizeはStage3保存価格があり、Gを作るには別承認が必要 |

再利用候補：`rte_compiled_cost`のmetric名称・compiler hash・既存Qiskit定義、長時間nativeのtrial識別、Pauliのprice result schema、P3 `matched_partition_stage4`のmetric別paired会計。`transpile_and_measure_cost`は再利用する定義を読むだけで、B1では呼ばない。保存CXに新compileは不要。欠測した副指標をRZから換算しない。

公平性条件：同H/state/reference/rank・fragment順・T・precision/allowance、同じcandidate・cohort/index・axis・compiler hash・control・full-wrapper scope。価格共有された5精度を独立標本として扱わない。独立確認と探索、確認後minと事前nomineeを分離する。cos/sinの共分散を保つ。SE未保存・raw未取得ならSEはUNKNOWNで、0にしない。

depthは保存Qiskit回路指標であり、ハードウェア実行時間ではない。N×depthを採用するなら新しい資源目的の意味を明示する。SXはP3の保存output_histogramとH4 centralの代表posttranspile_histogramに存在するが、全系列で独立統計が揃うとは限らない。非Clifford回転/T countは既存RZの別名にしない。

非Clifford角を調べる場合は、保存QPYのcohort全体coverage、πの倍数の判定規約、許容誤差、scalar/control位相、synthesis精度を先に固定する。QPY decodeからの新gate計数も別の科学再解析。DFの代表index0 QPYだけから全価格標本の平均を作れない。必要回路のQPYが不足する場合、compileの別承認が必要になる可能性はあるが、今回はcompileを推奨・実行していない。

## 共通STOPと不要な処理

別承認後も、入力やSOURCE不一致、candidate/cohortの曖昧な結合、必要rawの欠落、比較scopeの変更、未許可の新compile/samplingが必要となった時点で該当対象をSTOPする。元の結果・STOPを修正しない。新しいgrant発行や承認証明packetは作らない。

今回の目的には、Hamiltonian再生成、状態作用、RNG replay、旧jobのretry/resume、追加候補探索、全repository再棚卸し、図表再生成は不要。文献上の直接重複は追加計算なしで判断材料となる。指標をRZに限定した記述だけを採る場合、B1未実行という限界を残して先へ進む選択もユーザーにある。

出力を確認したユーザーが、B1/B2の範囲・受入条件・新規性の位置付け・案Xの採否を選ぶ。本提案自体は科学実行の許可ではない。
