# PR-DF LD1：全13 roundの期待compiled RZ数 E[C] 評価計画

2026-11-15 JST。**計画のみ。科学数値の新解析・保存価格の再集計・fit/予測/CI・sampling・実回路build/compile・Q3再評価・N/G・winner判定は全て0。** 既存D2 branch `codex/d2-pr-df-ld1-m3-acquisition-20261115` を継続し、開始HEAD `063f2b3b6ea271a179a469be91252fa4e5a7d096` のm3/m4成果・SOURCE・旧STOPを保全する。

- **保存証拠で現在言えること**：全26軸にideal source lawのE_I[S]があり、m0/m3/m6の較正とm4の独立holdoutは元S/C・receiptまで追跡できる。m4の約−0.544502%は各軸4 path平均に対する保存済み予測差である。C最終受入済みでも、対象lawの正式E[C]は全26軸で未資格である。
- **正式なE[C]に進むための主要な不足**：BのQ3契約はideal finite-RTE lawを対象とする一方、残差はPCG64/CDFの指定有限cohortから得ている。SOURCEの生成構造対応は確認済みだが、実H4のlaw距離・実PRNG joint-law・価格への移送根拠は未取得。さらに有限pathの誤差、rare残差、共有係数の不確かさ、未観測roundへのモデル移送、必要価格精度/信頼水準が残る。
- **推奨する次の1作業**：既存の **F0「保存確率だけのcategorical-law診断」** を再利用して実H4に初めて適用する有限案を推奨する。1つの324-component表＋13 order表＝最大14診断、13個の条件付きevent/path TV、対象は26軸、新path0・新compile0。今回は実行しない。CのFull-H凍結やC2比較規則の完了を待つ依存はない。

正本：[26軸の差分索引](AXIS_EVIDENCE.json)／[26行CSV](AXIS_EVIDENCE.csv)／[次の有限案](NEXT_F0_PLAN.json)／[C HANDOFF](HANDOFF_C.md)／[metadata照合](CHECKS.json)。E[S]値・価格・係数を再生成せず、既存C26の各行へ差分のlocatorを加えた。

## 1. 固定対象・価格quantity・一次資料

保存linear H4/1.00 Å/STO-3G、tol-only DF10、PR-DF LD1/S2/Δ=.1/K2/h=.1、m0〜m12、元exactκと元H/state/snapshot、DF fragment/basis/qubit順を維持する。価格は各軸の**状態準備なし・測定付きfull controlled Hadamard wrapperのcompiled RZ count**。scalar c+s_Rを総Tに一回、新ZZ `CONTROLLED_ZZ_CX_CRZ_CX_V1`、symmetric-directional control、P0、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17を基準とする。構造Sは保存の元Accountingの量であり、compiler価格と同じ量へ改名しない。

| 略記 | 固定SOURCE・入口 | 本計画での用途 |
|---|---|---|
| B | [B_JOIN](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a82b0f0ec7f0f11d9981ddb2343fcd5378a87503/docs/research/pr_df_fullh_q3_fixed_results_20261115/C_RESULT_JOIN.json) | Q3済み候補・元/total ID・固定配分への対応 |
| B-law | [B_RESULT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a82b0f0ec7f0f11d9981ddb2343fcd5378a87503/docs/research/pr_df_fullh_q3_fixed_results_20261115/run01/PR_DF_LD1/RESULT.json) | statistical_contract.simulation_policy.target_law と元/total law_binding |
| C26 | [C26](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c99d511e2422b8f250490b11054f754a81e536c5/docs/research/pr_df_ld1_q3_price_admission_20261115/ADMISSION.json) | 既存26軸表の正本。構造とP1/P2・元receiptを再利用 |
| C手順 | [C_HANDOFF](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8eef0ad7a9b04e04d766f5e05b3b32eee50a794e/docs/research/pr_df_ld1_q3_price_admission_20261115/HANDOFF.md) | 8eef0ad…の引渡しがc99d511…受入索引を指定 |
| S26 | [ES](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f/docs/research/h4_structural_schedule_results_20261113/run01/COVERAGE.json) | E_I[S]と数値区間。新集計なし |
| M3 | [M3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2592d442ae8f36f71e42820a26e847b1fa38b59d/docs/research/d_pr_df_ld1_m3_results_20261115/C_HANDOFF.json) | 較正4 path×X/Y、元生成SOURCE/receipt |
| F | [FREEZE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cac53da1a4a8b2228245593c74c1ff9ec1f5212/docs/research/pr_df_ld1_residual_freeze_20261115/run01/FROZEN_MODELS.json) | 軸別凍結モデルとm4観測前予測のみ |
| F-input | [FIT_INPUTS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d83737b6d5241962ce39c81d39c56af3506c0cdd/docs/research/pr_df_ld1_residual_freeze_20261115/INPUTS.json) | Cがfit前に受入れた24組の元S/Cへの入口 |
| M4-D | [M4_D](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/063f2b3b6ea271a179a469be91252fa4e5a7d096/docs/research/d_pr_df_ld1_m4_holdout_results_20261115/C_HANDOFF.json) | 4 pathの独立holdoutと保存比較 |
| M4-C | [M4_C](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9edfa5d3201f22c23dd77569d102801be80dacf3/docs/research/pr_df_ld1_m4_holdout_admission_20261115/ADMISSION.json) | 9edfa5d…の最終consumer受入、8 receipt・2比較 |
| Law | [LAW_REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d72dc4e6228bb380f4d1d000e5e030794fce7164/docs/research/finite_k_sampler_law_alignment_20261113/REPORT.md) | 既存SOURCE対応、未実行F0、解決済み/未解決の境界 |
| Law数式 | [LAW_MATH](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d72dc4e6228bb380f4d1d000e5e030794fce7164/docs/research/finite_k_sampler_law_alignment_20261113/MATH.md) | P/Q/F/G、条件付きTVと期待値移送の式 |
| 旧F0 | [OLD_F0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d72dc4e6228bb380f4d1d000e5e030794fce7164/docs/research/finite_k_sampler_law_alignment_20261113/SCIENCE_PLAN.json) | 未採択・未実行の既存有限案 |
| F0 helper | [F0_HELPER](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d72dc4e6228bb380f4d1d000e5e030794fce7164/src/trottertracks/finite_k_sampler_law.py) | 人工12＋既存toy5件の記録がある既存実装 |

元価格configuration `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`、B/Cの元configuration `a4dda8f7e3fc07666ca2aabf634917e8bced14eb62151eb629ef23c0cce21d3d`、total configuration `9c3d392ec5c5ad373aaa1d96a8ba0828ddd268e6fef35da229029beec0229e8b` はC26が物理対応を受入済み。各軸の元価格canonical/元decoder/total decoder IDを別欄に保持した。新IDを古いreceiptへ上書きせず、Q3を価格の資格と取り違えない。

## 2. どのlawの期待値か

### 2.1 保存契約から確定する範囲

| 対象 | 固定されたlaw・SOURCE | 現時点の意味 |
|---|---|---|
| 今回BのQ3と同じ条件で総資源を比較する場合 | B-law `record.deterministic_input.statistical_contract.simulation_policy.target_law = ideal_finite_RTE_law`。各軸の元/total lawはC26 `original_law/total_law` からB保存RESULTへ戻る | 現契約を保つ計画の目標はI。actual sampler Qへの変更を暗黙に採択しない |
| 保存E[S] | S26 `expectation=IDEAL_FINITE_K_SOURCE_LAW`、kernel SOURCEはm0/m6が19b95d8…、他11 roundがde2427d… | 保存binary64 raw weightの厳密正規化比を含む数値enclosure。PCG64/CDFの期待値や統計CIではない |
| P1/P2/m3/m4のevent列 | carrier `155529e20a8fac39e00786c40b57830b71203c4f:scripts/h46_long_time_native.py:sample_requests` → provider `c31ef424175c02379eafebe8fc6c7548c1d9aabe` の `df_partial_s2` → `df_rte_circuit` → `rte.iter_sample_rte_events` | order/componentをNumPy1.26.4のchoiceで選ぶPCG64/CDF。outerごとにseedでGeneratorをreset。X/Yは同じevent列 |

B-lawの `sampler_law_transfer_certified=false`、`sufficient_N=null`、`G_qualified=null` を保持する。固定USER_SUPPLIED shot配分へのQ3条件付きモデル閾値適合は、実samplerへの移送や十分Nの認定ではない。

既存Law調査が生成経路・同一support/τ/phase/metadataを調べており、Bのlaw SOURCE `71ba49bb28b295aa28e51101bc3b1bf07d500767:src/trotterlib/rte.py` とprovider c31efの同fileは同じGit blob `53601b7c524653f5e1966b3955b9f51373e54002`。今回も一致を静的に確認した。**同じSOURCEであることと、CDF・有限乱数を含む実列のlawが同じであることは別である。** この確認をやり直す科学runは不要。

Iを厳密な解析実数のτ/libm重みまで含むlawと解釈する場合、その解析入力と保存binary64重みとの関係も別に必要。ここでのIは保存構造契約のideal source lawを指す。既存law_bindingは改名しない。Bの理想law宣言と保存数値評価を、解析実数の形式証明へ昇格しない。

Qにも区別がある。choice入力を正規化したcategorical分布、丸めCDFに独立な一様53-bit乱数を入れるモデル、実PCG64のjoint stream、固定seed表の決定論的cohortを同一視しない。固定cohort自体の平均は記述量として定義済みだが、母平均のQをいうならseed ensemble又は採用する確率モデルが必要である。seed namespace非重複はIID証明ではない。

### 2.2 Iをtargetにする場合

同じlaw Lと同じpath関数S/Cについて、D=C−Sとすれば E_L[C]=E_L[S]+E_L[D]。構造Sを変える必要はない。

仮にQの残差母平均が評価できても、Iに必要なのは E_I[D]。mixed値 M=E_I[S]+D_hat_Q の誤差は、

```text
M − E_I[C] = (D_hat_Q − E_Q[D]) + (E_Q[D] − E_I[D]).
```

第1項には有限path・推定/残差モデルの誤差、第2項にはlaw移送がある。現在はどちらも価格資格を支える評価が揃っていない。m4比較一件はこの二項を分離しない。

Q残差をIへ使うには、採用した確率モデルでlaw同一と示す、又は全path law距離と全supportのDの範囲で移送を抑える等の根拠が要る。既存Law数式の条件付き関係は `|E_I[D]−E_Q[D]| ≤ osc(D)·TV(I,Q)`。観測min/max・標本分散・E[S]をfull-supportのoscillationとして代用しない。SOURCE同一・K0の確率factorization誤差・truncation bound/normalization Bはその代替ではない。重要度重み付けも、実Qとsupport/尤度比の根拠なしには適用しない。現在のI→実Q価格systematicはnullのまま。

### 2.3 Qをtargetにする場合と、構造量を使う利点

同じmixed値のQ-target誤差は、

```text
M − E_Q[C] = (D_hat_Q − E_Q[D]) + (E_I[S] − E_Q[S]).
```

従って、保存E_I[S]をE_Q[S]へ無条件に代入できない。Qの構造平均又はそのlaw補正/上界が必要。**同じQでCを直接観測すれば、この構造law置換の項は生じない**が、有限標本/rare/信頼性と未観測roundの問題は残る。指定固定cohort平均をそのままE_Q[C]と呼ぶこともしない。

構造量は、同pathのS/C pairing、compiler由来残差の切分け、安価な構造期待値の利用、条件が合うときの制御変量として役立つ。E_Q[S]が既知なら E_Q[S]+平均(C−S) は直接C平均より分散を小さくできる可能性があるが、実際に小さくなることと精度は別途確認する。E_Q[S]の代わりに同じ標本の平均Sを使うだけなら、平均S＋平均(C−S)=平均Cであり、この書換えだけによる分散削減はない。今回は新しい分散・共分散を算出していない。

現在のB契約を維持するI-target計画には、I/Qの目標をもう一度ゼロから選ぶ必要はない。将来の主張を実装QのGへ変えるなら、価格側の置換だけでは足りず、Bのsignal/Q3と同じlawへの接続が必要で、別の科学判断となる。

## 3. 全26軸の差分：既存C表へ追加するもの

全行でE_I[S]と数値enclosureは保存済み、対象lawの正式E[C]は未資格。本文はX/Yが同状態のroundをまとめ、[AXIS_EVIDENCE.json](AXIS_EVIDENCE.json)と[CSV](AXIS_EVIDENCE.csv)は26軸を個別に保持する。R=255/510/1020/2040/4080/8160を2冪へ丸めない。

| m | q / R | 元physical ID | X/Y各軸の有限compiled証拠・用途 | 残るモデル検査 |
|---|---|---|---|---|
| 0 | 1 / 2 | `53e00d8b07d1547d19a7` | P2 fresh4：較正／P1 2：参照／replay1：参照 | 較正3点の一つ、独立検証ではない |
| 1 | 2 / 4 | `c00e00dc1e9c414fb7d9` | 該当cohortなし | 当該点の観測・凍結予測なし |
| 2 | 4 / 8 | `f2fcd66b1ef9ae2da5df` | 該当cohortなし | 当該点の観測・凍結予測なし |
| 3 | 8 / 16 | `045843b2429bece6f29b` | D2 fresh4：較正 | 較正3点の一つ、独立検証ではない |
| 4 | 16 / 32 | `3df0cfd3356c358aeda8` | D2 fresh4：独立holdout・C受入済み | 中間の未使用1点、記述的比較のみ |
| 5 | 32 / 64 | `57d641a831d76360b570` | 該当cohortなし | 当該点の観測・凍結予測なし |
| 6 | 64 / 128 | `2a8c6718220eec59d419` | P2 fresh4：較正／P1 2：参照／replay1：参照 | 較正3点の一つ、独立検証ではない |
| 7 | 128 / 255 | `b42287268e5249dfb4d1` | 該当cohortなし | 長q/R移送未検証 |
| 8 | 256 / 510 | `224aef0a626068b43184` | 該当cohortなし | 長q/R移送未検証 |
| 9 | 512 / 1020 | `06a78c7f9856f373336f` | 該当cohortなし | 長q/R移送未検証 |
| 10 | 1024 / 2040 | `b872ad1347a35c8f4315` | 該当cohortなし | 長q/R移送未検証 |
| 11 | 2048 / 4080 | `d48210cc27c599e268c2` | 該当cohortなし | 長q/R移送未検証 |
| 12 | 4096 / 8160 | `b55c521fdce6a448dbcb` | 該当cohortなし | 長q/R移送未検証 |

- m0/m6：P2 freshの各4 pathは、元role DEVELOPMENTを保ち、凍結モデル用途をCALIBRATIONとした。cohort `D_P2_DIRECT_PILOT_V1`。P1各2 path（`D_P1_EFFICIENT_CZZ_20261113`）とP2 replay各1 pathは参照のまま、較正へpoolしない。replayは保存eventの再利用で独立標本の追加ではない。
- m3：`D_PR_DF_LD1_M3_CAL_NORMAL_20261115_V1` の4 path×X/Y。生成SOURCE2f423bef…、raw4031dbab…をM3から参照。Cのfit INPUTSが受入れた同path S/Cへ戻る。
- m4：`D_PR_DF_LD1_M4_HOLDOUT_NORMAL_20261115_V1` の4 path×X/Y。生成SOURCE7bdfc6f6…、raw3042e6e6…、M4-D公開063f2b3b…、最終C受入9edfa5d3…。**独立holdoutのまま保持し、較正へ戻さない。**
- 凍結モデルは軸別 `theta0 + thetaq*q + thetaL*qR`。根拠はm0/m3/m6の3点、各軸rank3・係数3・lack_of_fit_DOF=0という保存結果。較正点での一致は独立なモデル検証ではない。実際に保存された新点予測はm4の二件のみで、他roundの予測は今回作成していない。

各軸の `base_C_row`→構造source/law、`legacy_C_path_refs`→P1/P2の元receipt、`D2_path_refs`→m3/m4の元request/seed/digest/軸別receipt、`accepted_model_input_refs`→Cの較正受入、`C_holdout_receipts/comparison`→m4最終受入をたどれる。元価格canonical IDとtotal decoder IDは別に保存した。X/Yは一つのpathの対応応答で、標本数を二倍にしない。元gateは取得/identity/実装の範囲で、full-wrapper statevector検証=false、数値action gateは元P2/m3/m4の未実施範囲を維持する。

### m4の結果が検査したこと／残したこと

M4-Cが受け入れた保存比較は両軸でP=217164876899444503/1683627180032、O=259385/2、符号付き相対差=−1188941146855657/218353818046300160（D2表示約−0.544502%）。これは引用であり再計算ではない。

検査したのは、観測前に凍結したI構造＋Q較正残差の予測が、未使用のm4有限cohort平均にどれだけ合ったかという**中間の1物理点の記述的差**である。取得/receiptの対応と開封順序も確認済み。有限pathのばらつき、I/Q移送差、model誤差を別々には検査していない。CIや必要精度の達成、rare寄与、m7〜m12への長q/R移送も認定していない。X/Y一致は独立した二つのcohort再現ではない。

比較状態 `DESCRIPTIVE_COMPLETE` と `model_verdict=NOT_ASSESSED_NO_ADOPTED_TOLERANCE` を保持する。凍結JSONに残る観測前statusは履歴として保全し、後日の記述的受入を別に結び付ける。許容差の事後設定やm4の再較正はしない。

## 4. 総資源Gに使うための残作業と精度

目標law Lと固定配分Nに対する予定量は `G_L = sum_(m,a) N_(m,a) E_L[C_(m,a)]`。Bの固定配分を参照するが、今回その積和・順位・寄与率を計算していない。Q3で閾値に適合したUSER_SUPPLIED配分と、十分Nは区別する。

26軸それぞれの母平均に同時に有効な絶対誤差幅epsilon_(m,a)があれば、G誤差を `sum N_(m,a)*epsilon_(m,a)` で管理する設計が可能である。これは設計上の式であり、今回の上界又はCIではない。必要な精度を26軸一律の便宜的な割合で決めるより、比較目的の許容G誤差/候補差と固定Nを結び付ける。**必要精度・信頼水準・許容誤差は未採択**で、PF2%、他方式の閾値、95%、旧P2候補n/設計floorを流用しない。

残る誤差は、保存構造の数値enclosure、有限cohort残差の推定誤差、law移送、未観測rare、共有係数/model誤差、未観測roundへの移送を分ける。rareは既に全support boundや推定誤差へ含めたなら再度加算せず、範囲を明示する。X/Yは同path、各roundは共通係数を使うため、独立と仮定した二乗和でtotal CIを作らない。同時confidenceを必要とするなら、その水準と配分方法を別途決める。構造区間の狭さもm4の小さい観測差も、これらを埋めない。

E[C]だけが得られてもground/actual-law/十分Nの不足は別に残る。従って現時点で全26軸をGの正式価格入力へ渡せず、未知価格を0に置いた総資源やwinnerを出さない。

## 5. 次の1作業：既存F0の実H4有限診断（未採択・未実行）

### 選択理由

| 選択肢 | 現時点の判断 |
|---|---|
| 対象lawの文言だけを再確認 | 現B契約がIであることは本計画で確認済み。これだけの別作業は不要 |
| **既存F0を保存確率へ適用** | **推奨**。未取得の正規化/CDF/格子差を全roundで評価し、law接続の次の分岐を具体化。新compileなし |
| 短roundの追加normal path | 有限cohortの情報は増えるがI/Q接続は解決せず、必要nの精度基準も未採択。今回は件数を便宜的に置かない |
| 長roundの独立holdout | 移送検証として必要な候補だが、母平均精度の資格を単一少数pathに持たせられない。現時点でround/nを先に固定しない |

旧P2にはm7 fresh2 path・最大4 compileという**負荷診断**の案があるが、平均価格認定用のnではない。今回のE[C]の次案として自動採択しない。m12等の寄与が最大だという価格×Nのランキングも未計算であり、断定しない。

### 有限scopeと再利用

[NEXT_F0_PLAN.json](NEXT_F0_PLAN.json)を次の指示の範囲指定に使える。D2が本branchで担当可能で、Cの別worktreeや共通guardを変更する必要はない。

| 項目 | 提案 |
|---|---|
| 対象 | 同一preparation・324 component表を一回、m0〜m12の各2-order分布を13回。X/Y共有のlaw診断を26軸へ参照 |
| law | 保存ideal source weight比→choice入力→丸めCDF→一様53-bit格子の条件付きモデル。実PCG64 Q全体とは別 |
| 新規量 | 最大14 categorical audit、13 event TV・13 path TV。価格/期待Sは計算しない |
| path数／新規compile上限 | **0 path／0 compile**。sampling・event列生成・実H4 build・simulation 0 |
| 入力 | a637ed1b… `preparation.components` の `probability/probability_hex`（順序維持）、同RESULTの `laws[0/1]`（m0/m6）、9d95d3ea… `run01/LAWS.json#laws[0..10]`（残11round） |
| 参照field | 各lawの元 `order_weights/order_probabilities/tau/tau_hex/event_normalization/event_count`。全13selectorとphysical IDをNEXT_F0_PLANに固定 |
| 実装 | d72dc4e6…の `finite_k_sampler_law.py` と元carrierの `cdf_law` を再利用。新しいsamplerは作らない。最小readerは後続許可内で作成・人工経路確認後、実行前にSOURCE/入力をcommit固定 |
| 観測量 | P/Q/F/Gのexact正規化・CDF・格子TV、正/零support差、order2 mass、条件付きK2 event/path TV、有用性と未解決scope |
| 出力案 | `docs/research/d_pr_df_ld1_f0_law_diagnostic_20261115/run01`。現在未使用。旧結果を上書きしない |

計算量は保存support表に依存し、巨大なpath/event列を生成しない既存helperの経路を使う。NumPy1.26.4のCDFとFractionを用い、追加benchmarkや実compileによる性能測定を含めない。進捗は14診断の開始/終了/所要時間と終了codeで記録する。Python3.11.0rc1では既知不具合のあるfaulthandler定期dumpを使わない。

helper内のQは単一categoricalのchoice入力、Gは格子分布であり、本報告の実path law Q・総資源Gとは同じ記号でも別の対象。実PCG64のjoint-law/IID証明、解析実数入力誤差、価格oscillation、E[C]・CIは結果に自動付与しない。

### 結果から判断することとSTOP

F0で初めて、実H4の保存確率について正規化/CDF/格子差がどうなるか、rare supportが変わるか、条件付きpath boundが短/長roundで有用かを判断できる。これにより、単純なlaw距離＋価格rangeで進めるか、observableに即したcoupling/構造kernel又は別の確率モデルが必要かを次に提案できる。ただしrange未取得なら価格へのshift boundはnull。条件付きgrid結果が小さくてもactual QやE[C]の資格にはしない。

保存入力欠落・SOURCE/物理ID/順序/hex不一致、同等job/receiptの出所不明、CDF mass/単調性/nonfinite異常で該当分を停止する。元確率の再生成・補正再正規化・guard緩和・無断retryを行わない。大きい/無効な上界もそのまま報告し、追加path・effective-grid E[S]・fitへ拡張しない。有限F0公開後にmandatory STOP。

## 6. §10.5差分照合と残るユーザー判断

再利用：C26の元configuration/別canonical ID、同physical候補、P1/P2 fresh/replayの元trial/receipt、D2 m3較正、m4独立holdoutとC最終受入、旧F0のSOURCE/helper/人工記録を確認した。P1/replay/旧loweringの別domain価格は新較正や母平均へpoolしない。既存17件の人工/toy記録は参照のみで再実行0。

現在確認した関連公開ref：本branch063f2b3b…、C m5-m4受入branch81596ce7…、C r-variable branch3a8b37f1…、D residual branch26d4f23e…、D price bridge6e110c08…。9edfa5d3…以後のC差分はFull-H D7凍結/HANDOFF等、前回D bridge以後の差分は決定論PF取得/holdout等であり、PR-DFの追加価格又は実H4 F0の結果とは同等でない。値は開いて再解析せず、差分path/metadataで区別した。既知のF0系列は計画/helper/人工記録のみ。ローカルの関連Python script名では同一取得/law診断jobを確認せず、提案出力先も未使用だった。確認範囲と実refはNEXT_F0_PLAN/CHECKSに保存した。

外部のアクセス不能jobまで不存在とは主張しない。将来の実行直前は既存照合を再利用し、同条件の新診断・job・SOURCE/入力・出力先の差分だけを確認する。同等の受入可能な診断があれば14件の新規候補から減らす。既取得か判断できない分は保留し、資料未取得を未実施とみなして投入しない。他担当の通常更新を禁止するHEAD一括guardは設けない。

ユーザー判断として残るのは、(1)この保存科学数値のF0新解析の明示実行許可、(2)将来の正式E[C]/GでPCG64をどの確率モデル/保証水準で扱うか、(3)その比較に必要な価格精度・同時信頼性、(4)希望する場合だけQ-targetへの契約変更である。F0の条件付き診断値を報告するために仮の価格許容差を置く必要はない。対象law Iが既存Q3に書かれていること自体の再承認は求めない。

## 7. ロードマップへの影響と今回の終了点

26軸について「構造値が揃った／有限価格がある／較正／独立holdout／正式E[C]」の対応を整理し、m4のC受入待ちは解消済みとして反映した。新たな価格資格や長roundの検証を得たわけではない。

推奨順序は、**現在のI契約を維持 → 既存F0の実H4診断を別指示で実行 → 結果と正式確率モデル/価格精度に沿って不足証拠とn/roundを設計 → 必要な較正とは別の独立検証 → E[C]資格をC/Bへ供給 → 同じlaw/誤差契約と配分で将来Gを比較**。これは計画であり、後続実行を採択していない。Full-H凍結/C2規則を待たずにPR-DFのlaw課題を進められる。

公開前の静的検査では26軸の物理ID/役割、929件のlocator参照、元receipt/requestへの136参照と文書リンクを確認した。確率又は価格の算術、研究moduleのimport・科学testは行っていない。m3/m4の成果、scripts/src/tests、既存manifest項目が無変更であることを確認した（[CHECKS](CHECKS.json)）。このPASSはmetadata照合の状態でありE[C]の資格ではない。

ロードマップ本体は統合担当が更新する。本branchでは計画・差分索引と引渡しのみを追加し、元SOURCE/raw/係数・m4独立比較/旧STOPを保全する。計画をcommit/pushし、remote SHAと公開成果物一致を確認して **mandatory STOP**。F0・追加path・compile・fit/予測/CI・Q3/N/Gへ進まない。
