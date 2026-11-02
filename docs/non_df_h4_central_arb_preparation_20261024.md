# H4 Pauli central-control：Arb包含bridgeの準備

**技術実装・人工検証・PF cosine非実行契約の準備を完了し、mandatory STOP。**
実H4のArb評価・状態作用・科学grant発行/消費・compile/sampling・N/G再解析はすべて0。
新bridge人工10、QPY人工6、既存sine人工9の**25テストPASS**。
3qubit・49,000ゲートの人工列では全3精度が既知の結果を包含した。
この検証は、実H4の局所ρ上界や正式certificateの受理ではない。

## 1. 基点と固定対象

専用branch：`codex/non-df-h4-central-arb-preparation-20261024`。
基点branch：`codex/non-df-h4-central-r1-sine-20261024`。
基点commit：`36e993e698953f24415f4e90c4cac199a693ee16`。
GitHub connectorから基点commitを実取得しSHAを確認した。
[取得記録](../artifacts/non_df_h4_central_arb_preparation/2026-10-24/github_base_verification.json)。
過去のpush失敗記録は時点履歴として保持する。

対象候補は既存のlinear H4、隣接1.00 Å、STO-3G、system8＋ancilla wire8、
Pauli-JW非identity184項/prefix27、S2、T=.8/q8/r4/K4。
DF/H2研究とは統合しない。今回はPauli compiled回路の局所評価基盤のみ。

| 固定資料 | SOURCE・保存先 |
|---|---|
| C | `e80d7510880f8db993b777c627bb8e95b05a070d`、`h4_ball_evidence.py` |
| D2 PF compiled | `74515d8eb4f073c40fc783681a3775dbca89e9b1`、D2 run01 PF output.qpy |
| R0 cosine | `84a326598b006384d13c7a354a3eec6881ee841c` |
| R1 order2 cosine | `ca9a1c0e2426565ba63cc8bfd0247f73339d95b8` |
| R1 sine | `198d4916f7f62b775d9f60020c42ffb07d4c1c89` |

R0/R1の6件の経験的診断・source/control/phase・state/ray・C-target/cache対応を参照した。
今回新たにdecodeした実QPYは**PF cosine compiled 1個だけ**。
QPY SHA=`8537e01152d7b9a169bcea2dc20f4ce107d560b3af39b47b326e37c3c309a77d`、
fingerprint=`ac65c4de7d57b254b5fced776a039dd36a76b35fed0a33c79eb5478cdc44a173`。
既存histogramと一致：RZ15,028・SX7,170・CX26,768、計48,966ゲート＋最終measurement1。
[構造検査](../artifacts/non_df_h4_central_arb_preparation/2026-10-24/structural_inspection.json)。
全パラメータとglobal phaseをexact binary64 rational列へ変換して保存した。
これは保存回路の構造表現であり、新しい回路・state・signalの生成ではない。

## 2. 実装と既存資産の再利用

[bridge](../src/trotterlib/h4_arb_bridge.py)は既存Cの`real/complex_ball/phase/dyadic/endpoints`をlazy importする。
QPY decode・native検査・fingerprint・signatureはR0の処理を再利用した。
標準Qiskit gate classも照合し、同じ名前だけを持つcustom gateは拒否する。
実QPY検査用[CLI](../scripts/inspect_h4_arb_qpy.py)には状態作用・compile・grant機能がない。
[再利用一覧](../artifacts/non_df_h4_central_arb_preparation/2026-10-24/reuse_matrix.json)。

既存DF局所RZ/XXPlusYY kernelは、native SX/CXを含む長いPauli vector伝播に直接適合しない。
この差分だけを新実装した。旧C/R0/R1コード、controller、科学SOURCEやgrantは変更・起動しない。
数値public APIと内部kernelの双方が**ARTIFICIAL fixture・合計2〜3qubit**を要求する。
保存H4 domainとn8入力はArbのロード前に拒否し、`execution_request`も常に拒否する。
実H4を有効化するには、別SOURCE/controllerと明示承認が必要。

## 3. 初期値から期待値までの包含設計

Arbは区間を外側へ丸める演算を提供し、Python floatを渡した場合は、そのbinary64値が対象となる。
今回も十進の意図値へ置換せず、Fraction/fmpqを経由する。
[python-flintの入力・精度の説明](https://python-flint.readthedocs.io/en/latest/general.html)、
[arbの端点・mid/rad API](https://python-flint.readthedocs.io/en/stable/arb.html)。

### raw state・phase・wire

- 各保存binary64成分をexact dyadicとして入力。64 bit以上なら初期成分はexact、初期L2半径は0。
- 初期vectorは`[raw, zeros]`。Qiskit little endian、ancillaは最上位bit。
- RZは`diag(exp(-iθ/2), exp(iθ/2))`。θは保存binary64のexact値。
- SXは`[[1+i,1-i],[1-i,1+i]]/2`。固有global phaseを落とさない。
- X/CXはexact permutation。CX引数はcontrol→target、closed control=1。非隣接wireも同じ規約。
- 回路global phaseもexp(iφ)として作用させる。relative phaseやancilla条件付き位相はnative列をそのまま保つ。
- cosine/sineのwrapperを作り直さず、保存列の最後のmeasureだけをZ縮約に置き換える。
  sineのSdg-before-final-Hによる`+Im`は、既存sine人工9件を再実行して確認した。
  binary64 πを厳密πに置換してwrapperの誤差を消すことはしない。
- raw rayを再正規化しない。分母νは同じraw成分のexact二乗和。
  出力normによる除算、probability clipping、ν=1の仮定は使わない。

### 長い列でのL2包含半径

各段階で真の有限encoded programのstate xに対し、exact dyadic center mと
`||x−m||₂ ≤ e`を保持する。native gate Gは数学的にunitaryである。
ArbでGmを成分ごとに包含し、そのexact midpointをm′とする。
各複素成分の実部/虚部radiusの二乗和から、外側丸めで`r ≥ ||Gm−m′||₂`を求めると、

`||Gx−m′||₂ ≤ ||G(x−m)||₂ + ||Gm−m′||₂ ≤ e+r`。

e+rも上端dyadicへ外側丸めする。X/CXはcenterのexact permutationなので新しいr=0。
midpointへの置換に伴う誤差を捨てず、各gateで全L2半径へ回収する。
これはstateの物理的再正規化ではない。矩形ballを独立に膨らませ続けるwrappingを避ける。

最終観測量Z_ancのoperator normは1なので、

`|x†Zx−m†Zm| ≤ 2||m||₂e + e²`。

centerの二次形式・norm・この転送量・νによる除算をすべてArbで包含し、exact rational端点を出力する。
ν区間が0を除外しない場合、非有限値、未対応gate、入力不一致はPASSにしない。
この包含対象は**固定raw rayとfinite encoded native programの理想ゲート作用**。
物理デバイス誤差、真のFCI stateとの差、一般compilerの正しさをここで保証するものではない。

## 4. C-targetとρの所有範囲

局所量は`ρ_local = B_saved |m_compiled − m_C_selected_ideal|`。
PFではB_saved=1で、Cはdeterministicな同一PF ideal targetを持つ。
R0が保存した[PF区間](../artifacts/non_df_h4_central_arb_preparation/2026-10-24/PF_C_target_interval_QUOTED.json)を
**区間全体として引用**し、経験的診断用midpointを保証値へ格上げしない。
この区間はCのraw `corrected_encoded_mean.real`を同じexact νで除した既存保存値。
C実装のraw inner productとR0のscalar変換SOURCEを照合した。νの二重除算やB_upへの置換をしない。
将来はoutwardな区間差と絶対値から局所ρ区間を得る。今回の実H4ρは引き続きnull。

PR0/PR1の保存complex C-targetは**経験的point値**であり、現在は包含区間ではない。
PRの個別targetを保証するには、固定joinのselected eventだけを使い、Cと同じchronological順に
forward blocks→各event→reverse blocks→native scalarを作用する別の包含経路が必要。
Pauli作用はbit permutationとexact ±1/±i、block角度はC retained coefficient×step_time/2のexact積、
signed rotation、productの順序と負符号、order phase、native scalarのexact和を保持する。
同じunitary L2半径法を用い、最後はraw inner productの包含と同一νで縮約できる設計。
PR1のaggregate phase `.26518225073344787`をPR0の`.2651822507334482`に置き換えない。
この個別PR enclosureのproduction実装・実H4評価は今回の成果物に含めない。
個別targetをPR law meanで代用せず、B_savedをtargetへ先に掛けてρで二重加算しない。
全lawρ・未観測event・費用母平均tailには別の証拠が必要。

## 5. 人工検証と精度

既存private環境を変更せず使用した。Arb側Python3.12/python-flint0.9.0、
QPY側Python3.11/Qiskit1.3.0。単一環境へ無理に依存を追加しない。
人工QPY→exact descriptor→別Arb環境のend-to-end検査も含む。

| 検証 | 結果 |
|---|---|
| 新Arb人工 | 10 PASS：既知SX/RZ値、SX²=X/SX⁴=I、CX向き、raw ν、global phase、誤入力拒否、ρ scale |
| 高精度の別数値経路 | 合計2/3qubitで既存C.embed/acb_matの256bit dense区間を64/96/128bitのvector結果が包含 |
| 新QPY人工 | 6 PASS：exact parameter、standard gate class、SHA/fingerprint、最終measurement、環境間結合 |
| 既存sine人工 | 9 PASS：+Im/Sdg、raw nonunit norm、selected cache等 |
| 長い人工列 | 3qubit、49,000 native gate、known identity、全3精度で既知signal=1を包含 |

長い人工列は`RZ(a) SX SX X RZ(-a) CX CX`を7,000回反復、a=.125、global phase=.3。
実H4からgate/angle/stateを抽出していない。人工非unit raw stateを使用する。

| bit | 人工signal区間幅 | wall秒（今回の観測） |
|---|---:|---:|
| 64 | 9.42654e-15 | 1.52 |
| 96 | 2.25642e-24 | 1.54 |
| 128 | 5.11760e-34 | 1.55 |

[全ログ](../artifacts/non_df_h4_central_arb_preparation/2026-10-24/artificial_tests.json)、
[長い列のexact端点・時間](../artifacts/non_df_h4_central_arb_preparation/2026-10-24/artificial_long_sequence.json)。
小さい人工系の時間からH4の実測時間・幅を確定しない。
64bit開始を支持する技術材料は得たが、実H4の区間幅・C-target floorは未検証。
異なるdense経路による検査も主agent本人の検証であり、別reviewerの独立reviewではない。
人工fixtureのQiskit APIとmetadataの日本語path quotingは準備中に修正し、人工検証/保全確認を再実行した。
[技術回復記録](../artifacts/non_df_h4_central_arb_preparation/2026-10-24/technical_recovery.json)。実H4の科学retryは0。

## 6. 最小PF cosine pilot：実行無効の契約案

[非実行契約](../artifacts/non_df_h4_central_arb_preparation/2026-10-24/pilot_contract_DRAFT_DISABLED.json)は
`execution_authorized=false / dispatch_enabled=false / grant_valid=false`。
科学runner SOURCE・grant・launch commandはnull。新科学grantを発行・消費していない。

- **1論理対象：PF cosine、保存compiled QPY 1個、既存raw rayと既存C-target区間のみ。**
- 64bitから開始。最大64→96→128bit、各精度のwalk試行1回、合計最大3状態作用。
  対応する局所残差scalar縮約も最大3。途中失敗も試行に数える。
- 新input QPY作用0、新C-target作用0、sine/PR/DF作用0、compile/sampling/N_G0。
- 保存ε_sig=.0005のcosine guardをそのまま引用：
  `10107730069649565297/3815672118158673332666368`（約2.6490038338323596e-6）。
- 幅1e-8は**表示・判定安定性の設計目標**。旧allowance・headroom・閾値を変更する条件ではない。
  guardの上下関係が決まれば、幅が1e-8を超えてもそこで終了する。
- finiteなρ上端≤保存guard、または下端>保存guardなら局所関係を記録してSTOP。
  budgetを跨ぐ場合/非有限である場合だけ、未使用の次精度へ進む案とする。
  128bit後、または保存C-target幅が支配する場合はINCONCLUSIVEでSTOP。
  新target・追加精度・同精度科学retryへ進まない。

1e-8を必須にすると、科学的判定に不要な高精度化を招く。
一方、判定がguard近傍なら、幅目標達成だけでは十分ではない。
人工列では1e-8は十分実現可能だったが、実H4達成の断定は保留する。
局所関係が決まっても正式certificateやLOCAL_BUDGET_METの自動受理、N/G更新はしない。

次段階には、ユーザーの対象/精度/call範囲の明示承認、別production SOURCE/controllerの固定、
新しく承認されたone-shot grant、native exit/reap/STOP保存が必要。
今回の人工scope guardを手作業で回避して実行することは許可していない。

## 7. 保全・運用・未解決事項

旧証拠379ファイルと保存入力110 SHA、旧manifest185行を保全確認する。
旧SOURCE・raw・grant・ledger・marker・STOPは変更しない。
新しい技術STOPは今回の準備だけを閉じ、失われた元native終了証拠の代用にはしない。
旧N/G=null、元native UNKNOWN、旧compiler証拠不足、C/L COMBINATION_BLOCKED、
ρ保証・PR全law・費用tail・旧fingerprint FAILの原因UNKNOWNを維持する。

適用policyは最新版2026-10-23版の§8.4/§10/§10.1/§10.2と共通AGENTS。
当該ファイルに依頼で挙がった**§10.3は存在しない**ため、その内容は仮定しない。
[版・hash記録](../artifacts/non_df_h4_central_arb_preparation/2026-10-24/policy_observation.json)。
計算資源の任意上限、共有設定変更、他ユーザーprocessへの操作、AI subagent利用は0。
局所ゲート対の計算、phase cache、exact permutation、QPYの一度のdecodeで重複を避けた。

[検証器](../artifacts/non_df_h4_central_arb_preparation/2026-10-24/verify_preparation.py)はmetadata/SHAだけを読み、
Qiskit/Arb/科学runnerをimportしない。[STOP](../artifacts/non_df_h4_central_arb_preparation/2026-10-24/STOP.json)。
技術成果物commitと公開状態・remote SHAは[公開記録](../artifacts/non_df_h4_central_arb_preparation/2026-10-24/publication_status.json)を参照。

次の担当：**ユーザー**が最小PF cosine pilotの実行範囲を判断し明示承認する。
**Codex**は承認後だけproductionの固定と限定評価・終了監査を行う。現在は未実行、mandatory STOP。

## 公開状態の追記

技術準備commitは `fdb775ca7f1d38ff46cd30364b4739c5d10a34be`。
originへのpush1回はHTTPS Username取得不能でexit128。
[失敗log](../artifacts/non_df_h4_central_arb_preparation/2026-10-24/push_attempt01.log)、
[remote再確認](../artifacts/non_df_h4_central_arb_preparation/2026-10-24/remote_after_push.json)。
専用branch検索0件、remote SHA=null。認証可能な環境で同branchをpushする必要がある。
後続local metadata commitは公開状態を記録するだけで、bridge・契約・人工結果・旧証拠・STOPを変更しない。
実H4 pilotは未実行、mandatory STOPを維持する。
