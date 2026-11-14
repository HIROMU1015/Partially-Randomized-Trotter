# PR-Pauli B1-CAL：B1-W完了後の優先順位と実行前の不足

2026-11-15 JST、Codex D2。**現時点の推奨は「CALを保留」**。D2の非configuration較正cellにはC入力recordの生成経路がある一方、実path・compiled receipt・公開SOURCEに束縛した実装証拠の経路が不足している。まずこの限定した接続を整える。その後に別途採択を検討する最初の科学範囲は、**LD2の較正5 cell・18 compileだけ**を推奨する。LD1との一括48件、holdoutの同時取得、B1-Vの先行を前提にしない。

これは静的照合と計画の更新であり、CAL/Vの採択・実行許可ではない。code変更、project module import、test suite起動、sampling、H4 build/compile、statevector、数値再解析、fit、E[C]/N/G算出はすべて0。既存の科学結果・旧STOP・SCOPE_DEVIATIONとログは保全した。機械可読の転記は[CAL_AFTER_W_METADATA.json](CAL_AFTER_W_METADATA.json)。

## 1. 既に確認できたことと今回の差分

| 辞書 | 既存の実回路証拠（各1行） |
|---|---|
| LD1 B0 | 保存H4/1Å/STO-3G/DF10、LD1/S2/m0/T.1/q1/R3/K2のnormal1 pathでorder0×2・order2×1を観測し、head/bit順/control/scalar/full-space vector/X/Y/compiled-vector gateとC guardがPASS、X/Y各RZ1380。 |
| LD2 B1-W | 同系のLD2/S2/m0/T.1/q1/R2/K2・normal index0の1 pathでorder0 component34 `XZZXYZZY`・40 `IIXZIZXI`のみを観測し、観測path gateと公開SOURCE付きC guardがPASS、X/Y各RZ2236。order2 product・非自明位相は未観測UNKNOWN。 |

X/Yは各々同一pathの別測定軸で、独立2標本ではない。**B0/Wのpath・価格は今回のCALにもholdoutにも繰り入れない**。m0の価格差を辞書の期待価格・総費用の順位にしない。

元[NEXT_SCIENCE_PLAN](NEXT_SCIENCE_PLAN.md)と[HANDOFF](HANDOFF.md)の固定版`1a7eed91`ではWが未取得だった。今回変わったのは、W依存が公開receiptにより満たされたこと、BのQ3判定が両辞書で得られたこと、D2自身のSOURCEから以下の実装上の境界を具体化したことである。元PROPOSALS/JOB_TEMPLATES、科学cell・辞書・cohort・seedは変更しない。

### 固定資料

| SOURCE | 一次資料と用途 |
|---|---|
| `00f74f86d16cc6878e1d4c5977eb9abd9ce9c076` | `docs/research/d_pauli_ld1_m0_pilot_results_20261114/RECOVERY_REPORT.md`、`RECOVERY_HANDOFF.md`、run02元trial/receipt。B0 rawは`8502dd33ad16dc012fb3558668ef3e4cd8a4405c`。 |
| `1a7eed91dd96992cc6abcbfa1cd473a2044db16c` | 本dirのREPORT、PROPOSALS、NEXT_SCIENCE_PLAN、HANDOFF、SCOPE_DEVIATION、人工test記録。 |
| `bd1b1b2862da6d174fa521e7612051810a636e7f` | D2 caller・connection・人工testの実装SOURCE。現在の当該codeとbyte差分なし。 |
| `1e2639a1be46ebea618b15591fb1e886bc499d33` | [LD2 W REPORT](../d_pauli_ld2_b1_w_results_20261115/REPORT.md)、[C HANDOFF](../d_pauli_ld2_b1_w_results_20261115/HANDOFF_C.md)。rawは`6c4709f85d0f28ce7c6957e9a70c85e378d8cad0`の`docs/research/d_pauli_ld2_b1_w_results_20261115/run01/RESULT.json`。 |
| `50be68b803a90fd893a1499ebce74e033225144d` | `docs/research/pauli_q3_three_quarter_20261115/REPORT.md`、HANDOFF、SUMMARY、INPUTS。保存結果の読取だけ。 |
| `7a0c9468b2499fac9fa1cd279ae2980c9dab048e` | Aの`A2_PAULI_B1_PLAN.md`、DESIGNS、`finite_price_design.py`。元CAL設計の正本。 |
| `c628e9e8447c11c984064b922cbce6c52dc72d48` | Cのguard・Pauli planner bridge・`price_model_validation.py`、PRICE_VALIDATION_PLAN。 |
| `0b4b2d941cd669aaa45231e1236bc2d3158a9d12` | `docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.csv/json`。元/total IDと価格roleを区別。 |
| `73b23e01cd0cbc98cc65be2053b3965770813edf` | 統合ロードマップのPauli W/V/CALと5方式の不足を参照。本体は編集しない。 |

## 2. 辞書別の有限計画

共通条件は保存H4 linear 1.00 Å/STO-3G/tol-only DF10、S2、step time 0.1、K2、元normal sampler、`PR_PAULI_TAIL`、`PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF`、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、symmetric-directionalの測定付きfull controlled Hadamard wrapper、P0・状態準備費除外。quantityは`FULL_WRAPPER_MEAN_COMPILED_PRICE`、kindは`DIRECT_COMPILED_FINITE_SAMPLE`、有限cohort限定である。

| 辞書 | 元configuration | 辞書SHA-256 | DF head / 保存s_R / λ_P |
|---|---|---|---|
| LD1 | `d905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27` | `dbdde5a12791cd7b31b00d8ec6ae5074d56964b467389688163ece8bf186945c` | one-body, fragment0 / 0.4177175120626939 / 3.3066052942233157 |
| LD2 | `d7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3` | `dd986e52057aa01a330a83dd147316696dbae669dc8a51828ee2a54734016ea7` | one-body, fragment0,1 / 0.2223759094107018 / 2.0550998769244275 |

辞書はP0 `4220f3006d0d8d413e195ed948e250578513b557`の保存184項で、係数・順序・bit順・hashを保持する。scalarは保存c_Hと当該s_Rを総Tで一回だけ適用する。H/state/snapshotはW/B0と同じ保存context。非configuration cellのphysical IDは元`PAULI_PRICE_CELL:`を維持し、m5/m6は元canonical round IDを維持する。Bの新total IDへ価格IDを付け替えない。

### LD1

| cell | q | R | T | path数 | 軸 | 役割 | 新規compile |
|---|---:|---:|---:|---:|---|---|---:|
| S171 | 1 | 171 | 0.1 | 2 fresh | X/Y | CALIBRATION | 4 |
| S1361 | 1 | 1361 | 0.1 | 2 fresh | X/Y | CALIBRATION | 4 |
| S10885 | 1 | 10885 | 0.1 | 2 fresh | X/Y | CALIBRATION | 4 |
| Q8R171 | 8 | 171 | 0.8 | 2 fresh、他cellから独立 | X/Y | CALIBRATION | 4 |
| P2R171 | 2 | 171 | 0.2 | 1 paired（S171 path0→path1） | X/Y | CALIBRATION・境界検査 | 2 |
| **最初の較正範囲** | | | | **8 fresh＋1 derived** | | **5 cell** | **18** |
| M5_ROUND | 32 | 86 | 3.2 | 2 fresh、CALから独立 | X/Y | HOLDOUT、C凍結後の別指示 | 4 |
| M6_ROUND | 64 | 171 | 6.4 | 1 fresh、CALから独立 | X/Y | HOLDOUT、C凍結後の別指示 | 2 |
| **独立holdoutまで含む総件数** | | | | | | | **24＝18＋6** |

### LD2

| cell | q | R | T | path数 | 軸 | 役割 | 新規compile |
|---|---:|---:|---:|---:|---|---|---:|
| S66 | 1 | 66 | 0.1 | 2 fresh | X/Y | CALIBRATION | 4 |
| S526 | 1 | 526 | 0.1 | 2 fresh | X/Y | CALIBRATION | 4 |
| S4205 | 1 | 4205 | 0.1 | 2 fresh | X/Y | CALIBRATION | 4 |
| Q8R66 | 8 | 66 | 0.8 | 2 fresh、他cellから独立 | X/Y | CALIBRATION | 4 |
| P2R66 | 2 | 66 | 0.2 | 1 paired（S66 path0→path1） | X/Y | CALIBRATION・境界検査 | 2 |
| **最初の較正範囲** | | | | **8 fresh＋1 derived** | | **5 cell** | **18** |
| M5_ROUND | 32 | 33 | 3.2 | 2 fresh、CALから独立 | X/Y | HOLDOUT、C凍結後の別指示 | 4 |
| M6_ROUND | 64 | 66 | 6.4 | 1 fresh、CALから独立 | X/Y | HOLDOUT、C凍結後の別指示 | 2 |
| **独立holdoutまで含む総件数** | | | | | | | **24＝18＋6** |

件数・rank3は元PROPOSALSと保存人工test記録から確認した。新たに設計rank・精度・価格を数値評価していない。較正は各軸5 cell meanを既存の3特徴量`PR_STEP_V1=(1,q,qR)`へ渡す案。pairedは親2本と相関するderived pathであり、独立9本ではない。X/Yでも独立標本を倍に数えない。holdoutは別cohort/seedの計3 path・6 compile。いずれの辞書も24件一括の実行依頼にはせず、両辞書48件の一括実行も前提にしない。

## 3. §10.5の重複・条件照合

予定14 cellの元physical IDを、D2のPROPOSALS/B0/W、Aの設計・最新索引`1adb7e65a`、Cの価格plan/coverageと候補表、B Q3報告に照合した。同じIDのhitはPROPOSALSとC対応表のmetadataで、CALのcompiled元trial・receiptは確認できなかった。IDだけの検索で同等性を断定せず、Aの既存監査とB0/W一次記録の条件差を併用した。

| 保存成果・計画 | 辞書/law・head/tail・scalar・物理条件・cohort/seed・lowering/wrapper/quantityの判定 | 採否 |
|---|---|---|
| LD1 B0、LD2 W | 対応辞書のm0配線証拠。CALのR・時間/cohort/seedと異なる。Wは同じ辞書でもorder0×2のみ | wiring前提として参照。**CAL標本0件** |
| B1 PROPOSALS、A DESIGNS | 今回のcell/cohort/seed・domainの元定義そのもの。実価格ではない | 計画を再利用、再samplingしない |
| 保存P0辞書・H/state/snapshot、native/D2/wrapper | 同じ入力・実装を再利用する計画。LD1/LD2の辞書とs_R・headは別 | 元SOURCEを維持。P0演算子の独立数値再検査なし |
| P3、旧Pauli-JW | P3はresidual/K4/T.8等、Pauli-JWはPauli headで、今回のDF head＋保存Pauli tail/K2と異なる | 直接価格として再利用不可。別IDでもhead/law等が不一致 |
| PR-DF E[S]/C1/P2、Full-H、決定論PF | tail/route・law・loweringまたはquantityが異なる | Pauliの較正点にしない。PFの2%規則も移送しない |
| B `50be68b80` Q3 | 元signal/lawとtotal-view bridgeを保つ精度側の保存評価。価格call0 | 精度側の進捗を参照。価格・十分Nではない |
| 関連公開branch・local process | Pauli D2 tipはW完了`1e2639a1b`、B0 tipは`00f74f86d`。A/Cの関連plan、他D価格branchも参照。照合時にPauli価格の実行中local jobなし | 確認範囲に同一CAL job・価格成果なし。外部hostの未公開jobまで確認済みとはしない |

公開refsと照合範囲はMETADATAに記載。新しい実行直前には関連job・table/catalog・別ID同等receipt・出力namespaceの変化だけを再確認する。同等保存成果が見つかればSOURCE付き受入を先に判断し、証拠不足・矛盾なら当該項目をSTOP。計画時点の未発見を、未確認の保存先まで含む「過去に未実施」としない。

## 4. D2のpaired q2は親eventの直接連結ではない

確認対象はD2自身の[caller](../../../scripts/run_pauli_b1_prices.py) `seed_table:129–136`、`execute:295–304`、`child:372–387`、`check_composition:139–149`と[connection](../../../src/trottertracks/pauli_tail_price_connection.py) `sample_steps:122–134`。D1の不具合をD2へ推定していない。

1. 入力はinlineの`parents`配列2件（親path0、path1の順）、各親のrequest/cohort/index/result、およびpairedの明示`step_seeds`。公開親receiptのlocatorをこの入口が自動取得する機能はない。
2. `seed_table`は`composed_of=[Sxxx#p0#g0,Sxxx#p1#g0]`から親seedを順に取る。`execute`は親数・seed順・親request/cohort/indexを検査する。
3. `child`は親event objectを回路へ渡さず、`acquire_path`→`pilot_core`→`sample_steps`を呼ぶ。明示seedで元normal samplerからq=2のeventを**再生成**し、head/tail/scalarを構成してX/Yをcompileする。pairedのcohort名から新seedを引くのではない。
4. その後の`check_composition`は親2本の`event_metadata`を順に並べ、`outer`だけを除外してJSON完全一致を要求する。event順・選択label・order/product・係数/phase・step seed・D2 aggregate phase/mappingが対象で、scalar値も親と一致が必要。per-step scalar=0、最後にtotal Tで一回という構成を維持する。

LD1のpaired seedは`[12446388743038358340,2111155141631760768]`、LD2は`[3400578458722811300,11774002175198634805]`。これらは元PROPOSALSからの転記で、今回新生成していない。

既存[test_paired_steps_retain_labels_products_phase_and_scalar_once](../../../tests/check_pauli_b1_circuits.py)はtoy parentをseed11/12で取得し、同seedのq2を再生成し、保存JSON境界後の比較・親順逆転/異scalar拒否を検査している。[元3件PASSログ](d2-b1-circuit-tests.txt)はこの実装を支持する。**実H4 CALのpaired receiptによる検証、公開親SOURCEの解決、保存event直接連結、compile前の比較を証明したログではない。** 今回再実行していない。

同じSOURCE・辞書・event時間・R・K・sampler/runtimeなら同seed再生成で同じevent列を得る経路は確認できるが、「再samplingなしの親receipt直接連結」とは報告しない。現状は比較がcompile後で、不一致なら既にcompile枠を消費し得る。

**次回前の限定提案**：親receiptを公開`{commit,path,selector}`から取得し、元SOURCE/request/cohort/index/seedを保持する。採択するpaired仕様を、(a)固定同seed replayを許容し、event列の親一致をbuild/compile前に検査、または(b)保存event列を直接composeへ渡す、のいずれかに明記する。元A案は明示step seed表を要求しており、D2に(a)の経路は既にある。(b)を要求するなら未実装部分だけを追加する。どちらも今回の計画更新では実装・実行しない。

## 5. 非configuration cellのC接続：recordはあるが公開証拠の束縛が不足

D2 `records:225–254`はpath集合・request/cohort/seed・scalar一致を検査し、X/Y別の有限cohort平均を`price_model_validation.record`へ渡す。`PRICE_CALIBRATION_RECORD_V1`にはcell_id、role、axis、特徴量、rz_count、law、n_paths、implementation、profile、quantity、sourceが入る。

ただし**5つの較正cellはすべてm=None**で、canonical round用分岐`if physical['m'] is not None`に入らない。このため既存`execute`の非configuration出力は次の通り。

| 出力・受入 | 現状のD2経路 |
|---|---|
| RESULT.json | child.pathsに実request・event metadata・作用gate・X/Y metrics・fingerprintを保存する経路がある |
| private compiled_cache | native.compile_cachedがfingerprint別のcompile receiptとwrapper/compiled QPYを保存する経路がある |
| C_INPUTS.json | X/Y別のrecordを出す。source.commitは実行時code HEAD、result_pathは新output、cohort/step_seedsも保持 |
| PRICES.json / SIDECARS_UNBOUND.json | 非configurationでは空配列になる |
| IMPLEMENTATION_EVIDENCE.json | 非configurationでは空objectになる。evidence_path引数はこの分岐では使われない |
| check_round_records | m=Noneを明示拒否。W/m5/m6用のC guard成功をこの較正cellへ適用できない |

source.commitは**実行codeのSHA**で、新しく生成されるRESULTやreceiptはそのcommit内に存在しない。実行SOURCEと結果の公開SOURCEは区別が必要。Wで手動作成したRECEIPT_TRACE/COMPILED_X/Yと公開sidecarの手順は、非configuration CAL callerには自動接続されていない。

既存のC consumerは[finite_price_design.fit_identified](../../../src/trottertracks/finite_price_design.py) `202–226`→`c_model(root).fit`（固定C `c628e9e8`の[price_model_validation](../../../src/trottertracks/price_model_validation.py)）。fit_identifiedは一軸の予定5 CAL cell・特徴量・kind/n_paths・domainを検査する。C `record`はsourceが非空等を要求するが、公開locatorを開いて実path/receiptを照合する処理ではない。**既存のfit入口は使えるが、fitの前に必要な実装証拠の受入は完成していない。**

既存`test_fictional_C_records_no_promotion_and_path_mix_rejected`は架空価格と`source={fixture:ARTIFICIAL}`でrecord生成・混在拒否を検査する。`test_C_fixed_guard_canonical_LD2_wiring_and_wrong_axis_rejection`はcanonical Wの人工proofをmock読取してguardを検査する。[元11件PASSログ](d2-b1-prep-tests.txt)はこの範囲の技術証拠であり、非configuration実価格の公開receipt束縛PASSではない。

### 最小の修正・引渡し案（今回は提案のみ）

1. **D2出力**：各cell/axisについて、完全なphysical request・辞書hash・H/state/snapshot、cohort/path index/seed、実scalar・native/wrapper/D2/compiler SOURCE、測定descriptor、作用gate、per-path X/Y fingerprint/metricsと元compiled receipt locatorを実装証拠へ結ぶ。既存RESULT/cacheを再利用し、別の回路生成器・承認台帳を作らない。
2. **公開後binding**：rawのRESULT、C_INPUTS、実装証拠、receiptを先にcommit/pushする。rawを改変せず別の公開用C入力/sidecarに`published_result={commit,path,selector}`、`implementation_evidence={commit,path,selector}`、軸別path receipt locatorを追加する。`source.commit`の実行codeという意味を保ち、source.cohort/step_seedsを残す。Wと同様に、結果生成前のcode SHAを結果公開SHAとして偽装しない。
3. **Cのfit前受入**：上記locatorから実path/axis/metrics/fingerprint・law/dictionary・実装/物理contextを照合する小さなcell用reader/checkを追加または適合する既存readerへ接続する。canonical m0/m5/m6へ偽装せず、非configuration physical IDを保つ。`check_round_records`をそのまま呼べるとはしない。受入済みrecordのみを既存`fit_identified(...,axis='X'/'Y')`へ渡す。
4. **凍結とholdout**：Cは許可された場合だけ一軸5 recordから各一モデルをfitし、domain・特徴量・使用CAL receipt/cohort/seed・paired親関係・係数をSOURCE付きで公開/凍結する。既存frozen_model_gateが要求するsource.cohort/step_seedsを保持する。その後の別指示で独立m5/m6を取得し、canonical round guardにも公開証拠を束縛する。

paired recordと親の相関を消さず、Cの既存exact OLSの点fitをCIや独立9標本の推定に読み替えない。必要な人工metadata/toy検査は、上記限定修正の別作業で行う提案とし、保存H4 P0数値caseを含むsuiteを起動しない。今回は修正もtestも行っていない。

## 6. 優先順位の比較と最初の範囲

B `50be68b80`では両辞書とも、両軸共通vector`[48,45,42,39,36,33,29,26,23,19,16,13,9]`に対し`MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`。これは保存DF10 solver proxy・ideal finite-K raw/IID・採択Z1＋T1/Q3の限定モデル評価で、十分N/最適N・物理的保証ではない。今回はその判定を読取っただけで、MSE・Nを再計算していない。両者の同じshot配分と閾値適合から、価格の優劣やwinnerを決めない。

| 選択肢 | 5方式比較で最初に埋める不足 | 利点 | 残る不足・判断 |
|---|---|---|---|
| LD1を先行 | PR-Pauli LD1のrank3有限較正価格とC fit入力 | 元A順序案を維持し、B0でorder2を含む観測pathの実作用が確認済み | 同じ非configuration証拠接続の不足。大R/長q実資源は未測定。LD2へ係数移送不可 |
| LD2を先行 | Q3閾値適合済みPauli LD2に、初のrank3有限較正価格とC fit入力 | Wの配線依存が解消。予定R=66/526/4205、q8/R66はLD1より小さいtail列の計画 | headはLD1より大きく、価格/時間が安いとは未確認。order2/非自明位相はUNKNOWN。証拠接続不足は共通 |
| **現時点ではCALを保留（推奨）** | 実価格を取る前に、比較へ使うC入力の実path/receipt/公開SOURCE接続を整える | 取得後に出所・実装を結び直す不足を先に解消できる。paired仕様も明示できる | 今回の静的作業ではcode修正しない。科学取得の不足自体は残る |

**技術不足の解消後に最初に依頼する候補範囲**はLD2のS66/S526/S4205/Q8R66/P2R66、X/Y合計**18 new compile**。Rの小ささは取得規模を選ぶためのmetadata上の理由であり、期待価格・wall/RSS・総Gの予測ではない。既存B0のorder2証拠をLD2へ移すこともない。LD1を含めた科学的な最良候補選定は未実施である。

S66の親2本を取得・保全してからP2R66へ渡す。他の独立cellは共有資源を観測して適切に処理する。各cellを扱う既存callerを再利用する。将来この18件を実行依頼する場合は、5 cellをその有限範囲として明示し、全較正完了後にSTOPする。現在はその実行依頼もjobも発行していない。LD2のfit/凍結が済む前にm5/m6を同時取得しない。

Vは今回の推奨前提にしない。m0の追加pathは同一cellの実装再現性・ばらつきの目安を増やすが、3係数を識別する設計の不足を埋めず、全law精度の認定にもならない。追加pathでorder2が出る保証もない。将来Vを選ぶなら「m0での第二観測により何を判断するか」を別途定め、元B0/WのSOURCE/cohort/index0/seedを保持する。

## 7. 依存順序・STOP・未確定事項

**較正取得18 → Cのfit・公開凍結 → 別指示で独立holdout6。** 辞書ごとに独立したモデル・採択・実行範囲を保つ。最初の18と総24を区別し、後者は許可済み予算ではない。

- 次の科学取得前：採択辞書/5 cell、使用SOURCE、入力・cohort/seed、paired方式、実装証拠/receipt受入、fresh namespaceとone-shot/retry範囲を固定する。同等保存成果の証拠不足、実行SOURCE/入力/辞書・scalar・lowering・wrapper・axis不一致、作用gate不成立は該当項目STOP。別seed・別backend・追加pathへ置換しない。
- 較正完了後：有限cohort資格のまま公開・C引渡しでSTOP。Cのfit/凍結は別の明示的許可の範囲とし、自動実行しない。fit後も`CALIBRATED_UNVALIDATED`、qualified_for_candidate=falseを維持する。
- holdout前：Cの固定SOURCE・モデル・使用CAL・未開封の独立cohort/seedとPauli用許容差を決める。**決定論PFの2%条件はPauliには未採択**。別指示がなければholdoutを実行しない。失敗時は旧結果を保持し、許容差変更・自動refit・追加取得へ進まない。
- 未確定：Pauliの精度/許容差・interval contract・confidence、paired相関を含む不確かさ、rare寄与、実samplerとideal lawの移送、LD2 order2/非自明位相、モデルの長q/大R/全round外挿、物理的十分N・正式Gのclaim水準。有限cohortの較正やholdout適合だけで全law E[C]/CIを認定しない。

ロードマップへの影響：§7/§10の「Pauli CALは別判断」を、**W完了済み、D2非configuration証拠接続を先に補完、直近推奨はCAL保留、その後の第一候補はLD2較正18、C凍結後に別指示holdout6**と具体化できる。科学的資格や価格coverageは今回増えていない。B Q3の両辞書閾値適合と十分N/正式G未資格は維持する。統合ロードマップ本体は編集しない。

**計画のcommit/push・remote照合後mandatory STOP。** V/CAL採択、新sampling/build/compile/statevector/数値再解析/fit/E[C]/N/Gには進まない。
