# PR-Pauli価格不足からの候補固有較正要求

固定技術SOURCE：`7d6a8c385fc057de012c7e1e603bc0bc358f313c`。
既存A branch `codex/a-pauli-on-demand-preparation-20261116`を継続した。

Aの公開`7cf4e1f05`応答と、同じ候補に対するD2 `PriceRules`のMODEL応答を受ける入口を追加した。保存価格をそのまま再利用し、不足する候補について既存S0-r1較正driver用のspecを機械的に生成する。新しいL_DごとのSPECS追加は不要。

- [API/HANDOFF](HANDOFF.md)、[SOURCE](SOURCE_INDEX.json)、[受入結果](ACCEPTANCE.json)
- [入口](../../../scripts/prepare_pauli_calibration.py)、[adapter](../../../src/trottertracks/pauli_calibration_request.py)
- [既存較正driverのrequest接続](../../../scripts/run_pauli_split_cal.py)

## §10.5の照合

A価格成果`b8bcdf97b`、A候補準備`7cf4e1f05`、B2 `f01ce68ba`、D2価格規則`c1f95e080`、C catalog `dbeb6480d`と関連driver履歴を確認した。調査範囲に同等の汎用較正要求生成器は見つからず、既存`run_pauli_split_cal.py`はLD3/LD5 SPECS方式だった。B2はAの構成生成とD2不足内容の搬送まで実装済み。観測時に関連較正processはなく、他担当のworktree/processは変更していない。非公開の別サーバー成果まで不存在と断定しない。

同じ科学cellが保存済みなら再取得しない。LD3/LD5のbindingを、prefixではなく生成した物理cell requestと辞書identityの一致で検索する。D2のcompleteな価格は較正へ進めず、部分completeなら取得済み軸を保全する。

## SPECSから機械的に導けるもの

| 従来の項目 | 取得元・規則 |
|---|---|
| prefix、config ID・本体・13round | A応答/B canonical。入力scheduleと全field照合 |
| 辞書・λ_P・s_R | 当該P0 dictionaryとgate SOURCE。DF λ_Rで代用しない |
| tail/head operators | 辞書と同じP0 carrierの`LDn_OPERATORS.npz` |
| common operators | 保存`4220f3006`のfull-Fock共通operator |
| cell | 既存S0-r1テンプレート：q=8、T=0.8、K=2、R=canonical m12のR |
| cell ID | 固定LD3 helper `cell_request`。`m=null`の非round cell |
| compiler/control/profile | canonical profile全体を保存profileへ照合 |
| scalar | 保存Hamiltonian constant＋当該辞書s_R、total Tで一回 |
| cohort・seed | 一致する保存bindingは元値を維持。新規はcell ID由来namespaceと既存seed関数 |
| output・private・実行branch | config/cell由来の既定値とAの現branch。任意の運用overrideも一括入力可 |
| plan/total_view | 来歴と任意のB view。係数やL_Dを決める材料にはしない |

q8・2 path・X/Y・最大4 compileは既存S0-r1の科学テンプレートであり、新しい探索・最適化規則ではない。どの候補を較正するかはB2が決める。

## 実装契約と適用範囲

旧`proposal()/validate()`のLD1/2制約を新分割へ流用しない。固定`13ab1acfa`の`acquire_path`は、当該辞書とexplicit operators/blocksを渡すとLD専用default探索を使わず、既存pilot coreへ接続する。Pauli connection、D2 lowering `9c4006bf`、native/wrapper `af284d85`、provider `c31ef424`を候補固有の辞書・profile・operatorに束縛する。

これはSOURCEから同定した実装recipeであり、新L_Dの数値bit-order/vector検証を終えたという意味ではない。実科学実行前の既存`input_preflight`が別途必要である。同関数はrestore/prepareを呼ぶため、今回のH4検収では起動していない。

対象はS2、K2、signed weights `[1]`、actual Δがbinary64 `0.1`、既存compiler/control/測定付きfull Hadamard wrapper。範囲外は`NEEDS_RULE_OR_IMPLEMENTATION_EXTENSION`、別lowering等は`NEEDS_IMPLEMENTATION_CONTRACT`として必要事項を返す。厳密1/10や別幅へ勝手に丸めない。

## 既存driverへの接続

LD3 helperは`158bbcc7b`とbyte一致で復元。split driverは`1097c37f9`を再利用し、`--request`と次を追加した。

- embedded canonical bodyを使うmetadata入力検査。
- generated specをJOBへ保存し、workerで消費する。SPECS[prefix]への逆戻りを防ぐ。
- specとfrozen bindingのprefix/config/dictionary/cell/cohort/H/state/profile/λ_P/operatorsの一致確認。
- 保存cellはreuse-onlyとしてfreeze/再取得を拒否。
- 新requestのSOURCE照合、候補実行branch、階層private directoryに対応。旧手書きSPECS経路は保持。

新requestに過去runのBLAS/OMP上限を恒常的に課さず、job環境を継承する。旧経路の固定条件は保持。取得時のC2段階log、process並列、既存acceleration、compile/receipt数、gate/STOPは再利用した。今回実H4で新しい性能測定はしていない。

## 検収結果

新規6 test＋A候補準備の既存11 test、計17 PASS。
[TESTS](TESTS.txt)／[A回帰](REGRESSION_A_PREPARATION.txt)。

| 対象 | 結果 |
|---|---|
| LD3 | 公開Q8R660のrequest ID、λ_P、辞書、scalar、16 seed、profile、m12対応、operator SOURCEを再現 |
| LD5 | 公開Q8R4について同じ科学入力を再現 |
| 保存価格 | 各26軸のexact値・SOURCEをコピー。新quote・fit・価格再解析なし |
| 部分価格 | 取得済み24軸を保全、欠測2軸と保存較正の再利用を返す |
| 人工未登録LD4 | 手書きSPECS追加なしで生成requestが実driver metadataとworker選択検査を通過 |
| 拒否ケース | 別ID/q/axis、別lowering、MODEL/DIRECT混在、異なるPF/K/幅、SOURCE欠測、LD4 spec＋LD3 binding混在 |

LD4 fixtureは辞書/SOURCE readerを人工データへ差し替えた入力検査である。実H4のLD4辞書やoperator、sampler、compiled vectorの科学的資格ではない。

公開B2のPRICE_RULES記録は要約で、raw 26軸応答は未保存だった。検収fixtureは公開PRICE_26AXIS carrierからexact値をfieldコピーし、`SAVED_PRICE_CARRIER_PROJECTION_NO_NEW_QUOTE`と明示した。新しいD2 quoteを実行したとは扱わない。

主担当がadapterと全受入・回帰を実施。補助agentはSOURCE/契約の読取調査とdriver拡張を担当し、契約reuse・worker binding・policy混在の技術的指摘は主担当が修正し人工回帰に反映した。独立科学レビューを実施したとの主張ではない。

## 引渡しとSTOP

同定した実装契約、候補固有cell、driver metadata、SOURCE、wrapper auditが必要なTをB2/D2へ返す。既存価格が揃うなら要求数0。新規較正の後も、当該候補のS0-r1分解・全requested Tのwrapper margin監査・D2登録/適用判定は必要であり、他L_DのA/H/Eを移植しない。

新しいH4 path・compile・fit・Q3は0。B2探索規則・実行run・他担当の原結果・固定SOURCE/KEEP/STOPは未変更。公開・remote照合後mandatory STOP。将来のH4 input_preflight/path/compileは別途明示承認後に行う。
