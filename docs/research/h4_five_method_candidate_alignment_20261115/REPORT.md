# H4の5方式：信号・精度・価格の物理候補対応

2026-11-15 JST。Cのconsumer側技術照合。**新しい科学計算・数値再解析・価格fit・compile・G・候補選定は0件**。

5方式を14の独立scheduleに分け、13round×X/Yの**364行**を照合した。主成果は[ALIGNMENT.json](ALIGNMENT.json)（SOURCE・receipt・cohort付き）、[CSV](ALIGNMENT.csv)、[HANDOFF](HANDOFF.md)、[最小追加検証](NEXT_VALIDATION.md)、[ロードマップへの影響](ROADMAP_IMPACT.md)。G・十分N・新しいE[C]評価値はnullを維持する。

## 結論

既存価格のある物理点と信号側の物理点は対応付けられる。しかし、全13roundを同一候補の適格Nと適格価格で埋めた方式は、確認した固定成果にない。主な障害は次の三つである。

- 決定論PF：価格shortlistの信号が全roundのpacketになっていない。p7 S2・幅0.1の既存packetを別幅・別公式のshortlistへ流用できない。
- ランダム方式：PR-DFのE[S]、P1/P2の有限cohort、Pauli B0、Full-H P-1は、それぞれ別種の証拠である。適格な全law平均compiled価格ではない。
- 精度：Bの保存モデルMSEと使用済みUSER_SUPPLIED Nは、正式な十分Nとは別である。新しいtotal decoderへのmetadata接続が成立しても、その科学契約の採択・十分Nの資格を代行しない。

## 固定SOURCEと再利用

| SOURCE | 本作業での利用 |
|---|---|
| C `855573064` | 52軸PR-DF coverage、4件の保存receipt registry、CELL_PLAN V2を再利用 |
| C `c628e9e84` | 元guard、canonical bridge、price context、model validation。実装・資格を変更しない |
| D `acea1118a` | **ADMISSION_UPDATE.md**の保存4＋新規20 calibration＋12 holdoutを現行として採用。旧HANDOFFの「p7受入待ち」は履歴 |
| B `999304918` | readiness、元/total canonical対応、Pauliの保存条件付きMSE、PR-DF旧partial DPのSOURCE・使用vector |
| **B `9c0e12d87`（最終照合時に公開）** | PR-DF fixed-gridを保つ版付きtotal bridgeと14構成の最新readiness。PR-DFのtotal MSEは未評価。R448も独立metadata受入済み |
| A `714247619` | PR-DF/Pauliの全round packet、Full PF follow-up原本 |
| A `7a0c9468b` | Full-H P-2、Pauli B1の未承認設計。価格結果と扱わない |
| **A `c212cee85`（作業中に公開）** | [信号原本調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c212cee857d639d2400819bbca3afa1dc052a9f4/docs/research/a_signal_candidate_identity_coverage_20261114/REPORT.md)と364行を直接再利用。Aの原本調査を再実装しない |
| roadmap `fa84c0d6` | 指定ロードマップとの課題差分を別紙に整理。本体・他担当branchを変更しない |

完全SHA・path・selectorはALIGNMENT.jsonのsources、各行のsignal/decoder/MSE/N/priceを参照。JSON locatorのselectorはJSON key/index、A CSV locatorのselectorはheaderを除く0始まりのdata row。原本の科学数値や価格平均を新規集計していない。

## 候補別対応表

S1＝全round packet保存、S2＝現行SOURCEの保存mean（packet化・受入が必要）、S2L＝旧SOURCEの保存mean（再資格化判断が必要）、S3＝保存stepはあるが新meanが必要。件数・所在はA c212の調査結果をそのまま参照する。

| route / schedule | 信号・decoder/MSE/N | 価格証拠 | 結合と主な不足 |
|---|---|---|---|
| PR-DF LD1 `a4dda8f7` | 13raw。旧partial DP MSEあり。旧USER_SUPPLIED vector。**B9c0の版付きtotal `9c3d392e`へmetadata受入済み**。total MSE/N未評価 | ideal E[S]26軸。m0/m6 X/YにP1 n2・P2 n4（別cohort）。旧C1は別lowering/domain | physical IDは一致。E[S]・cohortの価格consumer受入とscalar SOURCE、law差、全law残差・CI・モデル独立検証が未資格 |
| PR-DF LD2 `1fa33556` | 13raw。total `e6fe7e68`へmetadata受入済み、total MSE/N未評価。旧partial結果は別 | 同条件E[S]・compiled価格なし | LD1価格・kernelを移植しない。必要な科学価格範囲を選ぶ |
| PR-Pauli LD1 `d905d219` | S1 13round。total `9fa3c47d`へのdecoder-only bridgeあり。条件付きopt-in MSEあり、十分Nなし | m0 X/YのB0 n1（wiring）。Pauli D2・辞書/law付き | 元identity/law/contextが一致。全law平均・価格modelなし。B1は未承認 |
| PR-Pauli LD2 `d7dd4f52` | S1 13round。total `285bf481`。同様のMSE・N区分 | 同条件compiled価格なし | LD1辞書・B0の流用不可。B1-W/V/CALは設計案 |
| Full PF S2 16h/7 `11cf8ca6` | S2:m0,1,4,12、S2L:m3、S3:m2,5〜11。全13packetなし | m3旧locator未受入。m4/5較正、m6holdoutは計画 | A scheduleとD CELL_PLANの物理条件は一致。未取得signal・価格、packet/decoder、MSE/Nが必要 |
| Full PF Y4 16h/3 `a799697f` | S2:m0,1,2,4,12、S2L:m3、S3:m5〜11 | m3旧locator未受入。較正/holdout計画 | 同上。構造piece数で除外しない |
| Full PF M5 16h `70f7b317` | S2:m0〜4,12、S3:m5〜11 | **m4 X/Yの保存deterministic receipt受入済み**。m3旧locator。m5/6は計画 | m4の価格と保存meanは同物理条件。meanのpacket受入、他round価格資格、Nは別問題 |
| Truncated p7 B0I S2・0.1 `e6754564` | S1 13round。total `b8ceb775`。MSE/N未評価 | **m3/m4に旧価格locatorは存在**。現行full measured wrapper receiptとして未受入 | 価格shortlistとは別schedule。元receipt/wrapper/axis/sourceの受入不足を「価格原本が皆無」と混同しない |
| Truncated p7 S2 16h/7 `a6e55777` | Full S2 shortlistと同じS2/S2L/S3構成 | m3旧locator。m4/5較正、m6holdout計画 | p7 S2/.1 packetのm0だけ同物理点。schedule・canonical・N全体の流用不可 |
| Truncated p7 Y4 16h/3 `defc5e88` | Full Y4 shortlistと同様 | m3旧locator。較正/holdout計画 | 元full DF-H targetを維持。切断Hamiltonianのenergy推定に変更しない |
| Truncated p7 M5 16h `424eecbf` | S2:m0〜4,12、S3:m5〜11 | **m4 X/Y受入済み**。m3旧locator。m5/6計画 | 信号・価格の物理点は一致。p7 S2/.1の別scheduleと結合しない |
| Full-H κ2 `54ac3eb3` | S1 13round。total `265b90dc`。MSE/N未評価 | m3 X/Y P-1 n1。元停止＋別revalidation PASSを保全 | identity/law/context一致。P-1はwiring専用、P-2較正点・E[C]へ昇格不可 |
| Full-H R448 `cfa6aee3`（補助） | S1 13round。B9c0でtotal `7e792803`への独立metadata受入済み、MSE/N未評価 | 同候補価格なし | κ2と異なるR/event law/normalizer。正式比較採択なし |
| Full PF S2・0.1 `f2d58f32`（補助） | S2:m0,12、S2L:m3,4,6、S3:その他8round | m3/m4旧D receipt metadata/価格locator。現行guard未受入 | 旧39round計画の未実行系列。shortlistとは別候補 |

全候補のH/state/snapshot・formula・prefix・要求幅・exact κ・profileはJSONのcandidatesに保持し、actual T/q/δ/R/K・元canonical ID・signed structureはrowsに保持する。Full-Hでは**group内allocationと全event位置数を分離**する（m3 κ2はgroup q8/R71、canonical event位置568）。binding内のT=.8は歴史的入力識別子であり、実行round Tではない。

A監査のS1B/NO_INTAKEというsignal_statusはその公開時の分類として保持した。**現在のdecoder状態は別欄のB9c0対応を優先**する。旧PR-DF/E[S]/価格cohortの「consumer bridge待ち」は価格受入側の不足であり、解消済みのdecoder bridge不足と混同しない。Full PF S2/.1はBでtotal ID `79407297`を生成済みだが、signal intakeは未完了。

## 物理照合の範囲と資格

- 単なるID一致で結合しない。既存Cのprice_contextを構成し、保存価格のidentity、law_binding、physical_candidate、H/state/snapshot等のbinding、split/scalar、profileを既存guard comparatorで照合した。Pauliのdictionaryを含むlawを保全する。
- XとYをcosine/Re、sine/Im、P0−P1へ個別に対応付ける。各保存recordのsource・cohort・implementation sidecar・元receipt/fingerprintは固定locatorのまま保持する。既存guard結果の再利用であり、新しい回路意味の実験証明ではない。
- Full PF/p7の4 deterministic point receiptは受入済みtierを維持する。本表は価格値を新計算・平均せず、保存recordへ参照する。全roundの価格model資格を与えない。
- PR-DFのE[S]は構造RZ期待値。旧C1は異なるq/R・generic loweringであり、現在のnew ZZ・LD2へ係数移植しない。P1/P2のrole、cohort、n、replayとの区別を変更しない。
- Pauli B0は`PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF`、DF/Full-Hの新loweringは`CONTROLLED_ZZ_CX_CRZ_CX_V1`。PFはRTE lowering対象外。全RZ数量、controlled・測定wrapper・P0を維持し、normalization Bを価格へ掛けない。
- Pauli/PR-DFの使用N_X=N_Yベクトルは、元SOURCEの`64,60,56,52,47,43,38,34,30,25,21,17,12`。quantum Nと価格cohort nを分離する。Pauliの保存total-model上界、DFの旧partial-decoder上界を取り違えない。nominal/最適/十分Nとは記載しない。
- ideal law、PCG64/CDF実sampler、有限cohort、全law期待値、ground/state/numerical certificate、CI/covarianceの未資格を維持する。metadata一致はそれらの科学保証ではない。

## 分類・検証・保全

ALIGNMENT.rows.compatibilityはblockerを複数保持する。MATCHED_EVIDENCEは保存recordとの技術的対応、MATCHED_BUT_UNQUALIFIEDは対応があるがGへ科学的に採用できない状態。SIGNAL/PRICE_EVIDENCE_MISSING、DECODER_OR_CANONICAL_BRIDGE_PENDING、INSUFFICIENT_SOURCE_EVIDENCEを分ける。cross_candidate_linksにPHYSICAL_CANDIDATE_MISMATCHと、p7 m0の限定的な一致を明示した。

追加コードはmetadata index builderとそのtestsだけ。Aの364行、Cの52軸coverage・receipt registry、Dの固定metadata生成関数、Bのreadiness/bindingを再利用した。controller、decoder、N solver、価格計算器、fitを追加・変更していない。

[TESTS.txt](TESTS.txt)は新規13 test methodと既存receipt/guardの回帰29件、**合計42件すべてPASS、skip 0**。[METADATA_CHECKS.json](METADATA_CHECKS.json)は14 schedule/364軸、null維持を確認する。人工mutationは異なるR/q、SOURCE、H/state、辞書law、scalar、lowering/compiler/wrapper、測定軸を拒否する。作者自身の技術確認であり、独立科学レビューではない。報告前の公開A/B/D ref照合は[PUBLICATION_CHECK.json](PUBLICATION_CHECK.json)。

Cの既存branch `codex/m5-m4-receipt-admission-20261114`を継続。他担当worktree・科学入力・原本・raw・grant・ledger・marker・KEEP・STOPに変更なし。旧計画・受入SOURCEを上書きしない。**mandatory STOP：新科学計算、実fit/compile、G、winnerへ進まない。**
