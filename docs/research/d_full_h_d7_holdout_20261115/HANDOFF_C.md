# C向け引渡し：Full-H κ2 D7独立holdoutと固定予測の記述的比較

**DESCRIPTIVE_COMPLETE、4比較有効・欠測0。取得/実装gateは6軸PASS。モデルPASS/FAILなし、tolerance=null、model_verdict=NOT_ASSESSED_NO_ADOPTED_TOLERANCE。** 有限cohortの観測差だけを記述し、Cモデルの資格を自動昇格しない。

科学実行9cbabc617e95a81810924413c1b2b06ced329514は `9cbabc617e95a81810924413c1b2b06ced329514`、全6軸の固定結果SHAは `b5fac0ee4eac83f027f883e5ea5630623a98bf29`。Cモデル/予測はb5165f11bb63f9eb2a6bfe385ba75d8003b7e724、C2規則はf0765067a5a8b39888e36f3a8b80f1bdbda69223。C/C2原本・較正raw・旧STOPは無変更。

| 比較 | P（exact） | O（exact） | P−O（exact） | (P−O)/O（表示%） |
|---|---|---|---|---|
| M5_ROUND-X | 27894546580642075/20115111656 | 1392948 | -124757970359813/20115111656 | -0.445257% |
| M5_ROUND-Y | 27894546580642075/20115111656 | 1392948 | -124757970359813/20115111656 | -0.445257% |
| M6_ROUND-X | 111815248527103419/20115111656 | 5551296 | 150309651597243/20115111656 | 0.134608% |
| M6_ROUND-Y | 111815248527103419/20115111656 | 5551296 | 150309651597243/20115111656 | 0.134608% |


m5 Oは同軸path0/1の算術平均、m6 Oは同軸path0の値。X/Yの標本数は合算しない。exact有理数の正本は[COMPARISON.json](COMPARISON.json)と[CSV](COMPARISON.csv)。相対誤差のexact分子/分母も保存し、%だけ表示用に丸める。

## 固定locatorとCが確認できる内容

[HANDOFF_C.json](HANDOFF_C.json)のcomparison_rowsに4比較とCのmodel/prediction selector、各Oの個別path/軸の観測locatorを持つ。COMPARISON.jsonの各行も同じprediction_source/observation_sourcesを持つ。

- prediction_source：b516のFROZEN_MODELS.json、m5-Xはpredictions[0]、m6-Xは[1]、m5-Yは[2]、m6-Yは[3]。model_sourceはmodels.X/Y。Pは保存分子/分母を直接使用し再予測0。
- price：結果b5fac0ee4eac83f027f883e5ea5630623a98bf29のAXIS_PRICES.jsonの個別path/軸selector。m5はp0X/p0Y/p1X/p1Yの順、m6はp0X/p0Y。
- compile_record：同SHAの元COMPILE_RECORDS/p<i>_<axis>.json、原bytes/SHA/fingerprint/compiler/metrics。
- implementation_evidence：同SHAのAXIS_EVIDENCE.json、実際の数値/phase/parity/scalar/one-body/XY gateと資源。
- raw_receipt：同SHAのRAW_RECEIPT.json。READ_RECEIPT.py:read_receiptでlossless復号し、child.paths[index]と対応するcompile_receipts.cosine/sineへ辿る。

公開後D1検査：固定rawのlossless hash、元callerから再生成したrecordと公開JSONの完全一致、全6軸の原compile bytes/hash/metadata、必要gateとphysical/cohort/seed/9cbabc617e95a81810924413c1b2b06ced329514対応。C担当の科学的受入はNOT_PERFORMED。D1本人の検査を独立reviewやモデル合格と扱わない。

## 取得・資源・制約

予定3 fresh path/6 compile/4比較、事前同等結果の再利用0、新規6 compile、有効6軸/4比較、欠測・不一致・FAIL0。固定seedで各path一度samplingし、X/Y別に各一回compileした。科学retry・seed差替え・追加path0。m5 physical=FULL_H_bc8aab34256b73b68025（q32/R283/T3.2、path0/1）、m6=FULL_H_4ec8d73571760a6830b8（q64/R566/T6.4、path0）。固定cohort/step_seeds全文はHANDOFF_C.jsonへ引き継ぐ。

m5 wall22:05.61/maxRSS4278444KiB、m6 wall1:26:15/maxRSS16338552KiB。親/子/compile worker全exit0。資源の元file・単位とpath別詳細はHANDOFF_C.jsonのresources。maxRSSを同時workerの合計memoryとは扱わない。

index0各cellの所定7数値gate、全pathの実装/phase/parity/XY gateを保持。index1へindex0の数値gateを転用しない。resolved scalarとexpected scalarは各pathで一致。P0除外・測定付きfull wrapper・新ZZ lowering・固定compiler条件を保持した。

## ロードマップ差分とmandatory STOP

「Full-H κ2 D7の指定m5/m6独立holdout取得と観測前固定予測の4記述比較が完了」に更新できる。モデルの数値許容差・PASS/FAIL・期待価格資格は未確定のまま。Cは本結果を受け取るが再fitや資格昇格をこのhandoffが許可しない。ロードマップ本体は統合担当へ委ねた。

m6 n2、追加round、再fit、事後閾値、全law E[C]/CI、N/G/winnerは未実施・範囲外。Cモデル/規則/較正/旧STOPを保全。指定成果物の公開照合後mandatory STOP。
