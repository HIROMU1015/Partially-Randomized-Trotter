# 決定論PF長q検証：m9の12予測と判定規則の観測前固定

2026-11-15 JST。**6候補×X/Yのm9予測12件を元の凍結係数からexactに計算し、新しい成果物へ固定した。公開対象12、保留0。** 独立観測前の確認範囲は下記のmetadataに限定し、m9観測値の取得・開封・合否判定は行っていない。

利用者の今回の指示により、12予測の科学計算・公開とm9限定の判定規則を実施/採択した。実回路取得はDの後続作業。m9以外への規則拡張や科学的な価格資格を採択したものではない。

## 正本とSOURCE

- [run01/PREDICTIONS.json](run01/PREDICTIONS.json)：12個別model_group・configuration/canonical/physical ID、axis、q、actual幅、元係数の固定locator、係数とexact予測RZ分子/分母、判定規則、実行直前のmetadata観測。
- [run01/PREDICTIONS.csv](run01/PREDICTIONS.csv)：D向け12行表。分子・分母が比較の正本で、丸め表示を使わない。
- [M9_POLICY.json](M9_POLICY.json)：12点だけに適用する事前固定規則。
- [PREOPEN_CHECK.json](PREOPEN_CHECK.json)、[PLAN.json](PLAN.json)：SOURCE/重複/取得開始のmetadata照合と固定対象。実行時にも確認し、価格の内容を開かない。
- [TESTS.txt](TESTS.txt)：人工係数によるexact予測、固定metadata、scope/欠測保持等の**7 test method PASS**。テストでは実12予測を禁止している。
- D向けHANDOFFでは予測結果の固定commitとpathを提示する。元モデルを新モデルとして再発行しない。

科学実行SOURCE：**`375f026a30e3c1ed3179a73cbbe333578abff00f`**。

元モデル：`12372fa6b53c878dc5dd124e036572ba0cd82061:docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json`。既存`price_model_validation.predict`のexact Fraction演算を無変更で再利用し、各groupで1回、計12回だけ呼んだ。fit=0、観測開封=0、判定=0、compile=0、N/G=0。

対象正本：`bcb67b2b2bb78776db1dd8d0625a3facb37cd298:docs/research/d_pf_allround_price_coverage_plan_20261115/NEXT_VALIDATION_PLAN.md`のm9表と、元C `0b4b2d941`のALIGNMENT。X/Y行36/37、38/39、88/89、90/91、140/141、142/143を固定し、configuration、route、公式、prefix、T/q/actual幅、axis、model_groupを元係数domainへ照合した。

## 観測前固定のmetadata確認

共通指示§10.5に従い、以下を実行前に確認した。

1. C branchは起点`3b1301738507d612694bf865d904b9376e038f69`で公開refと一致。関連C結果path、m9関連公開ref、D価格branchの公開refを確認した。同じ凍結係数・m9物理点・軸の公開予測は確認範囲になかった。
2. D価格branch/worktreeは固定計画`bcb67b2b…`。公開済みm9取得結果・別m9 refはなかった。D1 Full-Hの別branch進行は今回の候補と区別し、待機・変更していない。
3. 同Linuxユーザーの関連Python runner名にm9予測/PF価格取得jobを観測しなかった。processの停止・変更はない。
4. Dの既知Full PF/p7 private namespaceはtop-level名だけを読んだ。calibration/holdoutのみで、m9/longq namespaceはなかった。元legacy costの指定6 physical ID×EXPLORATION/CONFIRMATIONの正確なresult pathも存在しなかった。ファイル内容、QPY、cache、vector、観測価格は読んでいない。
5. 既存PR-DF m9のK0/K1等は別physical/law・構造量であり、今回のFull PF/p7 compiled価格予測ではない。ファイル名と既存SOURCEで区別し、その数値を再解析していない。

各点について同じ観測前確認が通り、保留は0点。出所不明のjob、同じ予測、取得開始marker等がある場合に該当候補を保留し、予測計算を起動しない経路も人工testで確認した。

これは、指定された公開SOURCE・既知server metadataでの確認である。未列挙の別名archive・別サーバーを網羅していない。その限界を`preopening.scope/limitation`に保持し、全履歴で未実施と断定しない。確認不能な反対証拠を0扱いしてはいない。

## 12個別予測

固定H4 DF10、linear 1.00 Å、STO-3G、元H/state/snapshot・fragment順、Full10又はp7 B0I、P0、symmetric-directional control、full measured Hadamard wrapper。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、元native/lowering・scalar一回を保持する。全点m9、T=51.2、R/Kなし。

`C_g(q) = beta_one,g + beta_q,g*q`。X/Yを独立に12回評価した。全予測は分母1だが、JSON/CSVではnumerator/denominatorを別々に保存する。configurationの完全ID、binary64 actual幅のhex表現、model_group、元係数selector、scalar/source/profileは機械可読表にある。

| route / formula | physical ID | axis | q | actual幅 | 予測RZ（分母1） |
|---|---|---|---:|---|---:|
| Full S2 | a4ec53ff4e60d60a2930 | X | 224 | .2285714285714286 | 1745491 |
| Full S2 | a4ec53ff4e60d60a2930 | Y | 224 | .2285714285714286 | 1745491 |
| p7 S2 | 1bd0a5dd4bbf876da1d6 | X | 224 | .2285714285714286 | 1235219 |
| p7 S2 | 1bd0a5dd4bbf876da1d6 | Y | 224 | .2285714285714286 | 1235219 |
| Full Y4 | b2c479e7805c4d1f0b30 | X | 96 | .5333333333333333 | 2259667 |
| Full Y4 | b2c479e7805c4d1f0b30 | Y | 96 | .5333333333333333 | 2259667 |
| p7 Y4 | 9924c226ee246283ec54 | X | 96 | .5333333333333333 | 1588051 |
| p7 Y4 | 9924c226ee246283ec54 | Y | 96 | .5333333333333333 | 1588051 |
| Full M5 | a53547a5fb707b9550d5 | X | 32 | 1.6 | 2741203 |
| Full M5 | a53547a5fb707b9550d5 | Y | 32 | 1.6 | 2741203 |
| p7 M5 | 60111109e0024d1a1064 | X | 32 | 1.6 | 1940883 |
| p7 M5 | 60111109e0024d1a1064 | Y | 32 | 1.6 | 1940883 |

これは未検証のモデル予測であり、新しい直接価格、compiled価格の観測、価格優劣、モデルのm9 PASSではない。原係数を一切変更せず、m3価格・m6観測をfit/補正に使っていない。

## 採択・固定したm9限定規則

`r = abs(frozen_predicted_RZ - observed_RZ) / observed_RZ`を**観測値分母・丸め前のexact値**で比較する。

- 有効な観測、同一物理点/axis/SOURCE・compiler/wrapper/scalar等の実装gateが成立し、`r <= 1/50`：個別PASS。
- 有効な観測で`r > 1/50`：個別FAIL。
- 取得・出所・実装gateが成立しない、又は必要な値が欠測：UNKNOWN。価格差のFAILへ置換しない。
- **指定12点すべての個別PASSでm9 campaign成功**。X/Y、route、公式を平均・poolしない。未確認/UNKNOWNがあれば全PASSではない。
- 自動refit・特徴量/閾値変更はしない。m9以外への規則・資格拡張はしない。2%は統計的95%保証ではない。

この規則だけを今回採択した。Dの実回路取得・開封・判定は後続の別作業で、Cでは実行しない。現在の`observed_RZ/decision/campaign_pass`はnull、`qualified_for_candidate=false`。

## 保全・ロードマップへの影響

元FROZEN_MODELS、m3保存12軸、m4/m5較正24軸、m6独立holdout12軸は、それぞれのSOURCE・元role・結果・STOPを保持する。m3は直接観測点で較正/holdoutに追加せず、m6は既存独立holdoutとして保持し再判定しない。

ロードマップは「m9予測と判定規則が未固定」から**「12予測とm9限定規則を事前固定済み、Dの同物理点取得・独立判定待ち」**へ更新するのが差分。本体は統合担当が更新する。既存48直接価格は増えておらず、今回追加したのは12モデル予測だけ。m7/m8・m10～m12、別幅、他方式、十分N・G・winnerの資格は増えていない。

**mandatory STOP：commit/push・remote照合とDへの引渡しで終了。観測取得・開封・判定、他round予測、N/Gへ自動進行しない。**
