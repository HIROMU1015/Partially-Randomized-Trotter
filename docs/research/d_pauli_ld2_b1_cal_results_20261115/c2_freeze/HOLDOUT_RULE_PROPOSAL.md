# PR-Pauli LD2 m5/m6独立holdout：観測前の規則案

**提案・未採択。推奨は記述的検証：4件の誤差を報告し、モデルのPASS/FAILを付けない。**
本書の公開は規則の採択、holdout取得・開封・比較計算の実行許可を意味しない。

## 1. 用途・精度・推奨理由

用途は、固定した価格モデルを同条件のm5/m6へ適用した際の有限cohortでの予測差を調べる
探索的診断とする。価格量は測定付きfull controlled Hadamard wrapperのcompiled RZ数
（状態準備費を除外）。H4直線1Å、STO-3G、保存DF rank10、L_D=2、K=2、DF head S2＋Pauli tail、
辞書・lowering・scalar・compiler profileは[凍結HANDOFF](HANDOFF_D2.md)の固定条件を維持する。

算術平均・誤差は整数／有理数でexactに保存し、小数・%表示の丸めを判定に使わない。
**予測精度の許容%は設定しない。算術のexact性はモデルの予測精度の保証ではない。**

数値閾値を今回推奨しない理由は次のとおり。

- 固定資料には、このPauliモデルを用いる判断の許容損失、見分けるべき価格差、または
  下流の価格誤差予算がない。2%、5%等を採る研究上の基準を特定できない。
- 較正は各軸5 cellの小cohortで、P2R66はS66の親pathに依存する。較正残差は訓練点の診断であり、
  独立holdoutの許容差や誤差分布を与えない。m5は2 path、m6は1 pathで、母平均・同時区間の
  coverage契約もない。観測との差には有限pathの変動とモデルのずれの両方が含まれ得る。
- [既存価格検証計画§7](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/87c525a7b72d70314d87f59042efd663d5566735/docs/research/paper_g_price_routes_20261114/PRICE_VALIDATION_PLAN.md)
  は許容差を選択肢として留保し、Pauli用の区間契約も未設定。
  決定論PFの2%や別domainの過去基準をPauliへ移送する根拠はない。

工学的な閾値は用途に基づいて採択できる。少数pathであることだけで閾値設定が不可能になる
わけではないが、現在の固定資料から特定の値を正当化するには用途側の条件が不足している。

## 2. 固定比較単位と観測量

予測の正本は`b74e4e536392c2bbe94cd371436615fc56c0564f`の
`docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json`。
`predictions`を配列順ではなく`(m, axis)`で対応付け、保存済み`predicted_RZ`をそのまま使う。
係数からの予測再生成・再fitは行わない。

| 比較単位 | q / R / T | 観測量 O（同じ軸だけ） | 凍結予測 P |
|---|---|---|---|
| m5-X | 32 / 33 / 3.2 | 固定fresh path0-Xとpath1-Xの算術平均 | 6169957875465/90343469 |
| m5-Y | 32 / 33 / 3.2 | 固定fresh path0-Yとpath1-Yの算術平均 | 6169957875465/90343469 |
| m6-X | 64 / 66 / 6.4 | 固定fresh path0-Xの値 | 13521761923033/90343469 |
| m6-Y | 64 / 66 / 6.4 | 固定fresh path0-Yの値 | 13521761923033/90343469 |

canonical physical IDはm5が`RMSE:01489f6e391e9917e96eaa56a0006a91b6099e60956cb5fcea3a516b24df5802`、
m6が`RMSE:8826b5c0b36fdb91115109a85c6f2eca40cf1989bb3b3e67d8cbcd1598043529`。
cohort・seed・特徴量・n_pathsは各予測の`metadata_source`（固定計画
`043c7567451e8f502a1cf0f425d32efe0a50f5dc`）に一致させる。

**個別6 receiptを保存**する：m5はp0X/p1X/p0Y/p1Y、m6はp0X/p0Y。
各receiptに固定locator、実行SOURCEとraw公開SOURCE、round、軸、physical ID、cohort、
path index、seed、fingerprint、RZ値、既存gate結果を対応付ける。
X/Yは同一pathの別測定軸で、6独立pathではない。3 fresh path・4比較単位・6 receiptを区別する。
X/Yをpoolせず、同値でも片軸を省略しない。m5の片方が欠けたら1 path平均へ変更しない。

## 3. D2次回指示に転記する比較式・状態・STOP条件

以下は**本案の採択と、別途の取得・開封・比較計算の実行許可があった場合**の指示案である。

```text
対象集合 U = {(5,X), (5,Y), (6,X), (6,Y)}。P[m,a]は固定正本の保存予測。
Z[m,a,i] = 指定fresh path i・軸aの、gateで有効と確認されたcompiled RZ数。
O[5,a] = (Z[5,a,0] + Z[5,a,1]) / 2
O[6,a] = Z[6,a,0]

各有効単位について、
  signed_error   d = P - O       # 正：過大予測、負：過小予測
  absolute_error A = |d|         # 単位：RZ
  signed_relative_error = d / O  # O > 0 のとき
  relative_error        = A / O  # O > 0 のとき。%は100倍して表示
分母は観測価格O。P、max(P,O)、較正価格、SE等へ切り替えない。
O=0なら相対誤差はnull、理由ZERO_OBSERVED_PRICE。dとAは報告する。
0/0も相対誤差0とはせず、epsilonや分母1への置換は行わない。
d=0はexact一致として記録するが、モデルの合格判定にはしない。

receiptと既存作用・実装gateの確認を、観測価格の平均・誤差算術より先に行う。
round/軸/physical ID、q/R/T、cohort/seed、辞書/law、SOURCE、profile/lowering/scalarを固定条件と照合する。
不足・不一致・gate失敗は該当単位を無効または欠測とし、価格0や予測値で補完しない。
取得失敗や入力不成立ではSTOPし、自動retry・再sampling・path差替えはしない。
既に有効と確認できた単位の行は保全し、不足単位を除いてcampaign成功とはしない。

全4単位が有効でO>0：campaign = DESCRIPTIVE_COMPLETE
全4単位が有効でO=0を含む：campaign = DESCRIPTIVE_COMPLETE_RELATIVE_UNDEFINED
いずれかが無効／欠測：campaign = INCOMPLETE_OR_INVALID（理由を各単位に記載）
いずれの場合も model_verdict = NOT_ASSESSED_NO_ADOPTED_TOLERANCE。
モデルのstatus/qualified_for_candidateは凍結時の未検証・未資格状態を維持する。
4行を必ず報告し、無効／欠測行のO・誤差はnull。X/Yやround間の平均誤差で代替しない。
記述報告を終えたらmandatory STOP。追加sampling/compile、再fit、feature変更へ進まない。
```

相対誤差の分母は既存C `price_model_validation.validate`の観測価格基準とそろえた。
同APIは正の許容差を要求するため、本案では呼ばず、架空のtoleranceも渡さない。
取得・作用gateのPASSはモデルのPASSではなく、`DESCRIPTIVE_COMPLETE`も精度認定ではない。

**有効観測の許容差超過との区別**：本推奨案は閾値を置かないため、モデル超過判定は行わない。
将来ユーザーが開封前に別案として相対許容差τ>0を採択する場合には、同じ分母で
`A <= τ O`を境界込みの許容内、`A > τ O`を有効観測での精度不適合として区別する。
4件すべて有効・定義可能かつ許容内の場合だけcampaign許容内とする。
全4件が有効・定義可能で1件でも超過ならcampaignは精度FAILとし、再fitせずSTOPする。
欠測・gate失敗・identity不一致は精度FAILへ読み替えず、campaignは未完了／無効とする。
O=0の相対判定も未定義のままとする。これは本案でτを採択したことを意味しない。

## 4. 報告から主張できる範囲

主張は「凍結予測を指定fresh有限cohortの4単位と、事前宣言した式で比較した」まで。
本案には合格条件を設けない。小さい誤差やexact一致が得られても、モデルの資格は昇格しない。
m6の1 pathから標本分散・SE・CIを推定せず、X/Yを独立標本として代用しない。
全lawのE[C]、長roundの精度、G、LD1移送、方式間の総費用順位は認定しない。
実LD2のorder2 product・非自明位相の未観測UNKNOWNも本提案では更新しない。

**事後閾値の扱い**：同じ4観測を見て選んだ閾値による判定は探索的・事後的と明記し、
事前固定の独立検証として扱わない。凍結済みの係数・予測があっても、この区別は変わらない。
その閾値を事前固定した独立検証が必要なら、別に定めた検証対象・許可が必要になる。

本書は提案のみ。今回は観測値の取得・開封・数値再解析・再fit・sampling・compileを行わず、
規則案をcommit/pushして**mandatory STOP**する。採否とロードマップ影響は
[HANDOFF](HANDOFF_HOLDOUT_RULE.md)を参照。
