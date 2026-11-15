# C/D・統合担当へ：幅hの粗幅予算内2配分Q3

**今回の2列では、細幅hで粗幅達成費用435,481,090 RZ以下かつQ3適合の配分は見つからなかった。** 数値gateは通過し、どちらもMSE下界>τのモデル不適合。これを幅h全体の不可能性や粗幅の最適性へ一般化しない。

| 規則 | N_X=N_Y | モデル費用 RZ | 粗幅との差 RZ | MSE区間 Ha²（表示） | 判定 |
|---|---|---:|---:|---|---|
| A_SHAPE | [21,19,18,17,15,14,12,11,9,8,6,5,3] | 421,323,290 | −14,157,800 | [2.6431698095527471E−5, 2.6431698122879336E−5] | 不適合 |
| B_PRICE | [59,55,51,47,43,39,35,31,27,16,7,2,1] | 434,203,864 | −1,277,226 | 約1.0050102793485435E−5（exact区間は下記） | 不適合 |

正本は[REPORT](REPORT.md)、exactの[EXACT_RESULTS](EXACT_RESULTS.json)、表示[SUMMARY](SUMMARY.json)、実行前固定[INPUTS](INPUTS.json)と[ALLOCATIONS](ALLOCATIONS.md)。科学実行SOURCE **8f9bf401009329683d1244675d6fdfd663689159**、caller `scripts/run_p7_s2_h_budget_q3_fixed.py`。両件の原API・partials・資源を保存し、callは各1回・計2回、retry/UNKNOWN/INCONCLUSIVEはいずれも0。

予算は粗幅b4296b110のB_PRICE保存達成額を引用。capは元N0（末尾15,11）で、直近適合列（末尾13,4）へ変更していない。元configuration e6754564…、total view b8ceb775…、binding 8e1e602d…、幅h固有schedule/13round packetを保全し、粗幅のreceipt/physical IDへ付け替えていない。decoder/target/IID raw/loss/exact τはbec032fc2を継承。

## 価格側・Cへの接続

[C_RESULT_JOIN](C_RESULT_JOIN.json)は、各allocationから元/total configuration、signal locator、13round×X/Y、元physical/canonical候補ID、N、元価格table/receipt/sidecar、Q3原API SOURCEへたどれる索引。入力価格はecf45508aの26軸だけで、価格取得・予測・fitは行っていない。

| 配分 | m0〜4直接 RZ | m5〜12外挿 RZ | 当該配分の外挿割合 |
|---|---:|---:|---:|
| A_SHAPE | 5,645,442 | 415,677,848 | 98.6600688512% |
| B_PRICE | 16,000,932 | 418,202,932 | 96.3148803300% |

直接10軸・外挿16軸を保持。旧配分の98.388806%は使用していない。外挿のモデル誤差上界・CIはnullで、ここでのTは採用価格規則の条件付きモデル費用。両Tは予算より厳密に小さいがQ3不適合のため、達成費用改善とは呼ばない。正式G入口を緩めず、十分N/G_qualified/winnerはnull。

## ロードマップへの影響

ロードマップ本体は統合担当が更新する。

1. 幅hの精度余裕を使い粗幅達成費用まで下げる、最初の2規則による有限検証が完了した。現在この予算以下で確認された幅h適合点は今回menuにはない。
2. 粗幅の435,481,090 RZ適合点と、幅hの直前配分の適合結果は不変。今回新2点の不適合を、公式・幅全体の棄却や最小必要Nの認定に使わない。
3. 追加配分・予算・N探索、長q価格の独立資格、十分N/物理RMSE/Gは未評価。今回の結果を根拠に自動実行しない。

zero/tieはtotal規則で会計、残余mass/lossを保存。AδEを再加算せず、元Minkowski・ground/state/実law/数値certificateの未資格を維持する。旧SOURCE/raw/結果/STOPを保全し、公開・remote照合後 **mandatory STOP**。
