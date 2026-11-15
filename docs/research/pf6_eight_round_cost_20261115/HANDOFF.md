# D/C向け：PF6既知8round費用と、未取得5roundの判断境界

[REPORT](REPORT.md)／[全96軸](AXIS_PRODUCTS.json)／[6候補の部分和](PARTIAL_SUMS.json)／[15候補対](PAIRWISE_CONDITIONS.json)。固定解析SOURCEは **732c703b0ebfd6ca4fd32195fa0d60dd04298c3e**、B固定配分/対応は8dba096d、C正本alignmentは0b4b2d94。結果公開commitは解析SOURCEと区別し、公開時の固定URLを使用する。

## 判断を変え得る追加証拠

未知部分は各候補について

```text
U_i = 31(C_i,7,X+C_i,7,Y) + 27(C_i,8,X+C_i,8,Y)
    + 19(C_i,10,X+C_i,10,Y) + 15(C_i,11,X+C_i,11,Y)
    + 11(C_i,12,X+C_i,12,Y)
```

と保持している。これは価格予測でも数値補完でもない。各候補の10軸、計60軸はINPUTSにphysical/canonical/round/axis/N付きでC=nullとして残した。X/Yの価格一致・順位・単調性・上限を仮定しない。

既知差が小さい対から見ると、次が判断境界となる。

| i（既知部分が安い） | j（既知部分が高い） | K_i | K_j | 必要な未知差Δ=K_j−K_i |
|---|---|---:|---:|---:|
| T-M5 | F-Y4 | 146,850,902 | 153,106,880 | 6,255,978 |
| T-Y4 | F-S2 | 107,719,628 | 115,251,570 | 7,531,942 |
| T-S2 | T-Y4 | 81,621,086 | 107,719,628 | 26,098,542 |

**全15組**はREPORTとPAIRWISE_CONDITIONSにある。各行で`U_i−U_j=Δ`が同額、`>Δ`ならiが高くなり、`<Δ`ならiが安い。例えば最も近いT-M5/F-Y4では、未知5round全体でT-M5側の加重費用がF-Y4より6,255,978を超えて大きければ順序が逆転する。T-Y4/F-S2では7,531,942。これは未取得の価格を予測した値ではなく、保存既知差からのexactな境界である。

最小部分和のT-S2は81,621,086だが、全13round最小とは認定しない。最も近い差の候補対が、最終winnerを決める最重要対だとも断定しない。KiとUiの分布・価格domainが未確定なので、この差は次の取得判断の材料である。全15組が未知加重差に依存し、現資料だけでは全体順序を確定できない。

## なぜ1軸を指定して決着できないか

各pairの未知差は**2候補×5round×X/Y**の符号付き和。仮に1軸を測っても、残りの項には有限の上限・下限付き差や相関条件がなく、任意の符号の残差が残る。したがって、現在の証拠から「この1軸だけで決着する」とは言えない。大きいN_mの軸が常に判断を決めるとも限らず、物理・実装を含む価格差が必要。

D/Cが今後取得・受入する場合は、該当pairの未知加重差を実証する価格、又は科学的に資格ある**同時**区間/差の範囲が必要になる。価格区間を今回新たに設定せず、m6/m9の個別2%モデルPASSから未取得5roundの区間を作らない。新取得・モデル資格付与・全G集計は今回の作業範囲外。

## 元receiptとdecoder IDの対応

AXIS_PRODUCTS各行はB original configuration、B total configuration、physical ID、original/total candidate IDを分け、元price_generation_configuration_id/candidate_idとsourceをそのまま保持する。`input_locators`のindex/receipt/sidecar/guard、implementation_evidenceからD/Cの元軸まで戻れる。旧cohort nullやEXPLORATIONも元のまま。

6候補のH/state/snapshot・law・formula・q/actual幅・scalar・compiler/lowering・full measured wrapper/P0を既存C guardで照合済み。今回のprice_classはDIRECT_COMPILED_DETERMINISTIC、同一の決定論回路の直接価格であり、別roundのモデル予測や有限cohort平均ではない。各軸Nを1度だけ掛け、normalization・古典価格標本数を加算/乗算しない。

Bの固定Q3結果は保存componentからの閾値適合と、原API return・CPU/RAM欠測の部分保存/運用UNKNOWNを維持する。今回の部分費用分析は不足したQ3記録を回復せず、N資格producerも発行しない。十分N、global RMSE、正式G/winnerは未認定。

## ロードマップへの差分

統合担当は、決定論PF6の**direct coverage96軸と固定shot配分の接続・既知8round整数部分和・15逆転条件が公開済み**という成果を追加できる。未取得m7/m8/m10/m11/m12の60軸、全13roundのG比較、価格モデルの外挿資格、十分N/物理保証、PF6原API/資源記録の欠測は未解消として残す。既知部分順位を全体winnerとして表示しない。ロードマップ本体は編集していない。

固定SOURCE/input/raw/旧結果/grant/ledger/marker/STOP、D/C worktreeとjobを保全。**公開・remote照合後mandatory STOP。** 今回の価格取得・Q3再実行・fit・予測・shot探索は0件。
