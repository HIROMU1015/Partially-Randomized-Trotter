# C・統合担当へ：p7/S2粗幅の初回局所shot探索

**厳密に安いQ3適合点あり、新規Q3は1件で早期終了。** 正本は[REPORT](REPORT.md)、[SEARCH_EXACT](run01/SEARCH_EXACT.json)、[SUMMARY](SUMMARY.json)、[探索履歴CSV](SEARCH_HISTORY.csv)、[C_RESULT_JOIN](C_RESULT_JOIN.json)。科学実行SOURCEは **e7d24c2192d4640137bdaf993a4e954a49f7932c**、結果公開SHAとは別である。

```text
対象：B0I p7/S2・要求幅16h/7、元configuration a6e55777…
変更：m12のX/Y各軸4→3、他roundは元B_PRICE点のまま
N_X=N_Y=[59,55,51,47,43,39,35,31,27,23,19,13,3]
条件付き費用：415,721,884 RZ
元435,481,090 RZからの削減：19,759,206 RZ
MSE区間（表示）：[1.951734976728839865403189E−6,
                   1.951734976728839865403194E−6] Ha²
判定：MODEL_BOUND_MEETS_ADOPTED_THRESHOLD
停止：FIRST_STRICTLY_CHEAPER_MODEL_FEASIBLE_POINT
```

MSEは元点より大きいが同じτ以下であり、改善した量は費用である。新call/retry/振替は1/0/0件。m12：4→2、m11：13→12/11、振替poolは未評価。計画上の残枠を使って最小化を継続せず、未評価案の適否・最小N・局所最適を推定しない。

C_RESULT_JOINは元/total configuration・binding、carrier06a15ee4…の13round packet、allocation、元physical/候補ID、X/YごとのN/C/積、1002f1dfeaの元価格行、今回の結果SOURCEを対応付ける。価格は**16直接軸＋10条件付き閉包軸**で、今回の費用内訳は81,621,086＋334,100,798 RZ。receiptの付け替えや価格再取得はしていない。閉包の条件・価格誤差/CI=nullを維持し、正式G資格を新たに与えない。

採択済みZ1＋T1、保存solver proxy、deterministic rawのideal IID二項、全事象平方loss、exact τを維持。新閉包・質量・残余・MSE/moments gateを確認し、原API/partials/資源を完全保存。callerと記録済み子99件は終了code0。14人工/回帰test PASS。実装は既存kernel/workerの再利用と有限選択順序のみで、旧結果は不変。

ロードマップ本体は統合担当が更新する差分：

1. p7/S2・16h/7の固定配分比較に、新しい条件付き達成費用415,721,884 RZの点を追加できる。元435,481,090 RZの点も保存参照として残す。
2. これは最大8件の有限計画で最初に見つけた改善点で、有限menu全体や配分空間の最小費用ではない。他方式との新費用比も今回未計算。
3. 十分N・物理的RMSE・ground/実law/数値certificate・G_qualifiedは未認定。追加配分や他方式への展開は別判断とし、今回の残枠を自動流用しない。

元Minkowski・AδE非二重加算、旧SOURCE/raw/結果/STOPを保全。公開・remote照合後 **mandatory STOP**。
