# 保存H4 q4_sample00／q64_sample00：限定資源照合

2026-11-09 JST。**指定2 pathのStrong／Reference X/Y、計8保存compiled QPYで、絶対価格の7指標すべてが保存値と一致した。** これはcompiled列の再計数であり、全pre-P0構造から価格を予測できた結果ではない。H4の新compile・simulation・全入力correctness・samplingは0。

## 科学条件・原本identity

Linear H4／隣接1.00 Å／STO-3G／Nα=Nβ=2、元H、number-CDF8／元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1。元Hの厳密groundを仮定し状態準備費は除外。q4/cut2とq64/cut32、C1追加融合OFFの保存価格である。強いsymmetricのGaussian共有・directional control・head relative fusion・tail共有・step融合・scalar集約とwhole-path Referenceを維持する。

P0はQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approx1／initially_zero=False／topology-free／宣言位相保持。元hashで入力を固定し、SOURCE・policy・cut・compilerを確認した。

- q4：C6a実行SOURCE6b666702、archive 6,950,831 bytes、SHA256 `a631cd394073f7770d2723eb0cfa93fc6a8f703534b9cdce88b7c126381a007b`。公開payload_archiveの選択raw/compiled各member hashとも一致。原本archiveを展開保存・変更せず、必要memberだけread-onlyで読んだ。
- q64：sampling/raw/初期P0 SOURCE6b1d613dの保存sample00。最終fe361b20のpayload_index／prices／SOURCEとbytesを照合し、選択raw4／compiled4のhashと一致。旧postcheckのnofusion修正や最終paired/χ PASSは保存証拠として維持し、今回再実行しない。
- H0実行SOURCE：2712a73f6e609e0f883563c400b4d1facecd1aa8。[SOURCE](../../../artifacts/track_d_t1_h0_p0_bridge/2026-11-09/run01/SOURCE.json)／[前後照合](../../../artifacts/track_d_t1_h0_p0_bridge/2026-11-09/run01/preservation.json)。private絶対pathは公開しない。

q4価格recordにはmetrics内qubit_countもある一方、q64では幅をpeak_qubitsに分離している。両者とも保存peak_qubitsへbindし、q4の重複fieldも一致を確認した。schema差を科学条件変更として扱っていない。

## 保存compiledの絶対資源

X/Yは各々を独立に検査した結果、下表の全値が軸間で等しかったのでまとめて表示する。X/Yを独立pathへ倍増した標本統計は作らない。

| path | 方式 | Axis | RZ count | CX count | RZ depth | CX depth | Total depth | Size | Qubits |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| q4_sample00 | Strong | X/Y各同値 | 21020 | 8024 | 5013 | 3874 | 11671 | 42839 | 9 |
| q4_sample00 | Reference | X/Y各同値 | 21239 | 7336 | 2571 | 1590 | 5686 | 42562 | 17 |
| q64_sample00 | Strong | X/Y各同値 | 331655 | 127232 | 80644 | 62494 | 188087 | 677752 | 9 |
| q64_sample00 | Reference | X/Y各同値 | 328327 | 113104 | 38316 | 23536 | 84948 | 660488 | 17 |

全8の再計数は保存絶対値と一致し、compiled列へbridgeを適用しても追加CX削減は0だった。q4ではReferenceのRZは219増、q64では3328減、幅は両方で8増える。CX／depthとRZ／幅を一つの単価へ統合しない。RZ→T、topology-free depth→実機wall timeの換算もしない。

これらは2固定pathであり、C7-4の88標本やC8-1の8標本を新平均・信頼区間へ再集計したものではない。208−224qの定理・全sampling lawの期待価格・全RPE総費用は認定しない。

## Pre-P0情報だけで取得できた部分

rawのnative連続segment内に限って同じCX ruleを適用した。native部分以外を削除したfull回路の価格として扱わず、不適格命令のたびにfrontierを切った。

| path | 方式 | Native連続segment数 | Raw native部分のCX | 局所ruleの削除CX | Full P0予測 |
|---|---|---:|---:|---:|---|
| q4_sample00 | Strong | 102 | 1546 | 14 | 未資格・null |
| q4_sample00 | Reference | 107 | 250 | 14 | 未資格・null |
| q64_sample00 | Strong | 1649 | 24380 | 134 | 未資格・null |
| q64_sample00 | Reference | 1714 | 3884 | 134 | 未資格・null |

X/Yでこの局所集計も同じだが、Y readoutのsdgという追加未対応opcodeは個別記録へ残した。これらの削除数は両方式共通で、Reference固有の追加利益を示さない。また既存P0が既に行う相殺を含み得るため、**保存compiled CXから14／134をさらに引かない。**

rawにはXXPlusYY Gaussian、RZZ、CRZ、CSWAP、P、H、Y側Sdg等が残る。Gaussian／relative transition／step融合／A†/B／χを既存emitterで保ったままnative化し、P0の1Q再合成と界面を閉じる証拠が不足している。native部分のdepthを足してfull-wrapper depthとする操作もしていない。

## 結論と保全

compiled読み出しによる厳密照合と、pre-P0 raw局所の意味を保つCX変換は取得した。full H4のcompile不要価格評価器は未完成である。原本入手不能によるHOLDは0、別pathへの置換も0。原本event/QPY/NPZは非公開保全し、公開はscalar・hash・この比較のみとする。[全8 record](../../../artifacts/track_d_t1_h0_p0_bridge/2026-11-09/run01/saved_H4_comparison.json)。
