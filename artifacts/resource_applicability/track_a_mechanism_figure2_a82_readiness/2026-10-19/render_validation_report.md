# Figure 2の入力・SOURCE・描画検証

描画SOURCEと入力4 CSV／出力SHAはrender_source_manifest.jsonを参照。

Figure 2Aの90点と2Bの26点、計116点すべてについて、contrast・axis・metricを固定元表へ結合し、
元tokenの文字列一致、float(Fraction(token))座標、5桁／13桁の表示丸め、PF class・cohort・nの一致を確認した。
plot_values.csvは実際にscatterへ渡した変換値を同じ関数で記録したもの。新しい比・積・平均・SE・統計・科学的分解は0。
SVGのXML、PNG signatureと画素寸法、PDF signatureを確認した。最終PNG2枚を主担当が目視し、
ラベルの可読性、model中抜き、cohort区分、H4 P注記、A82欠測注記を確認した。
初回B注記の重なりを修正し、初回出力はlocal監査へ保全した。実データに合わせた科学条件や選択規則の変更はない。
H4 Pの値は主表に混ぜず別表へ転記し、両図のH4数値は元n8/n32のまま。A82説明因子を0補完していない。
これは作者自身の技術確認であり、独立科学再現・immutable CIではない。
