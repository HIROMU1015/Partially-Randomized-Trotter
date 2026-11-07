# D内部PF：固定H4の実験設計

基点48209b088ca0dffd22a8d3fddcb82888eb29d3a5。既存Track C H4/LD3、元H保存ground、18round/.02、epsilon1.5936e-4Ha、alpha.05、beta.4を維持。旧SOURCE・結果・契約・STOPは保全し、新出力だけへ保存する。ユーザーの今回の一括実行許可内で計算・修正・比較を進める。

旧C-N/S/P・C2-A/B・compiler A保証・混合公式の保存結果を参照した。旧CはDF生成順0123の片側D sweepと逆sweep、全体Y4、高次/processingを比較しており、LD3の全24内部順序、DだけのY4、fragment時間の対称再配分はこの固定入力で未検証。B2条件は使用しない。

段階Aは24順序を全列挙し、旧多項式logによるground H2をD内部・mixedへ分解。既存Dのself-adjoint S2/Y4をD(h/2)R(h)D(h/2)へ接続する対照も評価。RTEに負時間を追加しないD-only高次が、全体Y4との新しい自由度になる。

同じ主要仮説内で、符号の異なる順序があれば、正時間palindromicな3段合成のground三次相殺を調べる。全体generatorが同じなのでH2は各stageの時間係数の三乗和となる。必要ならfragment1個の対称時間再配分を追加し、全PRのground主要係数を評価する。係数選択はBCHから行い、別h/長時間/有限RTEでholdoutする。作用素次数向上・普遍的新規性とは扱わない。

科学的に有望なものに同じh/r機会、半刻み対照とFull S2/Y4の強baselineを与え、有限RTE・branch・native資源・総Gを評価する。compiler Aと既存event builderを再利用し、旧価格モデルの別回路族への無検証移送はしない。未保証のcompiler diskを0扱いしない。合理的候補が尽きたら負結果も保存し、公開照合後mandatory STOP。詳細GPT科学レビューなし。
