# Claim–Evidence Matrix 更新案

固定base：bf8c19057fd82c900c6234152e152e79cb52e707。旧6行の全文はJSONのoriginal_rowに保全し、採択前の追補案として保存する。

成立した主張とは保存条件に対する対応が確認できた範囲を指す。母平均優位・厳密保証や論文採否を新認定しない。

|ID|提案する主張|証拠が成立する範囲|未解決・主張の上限|
|---|---|---|---|
|CE1|強化PF比較集合による点費用結論の変化を、保存domainとcost class付きで記述する。|従来のH6/H8保存比較を維持。H4はDで保存代表10/10不変という別provenanceを追記。|全domain実測最良・母平均最良・登録外最適性は未認定。旧32の数値を書き換えない。|
|CE2|H8探索n8の両端改善と独立固定n48の2比較を別の証拠として示す。|n48通常評価の保存点削減率2.935924%／3.302597%。候補と固定PF、n8との分離を明示。|高次event未観測、母平均優位未確定。旧union n32のA82説明は依然欠測。|
|CE3|中間精度の保存点費用削減を、固定候補・cohort・model条件付きで引用する。|保存された点比較にexact費用分解と31比較の条件付き説明を追加できる。新winnerを選ばない。|連続精度窓・全域の利益・rare頑健性・selection-adjusted保証は主張しない。|
|CE4|32比較のexact費用分解と31比較のnormalization/headroom/丸め因子が保存値を説明する。|R_G=R_N×R_Ceff、PF軸重み分解は既存exact結果。説明因子はempirical allowance下のbinary64/log整合。|A82.PRは原入力回収のみ、未検証。Ceffのshot構成依存を因果compiler効果と呼ばない。|
|CE5|signedと経験的非相殺の保存結果を別会計として維持する。|今回のcore32・説明31は通常signed/factor1のみ。H4 Dの非相殺5 endpointは別の転記表。|H6旧12候補分解欠測と全136非相殺null、factor10欠測等は未解消。新比較を追加しない。|
|CE6|有限時間signal・保存DF・P=0・指定compilerのRZ点費用という適用範囲を保持する。|local source-bound結果と技術検査、探索／選択後確認／独立固定確認を区別する。|母平均優位・rigorous bias・FT T資源・state prep/noise/backend・energy/RPE全roundへ拡張しない。|
|CE7|H4 D完了を追加provenanceとし、保存PF代表10/10不変を記録する。|通常signed5＋経験的非相殺5。登録104候補、費用あり71／保存不適格除外33。元core10行は不変。|P未完了。PR独立確認や大域最適性を認定しない。|
|CE8|A82原signalをreceipt SHA同一で回収し、欠測入力2項目の所在を確認した。|SOURCE/candidate/DF28/T=.8/旧n32のmetadata同一性。数値再解析0。|旧A82説明statusは欠測保持。新SOURCE/adapter固定と別実行承認が必要。|

各行の固定commit／SOURCEと元claim全文はJSON、対応する図表fieldはdata_manifest.jsonを参照。
H8 n48の費用削減率は既存説明結果の表示値の転記で、新しい統計量・差・rankingではない。
