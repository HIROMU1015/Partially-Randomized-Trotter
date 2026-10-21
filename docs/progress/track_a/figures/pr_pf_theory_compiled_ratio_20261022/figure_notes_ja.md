# 図の注記・限界

1. 主要求は現行最良2次PRと最良2次PF（Full/B0/B0+I）の比較である。指定固定資料では、
   理論・compiledで同一候補・同じ保存shotを確認できる4値が現行20条件のいずれにもそろわない。
   空の主図は作らず、旧固定H4の参考図と充足性表を保存した。
2. H2はtol-only DF3、現行H4はDF10、H6はDF19、H8はDF28。全て1 Å・STO-3G・T=.8・P=0、
   signed bias・倍率1・両軸会計である。理論参考は旧H4 legacy固定DF12。
   split L_D・q・δ=.8/q・r・K・input/state/snapshot/compiler hashはfigure_data.jsonの各保存recordにある。
   これは固定時間T=.8の信号精度会計であり、PF係数Cのdelta-fit窓や長RPE全round費用の評価ではない。
3. PR選択は従来全方式図の探索A n8をそのまま使う。PFはH2/H4の保存winnerが既にS2の場合に転記し、
   H6/H8では既存replacement順位のordinal順で最初のS2を転記した。費用を新しくsort・rank・argminしない。
   H4 ε=.001の全体保存winnerはS4であり、S2限定の保存winnerがないため未確定。
   3方式別S2最小の保存点は消さずに全て残した。新しい候補探索を意味しない。
4. B0/B0+Iのcompiled費用はあるが、同じ候補についてnative理論総費用は確認した資料にない。
   PF q依存compiled単価モデル、直接計数単価、ALL_SAVED458置換単価を理論ゲート費用へ改称しない。
   既存単価にshotを掛けてGを新規構成することも行わない。
5. H6/H8のALL_SAVED458のcompiled価格・保存順位は従来図から継承した条件付き値。
   profile/runtime同一性未証明などの留保は保存recordのevidence_detailsに保持する。
   同じcommitのStage Bはnoncancellationの旧n32 PRを扱う補足で、今回のsigned新探索Aには使わない。
6. H6/H8新A公開CSVにはshot原表が公開されていない旨がREADMEに明記されている。
   PR shotをnullとして欠測理由を保持し、旧n8/n32のshotから復元しない。
   この点も含め、旧理論資料と現行を無条件に接続できない。
7. 参考H4 q1はPR shots=(12305,8258)、q8は(7248,7212)、PFは(11057,7414)。
   方式同士のshot数を同じにする条件ではなく、各候補の理論／compiledで同じ保存shotを使う条件。
   保存Gをそのまま転記し、shot×単価による再会計はしていない。
8. NATIVE_MIXEDはPRのprefix A＋各closed finite-RTE tail B、PFはAと同じnative分解規約。
   元PR論文のFT Toffoli資源式ではなく、全回路B・最適compilerの費用・compiledの厳密上界でもない。
   lower/upperの両保存値を残し、今回描く2組では一致するので代表中点は作らない。
9. PRの法則期待値と保存compiled有限標本平均を比べる。同一列の理論→構築→compiledの照合は別資料。
   差にはevent構成・lowering・opt1等が含まれ、compiler最適化単独の効果へ分解しない。
   old_compiler_runtime_bit_identity_certified=false、population_mean_certified=falseを保持する。
10. 旧H4 S2 PFはε=.05だけ適格。他4精度は保存不適格を空欄のまま残す。
    H6旧保存PF理論対照はS4だけなので、今回の二次比較では除外する。
    スライドA.8のPR/PF順位逆転例もS4対照なので転用しない。
11. 主比較の理論G欠測、旧固定候補の不適格、S2採用未確定、条件非整合を別々のfieldで保持する。
    JSON null／CSV空欄は0ではなく、線や棒で架空値を表示しない。
12. 参考図ではPF削減率がPRより大きいが、現行最良B0/B0+Iの仮説A/Bについて結論を出せない。
    q1とq8の旧固定比較でも理論段階の優劣が異なる。原因・普遍的換算係数は認定しない。
13. PNG/SVG/PDFの表示と構造の自己確認を実施する。独立科学レビュー、追加science stage、evidence freezeではない。
    元資料、原図、スライド、論文、SOURCE、raw、result、grant、ledger、marker、STOPは変更しない。
