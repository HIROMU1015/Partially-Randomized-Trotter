# 2026-11-16：D2 PR-DFの次分割選択

[REPORT](../d_pr_df_next_split_choice_20261116/REPORT.md)／[HANDOFF](../d_pr_df_next_split_choice_20261116/HANDOFF.md)。LD3と遠いLD7を最終2案にし、PR-DF LD7を先行提案。保存prefix3/5/7/9のλ/scalar・旧signal/price条件を差分照合し、費用順位は未評価。LD2長round1点は候補選択を変えないため後回し。後続は新R列の全R1確認を条件に、signal13 round・own価格20 compile・共通V0→独自A/B最大3 Q3の有限案。候補/schedule/科学数値は新生成0、計画公開後STOP。

旧HANDOFFの長round確認優先を、今回の「次にどの分割を調べるか」という問いで再検討した。LD2単一長roundの上下どちらの価格でも、未知の大head/小tail領域を調べる選択はLD7のままなので、この探索順の先行依存にはしない。これは既存金額順位の不確かさを解消した判断ではない。LD7が将来安ければLD9、高い/同額ならLD3へ、という次の選択規則を提案。LD3/5/9は未評価で残す。

SOURCEのceil規則は読んだだけでR列を新計算していない。LD7の価格案(1,q)は全R1が後続metadataで成立した場合に限定。合わなければ科学call前STOPとし、κ/R/modelを勝手に変えない。c+s_Rも未計算。保存source/raw/6点比較・m4独立holdout・旧STOPを保全し、ロードマップ本体は統合担当へ渡す。
