# 現行S2理論／compiled比較：選択確定と入力不足（2026-10-22）

現行H-chain比較の理論G取得と作図の新規実行承認を受け、専用branchで実行可能な保存値処理と技術検証を行った。
[図・全データ・不足入力・STOP](../progress/track_a/figures/pr_pf_s2_theory_compiled_main_20261022/README.md)。

H4・ε=.001は、現行tol-only DF10・linear1Å/STO-3G/T=.8/P=0・signed/factor1・両軸full wrapper RZ会計において、
登録Full/B0/B0+Iの保存適格S2候補21件から既存exact G最小・同点ID規則でB0_p7_q8_2ndを選んだ。
保存compiled Gは1,666,873,406,028、保存shotはcos=22,035,273、sin=18,163,285。shot数の再計算なし。
他19条件のPR探索A n8・PF選択は前回packetの値を維持。

40人工counter/adapterテスト、旧固定S2理論単価と保存shotによる17条件のexact総費用回帰はPASS。
新SOURCEは`e7838a8c1a816ba39b4a2e9e4a9488f56cd746a5`、新grant・one-shot marker・ledgerを使用。原科学SOURCE/旧grant/STOPは不変。
元native counter（f01fa5fb）の数学的定義・bytesは保全。
B0Iは保存scalarを末尾P(-scalar*T)で1回扱うという技術規約を人工テストで確認した。

**現行の新理論Gは0/20条件、現行物理入力に対するcounter実評価は0。**
原preparationの順序付きbasis操作・対角係数・完全なTail法則、H6/H8新Aのshot原表がこの環境で取得できない。
ユーザーも元保存先がこの環境にないと確認した。hashのみから復元せず、別cohort・旧DF12・compiled単価モデルを代用しない。
入力が取得可能なら実行する承認はあるが、必要な原データが欠けるため理論Gの新評価には到達していない。

図Aのcompiled系列のみ5図をCompiled onlyと明記して保存した（15形式出力）。Theory曲線と図Bは未作成。
仮説A/B・原因・母平均・最終全RPE費用を判断しない。
独立確認、科学的詳細レビュー、新science stage、最終evidence freeze、スライド/論文本文変更は行わない。

次段はユーザーが元入力の保持環境/取得先を指定し、Codexが同じ候補・SOURCE・cohort・hashを照合した後に不足理論Gを評価する段階。
追加生成や別データでの穴埋めは自動実行しない。ローカルcommitまで、push未承認。mandatory STOP。
