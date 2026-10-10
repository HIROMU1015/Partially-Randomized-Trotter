# 2026-10-10 Track A：H4-P読込gate保全修正・保存再監査

利用者の「作業を進めて」を、直前のread gate修正・合成容量検査・保存receipt再監査へ適用する。
[新v2報告](../track_a_ax2b_h4_native_receipt_reaudit_v2.md)を追加し、旧親STOP・source・結果を変更しない。

原4MiB per-JSON gateを編集せず、原16MiB aggregate budgetを残りbudgetとして各readへ渡す新保存検証経路を用意する。
理由はB3の4,443,419-byte JSONが原aggregate budget内にあるのに旧readerだけでSTOPしたため。
historical source closureは旧実行commitのtreeから復元し、新audit sourceとは別にhash固定する。
追加moduleを理由に、原source bytesの照合を省略したり新source実行だったことへ読み替えたりしない。

36合成testsをlocal engineering evidenceとして保存。保存再監査はstdlib-only、分子再計算・H4-P再実行0。
PASSは保存資料の整合性だけで、元の実行statusや科学的認定を変更しない。
再監査後STOPし、science seal・launchは後続の別scopeと認可へ分離する。
`H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`を維持する。

保存再監査完了：PASS。旧16 file/8 cell・実行時source170件をcommit bytesまで照合し、新audit source172件/継承freeze187件と区別した。
原STOPは不変。新分子処理0、H4-P再実行0、science seal0、H4/H6 launch認可0。mandatory STOPで次の指示へ戻す。
