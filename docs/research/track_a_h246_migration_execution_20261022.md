# H246独立確認：入力再構築と新規one-shot実行契約

2026-10-22 JST。ユーザーは欠落入力のSCF/DF/state/reference再生成、必要最小限のsignal/shot再計算、新しいprivate契約・seed namespaceでの固定独立確認を明示承認した。前回STOPは変更しない。

元基準c7c1d8229508253390212588ae2b25e504eb4915、元科学SOURCE8825391ff73b531be5834bf852b9bf2479d1799b。H6原snapshotを過去commit f52787a2から回収し、元SHA99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005との一致を確認した。providerは元H2入力契約の217c971dd537b1cc7fbc13742bc0f6c263ff7f45をbyte復元する。H2/H4は元生成関数37b1cf72/dfef38fcを数値式不変で使用する。

## 別枠入力生成

H2/H4それぞれSCF1・積分生成1・tol-only DF1・sector固有値計算1。H2はsector作用4列と独立occupation行列1件、H4はsector作用36列と独立occupation作用36列。元のgeometry/basis/SCF1e-9・50cycle/DFtol1e-8、rank3/10、generation order、signed weights、cutoff0、weighted Hermitian projection9.9e-11、canonical state phaseを維持する。H6のSCF/DF/state再生成は0。新しい生成grant/ledger/markerを各系に封印し、使用済みgrantを再利用しない。

元snapshotと異なるSHAは原本復元と呼ばない。rank・元基底エネルギー1e-9以内・元規約の行列/oracle/残差/射影検査を通し、その後固定2候補のsignal/bias/normalizationとtailの比較を行う。同等性未確認なら独立確認を開始しない。必要な後段reference/signal/shot callは別枠で事前封印する。

## 固定独立確認

linear H2/H4/H6、隣接1Å、STO-3G、tol-only1e-8、rank3/10/19、prefix1/3/5、T=.8/P=0/K4/ABS_TIME_PER_OUTER=32。H2 S2/q4 24本対Yoshida4/q2 28本、H4 S2/q8 25本対m5_best/q1 9本、H6 S2/q8 23本対m5_best/q1 9本。計118trajectory/236軸。元の候補、PF係数、scalar-once、directional control、Qiskit1.3.0/rz-sx-x-cx/opt1/seed17とcall予算を維持する。候補拡張・top-up・補助H4比較・production・pushは含めない。

旧private packとseed除外資料は未回収。元契約を継承した新しいpackと新namespaceを作り、旧seed全体との完全非衝突は未検証と明記する。結果の逆転や未確定をretry理由にしない。取得後の固定比較、paired共分散を含む平均・不偏分散・SE、有理数の独立照合、Bonferroni正規近似区間、希少層null、終了監査まで実施してmandatory STOPする。一次担当の技術・数値照合であり、別担当の独立科学レビューではない。

実装：src/trotterlib/h246_migration.py、scripts/run_h246_migration.py、tests/test_h246_migration.py。新旧のgrant・入力SHA・SOURCE・provenanceを分離する。人工検証34PASS/20subtest（元provider上の3公式native/compile/archive、H2guard両軸を含む）。この文書の初版時点で新実分子callは0、入力生成packは次のSOURCE固定後に発行する。
