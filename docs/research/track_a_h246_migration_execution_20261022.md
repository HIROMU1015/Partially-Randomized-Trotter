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

## 入力生成完了・欠落reference復元の別枠予算

SOURCE88cd0907でH2/H4生成を各一回完了。両系とも元の全科学配列hash、Hamiltonian hash、state hash、基底エネルギーが完全一致した。NPZ容器SHAは異なるため、原本復元とは区別する。H6は原snapshot SHA一致。

次の別枠は3系各1回のsnapshot restore、occupation oracle context、full/prefix preparation、tail spectrum、full-H reference生成。H2だけ欠落した固定2候補のsignalを元のspectral法で再計算する。H4/H6は元の公開signalを厳密一致する科学配列に結合して再利用し、新referenceとの整合も確認する。shot再計算0・新sampling0・compile0。各系別の新grantをSOURCE固定後に発行し、予算を全予約してから起動する。

## Reference復元完了・新契約への接続

SOURCEeb4a2411でreference3件と欠落H2signal2件を別枠で取得した。3系の保存Hamiltonian/state fingerprintは元値と一致、H4/H6のcut preparation hashも元signalと一致。固定6候補の保存bias/log_Bとの差は0であり、元の凍結shotを保持する。新shot再計算0。

新しいh246_reconstructed_contract.pyは、旧proposal02を装わず、元固定候補・n・取得call・analysis scopeを保持した新規packを発行する。元SOURCE68ファイルとproviderをSHA照合し、元のnative/runner/auditor/evaluatorを再結合する。数値関数・PF係数・compiler・archive方式を変更しない。CLIの転送先だけを新runnerへ結合する。新namespaceのseed846個はstage間と復元可能な旧提案seed846個に非衝突。その他の旧約25.65万除外は未回収で完全非衝突未保証。各stageに新grant/ledger/marker/承認receiptを用いる。実空きCPU/RAM/diskによる旧allocatorを使い、固定worker上限を設けない。

新接続の人工検証5PASS/3subtest。人工段階でcohort文字列変換helper自身の再変換によるラベル接続不具合を発見し、元helperを保持して修正した。数値関数は変更せず、実確認grant発行前に3公式native/compile/archiveの再検証を通過した。合計の人工検証は34+5PASS。
