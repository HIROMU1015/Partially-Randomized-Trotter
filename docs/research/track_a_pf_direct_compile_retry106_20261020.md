# PF直接compile：残106軸のcontrol参照修正・再実行

2026-10-20 JST。基点`636208ba6d74af13e7c405970c896b621022af65`。ユーザーの「修正を行って再計算して」を、残106軸の修正・control/compile再実行承認として記録した。保存受入682軸と過去のSOURCE/raw/grant/ledger/marker/STOPを保全する。現在はseal前で、実分子の再計算は未開始。

[packet](../../artifacts/resource_applicability/track_a_pf_direct_compile_retry106/2026-10-20/README.md)。linear H2/H4、隣接1 Å、STO-3G、tol-only DF3/10、T=.8、P=0、symmetric_directional。対象H2 46軸/H4 60軸、計106軸＝53 cosine/sine pair。旧profile照合待ち48軸・H6/H8/legacyの追加実行は対象外。compilerは既存Qiskit1.3.0、rz/sx/x/cx、optimization_level1、seed17と同一hash。指標は状態準備を除くfull Hadamard wrapperのcompiled整数ゲート数で、shot重み付け・新費用ランキングは行わない。

## 原因の訂正と修正

前回の「B0Iのpreparation選択が原因」という説明は、generic MatchedPortと、このD系列のoverlay生成を取り違えた仮説だった。保存D生成SOURCEを追うと、B0IはB0 discard preparationへ保存scalar Cを加える方式で、現preparationを変更する必要はない。h2_d_recovery_scienceもこの対応を明示している。原コードと過去の診断を保全し、この新来歴で訂正する。

96軸のB0Iでは、overlayがsignalsだけをexp(-i*T*C)で回転し、independent_forward_signalsを未回転のまま残す。旧controlは回転済みprimaryと未回転forwardの両方へ一致を要求していた。新controlは既存の保存Cでforwardを一度だけ回転し、同じ位相規約で比較する。full-H reference・保存primary・native scalarは変更しない。

H2 prefix0/S4では、古いmerged iteratorが単一項でも接続項を余分に出力する。w=1/(2-2^(1/3)), v=1-2wのとき、旧列の重み和は3w+2v=2-w、正しいunmerged Yoshida列は2w+v=1。現在のnative回路は後者を用いている。B0の10軸に加え、B0Iの10軸も同じ問題を継承している。この20軸に限って、既に保存されたexact_truncatedをcontrol参照へ固定する。単一項では全指数が可換で、S4は時間Tの単一項発展に一致するためである。B0I forwardのみ保存Cで一度回転する。

この参照選択は実結果を見る前にtask IDごとに固定する。許容誤差は従来の1e-9を維持する。旧corrected signal・適格性・shot・モデル値を修正せず、20軸はrequires_saved_classical_signal_reconciliationをCSV/JSONへ明示する。これらを旧signalと整合済みの費用ランキングへ自動使用しない。旧signalの修正は別作業で、本契約では行わない。

## 検査と実行契約

新13人工検査と既存境界13件PASS。人工referenceで回転の一度適用・保存record不変・不正field拒否、実nativeを使うtoy単一項回路でq1/2/4/8/16・scalar shift0/.23と解析解の一致を確認した。実分子入力load・実分子compileは検査で行わず、承認された本契約のcontrolで確認する。

既存loader・native builder・compiled metrics・並列admission・分離observer・実wait4終了監査を再利用する。新たに失敗時にもlabel付きcontrol誤差と選択した保存参照を保存する。新grant/ledger/marker/STOPで1回実行し、native53/control212/wrapper build/compile各106に固定する。sampling/fit/bootstrap/shot再計算/費用再ランキング0。科学条件・call範囲を拡張せず、失敗taskはSTOPし自動retryしない。

SOURCE/input/provider/compiler/runtime/profile・旧682軸の結果と終了証拠をbyte照合してsealする。旧572軸のrun全体最終gate欠測は継承し、旧110軸の独立最終gate PASSと区別する。既存OS制御の範囲で実affinity/cgroup・空きメモリに応じて並列化し、独自CPU/RAM/worker上限・一時PSI上昇による強制終了・無関係worktreeの凍結を設けない。

次の担当はCodex。seal後106軸を一度実行し、終了監査・結果commit・簡潔な報告の後mandatory STOPする。新しい公開範囲は未承認で、科学実行承認と混同しない。詳細科学レビュー・最終evidence freezeは開始しない。
