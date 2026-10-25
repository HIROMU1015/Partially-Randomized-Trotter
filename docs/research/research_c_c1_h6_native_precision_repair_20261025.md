# C1：H6 native Gaussian精度修正・再実行条件

2026-10-25。起点 `7b1cc69f55da75cb50c20682e3fbb830f0c84106`。ユーザーの「修正を行って…修正して再実行してよい」に基づく、同一8経路の技術修正とfresh再実行。旧SOURCE `bf431765b1a83ca79d1c2fe15d30284e0b1f28c6`、旧run01・162-call STOPは変更しない。

[旧実行のFAIL](research_c_c1_h6_native_execution_20261025.md)／[新298-call契約](research_c_c1_h6_native_precision_contract_20261025.json)／[SOURCE・人工証拠](../../artifacts/research_c_c1_h6_native_precision/2026-10-25/technical/)。新契約も歴史的status文字列を保持するが、今回のチャット上の修正・再実行許可が適用される。承認原文保存・証明書・新grantは不要で、再確認は要求しない。

## 技術的な原因と修正

旧経路は既存NativeBuilder→`native_gaussian`→Qiskit Nature0.7.2の`BogoliubovTransform(U.T)`を使う。このnumber-conserving JW分解には、target成分とGivens引数に既定`np.isclose(...,0)`を使う箇所と、角度を`arccos(c)`で戻す箇所がある。前者は約1e-8以下の成分を省き、後者はcosが1に近い小角度で精度を失い得る。input validationのatolを指定しても、内部のこの省略条件は変更されない。

人工2×2回転のsingle-particle transferで再現した値は以下のとおり。H6保存値をdecodeした結果ではない。

| 回転角 | 旧max-entry | 修正max-entry |
|---:|---:|---:|
| 1e-10 | 1e-10 | 1.84e-26 |
| 1e-9 | 1e-9 | 1.84e-25 |
| 1e-8 | 1e-8 | 1.84e-24 |
| 1e-7 | 3.9971880624e-11 | 1.84e-23 |

[人工例原本](../../artifacts/research_c_c1_h6_native_precision/2026-10-25/technical/precision_examples.json)。Aer既定zero_thresholdを変える仮説も人工のtiny RY/RZで調べたが、既定値でも正しい小振幅が保持されたため、Aer設定・検証閾値は変更していない。旧H6 FAILへの因果帰属は、新しい同条件評価との比較後に限定して述べる。

新しいtask-local `c1_h6_native_precision.py` はQiskit Natureの既存number-conserving JW消去順序を再利用する数値修正版。全く別のGaussian合成法は導入しない。

- 列／行Givensの順序、U.T規約、RZ／XXPlusYYゲート種を保持する。
- exact-zero判定にし、既存SciPy LAPACK `zlartg`でGivensを計算する。厳密なa=0／b=0では旧方式の位相規約を保つ。
- `atan2(abs(s),c)`で小角度を保持する。
- 生成gateの単粒子transferを小行列で再構成し、元Uに対するmax-entry<=1e-10を同じGaussian build内で確認する。不一致時は明示FAIL、旧方式へfallbackしない。
- H6専用PrecisionBuilderの`_basis`だけを差し替え、既存NativeBuilderのschedule・一般K・κ・scalar・controlled phase・exact basis共有・角度結合・P0は再利用する。旧H2/H4 builder／インストール済みpackage・venvは変更しない。

精度修正により、従来省略されていた小角度のgateが残る可能性がある。これは新しい費用最適化ではない。同じ規則をCDF/XDFへ適用し、修正後の正しい回路の費用だけを比較する。単粒子checkをfull controlled全operatorの保証とは扱わない。

## Qiskit逆回路生成の競合修正

人工end-to-endで、controlled RZZの`inverse()`が内部BasisTranslatorを呼び、他候補のP0 compileと並行すると`RuntimeError: Already borrowed`になるケースを再現した。既存P0 mutexをactual inverseの**回路生成**にも適用する。逆回路のstate-actionはlock外で並列実行し、P0条件やactual inverseの意味は変更しない。

独立8候補のAer並列計算は維持する。compilerとinverse生成が重ならないことを人工の同時アクセス検知テストで確認した。科学callの追加はなく、inverse生成は元の8 adjoint logical unitsに含む。

kernel FAILが再発した場合に、全候補の例外詳細を保存する最小修正も加えた。旧実行で6件の詳細が欠落していた点への対応で、判定条件を変えない。

## 人工・回帰検証

| 検証 | 結果 |
|---|---:|
| eval：新精度・full人工Fock・state-action・正常／異常end-to-end・H2/H6関連 | 107 PASS |
| compiler/inverse共通mutexの人工同時アクセス回帰 | 1 PASS |
| 旧Stage2 native mock（独立process） | 27 PASS |
| 旧修正codec（既存ffsim専用環境） | 50 PASS |
| 合計 | **185 PASS、FAIL/skip 0** |

det±1、π近傍、1e-12～1e-7の小回転、実／複素2・6次元の全Fock exterior-power独立reference、12 modesで巨大Operatorを構築しないことを確認。人工S2/P0の8経路、phase・向き・frame順・compile・adjoint破損時STOP、SOURCE/input不一致、fresh output、未知合成誤差をnullに保つ挙動も検証した。

最初の組合せ試験は、ffsim未導入eval環境へcodec試験を混在させた12件、既存mockと実回路の同一process混在1件、新しい人工P0期待値2e-14が過度に厳しかった1件の計14 FAILを記録した。前二者は既存の適合環境・独立processへ分離し、後者は登録済みcomponent allowanceを使う検査へ合わせた（微小回転の保持は別の独立テストで検証）。次のevalでinverse/P0競合1件を再現し、上記mutex修正後に全PASS。これらの初回log/XMLも保全し、実H6科学callのretryとは数えない。**科学判定の1e-10／signal2e-10等は一切変更していない。**

## 固定した再実行範囲

直鎖H6・1.00 Å・STO-3G、Nα=Nβ3／sector400、保存最終CDF14・selected XDF14、全frame ordered S2 s1/s4、literal/sharedの8経路、δ0.1・m0..12、CA0.00159／DF reserve CA/8、own-DF ground oracle・状態準備費用除外を維持する。固定P0 Qiskit1.3.0／opt1／seed17／rz,sx,x,cx、approximation_degree1.0、qubits_initially_zero=falseも不変。

保存input25 hashと、元input／CDF fit／PFのSOURCE来歴をそのまま保持する。新fit・optimizer・DF/PF再計算・rank変更・shots・最終CA／総費用算定は0。新SOURCE manifestは44 files、同じ依存lockを使う。契約・manifestの実行前hashは[preflight](../../artifacts/research_c_c1_h6_native_precision/2026-10-25/technical/preflight.json)に保存した。SOURCEのcommitは実行前に公開し、execution identityへ固定する。

fresh outputは `artifacts/research_c_c1_h6_native_precision/2026-10-25/run01`。最大298 unitsの内訳は旧契約と同一で、局所30／build41／compile11／kernel8／adjoint8／Hadamard8／metrics11／profile104／比較8等。旧162 unitsと合わせて最大460 unitsであり、旧残136 unitsの流用ではない。今回のfresh再実行1回を独立に数え、run内retry/resumeは0。最初のcorrectness FAILで依存する後続を停止し、開始済み作業だけ回収する。

科学gateはcomponent/norm drift1e-10、complex signal2e-10、位相fitなし、同じ固定probe seed。成功してもfinite probesはglobal operator boundではなく、104行のprofileは`UNRESOLVED_FINITE_PROBES`・未資格boundをnullで保持する。費用対象は1 h-step kernelのみで、全roundのshots加重総費用へ外挿しない。

CPU/RAM/worker/wallの恣意的な上限は追加せず、実行直前の空き資源に応じた既存planを適用する。P0／inverse生成だけ排他、重いAerと独立referenceは並列。共有設定・venv・他jobは変更しない。

結果または部分証拠の公開照合後はmandatory STOP。合成誤差の全operator資格、CA、shots、総費用や追加科学レビューへは自動進行しない。
