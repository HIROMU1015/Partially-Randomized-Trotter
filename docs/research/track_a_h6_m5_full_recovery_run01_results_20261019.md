# H6 full m5_best：固定1構成の再計算run01・数値不一致の再現

2026-10-19。**cosineで元と同じ数値不一致を再現しSTOP。sineは未実行。** 今回は回路・比較ベクトル・診断情報の保存に成功した。共通位相差が主成分と保存診断から示唆されるが、位相差の発生箇所・原因は未確定。位相を外した診断値をPASS判定へ使わず、旧STOPも保持する。14:22:24 JST終了、全worker回収済み、mandatory STOP。

SOURCE：`f743a88225f7cb2c8d2c91d7e19813a5780f803b`。branch：`track-a-h6-m5-full-recovery-20261019`。元science SOURCE：`c7d53f04c3be131078ee91e8bef3c78ca434ed3e`、元結果：`b38898429af3a2a48d7057d9d828256c057685f8`。今回の起動資料：`2afcb79f1480f6574d3c96eca8e5a4a875c86336`。[起動契約](track_a_h6_m5_full_recovery_run01_start_20261019.md)、[元結果](track_a_h46_m5_tol_pilot_run01_results_20261019.md)、[今回summary](../../artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01/2026-10-19/summary.json)、[終了witness](../../artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01/2026-10-19/completion_witness.json)。

## 固定範囲と終了

H6直鎖1 Å/STO-3G、保存tol-only1e-8（DF19は観測rank）、full PF/p19/q2、T=.8、P=0、`PF_M5_BEST_20D64C2_BINARY64_V1`の11 signed S2 piece。symmetric-directional/full Hadamard wrapper、scalar一回。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17を維持した。成功済み58軸の再取得、新PR標本、SCF/DF/state生成、新full-H reference、候補探索は0。固定1点のsignal/controlのみ再計算し、元full-H referenceをそのまま使った。

|項目|結果|
|---|---|
|開始→終了|13:55:17→14:22:24 JST|
|wall|1,627.80秒、約27分08秒|
|新compile予約|cosine1軸|
|新受理費用|0軸|
|sine|予約0・compile0|
|worker|exit0・reap済み|
|launcher|exit1・外部wait4でreap済み|
|旧+今回の累積予約|59+1=60、旧消費の返却なし|
|追加retry/再fit/次Phase|0|

worker exit0は例外をSTOP記録として保存できたことを示し、数値PASSを意味しない。launcher exit1は `NUMERIC_DISAGREEMENT:compiled_vector_cosine` を反映する。

## 保存された診断

以下は実行中に固定adapterが計算・保存した値の転記であり、終了後にベクトルを読み直して再計算していない。[diagnostics原本](../../artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01/2026-10-19/diagnostics.json)、[STOP projection](../../artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01/2026-10-19/STOP_projection.json)。

|指標|保存値|用途・判定|
|---|---:|---|
|raw `norm(after-before)`|1.504043707445665e-9|固定1e-9未満に不適合|
|内積からの共通位相|1.5028193689500676e-9 rad|診断のみ|
|共通位相を外した差|6.067672593889436e-11|診断のみ、PASSへ流用しない|
|before norm|0.9999999999999337|診断のみ|
|after norm|0.9999999999987826|診断のみ|
|compile前Hadamard値|-0.8510893855918|診断のみ|
|compile後Hadamard値|-0.8510893856062353|診断のみ|
|独立expected Hadamard値|-0.8510893855919196|既存controlの参照|

raw差は元runと完全に同じ保存値だった。元の六検査値、compile前wrapper fingerprint、固定1点signalも一致した。新SOURCEで追加した保存hookがある状態でも、同条件の不一致が再現したと扱う。元のcompiled回路は保存されていなかったため、両runのcompiled QPYがbit一致したとは主張しない。

共通位相を外した残差が小さいことは、今回のraw差で共通位相が主成分であるという解釈を支持する。ただし、回路生成・phaseの累積・basis変換・1qubit再合成・statevector演算のどこで差が生じたかは、これだけでは特定できない。丸め誤差やcompiler bugを確定したという意味でもない。

受入条件は**finiteかつraw差1e-9未満**のまま。診断用のphase-aligned差を受入に使わず、full-H referenceの位相も合わせ直していない。今回を「数値修正でPASSした」「60軸取得完了」「m5が使えないと確定した」のいずれとも表示しない。

## 保存証拠・費用の未受理

privateの新runに次を保存し、receiptのSHA/bytesを終了後に照合した。

- 元wrapper QPY（36,202,434 bytes）と初期lowベクトル。
- compiled QPY（92,549,282 bytes）、global phase・layout・compiler identity。
- before/after/初期low/独立expected作用のNPZ（459,780 bytes）、complex128・shape8192の比較状態。
- raw/位相/norm/枝別/Hadamard診断、元の六検査値、STOP、1軸ledger、終了/reap/wait4証拠。

これらは `UNVALIDATED_NOT_FOR_ACCOUNTING`。compiled receiptのRZ **831,662**は未検証diagnostic費用で、受理単価・総費用の入力ではない。通常のaccepted cost fileは0件。欠測のsine費用を0補完・model補完しない。

[wrapper receipt](../../artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01/2026-10-19/wrapper_receipt.json)、[compiled receipt](../../artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01/2026-10-19/compiled_receipt.json)、[vectors receipt](../../artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01/2026-10-19/vectors_receipt.json)、[全raw file inventory](../../artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01/2026-10-19/raw_file_inventory.json)は公開可能なmetadata。QPY/NPZ・入力・private traceback原本は公開bundleへ含めず、専用private領域に保全した。

## 終了監査と保全

新SOURCE/input/runtime/原証拠330 file、旧SOURCE/結果/grant/ledger/marker/STOPを含む629保護file、無効grantを含む新proposal原本は不変。旧science16 fileと新SOURCE20 fileのGit bytesも照合する。原run01の29 PASS/58受理軸・1 STOPは変更しない。新compile1件は別ledgerに保持し、未使用のsine枠を自動再使用しない。

[監査summary](../../artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01/2026-10-19/validation_summary.json)は新しい原子・分子計算やQPY/ベクトルの数値再解析ではなく、保存record・SHA・scope・終了証拠の照合。これは作者自身の監査で、別reviewerによる独立検証ではない。旧15人工testを再実行・水増ししていない。共有環境・他worktreeへの書込やsignalは行っていない。

## 次の担当と判断

今回の再実行で同じraw不一致と、その診断証拠を得た。**すぐに同条件のcompileをもう一度繰り返す必要はない。** 保存QPY/ベクトルがあるため、次の診断は追加compileを避ける設計にできる。

**Codex（次の準備案、未実施）**：保存回路・phase処理・ベクトル診断から、共通位相差の発生箇所を切り分ける最小の追加解析計画を用意する。新規compileは計画に含めず、必要なら固定1点の保存値数値解析を別scopeとして定義する。

**ユーザー（次の判断）**：その保存証拠の追加解析を実行するか判断する。現行判定規則の変更を検討する場合も、物理的relative phase/scalar/control/referenceを保持する科学判断として別途採択し、旧STOPを遡及的にPASSへ変更しない。今回の承認から閾値変更・phase-aligned受入・主比較採用AD03・後続Coreを自動的に導かない。

結果をlocal commitしてmandatory STOP。公開push・追加解析・追加compile・費用再会計は行わない。
