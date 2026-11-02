# P3 Stage 3：H4 matched native価格の限定実行

2026-11-02 JST。ユーザーの「実行してよい」により、公開設計commit
`be3de04e6fb501fecc4a07f38ceea7c24ba10472`の初期案を実装・人工検証・限定実行する。
[価格設計](h4_p3_stage3_native_cost_design_20261102.md)と[preflight政策](h4_p3_stage3_preflight_readiness_20261102.md)を変更しない。
旧設計の「未承認」は当時の履歴として保全する。

対象はH4/1Å/STO-3G/中性singlet/system8/full256、保存DF tol-only1e-8/rank10、
案A/T.8/S2/q8/r4/K4、M_B/M_C、P=0。共通Gaussian保持を一度生成し、同じ命令を全armへ使う。
Stage 2保存辞書184/360とDF U/ηを引き継ぎ、再対角化・新state/reference・信号/B/N/G評価は行わない。

## 最小実装・再利用

[kernel](../../src/trotterlib/matched_partition_stage3.py)、[runner](../../scripts/run_h4_p3_stage3.py)、
[人工tests](../../tests/test_matched_partition_stage3.py)。
既存Givens lowering、DFBasisRegistry/DF event builder、Pauli event/D2制御、
Hadamard wrapper、compiler/metrics/fingerprint、Stage 1/2のnative occupation/normを再利用する。
T一体だけ8×8で対角化し、DF残差は保存Uを使う。dense Gaussian fallbackなし。
native operatorと、保存辞書/JW/native occupationによる独立oracleを照合する。
actual adjoint、ancilla0/1、scalar、T identity、event sign/orderを保つ。

人工検査後にSOURCEをcommit固定し、H4共通保持・basis・scalarのPASS後にだけ8 trajectoryを一度samplingする。
各arm4seed、32 event/trajectory。F8/E8/R1/Empty1のnative body確認が全てPASSした後、
同じevent列を両軸へ使い、F16/E16/R2/Empty2の36 compileを行う。
方式間paired標本・母平均・全lawρ・compiled作用保証を主張しない。

人工compiler回帰の初回検査ではDF測定wrapperのfull unitaryに全体位相πの差があった。
測定observableの差は約3.5e-14であり、回帰判定は評価対象であるancilla測定observableで行う。
この修正はcompiler条件や科学条件を変えず、global phaseの事後alignmentも行わない。
compile前のGaussian保持・controlled bodyはliteral operatorを位相合わせなしで照合し、
compiled QPYのglobal phaseは原値を保存する。H4の全compiled operator/ρ検査はscope外のまま。

## 精度・記録・停止

SOURCE固定前の最終人工検査は新規12件＋既存bit-copy回帰1件＝13 PASS。
初回の人工compiler検査で判定対象を測定observableへ修正した経緯は上記に保持する。
実H4のscientific retryではなく、許可範囲内のSOURCE固定前人工debugである。

complex128、基底・保持は契約のτ/4と4τ、長いprogramはnative局所作用とoracleの積誤差を伝播する。
native回路の行列はQiskitで評価し、誤差目安には局所gate matrixをcacheして
`|| |g||B| ||_F <= || |g| ||_2 ||B||_F`の評価を使う。
library内部のcall監査logは作らず、対象数と主な実評価件数を記録する。
独立oracleは実測matrix normを使い、unitary norm=1を丸め後に無条件仮定しない。
差を事後phase alignmentで除去せず、preflight非PASSなら価格取得へ進まない。

新しいstage3/run01へ入力・SOURCE・basis/preflight、全event、36 input/compiled QPY、metrics、終了状態を保存する。
QPYはgzipで可逆圧縮し、圧縮bytesと展開QPY両方のhashを対応づける。入力/出力programやgate定義を変更しない。
共有cache/環境は変更せず、独立taskだけを空き資源に合わせ並列化する。
科学失敗の自動retry、別seed/compiler/solver/precisionへの切替は行わない。

SOURCE固定時点では結果を先取りしない。実行結果・実件数・STOPは
[結果保存先](../../artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run01/REPORT.md)に記録する。
失敗時も原因と旧出力を保全し、公開・remote照合後mandatory STOP。
Stage 2の`STOP_STAGE2_UNRESOLVED_COMPONENTS`は保持し、Stage 4等への自動移行はない。
