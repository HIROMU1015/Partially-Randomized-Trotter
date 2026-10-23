# H4非DF/DF：C production実装・人工検証・技術SOURCE

2026-10-23 JST。基点 `b0f94185f93cd5614e8f183613a2a9313a5b9d4f`、専用branch `codex/non-df-h4-c-production-20261023`。旧科学SOURCEは `0c329b30a81accb35c212be8defa66fee0dd0f12`。利用者の承認範囲はproduction実装、人工検証、SOURCE/runtime固定、保存・終了監査である。**実H4の数値証拠生成・certificate生成・新N/G会計・sampling・compileは0。科学grant未発行、実行3 flag false。**

## 実装した経路

| Source | 役割 |
|---|---|
| `src/trotterlib/h4_legacy_export.py` | 封印済み旧runtimeで保存入力を読むbridge。旧Hamiltonian/preparation/basis hash・prefix・identity・演算順序を照合し、binary64を損失なくJSONへ運ぶ。RNG・新chemistry・回路compilerを呼ばない |
| `src/trotterlib/h4_ball_evidence.py` | Pauli-JWとDF-nativeのfull-Fock assembler、Arb/Acb演算、reference、S2、degree5有限法則、実装法則、bias・norm/演算子/法則差の包含 |
| `src/trotterlib/h4_certificate_control.py` | stdlib authority、SOURCE/runtime/input/参照先の固定、one-shot、非返却予算、排他的claim、独立witnessによるnative wait・子孫回収・STOP |
| `scripts/run_h4_numerical_evidence.py` | inspect / grant発行 / execute / worker / bridge / witnessの明示CLI。科学経路は有効な新grantなしで拒否 |

科学対象はlinear H4/隣接1.00 Å/STO-3G、8 qubit、共通保存状態Nα=Nβ=2。Pauli-JW184項/PR prefix27、DF tol-only1e-8/rank10/PR prefix1＋one-body。S2、T=.8、q8、r4/K4、5精度・両軸の40条件を維持した。未証明のsector射影、状態vectorの置換、候補・ランキング変更は行わない。

## 数値保証と意味の境界

保存binary64をexact integer ratioへ変換し、Arb/Acbへ取り込む。保存stateは再正規化せず、`ψ†ψ`の包含と同じrayの物理的期待値を別項目へ出す。DF one-body/gのencoded Hermiticity、保存full matrixとの再構成残差、basisの各local matrix fingerprint、preparation metadata、deterministic block順序を検査する。不足・不一致・破損・非有限値はSTOPとなる。

行列指数には保証付き `acb_mat.exp()`、有限lawには元のK4に対応するdegree5多項式を使う。非unitaryな多項式やencoded basisのnormを1と仮定しない。nominal meanと、完全なcomponent/rotation/CDFから因数分解する実装meanを分けて保存する。referenceやDF因子の指数は同じtier内でcacheする。精度64→96→128 bitsは有限の幅不足だけで進み、既に設計目標を満たしたcaseを再評価しない。例外・hash/model不一致・到達不能と証明された目標では精度を上げて続行しない。

FLINTのball演算とmatrix exponentialのAPIは公式資料に基づく。外向き端点をexact dyadicとして保存し、midpointだけで受理しない。[python-flint Arb](https://python-flint.readthedocs.io/en/latest/arb.html)、[Acb matrices](https://python-flint.readthedocs.io/en/latest/acb_mat.html)。

各軸はBの固定結果にある必要u上限の1/2と比較する。signed biasの保存点との差、state defect、法則対応・basis相殺・位相丸めの解析的偏差境界を区別する。これはCの新しい誤差診断であり、旧経験的allowanceやguardの変更ではない。`TARGET_MET`は**数値設計目標との比較結果**である。全出力の `formal_certificate_accepted=false` を維持し、旧20行N/G=null・40軸UNDETERMINEDを変更しない。

## PR normalizationと残る受理条件

全componentの絶対係数和からlambda、有限weights、B_K、累積B/log_Bをballで包含する。保存log_B_upperが包含上端を覆うかを判定し、saved Bとの差の信号への伝播も記録する。nominal normalizationの包含と、実装されたsampling lawの確率・角度・basisとの対応を別欄に保存する。

NumPy1.26.4のfloat64 cumsum・末尾除算と53-bit uniform格子のright-search massを再現し、旧runtimeで得るCDFと照合する。新しい乱数抽選やseed replayは行わない。IID uniform-wordという既存の統計仮定を残し、PCG64列の独立性を証明したとはしない。[NumPy固定source](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_generator.pyx)。

**encoded local matrix/lawに対する包含と、literal compiled wrapper・理想的なunitary eventへの転送は同一ではない。** DFの浮動小数点basisはexact unitarityと±Bのevent範囲を自動的に証明しない。basis相殺による偏差を保守的に包含するが、厳密event範囲の転送gateは未確定として明示する。通常floatの高精度再計算・人工一致・過去の経験的compiled PASSだけを完全certificateへ昇格させない。この制約により、次回H4で設計目標を満たしても、科学的certificateの正式受理やN/G会計を自動で行う経路はない。

## 実行制御と終了監視

新SOURCE、両runtime、入力manifest、実際にimport/readする旧root・Python・入力・結果・method-mapの参照先を結合する。別worktreeのHEAD/statusや一時的CPU/PSI競合は拒否条件にしない。予約を科学callより先に永続化し、異常後も予算を返却しない。使用済みgrant、重複launch/worker/bridge claim、STOP後の再起動を拒否する。

Pauli/DFの独立processごとにdetached witnessを置き、native waitが返るたびにexitをfsyncして保存する。旧runtime bridgeにもwitnessを設ける。主監視親の最後の集約処理だけには依存しない。Linux subreaperで自分の子孫だけを追跡・回収し、異常時に他利用者のprocessへ信号を送らない。予期せぬ残存子孫を停止した場合は正常終了扱いにしない。witness自体まで消失した場合の元nativeはUNKNOWNであり、事後PASSを作らない。

旧H4のnative exit/reap/STOP欠落は未回収、元監視消失原因UNKNOWNのまま。この新監視実装や人工での親消失試験は、旧runの終了証拠を代替しない。

## 技術検証と保全

人工・回帰は110 PASS、FAIL/ERROR/skip0（旧33＋新77）。最終結果は[tests_release/tests.json](../artifacts/non_df_h4_c_production/2026-10-23/tests_release/tests.json)。旧33件を含め、binary64 exact変換、区間指数、非可換順序、degree5、normalization、CDFの端点、state defect、basis/preparation/hash、重複軸・順序、有限precision進行、並列予算、無効grant、重複claim、native正常/例外/signal/子孫/主親消失を検査した。別の[backend13回帰](../artifacts/non_df_h4_c_production/2026-10-23/backend13_regression.json)もPASS。これらは主担当自身の技術検査であり、別reviewerの独立科学レビューではない。

旧runtimeで供給した人工2 qubitのPauli/DF preparationを同じproduction assemblerへ渡し、full-space・法則・bias・20軸ずつの設計比較まで検証した。人工targetは人工用であり、H4の必要精度を達成したという証拠ではない。開発初回のFrobenius境界の問題と修正履歴を[development_history.json](../artifacts/non_df_h4_c_production/2026-10-23/development_history.json)へ記録した。

実H4についてはZIP/NPY headerのshape/dtypeとSOURCE/receipt/入力SHAだけを静的照合し、array payloadを数値評価していない。旧SOURCE738・結果/claim1147・前回調査17・B/C保全inventory99ファイルの保存内容を照合した。旧runtime3428 hashも一致した。新しい依存環境はプロジェクトprivate環境のみで、共有venv・OS・他jobは変更していない。

## 固定SOURCE・契約・次の担当

[新技術SOURCE](../artifacts/non_df_h4_c_production/2026-10-23/SOURCE.json)、[完成契約](../artifacts/non_df_h4_c_production/2026-10-23/C_execution_contract.json)、[入力manifest](../artifacts/non_df_h4_c_production/2026-10-23/input_manifest.json)、[private runtime](../artifacts/non_df_h4_c_production/2026-10-23/runtime_manifest.json)、[旧runtime](../artifacts/non_df_h4_c_production/2026-10-23/legacy_runtime_manifest.json)。実装SOURCEは `49c18ccc35f14041df45bfeff455ba5e67543182`。封印後の監査・成果物commitと区別する。封印の完了状態はSOURCE.jsonとcompletion_audit.jsonを参照する。

次回の最大予算はoperator復元2 batches、probability対応2 batches、reference6、nominal mean12、実装PR mean6、PR normalization6、dense exp39 requests。今回これらの実H4科学callは消費していない。SCF/FCI/新DF/状態最適化/sampling/compile/shot/Gは次回契約でも0。

次は**ユーザー**が、このSOURCEと契約で実H4の数値証拠候補生成を進める範囲を明示承認する。特にencoded modelとevent範囲・compiled wrapper転送に残る受理制約を確認する。**Codex**は承認後に新契約/approval/namespaceを結合して新grantを一回だけ発行し、契約内の実行と監査を行う。N/G再会計・公開push・詳細科学レビューは別の承認対象。今回はlocal commitのみ、mandatory STOPし、自動実行しない。[引継ぎ・将来のgrant手順](../artifacts/non_df_h4_c_production/2026-10-23/CONTINUATION.md)。
