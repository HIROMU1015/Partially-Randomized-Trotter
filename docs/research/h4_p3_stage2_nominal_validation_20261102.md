# P3 Stage 2：H4 matched nominal信号の限定検証

2026-11-02 JST。ユーザーの明示許可により、契約commit
`495ed110ad8379663145791b788a63b64f05eedd`の初期scopeだけを実装・検証する。
[科学契約](h4_p3_stage2_nominal_signal_contract_20261102.md)と
[再利用対応](h4_p3_stage2_execution_readiness_20261102.md)の条件・精度・STOPは変更しない。
旧資料の「未承認」は作成時点の履歴であり、今回の実行許可と区別する。

固定条件はlinear H4/1Å/STO-3G/中性singlet、system8/full256、保存DF tol-only1e-8/rank10、
案A `T_eff/V_square`、T.8/S2/q8/r4/K4（5次）、M_B/M_Cだけ。
保存raw state・exact ν・保存DF C raw referenceを使い、新state/referenceを生成しない。
旧prefix27/prefix1/legacy DF12/長時間H4/H6は混ぜない。

## 最小実装と再利用

[kernel](../../src/trotterlib/matched_partition_stage2.py)、
[runner](../../scripts/run_h4_p3_stage2.py)、[人工tests](../../tests/test_matched_partition_stage2.py)。
Stage 1の修正済み入力validator、native occupation表、OpenFermion補正/JW変換、
`PauliHamiltonian`、`paired_polynomial_action`、`diag_hermitian`、
`exact_df_diagonal_coefficients`、`finite_rte_distribution`、`finite_taylor_operator`、
P1の保存enclosure読取を再利用する。旧runnerを再起動しない。

Pauliは元Wと−DのJW辞書、DFは保存Gのbasisからrotated Z/ZZ辞書、canonical Vはnative factor square。
T/state/scalarとoccupation tableは共有するが、残差辞書・計算済み名目平均を共有しない。
DF factor別生成は並列に行い、結果を保存fragment順へ戻す。cutoff0、basisを越えたsupport統合なし。
Gaussian回路は生成しない。

DF dense有限平均は既存`finite_taylor_operator`を一度呼び、r=4の二回のsquaringをcacheする。
`finite_rte_corrected_operator`内部と同じ処理を、積誤差とone-step normを保存するために明示接続した。
同関数を別途呼んで同じ多項式を二度構築しない。Pauliは既存vector recurrenceの別経路を維持する。
raw meanはC kernelのmoment代数をmatrix-onlyで接続し、nom/Bで定義しない。
flintをimportする旧C runner一式は使わず、normalizationは既存binary64関数と標準decimal50の3項検算だけを使う。

## 精度・出力・終了

SOURCE固定前の人工検査は新規9件＋既存P1/bit-copy回帰10件の計19件PASS。
小規模手指定行列だけを使い、回路生成・新分子入力・samplingなし。
全Stage 1 H4 batchは再実行していない。

matrix/vector complex128、共通expm2種と同精度spectral照合2種。
operator閾値は契約のscale依存τ/4と4τを守り、非PASSなら信号を開始しない。
normはSVD spectral、成分involution/Hermiticityの小残差にはFrobenius上側診断を使う。
信号誤差は契約のγ、行列積、exponential差、polynomial sensitivity、非unitary norm増幅、
内積・ν・phase・scalar丸めから伝播するengineering目安であり、rigorous certificateではない。
identity defectはresidual bindingの内訳とし、二重加算しない。

初期scopeは複素信号6件、normalization2bundle。cosine/+Imを別runにせず、
common operator・basis・exponentialをcacheする。途中結果を保存し、科学失敗を自動retryしない。
結果とSOURCE/入力来歴、辞書descriptor、DF basis、common T/Vを新しいstage2 namespaceへ保存する。
旧SOURCE/input/raw/result/grant/ledger/marker/STOPは更新しない。

実行結果・人工検査件数・科学SOURCE・STOPは
[結果報告](../../artifacts/df_pauli_matched_partition_validation/2026-11-02/stage2/run01/REPORT.md)に記録する。
これはSOURCE固定時点では将来の出力先であり、結果PASSを先取りしない。
結果が整合してもbias成分が未解像なら`STOP_STAGE2_UNRESOLVED_COMPONENTS`。
全予定比較が解像しても`STOP_STAGE2_NOMINAL_DIAGNOSTIC_COMPLETE_NO_AUTOMATIC_FOLLOWUP`で停止する。
native価格、実効law、実compiledρ、N/C/G、RPE全費用・一般的な優位性は未検証のまま。
公開・remote照合後mandatory STOPし、Stage 3/4等へ自動移行しない。
