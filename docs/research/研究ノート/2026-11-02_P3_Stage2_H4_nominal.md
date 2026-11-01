# 2026-11-02：P3 Stage 2の限定結果

ユーザーの明示許可により、契約`495ed110ad8379663145791b788a63b64f05eedd`、
科学SOURCE`4bfbbaad1f8178077489bf06b085377256d6fb80`からH4の1tupleだけを実行した。
条件はLinear H4/1Å/STO-3G/中性singlet/full256、保存DF tol-only1e-8/rank10、
案A/T.8/S2/q8/r4/K4、M_B/M_C、保存raw φ/ν/DF C reference。

[結果・精度・全値・来歴](../../../artifacts/df_pauli_matched_partition_validation/2026-11-02/stage2/run01/REPORT.md)。
operator preflight57、nominal/共通operator/raw-B照合5項目がengineering PASS。
補正平均のPauli/DF点差は約3.77e-15、Bは39.12591/34.52800。
同じ物理残差でも辞書normalizationとraw attenuationが異なり、補正nominal平均は整合した。
outer差は両軸解像、finite差は採用不確かさ約1.3e-9より小さく、両方式・両軸UNRESOLVED。
点差の符号・大きさやbias=0を採択しない。

人工19PASS、科学1batch/6複素信号/2normalization、7.32秒・exit0・worker join済み。
旧SOURCE/input/raw/result/Stage 1/P1/P2 STOPは保持。
実効law、native回路・価格、実compiledρ、N/C/G、RPE全費用、主要claim採択は未実施。
`STOP_STAGE2_UNRESOLVED_COMPONENTS`とし、追加精度・retry・Stage 3/4・詳細科学レビューへ移行しない。
次はユーザーが後続scopeの必要性を判断し、Codexは別の明示指示の範囲だけを扱う。
