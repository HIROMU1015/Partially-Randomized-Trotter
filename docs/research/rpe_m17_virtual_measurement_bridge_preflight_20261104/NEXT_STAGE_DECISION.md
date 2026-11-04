# m17仮想測定bridge：次段判断

2026-11-04 JST。**静的SOURCE調査と人工15 testsを完了。新しい実H4数値評価・sampling・RNG replay・compileは0。**

1. **接続できた範囲**：order/componentのCDF/53-bit law、signed event、F→events→R、Re/+Im、
   検査→clip→float確率→strict Bernoulliの処理を対応付けた。
   受理後の確率計算とIID uniform53 Bernoulliのmean差は、binary64/RNEの前提で解析的に≤3·2^-53。
   clippingはideal条件付きmeanとの距離を増やさない。これは実H4 state-actionの誤差上界ではない。
2. **重要な未保証bridge**：raw ψとnormalized ray、numeric P/F/R/trig/runtime actionとideal gateの差。
   さらに範囲検査はclipの前に例外を出すため、**全supportで±1測定まで完了するか**が未解決。
   平均上界が予算内でも完了保証とは限らず、現在のU_execはnull。
3. **最も効率的な第一検証**：候補1の「float64 state-actionの平均誤差majorant＋全supportのguard確認」。
   保存ideal包含とnumeric contextの差をoperator/normでboundし、q/rをscalarで伝播する。
   run02再実行・shot増加やm17の直接trajectoryは標準案にしない。
4. **IIDとPCG64**：Model IIは独立一様53-bit wordを明示仮定する。
   Model IIIの固定seed列は決定的で、role分離・seed衝突なし・有限標本の正常完了からIIDを証明しない。
   次の数値bridgeが成立しても、この確率モデルの境界は残る。
5. **研究判断の変わり方**：上界と完了域を確認できれば、m17数値仮想測定の条件付き局所保証へ進める。
   guardや保守的上界が不足なら、追加samplingでは解決できない不足を特定できる。
   いずれも旧INCONSISTENTの取消し、Model IIIの保証、全18roundのGO、最終費用評価にはしない。
6. **追加で必要なもの**：既存A runnerはm0/m3・r1用なので、m17用の新しい薄いSOURCEが必要。
   保存m17 F/R64、B raw state、latest local matrixから作るP_i64を新contextとして固定し、
   matvec/vdot等のkernel丸め上界とguard条件を正当化する。run02の未保存payload identityは未保証のまま。
   その後の実データによる残差/norm/平均majorant・U_L/予算比較は新科学計算であり今回は未実行。
7. **担当と許可**：次段案はCodexによる最小adapter・人工検査・実行SOURCEの公開固定。
   ユーザーはその固定SOURCEと[限定実行計画](MINIMAL_VALIDATION_PLAN.md)に対し、新科学実行を別途明示許可する。
   重要科学レビューの許可は独立。本preflightを実行許可に読み替えず、ここでmandatory STOP。

保存結果のscopeはlinear H4/1.00 Å/STO-3G/legacy DF12/B state/4電子sector70、
L_D3/単一δ=.02/m17 q131072/r32 K2、extract_identity_phase、cutoff1e-12。
latest ν_catはSOURCE replay成分/basis、IID uniform53、ideal未compile gate、normalized rayに限定され、
既存の両軸局所十分条件確認を維持する。N/α/β/ε、原SOURCE/raw/STOPは変更していない。

`CONDITIONAL_IDEAL_CATEGORICAL_SUFFICIENT_CONDITION_CONFIRMED`、
全体`RPE_tolerance_certified=false` / `scientific_status=UNDETERMINED`、
旧`numeric_diagnostic_status=INCONSISTENT` / `u_num_bound=null`を保持。

- [SOURCE対応と入力の可用性](VIRTUAL_MEASUREMENT_SOURCE_MAP.md)
- [確率モデル・bridge証拠・解析式](BRIDGE_EVIDENCE_AND_GAPS.md)
- [第一科学検証の入力・call範囲・判定・STOP案](MINIMAL_VALIDATION_PLAN.md)
- [人工試験・原本照合記録](VALIDATION.txt)

今回の公開branch：`codex/rpe-m17-virtual-measurement-bridge-preflight-20261104`。
baseは包含結果公開commit `34840ce9bf81d985b32764d9bc9d77645ed52e59`。
静的preflightの公開commit SHA・remote照合結果は最終チャットに記載する。
