# Track D：固定人工pathの最小検証結果

2026-11-06 JST。**TECHNICAL_PASS_MANDATORY_STOP**。固定した1人工wordでReferenceのX/Y測定同値性を数値確認し、3方式×X/Yの6 full-wrapper compileを完了した。強い対称制御対照に対してReferenceはCX/depthが小さい一方、RZ/qubitsは増えた。科学的GO、一般PRや全RPE総費用の優位は判定しない。

## 固定SOURCE・条件・証拠

- 実行SOURCE：[1cc8c87c297e3b526b4e18f20f44da3b70f86157](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/1cc8c87c297e3b526b4e18f20f44da3b70f86157)。SOURCEと[PLAN](PLAN.md)を実行前にcommit/pushし、remote SHAを確認した。その後SOURCEを変更・再実行していない。
- 入力：既存 `native-partial-closed:B`、2 system modes、one-body (.7,−.2)、DF squared η=(.4,−.9), λ=−.6、既存u/v frames、T=.8、q=2、scalar=.17、K=0、τ=.4、同frameのZplus→ZZminusを両stepへ置く。分子・geometry・AO basis・分子DF rank・L_Dは非該当。人工τ/outer時間は物理的PR scheduleとして資格認定していない。
- [機械可読結果](../../../artifacts/track_d_minimal_reference_validation/2026-11-06/result.json)：入力event・SOURCE hash・環境・物理operator・18 correctness check・負対照・各wrapperのraw/compiled metricsとcompile後検査を保存。
- [回路directory](../../../artifacts/track_d_minimal_reference_validation/2026-11-06/)：3方式×X/Yのrawとcompiled QPY、計12回路を保存。初期state準備を含まず、読出しまで含む。ローカルfont cacheは証拠から除外した。

Python3.11.0rc1、Qiskit1.3.0、Qiskit Nature0.7.2、NumPy1.26.4、SciPy1.14.1、OpenFermion1.6.1。compilerはrz/sx/x/cx、opt1、seed17、approximation_degree=1、qubits_initially_zero=False、backend/couplingなし。全方式に既存F2のphase保持loweringを使った。

## correctness

実行は1回、compileは登録した6回。18 precompile checkは全て絶対entry許容差 `3e-12` 内だった。独立なoccupation/Fock-minor参照、Vの真空位相、actual adjoint、既存nativeと共有・融合対照のdiag(I,V)、SE gadgetの分枝順序、元system上の4×4 X/Y測定演算子を照合した。特定の初期ψだけのsignal比較ではない。

| 検査 | 保存された結果 |
|---|---|
| precompile check最大entry差 | 1.176836406102666e-14 |
| χ=−cTの真空位相 | −0.136 rad、参照vacuum列との差0 |
| P(χ)省略の負対照 | X差0.12533801616727808、Y差0.13558114487825276で両方の不一致を検出 |
| compile後のX/Y測定演算子最大entry差 | 9.159339953157541e-15 |
| compile前後の全unitary最大entry差 | 3.626015263700829e-14。global phaseを無視しない比較 |

compile後6件も同じ許容差を通過。これはbinary64での局所数値検査であり、厳密誤差certificateや実機noise保証ではない。shot・sampling・分子計算は0。過去の42 testsや旧科学runは再実行していない。

## full interrogation価格

各方式のX/Yはそれぞれ1回compileし、下表の整数metricは両軸で一致した。H、Y軸Sdg、終端H、ancilla測定を含み、state準備を除く。RZ/CX/depthは別指標であり合算しない。

| 方式 | RZ count | CX count | total depth | circuit size | qubits |
|---|---:|---:|---:|---:|---:|
| diagonal-only ordinary＋共有・融合 | 288 | 152 | 382 | 559 | 3 |
| symmetric-directional＋共有・融合 | 197 | 96 | 265 | 396 | 3 |
| Reference | 202 | 84 | 154 | 401 | 5 |

両直接制御対照には同block D0のstep境界融合を与え、既存native出力との同値性も確認した。Referenceはcomplete-step cutを使い、cutを跨ぐ融合を費用から消していない。CSWAP前後2層、P(χ)、追加reference qubitsを含む。全armで同じcompiler最適化機会を与えたが、global最適価格を証明したものではない。

従って、この例は強baseline後にもCX/depthの追加削減が残り得る一方、RZでは有利にならないことを示す局所技術証拠である。T合成、routing、qubit維持費、全lawの期待価格、必要N、全RPE総費用は未評価。K0/same-basisでproduct・tail identityを含まない人工wordのため、一般PRへ改善率や順位を移さない。

## 実資源・再利用・停止

実行wall 3.4791518840938807 s、単一processの記録peak RSS 366,809,088 bytes。利用可能32 logical CPUに対して固定6 compileを6 threadsで処理し、小行列・fixtureを再利用した。小行列のBLASと入れ子parallelを抑えたが、job全体のCPU/RAM/wall capは追加していない。GPU driverは使えず、この小規模workloadはCPUで完了。serial比較をしていないため速度向上率は主張しない。

既存のfixture、Gaussian/Fock reference、DF primitive、RTE event builder、native provider、counterの隣接basis相殺、phase保持compileを再利用し、新規部分を固定scopeのassembler/gadget/誘導observable検査に限定した。元SOURCE、input、raw/results、grant、ledger、marker、旧STOPを変更していない。

**mandatory STOP：科学実行は完了・停止。** 本結果・QPY・文書の公開取得照合だけを行い、別word、別cut、K>0、分子、sampling、shot、全RPEへ進まない。次の担当はユーザー／GPT。主資源指標と研究継続の要否を判断し、必要なら別scopeの実行または独立した重要科学レビューを許可する。新しい研究方向は正式採択していない。
