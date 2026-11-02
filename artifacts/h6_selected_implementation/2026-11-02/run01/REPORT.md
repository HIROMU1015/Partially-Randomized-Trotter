# H6純RTE/Hybrid：保存3 trajectoryの局所実装診断結果

2026-11-02 JST。**固定15状態作用を正常終了し、6組すべてでselected-target→compiled差のengineering上側候補が保存局所budget内。**
直接差はη/δ/dの全てで数値分解能以下。0や厳密な実装誤差上界を認定していない。
これはconfirmation/index0000という固定3標本の局所診断で、全lawのρ、実装精度保証付きN/G、母平均価格順位の認定ではない。
[STOP](STOP.json)：`STOP_H6_SELECTED_IMPLEMENTATION_LOCAL_DIAGNOSTIC_COMPLETE_NO_AUTOMATIC_FOLLOWUP`。

## 固定条件・SOURCE・入力

linear H6、隣接1.00Å、STO-3G、中性singlet、保存Pauli-JW918非identity項＋constant、system12/full4096、Full13qubit/ancilla wire12。
T=.8/S2/q8/K2、P=0、central-rotation制御、cosine/+Im sineの測定付きfull Hadamard wrapper。
Qiskit1.3.0/NumPy1.26.4/SciPy1.14.1、保存compile profile rz,sx,x,cx/opt1/seed17、追加routingなし。DF rankは適用外。

| ID | candidate | L_D | r | 固定event位置数 | 固定標本 |
|---|---|---:|---:|---:|---|
| R64 | H6_p000_q8_r64_K2 | 0 | 64 | 512 | confirmation/0000 |
| H32 | PR_p0078_q8_r32_K2 | 78 | 32 | 256 | confirmation/0000 |
| H64 | PR_p0078_q8_r64_K2 | 78 | 64 | 512 | confirmation/0000 |

- 科学SOURCE `9d392de7c8c6f8ae35faba9aeb1c8c6f6a215fef`。人工9 tests PASS後に公開・8変更ファイル/treeを照合し、新namespaceへ1 batchだけ実行。
- 承認対象の[調査書§8](../../../../docs/research/h6_pure_rte_hybrid_implementation_evidence_inventory_20261102.md)・[固定INVENTORY](../../../h6_pure_rte_hybrid_evidence_audit/2026-11-02/INVENTORY.json)はcommit `b3bab573cf17b9dd8ddc1dca5f1a6a6c935ef6d1`。
- [実行条件と数値政策](../../../../docs/research/h6_selected_implementation_execution_20261102.md)。
- Hybrid保存結果 `00958ae1945c780718b2deef07d5dab1096a5cd8` / SOURCE `887b69e449cb3c378286f09d0d7626599850fb67`。
- 純RTE保存結果 `10cb3e5cca7834ebf75301bd91bd093ca51d4a6b` / SOURCE `e755ecf581600192fd2745d7066e410cebd933d3`。
- P2結果 `c9073274daee69c9b69d8db3dc332bf6132c2259` / SOURCE `11e50e262f95630b14fa4070ad30894ba3f64b74`の読込・意味論・価格照合経路を再利用。
- 原model SHA `6c66a2eeb4d99cf5bbd9137bbcb067275536be3201d5cc047225266dc5f58681`、raw state hash `ff1ba6309c6b17a263195eea718f1d5243cbe6b3422c6a9d1fbb0fe8b212ff0c`。

[SOURCE・archive part/member・raw/gzip QPY・runtime・15 taskの対応](source_and_inputs.json)。
保存raw stateをbit-copyし、exact νを確認して一度だけ除算した。stateの再生成・最適化・再正規化、full-H reference生成はない。
Bはraw各信号へ掛けず、差の会計時に方式別B_savedを一度だけ適用する。

## 作用と測定の境界

selected C_iは同じ保存event列の数学的作用で、law平均ではない。exact coefficient×exact saved delta/2の保持rotation、保存binary64 event角度、
符号付きPauli product→rotation→order位相、各q stepの保存scalarをそのまま適用した。新law/angle/samplingは生成していない。
native入力とcompiledは保存QPYをrestoreし、末尾のmeasureだけを評価用copyから除いてStatevectorで作用した。
raw p0−p1を元のνで割り、確率APIのclipや出力normでの再正規化はしない。full/global phaseは保持し、事後phase alignmentもない。
full-wrapperのglobal phaseはobservableに影響しないが、bodyのancilla-relative phaseを除去したわけではない。

P2 preflightでは12 QPYのraw/gzipと6 Fullのtoken/phase/wire/angle/wrapper/保存metricsを照合した。
[preflight.json](preflight.json)は状態作用前の確認。run内はpreflight12 decode＋作用worker12 restoreであり、新Full再生成/compileではない。

| selected-target（B補正前のphysical複素振幅） | Re | Im | engineering数値分解能 |
|---|---:|---:|---:|
| R64 | -0.90208062085851248 | 0.17994765324012305 | 1.199284497e-11 |
| H32 | -0.0034837391101215013 | 0.00014670510678145115 | 2.080322190e-11 |
| H64 | -0.80604214434325572 | 0.49285400903399473 | 2.307813330e-11 |

個別trajectoryの振幅が保存law平均と異なることは不整合ではない。特にH32のorder2を含む個別振幅をlaw平均へ置き換えて誤差を計算していない。
全15作用のraw値、数値分解能・vector誤差候補、PID/wall timeは[actions.json](actions.json)。

## B補正した6比較と保存budget

η=B|m_input−m_C|、δ=B|m_compiled−m_input|、d=B|m_compiled−m_C|。ηはselected-target→native接続、δはnative→compiled接続。
保存ε_sig=.0005の候補/軸別rho_guardをそのまま引用した。G・N・感度や新しい許容幅は計算していない。

| 対象 | 軸 | η点値 | δ点値 | d点値 | dのengineering上側候補 | 保存rho_guard |
|---|---|---:|---:|---:|---:|---:|
| H32 | cosine | 1.618288475e-15 | 1.204307702e-15 | 4.139807727e-16 | 1.302809186e-09 | 2.805804609e-06 |
| H32 | sine | 1.535824391e-16 | 2.125248887e-16 | 5.894244959e-17 | 1.302858676e-09 | 2.375479343e-06 |
| H64 | cosine | 5.873691598e-13 | 5.772096676e-13 | 1.015949219e-14 | 1.683810490e-09 | 2.805806633e-06 |
| H64 | sine | 3.605365471e-13 | 3.516313132e-13 | 8.905233898e-15 | 1.683751854e-09 | 2.375480245e-06 |
| R64 | cosine | 4.780620149e-13 | 4.012247685e-13 | 7.683724640e-14 | 1.006816423e-09 | 3.500412788e-06 |
| R64 | sine | 9.473590135e-14 | 7.978622388e-14 | 1.494967747e-14 | 1.006744373e-09 | 3.500441821e-06 |

全6行でENGINEERING_UPPER_WITHIN_SAVED_LOCAL_BUDGET。直接差はいずれも付与したengineering resolution以下で、解像していない。
η/δ/dの全ての分解能区間・exact Fraction比較・triangle確認は[results.json](results.json)、転記は[comparison.csv](comparison.csv)。
ηやδの点値と合成dの大小から、数値的相殺の原因やcompilerが誤差を修正したという説明を導かない。

## 数値不確かさと判定の意味

計算中心はcomplex128。ε=2^-52、γ_k=kε/(1−kε)に基づき、演算数・角度・vector長へ応じたengineering候補を伝播した。
selected C_iはPauli sign/swapをexactに扱い、角度binding/trig/線形和/scalar/dot/ν変換の誤差候補を付けた。
native/compiledは局所幅w≤4に対する`γ_(32w)*(1+Σ|params|)+γ_(8w)*sqrt(w)`、unitaryに相関する入力誤差とobservable還元を使う。
各差の分解能は両側の候補をB_savedで一度scaledして加えた。固定signal誤差1e-10などは置いていない。
**これはlibm・Qiskit内部全演算を包含した証明ではない。** `numerical_error_upper/rho_upper/certified_upper=null`を維持する。
保存law平均のtight L2包含も、今回のliteral gate列の包含として流用していない。
旧比較関数のLOCAL_BUDGET_MET=falseは正式保証の非認定という意味で保持し、今回のengineering所属判定と区別する。
実compiledρや全law誤差を取得したと認定せず、旧actual_N/actual_Gやρ=0会計を更新していない。

## この結果で支持できること・残ること

固定3 trajectory/両軸では、selected-target・native・compiledの間に保存許容幅を脅かす不整合は、このengineering診断では観測されなかった。
純RTEとHybrid双方の保存programの局所的な作用対応を追加確認した。保持prefixの単独因果効果・方式一般の優位・母平均費用順位は示さない。
R64をH32/R64とH64/R64で二重実行しておらず、同じindexでも方式間paired標本にはしていない。
H32にはorder2が1個、H64/R64はorder0だけ。K2のorder4はsupport外。
この6組から未観測event・他trajectory・全lawへ外挿しない。全lawに適用するには一様上界または妥当な重み付き転送が別に必要で、
価格tail・選抜後不確かさ・N/Gへの実ρ接続も別の研究scopeである。P3の精密化やDF/RPE全体の会計は開始していない。

## 人工検査・件数・保全・STOP

人工9 tests PASS：signed Pauli/endian、非可換products/rotations/±phase/保持/scalar/非unit raw norm、L0境界、両測定軸、
global phase不変性、2qubit人工compile対照/gate拒否、B一回/triangle/局所非認定、欠測・大分解能、15 unique tasks/R64共有。
人工compile1件と実H6 compile0を区別する。別reviewerの独立レビューではなく、作成者のcorrectness検査である。
科学batch1、15/15状態作用（selected3/native6/compiled6）、6比較、親exit0、wall 14.879804秒。
15 distinct worker PID、spawn/fresh workerの1作用ずつ、全workerをjoin。worker内BLAS1 threadは局所小gate処理をprocess並列へ配分する設定で、共有環境は変更していない。

```json
{
  "scientific_batches": 1,
  "state_actions_submitted": 15,
  "state_actions_completed": 15,
  "selected_actions": 3,
  "native_input_actions": 6,
  "compiled_actions": 6,
  "comparisons": 6,
  "new_sampling": 0,
  "RNG_replay": 0,
  "Full_regenerated": 0,
  "new_compile": 0,
  "new_law_mean": 0,
  "new_full_H_reference": 0,
  "new_N_G": 0,
  "new_price_tail": 0,
  "automatic_retries": 0
}
```

[verification.json](verification.json)：SOURCE16・固定local入力2・15作用・6比較・全回収を確認。
[publication_verification.json](publication_verification.json)：保存CSV/JSON転記と旧結果/STOP・manifest保全を記録。
旧P1/P2・Pauli比較・P3全段階のSOURCE/raw/結果/STOPは保全した。新N/G/SE/price tail・law平均・sampling/compile・自動retryは0。
次の担当はユーザー：この局所engineering確認を条件付きcase studyへ加える範囲と、全law誤差転送・価格tail等の追加要否を判断する。
Codexは新しい明示指示まで追加科学処理や会計更新を開始しない。
**結果公開・remote SHA/bytes照合後mandatory STOP。**
