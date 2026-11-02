# Track B G9 v2：matched-native結果・GPT handoff

## 完了とSTOP

**`G9_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE`、全11 rows / 22 axesの会計完了。mandatory STOP済み。**

利用者の明示one-shot指示に基づく別実行。runs=1 / retries=0。
旧G9 v1のtechnical result・消費済みmarkerはそのまま保存し、v2では別authorization・別markerを使用した。
完了後は保存値の照合と資料公開だけを行い、追加synthesis、generator、matrix、科学run、testsは行っていない。

- execution branch：`track-b-g9-v2-one-shot-execution-20261010`
- 固定source S：`0ef2b92738750a3c0d187743dd4b7c9b927db802`
- authorization-only / 実行HEAD A：`5ed597d8163af002d315a9e2b9eee0e9f14edce2`（唯一parent=S、2pathだけの変更）
- Aの公開remote SHAとclean worktreeを照合してからlaunch gateを通過。
- contract SHA256：`2b07deb4740b6d987fe88058d05fc6dde5f6970fdeb34b69a4ba495343e88531`
- authorization SHA256：`4341338c83a7ebf96781c4e406d194b4ea844eb31a6eae2d03a51821935d827a`
- marker SHA256：`5f895ad90418d9a662093724fc1348921dba22d5da95b5413a94ae8f8bde9dc4`
- raw result SHA256：`3eb8430014a7881b5189ca8779a58ccd7ab2f50c37a58795bd0458b174abda20`

## 固定対象と証拠範囲

known development `p=(1/5,3/10,1/2), x=5/7, m=5`の**full first operator moment P5**。
G8 review指定の3-system-qubit synthetic provider：
`Q0=Z0; V1=R_XX01(pi/4); V2=R_XX12(pi/4)R_ZZ01(pi/4); Qi=Vi†ZiVi`、右から作用。
分子、geometry、basis、DF rank/split/delta windowは適用外。この文書はPR/QPE最終総costや実分子の有利性ではない。
Pauli情報が安く収集できるI1 context。p/x/m/角度は既知であり、完全held-out/独立再現/I0取得優位と呼ばない。

Re/Im各1/200、22 axis alpha49/22000で0.049、11 resource rows beta1/11000で0.001、familywise0.05。
これは有限P5 meanの推定taskであり、exact exponentialのTaylor tailやQPE全体の精度保証へ置き換えない。
provider delta=0はexact Clifford+T model。G8の仮想delta10^-6達成実験ではない。
Rz strict epsilon10^-6、H160/K256、rho=eta10^-12、同direct構成・同CZ lowering・adjacent inverse cancellation。
実量子shot/trajectoryは0で、保存されたoperator/referenceとnative費用・十分shot予算のlocal evidenceである。

## Primary direct：6方式

期待費用は二axis合計。T欄はnative-eventの**intercept**、総primaryは`T intercept + K*T_prep/readout`。
状態準備/readoutは全方式共通の非負単価parameterとして残し、無料準備を科学的前提にしない。
CXはnative-eventの座標。1Qは全CSVでnative-eventと共通outer prep/readoutを別々に表示する。

| 方式 | N / axis | m2 upper | native T intercept | native CX | K（prep/readout） | workspace |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 1,246,046 | 2.256283543 | 348,759,216.423 | 20,586,846.721 | 2,492,092.000 | 1 |
| partial-return + tail | 949,142 | 1.718110908 | 258,865,913.425 | 15,975,424.661 | 1,898,284.000 | 1 |
| closed P3 + tail | 933,915 | 1.690512970 | 267,617,674.368 | 16,067,266.079 | 1,867,830.000 | 1 |
| general local full | 1,076,085 | 1.947847992 | 257,506,695.137 | 15,748,232.374 | 1,846,058.549 | 1 |
| closed P5 full | 917,129 | 1.660089378 | 255,860,636.626 | 15,647,565.043 | 1,834,258.000 | 1 |
| matched CTS | 1,596,220 | 2.891104970 | 364,311,583.986 | 9,766,516.587 | 3,192,440.000 | 1 |

保存値ではclosed P5が登録direct6方式のT interceptとKの両方で最小。
したがって、この保存されたcanonical law・固定精度・共通compiler・同confidence policy内で、
共通非負T_prep/readoutを足してもT-primaryの順序は変わらない。
全15組のexact Fraction差を[affine保存値表](../../../artifacts/track_b_g9_p5_native_result/2026-10-10/v2/saved_affine_T_comparison_v2.json)に保存した。
追加候補探索、cost-aware proposalの最適化、threshold変更、研究GO分類はしていない。

closed P5のT interceptはこのmatched CTSより表示上29.768734%小さい。
一方、CTSはnative CXで小さい。T/CX間の交換関係を保持し、全resource座標のstrict dominanceと呼ばない。
CTSのeven correction10 eventはT=0を含めて保存し、無料eventを恣意的に除外していない。

## P5 fast pathとgeneral local fullの分解

独立formal auditは31 parents / 63 events / 10 groupsでexact係数一致。
source preparationの独立first-adjacent deletion、任意Lの群式、local条件の証拠を保持し、runnerでもformal auditを完了した。
closed P5とgeneral local fullは同じ理想full-return ensembleを実現する。
有限bitのgroup/law丸めは別々なので、digital係数のbyte-exact同一性は要求しない。

- general local full：reference acceptance=0.857766137921、m2 ref=1.935363620466、m2 upper=1.947847992243。
- closed P5：acceptance=1、m2 ref=1.660089378201、m2 upper=1.660089378216。
- general local fullのK=1,846,058.549050、closed P5のK=1,834,258.000000。
- accepted event当たりTの表示はfull=139.489993570、P5=139.489993570。

native費用を実装へ戻してもこの固定登録表でfullはordinary/partial/P3対照より小さいT interceptを持つ。
closed P5はfullよりT interceptで表示上0.639229%小さい。
その差を独立競合法の発見や一般法の失敗と採点せず、normalizer・zero-fill・保守的budgetの違いとして示す。
小mのfast pathを許した後の一般mの役割・新規性・適用域はGPT判断。

O(L²)はclosed P5群の構築/root算術operationのscope。bit費用を無料にしない。
現在のdyadic group selectorは群lawをlinear scanし、全sampleをO(L)と主張しない。
local conditional部分はO(L)。G9は登録全event reference会計であり、production samplerの新しい長trace検証や大L実行ではない。

## Generic helper：接続診断5方式

次の5 rowはprimaryを置き換えず、全方式にdirectを許したうえで別診断として保持した。

| 方式 | N / axis | m2 upper | native T intercept | native CX | K（prep/readout） | workspace |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 1,246,046 | 2.256283543 | 355,525,243.613 | 39,103,085.102 | 2,492,092.000 | 2 |
| partial-return + tail | 949,142 | 1.718110908 | 264,051,616.707 | 30,143,399.225 | 1,898,284.000 | 2 |
| closed P3 + tail | 933,915 | 1.690512970 | 272,436,107.957 | 29,439,793.256 | 1,867,830.000 | 2 |
| general local full | 1,076,085 | 1.947847992 | 262,257,272.596 | 28,941,504.391 | 1,846,058.549 | 2 |
| closed P5 full | 917,129 | 1.660089378 | 260,580,846.997 | 28,756,501.785 | 1,834,258.000 | 2 |

workspaceはsystem外のancilla数。directでは1（total4 qubits）、helperでは2（total5 qubits）。
全回路最適化、最良compiler、cost-aware IS/synthesis精度配分最適性、世界的native優位は主張しない。

## Strict error・operator意味論・会計監査

19 keyすべてのsequence identity、T/T†/1Q/W count、保存strict error guardを照合。
18旧sequenceはangle/epsilon/tool/phase/hashが一致するものだけ再利用し、旧wrapper costは流用していない。
CTS用の登録new keyを一回取得し、T=68、sequence長=173、
SHA256=`848bdea8499ba7aae30e7517e784e77816b1217a5fa271b0040c06913e3bec87`、strict error upper表示=1.40101360638e-07。
negative rotationはactual adjoint、controlled relative phaseを保持。

全11 rowのweighted operator mean残差、全1,866 native-event bindingのideal phase/realized reference errorを実行中に確認した。
最大mean残差表示=2.03189578201e-15、
最大ideal native phase診断=6.4086262869e-15、
最大realized error診断=2.39006538095e-07。
float matrix tolerance10^-10は意味論のdiagnosticであり、confidence認証の代用品ではない。
confidenceには固定coefficient error<=8rho、event composition<=2epsilon、共通biasを使用し、実際に小さかったerrorやsignalでNを削減していない。

[保存値監査](../../../artifacts/track_b_g9_p5_native_result/2026-10-10/v2/saved_output_audit_v2.json)は**25 checks PASS**。
stdlibだけでprobability/weight/m2/range、native加算counts、固定Bernstein N、accepted capとhard attempts、affine Kを再照合。
operator/guard/synthesis/generatorを再実行しない。source critical80と過去982保護pathは不変。
source preparationのG9 v2 boundary19 testsとG9 v1 semantic23 testsは以前のlocal証拠として保持し、今回のSTOP後に再実行していない。
immutable CIや外部再現とは呼ばない。

## 資源上限と時刻区間

- Guard区間 wall=3.094704 s / CPU=3.094166 s / peak RSS=177184 KiB（173.031 MiB）。
- new key wall=0.300914 s / CPU=0.300754 s。
- reference/native会計区間=2.762905 s。
- raw result=11,341,870 bytes、output16 MiB以内。全time/CPU/RSS/call/cache/sequence/shot上限を通過。
- wallはguard区間の値で、pre-marker provenance検証や最終serializationを含む総command時間ではない。
- CPU区間にはsmall full-event referenceが含まれる。大Lの生成器費用・classical scalability・入力取得費用の優位を証明しない。
- accepted cap（resource failure付き）とhard attempts（deterministic上限）を分け、各max event Tを掛けたT upperをCSV/原resultへ保存。
  期待T、tail upper、hard upperを混同しない。local fullのcapはaccepted=1,864,070 / hard=2,152,170。

## 正本・固定source

- [raw result](../../../artifacts/track_b_g9_p5_native_result/2026-10-10/v2/result_v1.json)
- [one-shot marker](../../../artifacts/track_b_g9_p5_native_result/2026-10-10/v2/one_shot_consumed.json)
- [STOP](../../../artifacts/track_b_g9_p5_native_result/2026-10-10/v2/STOP.json)
- [全11 rows CSV](../../../artifacts/track_b_g9_p5_native_result/2026-10-10/v2/resource_rows_display_v2.csv)
- [manifest](../../../artifacts/track_b_g9_p5_native_result/2026-10-10/v2/evidence_manifest_v2.json)
- [saved-only verifier](../../../scripts/tracks/algorithm_codesign/audit_g9_v2_saved_outputs.py)
- [source S](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/0ef2b92738750a3c0d187743dd4b7c9b927db802)
- [authorization A](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/5ed597d8163af002d315a9e2b9eee0e9f14edce2)
- [旧v1失敗を保持した資料](g9_results_and_gpt_handoff_20261010.md)
- [v2 source準備](g9_v2_api_boundary_source_review_20261010.md)
- [採用GPT G8 review](../../research/track_b_G8_scientific_review_20261010.md)

結果公開commitはAの直接childで、追加saved-only audit・資料・結果と索引への追記のみ。
source code/contract/authorization/receipt/marker/旧resultは変更しない。

## GPTへ戻す判断

登録native contextではT-primaryの有効点が残るという保存表と、CTSのCX上の利点を渡す。
次はG8 review §14.6に沿い、一般m・入力構造のvalidationの必要性/範囲、fast pathと一般法の役割、
強い対照/最適samplingとの距離、oracle/access条件、独立新規性・論文着地点をGPTが判断する。

この一つのknown P5/合成providerの値だけで主methodや独立論文十分性を採択しない。
新p/x/m/provider/grid、DF/分子/NPZ、LP/fullv4、追加合成/precision、quantum shots/trajectory/GPUへ進まない。
**mandatory STOP。次stage・研究判断はGPT/利用者へ戻す。**
