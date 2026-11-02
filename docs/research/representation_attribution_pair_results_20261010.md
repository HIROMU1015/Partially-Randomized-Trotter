# Aのcore寄与分解・B′物理pair比較：限定結果とGPT review（2026-10-10）

同じ固定toyに安いwhole-Pauli対照を加えると、Aのcore導入を資源改善の中心とする根拠は残らなかった。
全占有対角化による小さいRZ改善はCX改善を伴わない。B′は物理pair恒等式とcontrolled primitiveを実装できたが、
小さいlambda/shot数の利益をbasisとcentral gateの費用が上回り、直接Pauliより高費用だった。
これを固定q1・一つのcompiler・binary64小系診断に限る識別結果とし、一般的不可能性や新規性否定の証明へ広げない。

`ATTRIBUTION_PAIR_COMPLETE_AWAITING_GPT_REVIEW`、mandatory STOP。
central_hypothesis_adopted=null、next_stage_authorized=false。科学batchはrun1のみ、C追加scan=0。

## 証拠と比較条件

base `c24dcede10ed9726e1e0b1030eca749ba7bc5cba`、source `3975650908c594ce6dddb4fc2086167caf5b6460`。
branch `representation-attribution-pair-20261010`。利用者提供[review §9](representation_attribution_pair_inputs/representation_construction_comparison_scientific_review_2026-10-10.md)を採用した
[結果前scope](representation_attribution_pair_scope.md)、[準備と初回test修正](representation_attribution_pair_preparation.md)、
[primary result](../../artifacts/representation_attribution_pair/2026-10-10/run1/result.json)、
[全native IR](../../artifacts/representation_attribution_pair/2026-10-10/run1/native_ir.json)、
[source/env/resource audit](../../artifacts/representation_attribution_pair/2026-10-10/run1/run_audit.json)、
[別実装保存監査](../../artifacts/representation_attribution_pair/2026-10-10/saved_evidence_audit.json)を一組に読む。
旧source/resultは保持し、新direct辞書の実装差と新次数別推定を旧8draw点推定と区別する。

actual costは結果前固定q1だけ。mean/bias/Gammaはq1/2/4の全30件。
epsilon_complex=.05/.02、axis epsilon=epsilon/√2、二軸合計失敗率.05、N=ceil[2Γ²log80/(epsilon/√2−b)²]。
bはexact小系operator診断で、形式的なbias certificateではない。
全Hadamard wrapperに状態準備・ordinary control・X/Y readoutを含める。測定1はIR外metadata。
Qiskit1.3.0、opt1、seed20261010、basis rz/sx/x/cx、coupling mapなし。
表のcostはX/Y各一shotの費用の和、workは同じ各軸Nを掛けた値。±はpaired engineering SEで厳密CIではない。
異なる資源を重みで一つにまとめない。全6native metricsと候補差SEはprimary JSONに残す。

## A：辞書とcoreの効果

3-mode JW、全Fock8、synthetic正平方DF rank2、geometry/化学basisなし。
H=ΣdΓ(g_l)²、g0=[[.8,.09,0],[.09,-.35,0],[0,0,.15]]、g1=[[-.2,0,0],[0,.6,.07],[0,.07,-.45]]。
独立onebody/constant correctionなし。native L_D0/1/2、whole-H all-R、計算基底core二種。
T=.4、q1/2/4、delta=.4/.2/.1、r1/K2（tailなしはdeterministic PF）。
準備X0のsignalは1粒子sectorでGaussian等価だが、適格性は全Fock biasで揃えた。相関化学の検証ではない。

| candidate | λ / scalar | Γ / bias (q1) | N(.05) | XY RZ/shot | XY CX/shot | XY depth/shot | RZ work ±SE | CX work ±SE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| native_df_ld0 | 0.900412 / 0.444 | 1.12820739 / 0.00069213421 | 9285 | 60.63325 | 19.72822 | 73.23305 | 562,979.749 ±2,607.203 | 183,176.525 ±844.049 |
| native_df_ld1 | 0.274862 / 0.1537 | 1.01207372 / 0.00013151067 | 7236 | 207.18979 | 119.44275 | 288.26265 | 1,499,225.311 ±38.907 | 864,287.748 ±17.914 |
| native_df_ld2 | 0 / 0 | 1.00000000 / 0.00012859386 | 7063 | 229.00000 | 148.00000 | 326.00000 | 1,617,427.000 ±0.000 | 1,045,324.000 ±0.000 |
| collected_whole_pauli | 0.979 / 0.444 | 1.15126735 / 0.00069213421 | 9668 | 14.37920 | 6.93305 | 20.18607 | 139,018.099 ±410.653 | 67,028.724 ±241.348 |
| frame_identity | 0.0915 / 0.0065 | 1.00133939 / 0.00029863003 | 7151 | 71.30369 | 81.10274 | 136.29047 | 509,892.718 ±5.499 | 579,965.686 ±3.020 |
| complete_occupation_core | 0.085 / 0 | 1.00115587 / 0.00029869445 | 7148 | 71.01131 | 81.30044 | 136.09082 | 507,588.821 ±5.155 | 581,135.568 ±2.849 |

whole-Pauliとnative all-Rのcorrected mean/bias/scalarは同じで、lambdaは.979対.900412とwhole-Pauliが大きい。
Nも9668対9285へ増える。それでもbasis不要の辞書が1shot費用を下げ、RZ work約75.31%、CX約63.41%減となった。
この比較では低いlambdaによるshot節約より、入力表現から実際に生成される回路費用が大きい。
native L_D1/2も全候補に残したがq1費用は高かった。qやcompilerを跨ぐ最適性は評価していない。

whole-Pauli対wholecoreの差は、coreによりNを9668→7148へ減らす一方、X/Y pair RZを14.379→71.011へ増やす。
したがって完全coreのworkはwhole-Pauliの約3.65倍RZ、約8.67倍CX、約4.98倍depthとなる。
whole-Pauli−complete-coreの差はRZ −368,570.722±409.100、CX −514,106.844±240.559。
安い対照を加えても残ったcore固有の資源利得、という解釈を支持しない。

全占有対角D*は従来D0へ
.0081(n0+n1−2n0n1)+.0049(n1+n2−2n1n2)を加える。追加のPauli supportは0。
λは.0915/10terms→.085/8terms、Gamma1.001339→1.001156、N7151→7148。
biasは.000298630→.000298694と僅かに増え、bias改善とはいえない。
旧core−完全coreのpaired差はRZ +2,303.897±2.763、CX −1,169.882±1.505。
完全coreのRZ work約0.452%減、CX約0.202%増、depth約0.188%減。費用差の代数的分解は
[保存値寄与表](../../artifacts/representation_attribution_pair/2026-10-10/attribution_summary.json)を参照。
これは既知の占有対角成分の正確な吸収による小さなcompiler条件付き変化で、新しい一般構成原理の証拠ではない。

source構造診断はbasisの呼出回数・操作数を別保存する。旧frame_identityのX/Y和12操作は、外側の
3つのRZ(0)を各方向・各軸で数えたもの。実際のorbital移動は恒等で、nativeのbasis費用と解釈しない。
whole-Pauli/完全coreのbasis操作0、native all-RのX/Y期待source basis操作20.559。
これはnative gate費用に加算する独立単位ではない。

## B′：物理pairとscalar shift

physical3+aux1、JW、同じV4の右積G01(π/8)G23(π/6)G12(π/10)、上3行isometry。
density edges01=.7、12=.4、23=.25、03=−.2。geometry/化学basis/DF rankは該当なし、分子fittingなし。
T=.2、q1/2/4、delta=.2/.1/.05、L_D0/r1/K2、準備X0X1・aux真空。
物理全Fock8の第一moment精度で比較し、反射法の個別trajectory leakageやreset/channel保護の費用とは区別する。

cx=Σxi ai、cy=Σyi aiをQRで正規直交化すると
cx†cy†cycx=Δ n_b1 n_b2、Δ=||x||²||y||²−|x†y|²。
constructorは安定なwedge和でΔを計算し、複素係数ではQR completionの共役をorbital frameとする。
各pairのω=wΔは.6832889870078078、.3000000000000000、.0059682189257829、−.0500000000000000。
Q=I−2n_b1n_b2はGaussian-conjugated CZで、H=(Σω/2)I−Σω Q/2。
Q辞書のscalarは.4696286029667954、λ=.5196286029667954。
Z/Z/ZZ辞書はscalar .2348143014833977、λ=.7794429044501932。
scalarを消さず、制御枝の位相をabsolute operatorとして照合した。

| candidate | λ / scalar | Γ / bias (q1) | N(.05) | XY RZ/shot | XY CX/shot | XY depth/shot | RZ work ±SE | CX work ±SE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| enlarged_generator_reflection | 0.9625 / 0.2875 | 1.03692582 / 1.2026901e-05 | 7544 | 87.26179 | 27.29269 | 76.67067 | 658,302.948 ±381.199 | 205,896.080 ±165.272 |
| coefficient_physical_pauli | 1.04535 / 0.234814 | 1.04352978 / 1.6412585e-05 | 7643 | 17.41391 | 7.03234 | 21.06932 | 133,094.499 ±162.415 | 53,748.197 ±79.520 |
| physical_pair_Z_ZZ | 0.779443 / 0.234814 | 1.02424471 / 1.6412585e-05 | 7363 | 67.71492 | 22.07018 | 81.40211 | 498,584.947 ±413.344 | 162,502.740 ±146.127 |
| physical_pair_Q | 0.519629 / 0.469629 | 1.01078929 / 4.7325949e-06 | 7166 | 81.24492 | 30.57812 | 96.98167 | 582,201.076 ±0.000 | 219,122.803 ±0.000 |

新directはquartic係数代数から生成する18primitive辞書。旧dense-reference31primitiveとの差は主に
floating roundoff由来の微小非零supportで、operator差4.4412e−16。truncate thresholdを追加していない。
以前の8draw費用を新directの厳密再現と称さず、同じ物理Hに対する明示した新constructorとして比較する。

Q化はN7643→7166へ6.24%減らすが、X/Y pair RZ17.414→81.245へ増やす。
QのRZ workはdirectの約4.37倍、CX約4.08倍、depth約4.32倍。
direct−Qの差はRZ −449,106.577±162.415、CX −165,374.607±79.520。
Qの小さいlambdaだけを改善と呼べない。Qは反射法よりRZが低いがCX/depthが高く、資源間の利害分岐も残る。
Z/Z/ZZ pairは反射よりRZ/CXを下げるがdepthは上がり、directより全表記三軸で高い。
本実装でQ化したcentral controlled pair product/rotationとbasis移動の費用が、shot節約を上回った。
最適completion/target backend/Clifford+T合成や、大系での優位性を判定したわけではない。

## 稀な次数・不確かさ・実行監査

次数0を全component列挙、次数2を条件付き96 IID draw、Qの次数2だけ4³=64全列挙。
共有96×3uniform seed20264110をinverse-CDFで各辞書へ写し、候補間とX/Y間の共分散を保持した。
raw draw933件、同一候補・次数内重複reuse308 wrapper、unique compile1558（preflight最大1866）。
無条件8drawの偶然の次数coverageに依存しない。Q辞書のcost SE0は全有限eventを列挙したためで、
量子shot noiseやbinary64/環境差が0という意味ではない。
native all-RのRZ workは旧603,525±88,364に対し新562,979.752±2,607.195。
旧値を誤りと断定せず、旧8drawの推定幅と新次数別推定を区別する。
source-bound sampling SEの範囲で対照との大差は残るが、厳密有意差検定・多重比較・compiler移送CIは行わない。

実行wall24.1247秒、CPU24.1327秒、peak RSS560.52 MiB、1 process/BLAS1。
source cap CPU600s/wall900s/AS4GiB/64MiB file/compile2048/5qubit/10000 native gatesを満たした。
native最大254 gates、総165,923 gates、1558 IR、177 source/input blobs、30 mean、933 drawを別実装監査PASS。
exterior-power minors、explicit JW、native行更新を使い、Qiskit/project module/sample/transpileを再実行していない。
saved uniform identityのseed replayだけを明示する。最大absolute wrapper差3.1303e−14、mean差2.2204e−16。
unsigned/resigned位相改変とshot整数改変の3チェックを拒否した。
85 focused local tests passed、18既存warnings、fail/skip0。初回専用testのfield名誤りは修正履歴を保存。
science failure0、再実行0、分子load/ground-state/GPU/量子shots0。immutable CIや外部科学再現ではない。
提供inline algebraも別artifactで再確認し、Aの有理数辞書はsupplied JSONと一致した。

## 限定prior-art照合とGPTへ戻す判断

既知のorbital-rotated density fragmentと、occupationをreflectionへ換えるLCU norm削減、basis rotationの
回路費用は既に扱われている。B′のpair QR後density実装はこのclassに入る。
本実装のQ centeringと各pair QRを合わせたRTE辞書の全algorithmが同論文と完全同一とまでは立証しないが、
代数やGaussian density primitiveを新規性として採択しない。[Martínez-Martínez et al., §2.3 / Eqs.(37)–(42)](https://arxiv.org/html/2210.10189v2)

THCには非直交factorを局所的なbasis rotationとSELECTへ接続する既存構成がある。
aux basisを使うだけの説明や、小さいlambdaだけで独立の研究寄与とはいえない。
同論文のqubitization/QROM/SELECT費用を今回のtopology-free sampled-Hadamard費用と同じ尺度として扱わない。
[Lee et al., §II.3–II.4 / III.3](https://arxiv.org/html/2011.03494v3)

ITHCはenlarged densityのaux-vacuum射影と、実際のaux resetを伴うstate simulationを扱う。
今回確認したのはその固定構造fixtureと物理pair第一momentで、分子圧縮rankの利益やreset付きchannel全体の比較ではない。
[Luo and Cirac, Eqs.(4)–(10)](https://arxiv.org/html/2407.04432v2)

これら3一次資料と提供reviewの限定照合であり、全世界のpriorityや全simulation methodを検索した証明ではない。
判定は「この固定taskではcore/pairを中心改善として支持せず、既知辞書とnative費用の説明が主要」。
主要研究方針はGPT担当のまま、以下をreviewへ返す。

1. Aのshared coreを比較primitiveとして保持し、別の構成原理へ探索を広げるか。現toyのパラメータ追加で成功を作らない。
2. B′は既知fermionic-density/projector primitiveとして保持するか。新規主題には別の方法論差分とtaskが必要。
3. 次taskを選ぶ際、1粒子Gaussian等価性を避ける必要と、大系/molecular fittingの必要性をどう条件付けるか。
4. Cは現低優先度を保持し、具体的な新cheap mixed-access案まで追加scanを止めるか。

次batchは未認可。旧Track A/Bのsource/契約/STOPと既存証拠を保護したままここで停止する。

## 精度.02の全候補q1会計

| group/candidate | N/axis | RZ work ±SE | CX work ±SE | depth work ±SE |
|---|---:|---:|---:|---:|
| A/native_df_ld0 | 61665 | 3,738,949.512 ±17,315.364 | 1,216,540.701 ±5,605.628 | 4,515,916.268 ±20,995.419 |
| A/native_df_ld1 | 45732 | 9,475,203.420 ±245.893 | 5,462,355.898 ±113.220 | 13,182,827.335 ±360.707 |
| A/native_df_ld2 | 44629 | 10,220,041.000 ±0.000 | 6,605,092.000 ±0.000 | 14,549,054.000 ±0.000 |
| A/collected_whole_pauli | 64212 | 923,317.148 ±2,727.435 | 445,184.986 ±1,602.962 | 1,296,187.726 ±4,793.062 |
| A/frame_identity | 45854 | 3,269,559.600 ±35.263 | 3,718,884.993 ±19.367 | 6,249,463.083 ±55.202 |
| A/complete_occupation_core | 45838 | 3,255,016.282 ±33.060 | 3,726,649.717 ±18.270 | 6,238,130.804 ±52.344 |
| B/enlarged_generator_reflection | 47197 | 4,118,494.732 ±2,384.869 | 1,288,133.254 ±1,033.980 | 3,618,625.421 ±3,950.753 |
| B/coefficient_physical_pauli | 47830 | 832,907.222 ±1,016.398 | 336,356.960 ±497.639 | 1,007,745.733 ±1,386.210 |
| B/physical_pair_Z_ZZ | 46078 | 3,120,168.026 ±2,586.727 | 1,016,949.784 ±914.472 | 3,750,846.202 ±3,365.651 |
| B/physical_pair_Q | 44801 | 3,639,853.534 ±0.000 | 1,369,930.326 ±0.000 | 4,344,875.872 ±0.000 |

計算系列日付は2026-10-10。日付を跨いだ2026-10-11に報告・公開確認を整理し、source/resultの系列pathは改名しない。

## GitHub取得先

[source3975650](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3975650908c594ce6dddb4fc2086167caf5b6460)、[専用branch](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/representation-attribution-pair-20261010)、[一次結果・全IR・監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/representation-attribution-pair-20261010/artifacts/representation_attribution_pair/2026-10-10/)。公開result commitと独立fetch照合のreceiptは同artifactディレクトリへ別保存する。

全体manifest checkerは既存relevant_commits[7]のdescription欠測で基点/currentとも同じ失敗。新entryのschema viewはPASSし、旧84 result entriesと以前のJSON書式を保持した。[manifest検査記録](../../artifacts/representation_attribution_pair/2026-10-10/manifest_check_audit.json)。元root HEADと1317 baseline pathsは同一、以前のstatus entryも保持。作業中にrootへ追加された別Track B review一件をそのまま残した。[保全audit](../../artifacts/representation_attribution_pair/2026-10-10/preservation_audit.json)。

公開result commit `e576f023db84e6ac40629fd0437cab9c19073381`を新しいbare repositoryから独立fetchし、変更37 filesとsource/input177 blobsをSHA照合PASS。[取得receipt](../../artifacts/representation_attribution_pair/2026-10-10/remote_retrieval_receipt.json)、[immutable一次結果・全IR](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e576f023db84e6ac40629fd0437cab9c19073381/artifacts/representation_attribution_pair/2026-10-10/run1/)。大きいIRはGit blob/raw Git取得で参照する。このreceipt整理は後続child commitで、科学再実行ではない。
