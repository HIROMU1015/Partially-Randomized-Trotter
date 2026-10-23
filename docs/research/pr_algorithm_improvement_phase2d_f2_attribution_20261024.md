# 限定F2：保存caseの回路資源寄与分析

**PPにはGaussian frameを含む回路群がなく、混在armには高費用のframe／Pauli群が残る。SP／PS差では、PSのframe出現確率は低いが、その群の条件付き費用が高いことが支配的だった。** これは保存されたfull-wrapper費用の記述的な分解であり、Gaussian変換単独の因果的overheadを測った結果ではない。

## 対象・再利用・照合

測定結果 `dc7378ab34015287215bcf4ce52147e46d94ecd9`、測定SOURCE `9468c7089727f5648d9f59cddbaebe0a42f5f0a0`、[静的原因調査](pr_algorithm_improvement_phase2d_f2_resource_causes_20261024.md) `3a782bd79967d23c47f27e75907027e2c2e87bf4` が起点。
4 mode・2粒子・occupied(0,1)、L_D=1、残差density-square 2fragmentの固定合成Hamiltonian。geometry／AO basis／分子DF rankは非該当、rank探索なし。T=1/5、q=1、δ=1/5の一点、r=2、h=1/10、K=0、faithful identity tail。
測定compilerはQiskit1.3.0／Nature0.7.2、opt1、seed17、rz/sx/x/cx、topology-free、approximation_degree=1、qubits_initially_zero=Falseの位相保持修正版。

保存 `CASE_RESOURCES.csv.gz`、`F2_laws.json`、Bの4arm atom recordをhash照合した。既存 `f2_execution.aggregate_metrics` を**変更せず再利用**し、分類・条件付き費用・差分の集計だけを `f2_attribution.py` と専用runnerへ追加した。既存関数には分類別集計がないため、その部分のみ追加している。

物理X/Y **13,674行**を全て使用し、compile aliasも省かない。共通診断2行は確率分布から除外し、full-wrapper費用からは何も差し引かない。
8種類の分類表×4arm×2axisの**64組**で、case数・確率mass・全8保存metric（主7指標＋measure数）の和を照合した。**公開Φとexact有理数で全件一致、差は0**。分類行は490、全分類でX/Yの確率・費用もexactに同じだった。以下は各axisにそれぞれ適用し、X/Yの平均ではない。

新しいHamiltonian・law生成、sampling、simulation、reference、build／compile、γ・N計算は0件。人工保存recordによる5testはPASS。これはlocal再集計と技術検査で、CI・外部再現・統計資格の取得ではない。

## 分類と計算法

保存された正規化済みfinite-bit確率をそのまま使い、ordered pairの重みを `p_i p_j` とした。Bの係数から理想確率を再生成していない。

\[
P(g)=\sum_{(i,j)\in g}p_i p_j,\qquad
A_m(g)=\sum_{(i,j)\in g}p_i p_j C_{ij,m},\qquad
\mathbb E[C_m\mid g]=A_m(g)/P(g).
\]

各分類表は単独で全caseを重複なく覆う。**異なる分類表の寄与を足してはいけない**。P=0の群は寄与0、条件付き費用は未定義としてCSVを空欄にする。

| 分類表 `view` | 排他的な分類規則 |
|---|---|
| `mechanism` | II、IF、IP、FF_same、FF_different、FP、PPの7群。I=identity、F=FRAME_Z、P=PHYSICAL_PAULI。順序は保持して集計し、この表の群名だけは両向きを併合 |
| `any_frame` | frameを1個以上含む／含まない |
| `basis_relation` | identityのみ、singleton、同frame full共有、異frame、frame/Pauli境界、同local Pauli共有／別local。PauliはZのみとX/Yありも区別 |
| `support_pair` | 各位置のI／Fのsupport数1・2／Pのsupport数1～4。`F1`は「frame support1」で、fragment名F1ではない |
| `origin_pair` | 各位置のorigin集合をそのまま保持。`F0+F1`を半分ずつ割り当てない |
| `frame_fragment` | frameなし、F0/F1 singleton、F0/F1同基底full共有、異frame pair |
| `frame_recipe` | 上記にsingletonのsupport数、異frame pairの順序・support数を追加 |
| `pauli_basis` | Pauliなし／選択Pauliが全てZのみ／XかYを含む。frameと同時選択される場合も含む |

`frame_recipe`のsingletonは「frame成分が1個」で、IFとFPの両方を含む。FPのframeにもsupport completionが使われる。同frame pairはfull basis共有であり、長いrunの検証ではない。

## 主分類の確率・条件付き費用・寄与

全てfull-wrapperの1-shot量。費用の並びは **RZ／CX／total depth**。表は丸め表示で、exact値はartifactに保存した。出現しない主分類はP=0・寄与0として省略している。

| arm | 群 | P(g) | 条件付き費用（RZ／CX／depth） | 寄与A（RZ／CX／depth） |
|---|---|---|---|---|
| SS | II | 0.069360 | 27.000000／16.000000／38.000000 | 1.872732／1.109767／2.635697 |
| SS | IF | 0.388007 | 61.643075／28.909848／76.275387 | 23.917916／11.217210／29.595349 |
| SS | FF_same | 0.273075 | 68.216021／32.886293／83.528106 | 18.628118／8.980438／22.809471 |
| SS | FF_different | 0.269558 | 97.962616／42.251776／118.391642 | 26.406569／11.389288／31.913369 |
| SP | II | 0.040589 | 27.000000／16.000000／38.000000 | 1.095894／0.649418／1.542369 |
| SP | IF | 0.122668 | 52.525471／26.442691／64.853325 | 6.443220／3.243685／7.955459 |
| SP | IP | 0.199087 | 32.046166／20.765101／45.693689 | 6.379966／4.134056／9.097007 |
| SP | FF_same | 0.092683 | 54.686524／29.528259／68.036074 | 5.068526／2.736775／6.305806 |
| SP | FP | 0.300844 | 57.329340／31.207791／74.437736 | 17.247160／9.388662／22.394111 |
| SP | PP | 0.244129 | 36.565911／25.402647／53.430590 | 8.926812／6.201532／13.043975 |
| PS | II | 0.042690 | 27.000000／16.000000／38.000000 | 1.152620／0.683034／1.622205 |
| PS | IF | 0.109790 | 72.357472／31.809085／89.697828 | 7.944142／3.492326／9.847943 |
| PS | IP | 0.218060 | 31.881740／19.941955／42.876931 | 6.952128／4.348540／9.349738 |
| PS | FF_same | 0.070590 | 86.899408／37.523528／104.921630 | 6.134255／2.648797／7.406449 |
| PS | FP | 0.280406 | 77.460001／35.751040／97.324903 | 21.720240／10.024802／27.290476 |
| PS | PP | 0.278464 | 36.082204／23.757500／49.910597 | 10.047600／6.615612／13.898311 |
| PP | II | 0.038847 | 27.000000／16.000000／38.000000 | 1.048876／0.621556／1.476196 |
| PP | IP | 0.316500 | 32.572958／20.733479／44.879326 | 10.309331／6.562139／14.204292 |
| PP | PP | 0.644653 | 37.364180／25.354865／52.659024 | 24.086933／16.345091／33.946801 |

SSでは異frame pairが確率26.9558%に対しRZ寄与26.406569と最大で、条件付き費用も97.962616 RZ／42.251776 CX。次にIFがRZ23.917916、同frame pairが18.628118を占める。共有は既に働く群として分離されているが、この群間差を「共有による削減量」とはしない。

SP／PSではFPがそれぞれRZ17.247160／21.720240を占める。PPのPauli pairの条件付きRZ37.364180に対し、FPはSP57.329340、PS77.460001である。**Gaussian frameを含む群が高費用で、PPにはその群がない**という説明を定量化できた。ただし異なる演算子・support・確率・角度の群なので、純粋なGaussian費用差にはならない。

公開済み総費用からの差も、主分類の寄与差の和としてexactに復元できた。

| 差 | RZ／CX／depth差 |
|---|---|
| SS−PP | 35.380195／9.167916／37.326597 |
| SP−PP | 9.716438／2.825341／10.711437 |
| PS−PP | 18.505844／4.284324／19.787832 |

## SP／PS差：出現頻度と条件付き費用

| arm | 群 | P(g) | 条件付き費用（RZ／CX／depth） | 寄与A（RZ／CX／depth） |
|---|---|---|---|---|
| SS | frame_present | 0.930640 | 74.091631／33.941106／90.602412 | 68.952603／31.586935／84.318189 |
| SS | no_frame | 0.069360 | 27.000000／16.000000／38.000000 | 1.872732／1.109767／2.635697 |
| SP | frame_present | 0.516195 | 55.713230／29.773853／71.010677 | 28.758906／15.369122／36.655375 |
| SP | no_frame | 0.483805 | 33.903497／22.705454／48.952293 | 16.402672／10.985006／23.683351 |
| PS | frame_present | 0.460786 | 77.690309／35.083339／96.671405 | 35.798636／16.165925／44.544867 |
| PS | no_frame | 0.539214 | 33.664482／21.600318／46.123194 | 18.152347／11.647186／24.870254 |
| PP | no_frame | 1.000000 | 35.445140／23.528787／49.627289 | 35.445140／23.528787／49.627289 |

PSはframeを含む確率が**46.0786%**で、SPの**51.6195%**より低い。それでもframe群の条件付きRZは**77.690309対55.713230**、CXは**35.083339対29.773853**で高い。frame群のRZ寄与は35.798636対28.758906となり、no-frame群の寄与差も加わってPS−SPはRZ8.789406となる。

PPのRZ平均35.445140は、混在armのno-frame群の平均33.903497（SP）／33.664482（PS）より高い。PPの総費用優位は、Pauli群内で一様に安いことからは説明できず、高費用のframe群を含まないという分布全体の違いに現れる。

主分類ごとのPS−SP差は次のとおり。FPがRZ差4.473079を占め、全RZ差の**50.8917%**に対応する。ただしこれは群への記述的な配賦である。

| 群 | PS−SPの寄与差（RZ／CX／depth） |
|---|---|
| II | 0.056726／0.033615／0.079837 |
| IF | 1.500922／0.248641／1.892484 |
| IP | 0.572162／0.214484／0.252731 |
| FF_same | 1.065729／-0.087979／1.100643 |
| FP | 4.473079／0.636140／4.896365 |
| PP | 1.120788／0.414080／0.854336 |
| 合計 | 8.789406／1.458982／9.076395 |

さらに`any_frame`の2群に対して、次の対称な代数的分解を行った。pは群確率、μはその群の条件付き費用である。

\[
\Delta C=\sum_g (p_{PS,g}-p_{SP,g})\frac{\mu_{PS,g}+\mu_{SP,g}}2
+\sum_g\frac{p_{PS,g}+p_{SP,g}}2(\mu_{PS,g}-\mu_{SP,g}).
\]

| metric | 確率構成の項 | 条件付き費用の項 | 合計PS−SP |
|---|---|---|---|
| rz_count | -1.823937 | 10.613343 | 8.789406 |
| cx_count | -0.569366 | 2.028348 | 1.458982 |
| total_depth | -2.011525 | 11.087920 | 9.076395 |

この分解では確率構成の差はPSを安くする方向で、**条件付き費用の差が上回る**。基準を一方のarmだけに置く順序依存を避けた恒等式だが、群内のsupport・Pauli partner・compiler挙動等は同時に変わる。「条件付き費用差＝F1 Gaussian変換の因果効果」ではない。

## Fragment・support・基底共有から見える箇所

SPに残るframeはF0、PSに残るframeはF1である。以下は`frame_recipe`からの抜粋。singletonは相手がIまたはPauliで、sharedは同frame2成分である。

| arm | 群 | P(g) | 条件付き費用（RZ／CX／depth） | 寄与A（RZ／CX／depth） |
|---|---|---|---|---|
| SP | F0_shared_full | 0.092683 | 54.686524／29.528259／68.036074 | 5.068526／2.736775／6.305806 |
| SP | F0_singleton_s1 | 0.219408 | 55.157871／28.378921／69.457993 | 12.102102／6.226575／15.239670 |
| SP | F0_singleton_s2 | 0.204104 | 56.776461／31.384909／74.030554 | 11.588278／6.405772／15.109899 |
| PS | F1_shared_full | 0.070590 | 86.899408／37.523528／104.921630 | 6.134255／2.648797／7.406449 |
| PS | F1_singleton_s1 | 0.205245 | 68.826819／32.066843／87.218050 | 14.126339／6.581549／17.901041 |
| PS | F1_singleton_s2 | 0.184951 | 84.011489／37.499467／104.013145 | 15.538043／6.935579／19.237377 |

PSのF1 singleton全体は確率39.0196%、RZ寄与**29.664382**／CX寄与13.517128／depth寄与37.138419。PSのRZ総費用のうち、同frame full共有群の6.134255より大きい。まずこのsingleton経路、とりわけF1 support2群を調べる価値がある。共有full群でもF1の条件付きRZ86.899408はF0の54.686524より高いが、これも変換単独の対照ではない。

PPでは同local Z基底のPauli pair群は確率10.0084%、条件付きRZ30.468250／CX22.479789、別local基底pair群は52.1158%、RZ38.704615／CX25.927019であった。共有の対象はあるが、PPの低費用を「大半が同基底共有」とは説明できない。

Pauli supportが増える場合も残る。PPでordered `I→P_s` 群の条件付きCXはsupport1/2/3/4で**18/20/22/24**、RZは**28/31.737600/35/35.597102**だった。supportのparity処理が反映され、Pauli化が無費用ではないことも確認できる。各群の確率・寄与と全ordered support組合せはCSVを参照する。

PPのorigin集合はcollect後のまま扱った。例えば両成分が`F0+F1`の群は確率46.1696%、RZ寄与15.797322。F0とF1へ半分ずつ配るような分解はできず、collect前後の因果効果も今回の記録からは得られない。

## 改善候補と、build／compileが必要になる条件

| 優先候補 | 今回得た根拠 | 次に必要な証拠 |
|---|---|---|
| **PSのF1 singletonのsupport限定合成** | 確率39.0196%、RZ寄与29.664382。support2群の条件付きRZ84.011489、寄与15.538043 | 現行completionと、同じ選択operator・順序・wrapperを保つ候補合成の対照。初めてここで新build／compileが必要 |
| **SSの異frame境界、混在armのframe/Pauli境界** | SSの異frame群RZ寄与26.406569。FP群が混在armの主要費用・SP/PS差の主要部分 | 境界を直接合成する候補に絞り、同じ軌道でoperator・符号・位相を確認する対照。イベントの並べ替えやlaw変更は別判断 |
| **費用とλを併せて扱う表現選択** | 安いPPと、小さいλ/Btotを持つSSのtradeoffは残る。単純なL最小化では1-shot費用を説明できない | 表現候補の科学的採択が先。認証Nを使う総費用選択には別途γ・統計資格が必要で、今回計算しない |

これらは技術的な改善候補であり、実装承認・科学的GO・独立新規性の認定ではない。**保存された実装のどの群が高費用かを判断するための新build／compileは不要だった。** Gaussian変換だけの削減量、共有の純効果、compiler passの寄与、候補回路で順位が変わるかを答える場合に限り、新しい対照が必要になる。

対象はこの合成fixture・K0/r2・固定compilerのper-shot費用まで。η／α／γ／interval、認証N、N加重総費用は未資格・未取得のまま。共通D、prep/readout、中心部、基底変換、compiler fusionが混ざるfull-wrapper費用を、部品の加法費用や同精度総資源優位へ読み替えない。

## 成果物・再現・保全・STOP

[GROUP_CONTRIBUTIONS.csv](../../artifacts/pr_algorithm_improvement_phase2d_f2_attribution/2026-10-24/GROUP_CONTRIBUTIONS.csv)は全490分類行についてP・条件付き費用・寄与とcase数を収録する。`probability_exact`と`contribution_*_exact`が正本で、条件付き費用もその比からexactに復元できる。decimal列は表示用。
[RESULTS.json](../../artifacts/pr_algorithm_improvement_phase2d_f2_attribution/2026-10-24/RESULTS.json)には全体・差分・対称分解のexact値、照合、hash、費用を保存した。
[ATOM_FEATURES.json](../../artifacts/pr_algorithm_improvement_phase2d_f2_attribution/2026-10-24/ATOM_FEATURES.json)は分類に必要な構造metadataだけを保存し、元の係数・行列・承認情報を含まない。B原本4fileのhashとの対応を保持する。

再集計wall **0.528368秒**、CPU0.528369秒、最大RSS28,108 KiB。32 CPUのaffinityを確認し、13,676行の小さい処理をstream・整数分子加算・X/Y間pair metadata cacheで実行した。恣意的なCPU/RAM/wall cap、共有環境変更はない。これは再集計だけの費用で、過去の回路生成・辞書生成費用ではない。

再利用関数の元SOURCEを含む107fileの一致を実行前に確認した。旧SOURCE・raw・grant・ledger・marker・STOPへは書き込まず、新しい出力へ保存した。新grantは消費していない。
再現入口は `scripts/run_pr_dictionary_f2_attribution_saved.py`。公開された構造metadataを用いる場合は次のとおり（新しい出力先を指定する）。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_pr_dictionary_f2_attribution_saved.py \
  --public-root artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1 \
  --features artifacts/pr_algorithm_improvement_phase2d_f2_attribution/2026-10-24/ATOM_FEATURES.json \
  --output /tmp/pr-f2-attribution-reproduction
```

**mandatory STOP。次はユーザー／GPT**が、F1 singleton合成または基底境界の限定対照へ進む価値を判断する。採択する場合は、候補と同値性条件を決めて新build／compileを別途明示承認する。Codexは実装・追加実験・γ研究・重要科学レビューへ自動移行しない。
