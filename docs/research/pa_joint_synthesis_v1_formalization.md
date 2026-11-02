# P-A v1 DP形式化・mechanism監査

最終更新：2026-09-25 JST

## 結論

P-A v1の有限候補問題、辞書式目的関数、DP漸化式、計算量、operator同値性条件を明文化した。
DPは、事前計算済みinterval候補を仮定すれば各source-basis run長$n$に対して$O(n^2)$時間で、
凍結済みの有限候補族におけるproxy目的の大域最適解を返す。

一方、完了済みblind evidenceをmechanism別に監査すると、48 holdout trajectoryと6 operator probeの
全てで、各source-basis runは1区間のままだった。run内部を複数区間へ分割したrecordは0件である。
また48 holdoutに含まれる256 eventは全てTaylor order 0で、6 probeもapplication数から非零orderを
含まないことが分かる。

従って、既存結果が直接支持するのは

> source-basis runごとにfull basisまたはsupport-union completionを選ぶと、現行
> `support_run_le_1`よりcompiled costを下げられる

という限定された機構である。interval分割層は、明示的な「1 source runにつき1区間」baselineから
追加利益を生んだことがまだない。blind検証の数値と全gate通過は有効だが、それだけから
「runを複数区間へ分けることが効いた」とは言えない。

本監査完了時点のstatusは

`pa_v1_formalized_but_interval_mechanism_not_empirically_distinguished`

であった。P-Aを条件付き候補へ戻し、P-Cを副候補として維持し、次はH12や長RPEではなく、
非退化な構造だけを使う小さいmechanism判別を事前登録する、とした。後続P-A検証は本書末尾の追記どおり
完了してinterval DPを停止した。その後のP-C tracking検証でもcurrent H4 familyが停止条件に達した。

## 1. 有限候補問題

非identity application列を、source basis ID/hashが等しい最大連続runへ分ける。run

$$
a_1,a_2,\ldots,a_n
$$

について、連続かつ非空な区間への分割

$$
[0=t_0,t_1),[t_1,t_2),\ldots,[t_{k-1},t_k=n)
$$

を選ぶ。各区間$I=[s,t)$では、次の2 modeだけを候補とする。

1. `full`：元のregistered source basisを使う。
2. `support_union`：区間内applicationのdiagonal supportの和集合
   $S_I=\bigcup_{j=s}^{t-1}S_j$に対応するsource-unitary列を保存し、残りを決定論的に直交補完する。

任意のGaussian completion、非連続group、pairwise relative-basis transition、routing/noise-aware候補は
v1の探索空間に含めない。

## 2. 目的関数とtie-break

区間$I$の選択basisのruntime basis-operation数を$g(I,m)$とする。v1が最小化する値は、実装順に

$$
\left(
2\sum_I g(I,m_I),
k,
\sum_I |S_I|,
\sum_I \mathbf 1[m_I=\mathrm{full}]
\right)
$$

の辞書式順序である。

- 第1項の2はbasisを開閉するforward/inverseの2回に対応する。
- 第2項は同じ第1項なら区間数を少なくする。
- 第3項はさらに同じならsupport unionの総サイズを小さくする。
- 第4項はさらに同じなら`support_union`を優先する。

従来の`selection_objective`文字列は最初の3項だけを記述していたが、凍結実装は第4項もtie-breakに
使用している。形式化では実装どおり4項を明示する。

この目的はtranspile後RZ数そのものではない。従って、DPの最適性をcompiled RZのglobal optimumと
読み替えない。

## 3. DP漸化式と最適性範囲

$D[t]$をprefix $[0,t)$の最良objectiveとする。$D[0]=(0,0,0,0)$として、

$$
D[t]
=
\min_{0\le s<t,\ m\in\{\mathrm{full},\mathrm{union}\}}
\left{
D[s]\oplus c([s,t),m)
\right}
$$

とする。$\oplus$は4成分の加算、$\min$は辞書式最小である。

最後の区間$[s,t)$を固定すると、それ以前のprefixは独立に最適でなければ全体も最適でない。この
optimal-substructureにより、漸化式は全連続分割と各区間の2 modeを漏れなく比較する。従ってv1は、

- 固定source-basis run
- full/support-unionの2候補
- 凍結basis-operation-count proxy
- 上記tie-break

という有限候補族の中では大域最適である。それより広い量子回路またはGaussian circuit全体に対する
最適性ではない。

run長$n$について、連続区間は$n(n+1)/2$個、各区間2 modeなので遷移数は厳密に

$$
n(n+1)
$$

である。interval optionを事前計算済みとすればDP時間は$O(n^2)$、predecessor保持は$O(n)$、
全option tableをmaterializeする場合は$O(n^2)$ memoryである。現在の実装ではsupport union生成と
unitary completionの費用が別途かかるため、$O(n^2)$はDP recurrence部分の計算量であり、basis生成を
含む全wall timeの漸近評価ではない。

artifactに保存された全54 recordについて、報告された`dp_transition_count`が
$\sum_r n_r(n_r+1)$と一致し、選択segmentから再構成した第1・第2目的も保存値と一致した。

## 4. operator同値性条件

source one-particle unitaryを$U$、support $S$のZ/ZZ applicationを$D_S$、選択completionを$V$とする。
v1は

$$
V_{:,p}=U_{:,p}\qquad(p\in S)
$$

を満たすように$V$を構成する。対象のconjugated diagonal operatorはsupport列だけで決まるため、
exact arithmeticでは

$$
B(V)^{-1}D_SB(V)=B(U)^{-1}D_SB(U)
$$

となる。区間でsupport unionを保存すれば、その区間内の全applicationについて同じ関係が成立する。
event順序を変えず、各application operatorが等しいので、sequence全体も等しい。

隣接する同一basisでは、先行区間のcloseと次区間のopenが逆演算として消える。controlled回路では
basis changeをcontrolせず中央diagonal actionだけをcontrolする。identity/event由来scalar phaseは
basis選択に依存せず、既存規則によりglobal phaseまたはancilla relative phaseへ同じように集約する。

実装では次を検査する。

- source ID/hash、application support、selected basis hashの一致
- selected basisのunitarity
- preserved-column residualが$10^{-12}$以下
- controlled phase規約の維持

さらにblind検証の6 dense operator probeは最大$3.126\times10^{-15}$で、$10^{-10}$基準とrelative
ancilla phase一致を通過した。これは保存条件の数値実装を確認するlocal evidenceであり、上の代数条件を
外れたcompletionを保証するものではない。

## 5. mechanism監査結果

| stratum | record | source run | selected segment | run内分割record | full segment | union segment | nonzero Taylor event |
|---|---:|---:|---:|---:|---:|---:|---:|
| H4 opt2 holdout | 24 | 56 | 56 | 0 | 1 | 55 | 0/128 |
| H4 opt2 probe | 3 | 5 | 5 | 0 | 0 | 5 | 0相当 |
| H5 physical holdout | 24 | 64 | 64 | 0 | 0 | 64 | 0/128 |
| H5 physical probe | 3 | 7 | 7 | 0 | 0 | 7 | 0相当 |

probeはevent order自体を保存していないが、application数がsequence lengthと全件一致するため、
product applicationを追加する非零Taylor orderは含まれない。

H4 opt2 holdoutのevent digestは元opt1 pilot holdoutと同一である。basis planはcompiler levelに依存しない
ため、元pilotのholdoutでも同じ「1 run = 1 segment」planだったと分かる。

事前gateの`multi_application interval observed`は、長さ2以上のsegmentが存在することを検査したが、
1 runを2区間以上へ分けたことは検査していなかった。従ってgate通過と今回の0 splitは矛盾しない。

## 6. 研究判断

blind transferで得たH5 -17.076%、H4 opt2 -6.598%というcompiled RZ改善は取り消さない。ただし、
同じplanは「各source runでfull/support-unionの安い方を1つだけ選ぶ」baselineでも生成できる。
現artifact上ではinterval partitioningの増分plan変更は0/54であり、その増分compiled利益も識別できない。

scoped prior-art auditではpartial basis rotation、completion、basis共有自体は既知と整理している。
従ってinterval分割が実証されないまま、run-level union completionだけをP-A固有の独立差分として
扱うことはできない。P-Aは主研究の確定テーマではなく、非退化mechanism検証待ちの条件付き候補とする。

次の1件は、小さいforced-structure比較を結果を見る前に固定する。

1. 明示的な`one_segment_per_source_run` baselineを追加する。
2. 同一source run内でsupportが変わり、複数区間がproxy上で有利になり得る列をtrainingとblindへ分ける。
3. 少なくとも一部にTaylor order 2のproduct applicationを含める。
4. interval DPがbaselineと異なるplanを選び、operator同値性を保ち、compiled RZで追加利益を持つか判定する。
5. 差が得られなければinterval claimを停止し、P-Cを主題へ戻す。

この判別にH12、長RPE、追加$q>32$、full wrapper、backend/noiseは不要である。

後続の[非退化mechanism validation](../research_direction_joint_synthesis_mechanism_validation.md)では、
上記5条件を事前登録し、forced-support Taylor-order-2のtraining 15、blind 15を実行した。
全30 taskでinterval DPと一区間baselineのplan・全compiled metricが一致し、run内分割は0件だった。
従って`stop_pa_interval_dp_as_primary_and_return_to_pc`とし、この時点ではP-Cへ戻った。
その後[P-C tracking・breakdown validation](../research_direction_geometry_tracking_breakdown.md)も完了し、
current H4 familyのP-Cを停止した。
本節の形式化と既存run-level改善は保持するが、interval subdivisionの独立寄与は主張しない。


## 7. 証拠

- artifact：
  `artifacts/research_direction_joint_synthesis_formalization/2026-09-25/pa_v1_formalization_and_mechanism_audit_v1.json`
- content fingerprint：
  `aaa5fdba8ddc6ec25fe1f286d886aba14a7a440c435f1ca5dd78ce33404b3676`
- file SHA-256：
  `276d14158fec50e037596b06b94c2b3ea19e8dc86a16993d6848e8a289771a78`
- implementation：`src/trotterlib/research_direction_joint_synthesis_formalization.py`
- runner：`scripts/run_research_direction_joint_synthesis_formalization.py`
- test：`tests/test_research_direction_joint_synthesis_formalization.py`
- evidence status：local dirty worktree、外部再現／immutable CIなし

## 8. 検証

- `py_compile`：implementationとrunnerが通過
- 専用test：4 passed
- 関連test：形式化、blind validation、pilot、theme selectionの計14 passed
- validation manifest：OK
- `git diff --check`：通過
