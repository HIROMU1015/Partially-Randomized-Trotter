# H4 Pauli-JW／DF-native：PF・PR回路費用逆転の構造解析

2026-10-24 JST。**最も有力なのは、prefixによって残る演算の種類と、tailを置き換えるeventの実装粒度の組合せである。** 保存Pauli prefix27はすべてZまたはZZで、PFに含まれるX/Yを持つstringの反復がPRでは必須回路から外れる。DFではPFの多数項をfragment内にまとめる利点がある一方、PRにもone-body・保持fragmentの基底変換と対角演算が残り、tail eventもGaussian basis付きのZ/ZZとして実装される。さらにPauliはbody全体をgeneric control、DFはbasisを制御せずcentral diagonalを制御する。

これはSOURCEと保存descriptorによる**機構の同定**である。どの成分が何RZ・何割を説明するかは未同定。旧4caseには成分別compiled価格・pre/post gate histogram・実pass traceが保存されていることを確認できず、全回路の価格を成分に配賦しない。特に「DFのbasisが主因」と単独で確定しない。

[成果物・引継ぎ](../artifacts/non_df_h4_cost_reversal_structure/2026-10-24/README.md)、
[静的照合](../artifacts/non_df_h4_cost_reversal_structure/2026-10-24/verification.json)、
[次の非実行契約](../artifacts/non_df_h4_cost_reversal_structure/2026-10-24/next_contracts_DRAFT.json)。
今回は新しい実H4の算術評価・科学データ再集計・sampling・回路生成・compile・N/G計算を一切実行していない。

## 1. 観測事実と固定条件

| 表現 | PF compiled RZ/shot | PR compiled RZ/shot（保存256本平均） | 保存削減率 |
|---|---:|---:|---:|
| Pauli-JW | 255,500 | 9,321.1015625 | 96.3518% |
| DF-native | 135,355 | 23,678.44140625 | 82.5064% |

数値は[固定結果報告 §1](non_df_h4_conditional_cng_results_20261024.md)と
[保存cost summary](../artifacts/non_df_h4_conditional_cng_results/2026-10-24/run01/cost_summary.json)からの引用で、今回再計算していない。
PFではDF、PRではPauliが安いという**この固定実装・有限cohortの順位逆転**を対象とする。
PR母平均、他のprefix、他のcompiler、総費用Gの普遍的順位は結論しない。

Linear H4、隣接1.00 Å、STO-3G、8 system qubit、共通保存state。
S2、T=.8、q8、r4、K4。
Pauli184非identity項・PR prefix27。DF tol-only1e-8、rank10・PR prefix1＋one-body。
Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なし。
費用対象はcosine/sine各full Hadamard wrapperで、状態準備を除く。
5精度ε=.05/.01/.005/.001/.0005は保持するが、今回の単価構造解析では精度行を再会計しない。

同じ未thresholdのMO積分からPauliとDFを生成し、同じ保存stateを用いたことは
`h4_molecular_inputs.build_h4_inputs`と
[入力receipt](../artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/receipt.json)で対応する。
DF生成は近似を許すtol-only方針だが、このrunの保存 `df_truncation_value=0.0`、
`DF_full_H_agreement.error=8.462237051574415e-15`、判定閾値1e-9が記録されている。
これは保存された数値的一致診断で、厳密同一のcertificateではない。
大きな費用逆転を「DF近似でHamiltonianが変わったため」と帰属する証拠はない。
また、近いfull HでもPauli項とDF fragmentではS2分解の単位が異なり、固定qの近似unitaryが同一になるわけではない。

### 来歴

| 役割 | 固定commit |
|---|---|
| 元H4科学SOURCE | `0c329b30a81accb35c212be8defa66fee0dd0f12` |
| C | `e80d7510880f8db993b777c627bb8e95b05a070d` |
| L | `eb195bf2dedd0aed9d80ad7f84948b59ea53d854` |
| C/L接続 | `0880f4af98b125d62599e970db89fc33634d1b61` |
| 条件付きCNG adapter | `b1984dcde06a874d0fd6e9fe6395b7f3fb7504d8` |
| 条件付き結果 | `5600f0e4b28c140a6a50b8aba47425860972a932` |
| 今回の基点・直前調査 | `1caaba2516dd3a30475e2acbd69672810a9fc05a` |

GitHubから基点commitと固定11fileの全文を取得し、Git blobとローカル固定refを照合した
（[取得記録](../artifacts/non_df_h4_cost_reversal_structure/2026-10-24/github_verification.json)）。
原H4の関連15 SOURCE／再利用assetを固定Git objectと照合し、Qiskit runtimeの7 sourceは既存C sealのlegacy-runtime SHAと一致した。
C/Lの科学結果は既存証拠として参照し、新しいDF診断を行っていない。
旧監視native UNKNOWN、C/L COMBINATION_BLOCKED、compiler MISSING_EVIDENCE、旧N/G nullは別の未完了問題として維持する。

## 2. 4方式の実際の回路経路

`h4_pilot_science` → `evaluate_h4_audited` → `audited_adapter` →
各trajectory → `hadamard_wrapper` → `TranspiledCircuitCostCache.get_or_transpile` →
`transpile_and_measure_cost` → post-transpile `rz` count。
監査adapterはproductionで返ったeventを記録し、監査用に再抽選しない。

| 構成 | 各stepで必ず実行する部分 | random部分 | 制御化 | 反復と費用保存 |
|---|---|---|---|---|
| Pauli PF | 全184項のforward halfとreverse half、constant phase | なし | system body全体の `to_gate().control(1)` | q step、PF1本の両軸full-wrapper RZ |
| Pauli PR | prefix27のforward/reverse half、constant phase | r event/step、tail Pauliのproductsとrotation | PFと同じgeneric whole-body control | q step、PR256本のpaired両軸RZ |
| DF PF | one-body＋全10fragmentのforward/reverse half、scalar phases | なし | 各basisは非制御、中央RZ/RZZとphaseを制御 | q stepのraw concatenation、PF1本 |
| DF PR | one-body＋fragment0のforward/reverse half、constant＋抽出identity phase | r event/step、tailのrotated Z/ZZ | 各basisは非制御、central products/rotationを制御 | q stepのraw concatenation、PR256本 |

原SOURCEの主な入口：
[Pauli adapter](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/pauli_signal_adapter.py)、
[DF adapter／wrapper](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/fixed_signal_evaluation.py)、
[監査評価](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/h4_audited_evaluation.py)。
関数・classの行位置とSHAはverification.jsonの `sources` に収録した。

両wrapperともancilla H → controlled body →（sineのみSdg）→ H → measurement。
共通の外側wrapperの**ゲート列**は同じだが、そのcompiled増分が常に一定とは限らない。
特にPauli wrapperでは「制御化の生成」も実行するため、その費用をH2個だけの固定wrapper項へ押し込まない。
保存cohortのcosine/sine価格が同じことは既存観測であり、Sdgのcompiled寄与が数学的にゼロであることを意味しない。

## 3. Prefixとevent law：同じ数では比較できないもの

### Prefixの選択基準

`h4_molecular_inputs.resolve_prefixes` は共通の
`non_df_h4_readiness.structural_prefix` を使う。
これは**降順weightの累積が総weightの50%以上になる最初のprefix**で、
正の非空tailを要求する。費用最小化や共通operator normによるmatchingではない。

| 表現 | 並び・weight | identity／one-body | 保存された選択 |
|---|---|---|---|
| Pauli | 降順 `abs(c)`、label tie-break | constantは別枠で常に保持、非identity係数だけをprefix選択に使用 | 27 Pauli項 |
| DF | 降順 `abs(lambda)*||g||_F²`、original index tie-break | one-bodyは常に保持しfragment rankingから除外。tail identityも抽出してdeterministic phaseへ | fragment0の1個＋one-body |

DFのranking weightは、tail抽出後のinvolution係数L1和 `exact_rte_lambda_r` とも異なる。
同じ「50%」であっても保持Hamiltonian・誤差・回路費用の等しい割合ではない。
27対1や27/184対1/10だけで公平性・圧縮率・原因を決めない。

### 今回のPauli prefixの具体的内容

入力receipt・pauli.json・PR result.modelの保存label列は一致し、
**すべてI/Zだけ、非identity supportはZまたはZZ**である。
例えば `IZIIIIII`、`ZIIIIIII`、`ZZIIIIII`、`IIIZIIIZ`。
PFの全列には `XXYYIIII` や `IXZXXZXI` などX/Yを含むstringも保存されている。

このdescriptor確認は係数の再集計・Hamiltonian再生成ではない。
Z/ZZ保持部にはPauliのX/YをZへ移す基底変更が必要なく、互いに可換である。
ただしbuilderは項ごとのPauliEvolutionGateをforward/reverseに逐次appendする。
可換性だけを理由に、旧compilerが保持部全体をまとめた・相殺したとはしない。
「保持27が安い」という説明を、単なる保持項数から一段具体化できた点が今回の主な成果である。

### Eventの分布と演算粒度

共通のpaired Taylor lawで偶数order k∈{0,2,4}を選ぶ。
tau=lambda_R*(T/q)/r、
weightは tau^k/k! * sqrt(1+(tau/(k+1))²) を正規化したもの。
componentはabsolute involution係数に比例して選び、
回路上は**k個のsigned product → 1個のrotation**、phase=(-1)^(k/2)、
rotation角はatan(tau/(k+1))である
（`rte._paired_order_weight`、`_make_event`）。

Pauli componentはtail Pauli string。DF componentはfragmentをZ/ZZへ展開したrotated involutionであり、
「DF eventがfragment全体の二乗diagonalを1回実行する」という説明は誤り。
DFのdeterministic fragment evolutionとrandom involutionは別の粒度である。
同じq/r/Kでもlambda_R・component law・角度が異なるため、同じrandom回路分布ではない。

既存C modelの保存order0/2/4確率：
Pauli [.9960625813453219, .003934819613987418, 2.5990406907583435e-6]、
DF [.9974498244090985, .002549087577779853, 1.0880131215414273e-6]。
既存結果報告 §5の保存event数はPauli8157/35/0、DF8167/25/0。
**DFの高費用を「event数や高次order頻度が大きいから」と説明する証拠はない。**
これは保存値の引用で、新しい確率計算・trajectory集計ではない。
未観測order4の費用や母平均tailを確定したものでもない。

## 4. Literal構造とcompiled費用の分解

以下の式は時間順の回路構造を表す。D_forward → random → D_reverseをq回繰り返す。
Lは保持Pauli項数、b_fはbasisのliteral命令数、d_fはcentral diagonal／phaseのliteral命令数。
**これらの式に実H4数値を代入して新しい価格を計算していない。**

### Pauli：長いstringの指数回路を毎回残す必要がなくなる

`append_pauli_evolution` は1項の `PauliEvolutionGate(cP,time)`。
一般的な単一Pauliの指数回路は、local basis変更Bとparity network Lで
B† L† RZ(2ct) L Bの形に書ける。
Z/ZZ保持部はX/Y用Bを必要としない。
PFは全項、PRはprefixだけをそれぞれhalf-timeでforward/reverseに実行する。

PR eventのproductは **Pauli stringそのもののlocal X/Y/Z列**で、
各productをもう1本のPauli指数回路として実装しない。
rotationだけがPauliEvolutionGateである。
負符号とorder phaseはglobal phaseに集め、generic control後はancilla相対phaseとなる。
この点も「k+1回の同じexpensive evolution」と数える説明を退ける。

SOURCE literalレベルではdeterministic PauliEvolutionGate呼出しが2qL、
event rotationがqr、productのlocal命令がΣ_event Σ_product |support(P)|という構造になる。
logicalなPauliEvolutionGate1個とDF primitive1個の重さは異なり、
この呼出し数をそのままRZ数や表現間の単価差に換算しない。

PFではX/Y・parityを持つtail指数回路も毎step必須である。
PRではこれらが有限eventによる抽出へ移り、常時実行する部分はZ/ZZに偏る。
**保持数減少＋保持演算の種類の変化＋product実装の軽さ**が、Pauliの大きな削減を説明する具体的機構である。
各機構のcompiled寄与を単独測定した証拠はない。

### DF：fragmentへの集約とeventでのbasis露出は両立する

`df_partial_s2._append_block` は各blockについて
basis inverse → central diagonal → basis forward。
One-bodyにはRZとphase、二乗fragmentにはRZ/RZZとphaseを使う。
各fragmentの多数の係数を**同じbasisの中でまとめて**実行できることがPFの圧縮機構になる。
全PFではone-body＋全fragment、PRではone-body＋fragment0が残る。
保存C modelではこのPRの2blockはPFの先頭2blockとdescriptor上同一で、basisにRZ／XXPlusYYが含まれる。
具体例として保持fragment0のbasisには、qubits=[4,3]、parameters=["1.340418848679431","1.5707963267948966"]のXXPlusYY descriptorが保存されている。
これは実際の保存基底列がlocal Zだけの列ではないという証拠で、当該gateのcompiled価格を計算したものではない。

random componentもbasis inverse → signed Z/ZZ productまたはrotation → basis forward。
Productのcentral制御はsupportごとのCZ、rotationはCRZまたはcontrolled RZZ。
Gaussian basisはPauliのlocal H/S変更とは異なる演算列で、
full registered basisを使う元経路ではcentral supportが1〜2qubitでもbasis全体の列が残り得る。
ただし保存gateのゼロ角・隣接inverse・最適化もあるので、すべてがcompiled費用として残るとは限らない。

`df_rte_qiskit._build` は同じbasis ID/hashが隣接した場合に再利用する。
r eventをまとめた**1stepのoccurrence内部**でのreuseであり、各eventで必ず2回basisを払うわけではない。
異なるbasisへ移ると旧basisを閉じ新basisを開き、identityではactive basisをflushし、末尾でも閉じる。
deterministic halfとの接続やq step間をまとめるreuseはこのbuilderの対象外。
原repeated policyは `raw_concatenation`。
存在する `boundary_optimized` やsupport-restricted basisの別設計を旧価格の生成経路として扱わない。
raw経路でも最終compilerによる相殺の可能性はある。

既存counterはbasis両側を必ず出す保守的なliteral上界を与える。
P_max=max_j(2b_j+support_j)、R_max=max_j(2b_j+1)として
event occurrence上界は r*(K*P_max+R_max)+phase、
deterministic halfをDとするとstep上界は2D+phase+occurrenceである。
これはSOURCEの `estimate_df_rte_structural_size_upper_bound` と
`estimate_df_partial_s2_structural_size_upper_bound` の記号的内容である。
basis reuseによる実際の削減、全wrapper compiled RZ、母平均上界には変換していない。

### Controlled化：表現以外の明確な実装差

Pauliはbodyをgateへ変換してからgeneric controlする。
封印runtime `qiskit/circuit/add_control.py:control` はbodyを
p/u/x/z/rx/ry/rz/cxへ展開し、各命令を制御する。
cx分岐では既存controlにancillaを足したmcxとなるため、
parity CXもそのまま非制御で残す専用実装とは異なる。
local basis変更もgenericなcontrolled命令になり得る。

DFは外側basisに制御を付けずcentralだけを制御するため、
Gaussian basis全体をgeneric controlする経路を避ける。
ただし**DFのcontrolled RZZ自体はRZZGate.control(1)**であり、
その内部decompositionにgeneric controlが入り得る。
「DFのparityネットワークはすべて非制御」「DF native controlは常に最小費用」とは言えない。

この差はPauli PFで多く残る非対角stringの費用を増やし得る。
Pauli PRではそれらが常時実行部分から外れるため、同じgeneric controlでも影響の受け方が変わる。
SOURCEからこの非対称は確認できるが、旧compiled CCX数やRZ差分は保存されていない。
最終optimizerがどれだけ取り戻したかも未確認である。

### Compilerとwrapper：加算配賦ができない部分

`rte_compiled_cost.transpile_and_measure_cost` はfull wrapperをtranspileし、
post gate histogramの `rz` を価格にする。
低層返却objectにはpre/post histogram等があるが、
`h4_audited_evaluation` が永続化するrowはRZ・seed・event/trajectory hash等に限られ、
そのhistogramやtranspiled circuitはcase結果へ保存する経路になっていない。

既知4caseのresult/validation/actual_call_accountingと全price_joinをschema検査し、
pre/post histogram、実basis switch/reuse counter、pass traceの対象fieldは未確認。
同4case内のQPY/QASMも未確認。
これは限定された保存範囲の調査結果であり、全filesystemに存在しないとの主張ではない。
対象のpattern・file数・file名列のSHAはverification.jsonの `bounded_inventory` とcheckerに記録した。

固定level1 SOURCEはOptimize1qGatesDecomposition・InverseCancellationと、
basis外を再translationする固定点loopを含む。
最適化はRZ単独の最小化保証ではなく、一般のcommuting block全体を最適合成する保証もない。
同じ設定・seedでも並びが異なればfusion・inverse cancellationの結果は異なり得る。
preset SOURCEから可能な機構は分かるが、旧programの実pass trace・各passでの価格差は分からない。

したがって
C_full ≠ C_deterministic + C_random + C_basis + C_control + C_wrapper
が一般的である。基底変更は既にdeterministic/randomの内部にあるので、
独立加算すると二重計上にもなる。
今回は各成分のcompiled価格をnullとし、全回路価格だけを観測済みとして扱う。

## 5. 仮説A〜Eの絞り込み

| 仮説 | SOURCEで成立する機構・保存観測 | 制約・反証となる証拠 | 交絡／現在の説明力 | 最小追加比較 |
|---|---|---|---|---|
| A 保持部 | Pauli27はZ/ZZだけ。DFはone-body＋fragment0のbasis/diagonalが必須。PF→PRの保持単位が異なる | 27対1は費用比ではなく、保持部単独のcompiled価格がない | **具体的支持あり、主要候補**。B/C/D/Eと不可分 | D1でPR保持部を隔離しfullと対応 |
| B event単価 | Pauli productはlocal string、rotationだけexp。DFはrotated Z/ZZとbasis reuseを伴う | 両者qr event、DFの高次確率・保存order2数はむしろ小さい。eventごとの価格・basis切替実数なし | **具体的支持あり、主要候補**。単価の大小やgapの大半までは未証明 | D1のevent-onlyと構造tag |
| C basis/control | Pauli generic whole-body controlとDF central-only controlが異なる。DF basisはGaussian列 | DFのRZZ内部controlにもgeneric経路がある。basis reuseが既に働く。保存histogram不足 | **有力な実装要因**。DF basis単独を最大原因にしない | D2で同じPauli unitaryの制御位置だけ変更。必要ならD3 |
| D compiler相互作用 | full wrapperを一括compile、raw step境界、局所fusion/cancellationの可能性 | 実pass trace・非最適化対照がない。literal相殺可能性はcompiled実績を保証しない | **存在は確実、差の方向・量は未確定** | D1でfullとisolatedの残差・traceを保存 |
| E prefix/law交絡 | 50%基準のweightが異なりone-bodyの扱いも異なる。component法則・tauも異なる | 同じq/r/K、同じ物理入力、full H一致診断はある。それでも分割・実装をmatchしていない | **識別不能の交絡として確定**。表現単独の因果主張を制限 | D1/D2後、必要なら新研究系列で物理的な保持基準を事前固定 |

「最も有力」は寄与率のランキングではない。
AとB/Cをまとめた**保持内容とevent実装の非対称**が観測に整合する説明であり、
A・B・C・Dのどれが最大かを定量的に並べる証拠はない。
DF-PRが高いことを、未観測高次event・DF近似・外側wrapperだけに帰属する説明は採らない。

## 6. Q1〜Q6への回答

- **Q1 Pauli PRはなぜ安いか。** 全項指数回路を各step反復するPFから、Z/ZZの保持部と有限eventへ変わる。Productは指数回路でなくPauli string列。非対角basis／parity／generic controlを毎step全項分払う構造が外れる。保存96.3518%の各成分への配賦は未確定。
- **Q2 DF PFは安いのにPR削減が相対的に小さい理由。** PFはfragment内でbasisと多数diagonal演算をまとめる利点を持つ。PRではその集約を単純に1/10へ縮小するのではなく、one-body＋保持fragmentを残し、tailはbasis付きinvolutionとして実行する。Pauliの「Z/ZZだけを保持しlocal productsへ置換」と異なる費用構造になる。保存82.5064%がこれだけで必然になるとは示していない。
- **Q3 最大要因はどこか。** A/B/Cの連動が第一候補。保持部floor、random basis、generic control、compilerの最大寄与は現在識別不能。新しい成分別対照が必要。
- **Q4 SOURCEで説明できる範囲。** prefix選択、保持label、event粒度・順序、basis reuse範囲、control位置、計測scopeが特定できる。実compiledの各成分内訳・最適化収支・母平均への外挿はできない。
- **Q5 表現単独の効果か。** 切り分けられない。同じ元積分の比較ではあるが、分割単位、weight、law、control実装、S2近似、compiler入力が同時に変わる。単純prefix数matchingでも解消しない。
- **Q6 次の最小実験。** 第一にD1の固定代表4caseに対する部品隔離＋full測定。次にD2の同一Pauli unitaryに対する制御位置対照。DF basis reuseの寄与が残ればD3。全cohort再compileやprefix探索は初手にしない。

## 7. 優先する非実行契約案

[機械可読DRAFT](../artifacts/non_df_h4_cost_reversal_structure/2026-10-24/next_contracts_DRAFT.json)。
以下は**未実行・未承認**。実装SOURCE/runtime/grantは未固定、実行flag false。
元grantは再利用せず、旧結果を上書きしない。
科学的call数の範囲とCPU/RAM等の資源方針を区別する。

### D1：保持部・event部・全体のpaired cost診断（優先1）

PF各1本、PRは各保存trajectory_index=0だけを事前固定する。
最小・最大価格による代表選択を避けるが、4caseは母平均の代表性を保証しない。
保存descriptorから追加抽選なしで再構成するadapterの技術検証を先行させ、
seedからのRNG replay・SCF/FCI/DF/state再生成を行わない。
現状そのadapterのsealは完成しておらず、元builderが数値的な再準備を要求する場合は自動拡張せずSTOPする。

cosineに限定して、full4case、PR2caseそれぞれ保持-onlyとevent-only、
empty wrapper共通1の**最大9 compile**。
phaseをどこへ置いたかを固定し、scalar phaseとancilla相対phaseを捨てない。
隔離回路は意味の変わる診断ablationなので、本来のPF/PR性能や精度の比較には使わない。
各full circuitの入力fingerprint・RZを旧同一case/trajectoryの保存値と照合し、
不一致なら再現失敗としてSTOPし、旧価格の因果説明へ転用しない。

bodyのcontrol前／後、transpile前／後のgate histogram、
category tag、basis reuse/switch、RZ/CX/depth、QPY/QASM、pass traceを保存する。
part価格はfull価格に加算配賦せず、
fullとisolatedを比較した「相互作用残差」として扱う。
partによって余分なwrapperが重複するのでempty対照の定義が必要。
N/G・精度判定・population costを計算しない。
Aが大きいか、B/Cが大きいか、Dの非加法性が無視できないかを最少の4構成で切り分ける。

### D2：Pauli control位置だけの同一unitary対照（優先2）

D1のPauli PF/PR各1本をbaselineとして再利用する。
各Pauli exponentialのlocal basis／parity networkは非制御にし、
central rotationだけを制御する、意味を保つ専用対照を作る。
productsの符号・constant・order phaseをancilla相対phaseとして保つ。
人工小系の代数・位相テストを先行し、実H4のstate/evolutionは計算しない。
追加cosine full-wrapper compileは最大2。
prefix・event列・q/r/K・compilerは固定する。
差はcontrol構成とそれに対するcompiler応答の効果であり、表現単独の効果ではない。
改善した場合も新しい実装系列であり、旧結果の数値を置換しない。

### D3：DF-PRのbuilder basis reuse対照（優先3、条件付き）

D1のDF-PR index0を再利用し、同じevent列・basis・角度・制御・compilerのまま
隣接equal basis reuseだけをoffにした対照を最大1 compile。
D1で実際に除去可能なequal-basis対がない場合は、この比較を実行しない。
元経路が既にどれだけreuseを行うかを測る検査であり、
「DF basisが全価格差を生んだ」という因果証明ではない。
q境界最適化やsupport-restricted basisまで同時に変更しない。

D1だけでも9回という有限診断単位で成立し、D2/D3は個別採否できる。
全256本の再compileや新samplingを既定にしない。
prefix/law感度を研究する場合は、同じ物理的保持基準・精度・費用目的を明示した別系列を事前設計する。
単純なprefix数の一致は対照条件にしない。
母平均tail保証やrho/certificate完成の仕事とは目的・承認を分離する。

## 8. 再利用・検証・保全・終了

既存 `native_df_gate_model`、その人工tests、DF structural counter、
`df_partial_s2_cost`／`df_partial_s2_repeated_cost`、
`rte_boundary_cost_validation`／`rte_cost_angle_invariance_validation`、
`saved_rte_cost_diagnostics` を適用範囲から検討した。
DF additive部品builderはD1の設計参考とし、今回は呼び出していない。
旧cost pipelineは実回路生成・compileまたは他条件の解析を含むため、今回の実H4入力には実行していない。

実行した再利用は、既存保全611-file SHA mapと、
純標準libraryの人工gate tests3件（basis相殺、位相を含むdeclared lowering、wrapper）である。
追加のprefix人工1件と合わせて4件PASS。
別assetのdeclared lowering profileはQiskitの最適化結果ではなく、
その人工RZ係数を今回の実H4費用へ代入していない。
新checkerは、Git object／SHA／descriptor／schemaの今回固有の結合だけを実装した。
主agent自身の検査であり、独立した科学レビューではない。

確認した保全範囲は旧611file、関連SOURCE15file、runtime7file、取得11file。
旧raw・SOURCE・grant・ledger・marker・STOPを変更せず、追加成果物と索引だけを更新した。
共有設定・他ユーザー・別研究の資産・processへの操作はない。
`research_project_instructions_unified_2026-10-23.md` §8.4と§10の再利用・資源方針を適用した。
実在する同版には§10.3がないため、添付が明示する研究優先・最小限の成果物という方針をそのまま適用した。
計算資源への独自の低い上限は設けず、今回の小規模な静的処理を実施した。

専用branchは `codex/non-df-h4-cost-reversal-structure-20261024`。
commit／push／remote SHAの記録は[公開状態](../artifacts/non_df_h4_cost_reversal_structure/2026-10-24/publication_status.json)。
[今回のSTOP](../artifacts/non_df_h4_cost_reversal_structure/2026-10-24/STOP.json)はこの静的調査の終了だけで、
元runのnative PASSや旧監視終了の証明を代替しない。

**次の担当：ユーザー**がD1、必要ならD2/D3の診断範囲を選び、新しい回路生成・compileを明示承認する。
**Codex**はその承認後に限り、保存descriptor replay adapterと人工位相検査を実装し、
新SOURCE/runtime・限定call契約・新grantを固定して実行する。
今回は静的構造解析・非実行案・保存・commit/pushまでで **mandatory STOP**。
新数値解析・科学計算・回路compileへ自動移行しない。
