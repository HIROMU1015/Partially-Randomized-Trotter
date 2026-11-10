# H4 DF10長時間compiled価格 V2：適合性調査と最小評価案

2026-11-10 JST。**技術調査・設計のみ。新予測・残差再解析・fit・sampling・実回路build/compile・N/G評価・詳細科学レビューは実施していない。** V1のモデル・結果・不一致判定を保全する。本書の推薦は未評価の候補であり、価格providerの採用やMain Claimの変更ではない。

**最初に検証する方法は、V1の保存構造価格に、DF10のfull-wrapper価格から校正する少数の補正を加える方式とする。** 既存fitと保存特徴を使えば追加compileなしで有効性を調べられる。過去のevent境界モデルは構造・検証方法を再利用し、係数は移植しない。精度達成と長q適用の確証はまだない。

## 1. 固定対象と既存の証拠

- Linear H4/1.00 Å/STO-3G、保存tol-only DF10、sector(2,2)。targetは保存DF-Hの基底状態、保存数値stateによる条件付き評価、状態準備P=0。
- εE=0.00159360010199040 Ha、αtotal=.05/方式、βtotal=.4 rad、h=.1 a.u.、m=0…12。価格はQiskit1.3.0、rz/sx/x/cx、opt1/seed17、測定付き元full Hadamard wrapperのcompiled RZ数。
- 初期評価対象はS2、δ=.1固定のPF B0/p7とPR p1/R_outer4/K2。PF B0は切捨てPF。別prefix・K・δ・signed高次pieceへ自動展開しない。
- 正本は[V1結果][V1]と[前設計][DESIGN]、[元価格][H46]。完全SHA・candidate・API・旧数値の出典は[SOURCE_INDEX](SOURCE_INDEX.md)。

以下はV1保存値の引用で、新しい集計ではない。残差の符号はcompiled−model。

| V1対象 | 保存compiled RZ/軸 | 保存M1/軸 | 保存残差/軸 |
|---|---:|---:|---:|
| PF q8/T.8、開発 | 44411 | 39557 | +4854 |
| PF q16/T1.6、V1留保 | 88515 | 78869 | +9646 |
| PR q8、探索9本の平均 | 16857.11111111111 | 15442 | +1415.111111111111 |
| PR q8、確認33本の平均 | 16947.78787878788 | 15514.545454545454 | +1433.2424242424242 |

確認PRのpath残差は+736…+1960。全留保68軸行で整数一致0というV1判定は維持する。X/Yは各pathの対応する応答で、独立標本数を倍にしない。非零残差だけから構造モデル一般の失敗や近似価格の不適格は結論しない。

## 2. Phase A：過去の境界・qと現在の対応

| 既存方法 | 元の価格単位・境界 | DF10への対応と再利用判断 |
|---|---|---|
| `rte_boundary_cost_validation` | 同一短時間lawから独立に選ぶRTE eventをL個連結した非制御回路。C1/C2/C3からpair/triple補正 | PRの各tail内のevent連結に対応する。ただし現在はcontrolled head/tailを含むwhole wrapper。旧価格係数を全体へ加算しない |
| `rte_boundary_pair_validation` | 同basis/異basisのevent対を条件付けて局所非加法を測定 | 同basis共有の有無だけでなく異basis境界も調べる発想を再利用。同じfragment名がK2の複数applicationすべてのinterface一致を意味するわけではない |
| `rte_connected_cluster_cost_validation` | 同一sample内の1/2/3-event局所窓を差し引くconnected補正、order/basis別の条件付け・確率重み | 小差の分散低減、希少orderを偶然の出現に頼らず扱う設計を再利用。DF10用局所compiledラベルがないので、今回の最初のfitには採用しない |
| `hierarchical_cost_validation` controlled-q | 固定δのcontrolled partial-S2 **時間発展subcircuit**をq回。q1/q2を校正しq4等を直接照合 | 対応するのはqPF。元はHadamard/readout/preparationを含まない。現在のfull-wrapperごとに校正・検証するならaffineの方法を使える |

eventモデルは、同じlaw・compiler・境界文脈に対して、μ1=C1、μ2=C2−2C1、μ3=C3−2C2+C1と置く。

\[
\widehat C_L^{(2)}=L\mu_1+(L-1)\mu_2,\qquad
\widehat C_L^{(3)}=L\mu_1+(L-1)\mu_2+(L-2)\mu_3\quad(L\ge2).
\]

L=1はC1を使う。三次式は(3−L)C2+(L−2)C3と同値。窓ごとのκ2=G12−G1−G2、κ3=G123−G12−G23+G2を同じ列で評価する旧connected方式は、独立推定値の差より分散を抑えるための方法であり、三次より遠いcompiler効果が消える保証ではない。

一方、反復モデルは固定δで\(\widehat D_q=(2-q)D_1+(q-1)D_2\)。event数Lの式と置換しない。現在はtail occurrenceごとにheadが入り、tail内境界はqPF(R_outer−1)、PF step間はqPF−1として別の文脈を持つ。全qPF R_outer eventを、headのない一本のIID列として旧式へ投入しない。

旧実例はH4/1Å/STO-3Gでも**DF12/LD3**、δ=.1/r4の短時間.025、K0またはK2、原則非制御tailである。controlled-q例もK0・旧boundary_optimized構成である。現DF10/p1/K2・symmetric_directional・full wrapperとは異なる。Qiskit1.3/opt1等の一致だけでは同じ価格族にならない。旧DF12には異なるpreparation hashの218/324成分資料もあり、混合しない。

旧保存結果にはpairよりtripleが改善した例、controlled-qの良好な有限holdoutがある一方、点推定が小誤差でも95%診断が広い例がある（[出典](SOURCE_INDEX.md#旧研究の保存結果)）。方法の有望性の証拠として扱い、その改善率・係数・旧5%等の閾値をDF10へ移さない。旧検証を再実行しない。

## 3. Phase B：V1で未説明のもの

| 対象 | 静的調査で分かったこと | V2への含意 |
|---|---|---|
| native分解・basis translation | M1はRY=SX RZ(−t)SXdg等の解析的展開。元translatorの選択出力との同一資格はない | 数学的等価性とRZ数一致を区別。正残差をすべて「未実装の削減」と呼べない |
| 1Q run再合成 | 公式1.3.0の`Optimize1qGatesDecomposition`はrunを再合成し、target誤差またはgate列長等で置換を判断する。RZ数だけを最小化する規則ではない | M1のRZ隣接加算だけでは不足し得る。どのpassが実H4残差を何個生んだかは未特定 |
| DF basis・step | M1は直に隣接する同basis逆変換対を消す。元builderは各signed pieceのq1 headとtailを順に合成する | Track Dのhead融合定数・q−1係数を持ち込まない。stepをまたぐcompiler最適化の実効値は未校正 |
| tail event・head/tail | 元tailは同basisを共有しidentityで閉じる。M1も保存application順を保持。探索basis group21…27、確認18…30を既に保存 | 保存`tail_basis_groups`を少数補正の候補にできる。ただし局所event価格や因果効果は保存whole価格から直接分離できない |
| CXと1Q列 | CXの消去・可換移動でrunの切れ目が変わり得る。M1は一般CX最適化・1Q再合成をしない | wire局所のRZ加算をcompiler固定点と同一視しない。現段階で全passを模倣する費用は正当化できない |
| scalar/identity/wrapper | 元SOURCEは全Tのscalar位相を一回処理し、PRはidentity抽出を含む。M1はcontrol位相とX/Y readoutまで会計 | wrapperの定数価格を別compileして足すと境界非加法を落とす。補正は測定付き全回路価格を応答にする |

Qiskitの記述は[固定1.3.0公式SOURCE][QISKIT]による。実回路の再synthesis・pass比較は行っていない。V1ではorder2のないpathにも正残差があり、希少orderだけを主因にはできない。現時点の説明は可能な機構の整理であり、主要機構の寄与率の測定ではない。

## 4. 最有力モデルと候補比較

推奨は**B：構造＋少数補正**。M1の保存値を\(C^{(1)}_{s,i,a}\)、保存PR basis-group数を\(B_i\)、sを方式、aを軸とする。まず次の固定候補だけを比較する。

\[
\begin{aligned}
M0 &: \widehat C=C^{(1)} \quad\text{（凍結baseline）},\\
Mq_s &: \widehat C_{s,i,a}=C^{(1)}_{s,i,a}+b_{s,a}q_{\rm PF},\\
MqB_{\rm PR} &: \widehat C_{{\rm PR},i,a}=C^{(1)}_{{\rm PR},i,a}
                     +b_{{\rm PR},a}q_{\rm PF}+c_a B_i .
\end{aligned}
\]

- PFはMqの1係数/軸、PRはMqの1係数/軸またはMqBの2係数/軸。PF/PRの係数を共有しない。価格差だけでなく両方式の絶対価格を出す。
- MqBのBは`prediction.features.tail_basis_groups`、baselineは`prediction.structural_rz`。再抽出・対角化・新構造予測なしで入力が揃う。MqBは「basis変換が残差原因」と断定するモデルではなく、保存path間の構造差を使う少数補正候補。
- PR開発はq8のみなので、q列は定数列と同じ。bはbulk傾きと端点offsetを識別できない。Mqのq比例延長も**仮説**であり、このfitで長q係数を同定したとはしない。
- q、q−1、総event数qR、定数、共有basis対、Bを一括投入しない。共線性とn9に対する過剰fitを避ける。order2は開発で一例のため専用係数をfitしない。

| 候補 | 利点と必要作業 | 今回の判断 |
|---|---|---|
| A：native/1Q構造を改善 | SOURCE由来の説明を増やせるが、局所再合成と全体境界の一致を別々に確認する必要 | Bの誤差構造から対象を絞れる場合に後続候補。compiler全再実装は目的にしない |
| B：M1＋少数補正 | 保存値だけで試せる。既存構造のpath差を残し、full-wrapperの総残差を直接校正 | 最初の最小評価案。補正後の精度・適用域は未評価 |
| C：絶対価格のaffine | 同一δ・同一回路族の複数qがあれば最小実装 | PF q8/q16を両方校正に使うと独立q検証が残らない。PR q8だけでは傾きを識別できず、現保存範囲で共通解にはならない |
| 旧pair/tripleのDF10再校正 | tail内の非加法とorder別寄与を測れる | 局所ラベルの新sampling/build/compile・control文脈の整合が必要。whole価格しかない初手ではBより重い |

任意の実qの二点u,vを使うaffineは\(\widehat C_q=\frac{v-q}{v-u}C_u+\frac{q-u}{v-u}C_v\)。旧q1/q2専用APIへq8/q16を番号だけ変えて渡さない。generic `track_d_c7_4_models.fit`のrank検査・最小二乗を再利用し、D固有のfeature抽出・36応答labels・集計器は流用しない。新しい管理基盤や校正frameworkは不要。

## 5. Phase C：近似価格の検証方法と保存データの限界

| データ | 許す役割（次の実行案） | 得られない資格 |
|---|---|---|
| PF q8、PR探索9 | 補正fitとtraining診断。cohort別、X/Yの対応を保持 | training誤差の小ささは予測性能ではない。PF1点のfit一致は自明 |
| PF q16、PR確認33 | 既知データで固定候補を再確認 | **V1で使用済み**。V2の未使用独立holdoutではない。ここで候補を選べば選択用証拠となる |
| その他の保存価格 | 科学条件・候補metadataの補助 | PR q32/T1.6はδ=.05で、q16/T1.6/δ=.1の代替ではない。固定Tのq sweepを固定δの反復外挿へ混ぜない |

評価するのはe=C−Ĉ、|e|、e/C（C=0はnull）、方式・軸・cohort別のbias/RMSE/最大絶対誤差、相対誤差の分布、PRのpath間残差SD。PFではq8→q16の変化、PRでは保存Bやorderの層別記述を行う。X/Yを一つのtrajectory行の二応答としてfitし、同一pathの二軸を独立n扱いしない。PFとPRに人工的なpaired標本も作らない。

PFは決定的回路であり標本SEを付けない。PRではpath間変動、有限cohortの平均不確かさ、fit係数の不確かさ、残る系統的discrepancyを区別する。9/33本の最大値はlaw全体の一様誤差上界ではない。旧predictorの校正SEだけを総予測CIへ読み替えない。今回案の小標本評価だけでは校正不確かさ込みのCI・長q CIを資格付けない。

整数一致は診断として残し、採用必須条件にしない。新しい一律1/2/5%閾値は置かない。採用精度は、将来のN重みとPF/PR総費用の比較差に対し価格誤差が判断を変えるかで決める。保存短qで小残差でもq4096や全lawへ保証を延長しない。

## 6. 承認対象にできる最小科学実行案 S1（今回未実行）

**範囲：固定V1 carrier `eec5bdf9cc391fe49b55b5673f3e385deeb20d42`の保存数値だけを使う、3つの小さいfitと既知データ照合。** 今回の指示はS1の実行許可を含まない。

1. 実装SOURCEを実行前にcommitし、本書の候補・データ役割・指標を固定する。入力はV1のDEVELOPMENT/HOLDOUTの`compiled_rz`、`prediction.structural_rz`、q、PRの`tail_basis_groups`、層別記述用の保存`event_order_*`、既存identity欄。cohort/candidate/path/axisの対応を保持し、原trajectoryやproviderを読み直さない。
2. 開発だけで、PF Mq、PR Mq、PR MqBの3つを残差への最小二乗でfitする。各fitはX/Yの二応答。既存generic fitのequal-q重みを用い、この開発範囲では同じq内を等重みとする。追加intercept・特徴探索・regularizationは行わない。
3. 固定M0/Mq/MqBを既知PF q16・PR確認33へ一括適用し、§5の指標とpath/axis別予測・残差、各係数、rankを保存する。PF q8とPR探索9のtraining診断を別に示し、cohortをpoolしない。実数予測を丸めて誤差を隠さない。
4. rank不足・binding不一致・非有限値は該当モデルの評価不能として残す。負価格は失格診断を記録し、0へのclampやデータ依存の候補変更をしない。有効候補の評価は同じ固定仕様で完了させる。保存元・V1ファイルを改変しない。
5. 係数と全既知結果を公開し、追加補正の有用性・残る不確かさを報告してSTOP。比較後のモデル推薦は探索的と表示し、全RPE providerへ自動採用しない。

新構造計数・局所再synthesis・bootstrap・law期待値計算・追加sampling・実回路build/compile・長q価格・signal/N/G・候補最適化・詳細科学レビューはS1に含めない。必要な小さいadapterと人工rank/axis/binding検査はS1承認後に既存fitへ接続する。今回は文書のみなので、重複する予測器・テストを新設していない。

**S1で得る情報：** 保存構造価格への少数補正が、既知の別q PFと別cohort PRの絶対価格誤差を改善するか、basis特徴の追加が必要か。**残る未資格：** PRのq方向、独立汎化、希少order/law期待値、未観測interface、長q、13round総費用・そのCI。改善しなければ、その失敗を残して次案を選び、同じデータで自動的に特徴を増やさない。

## 7. 追加fit・compileが必要になる条件

S1の実行には追加fitの明示許可が必要だが、追加compileは不要。Aの局所再合成を実H4で数値比較する場合も新しい科学実行範囲となる。

PRの反復傾き・head/tail端点を検証するには、同じδ=.1/p1/R4/K2/full-wrapperで別qの直接価格が必要になる。例えば欠けているqPF16/T1.6は次の候補である。モデル凍結後の新seed/cohortを独立に扱い、校正へ使えばその点をholdoutと呼ばない。PFも既存q16を選択に使った後の独立q検証が必要。q1などの端点、より長いqのprobe、event局所校正は、S1の誤差と必要な費用判断に応じて選ぶ。

追加する科学情報を明記した小さい計画へ別途承認を求める。一点の新qでも4096までの一様保証にはならず、最初から全13roundや旧研究の全局所校正を要求しない。今回は追加q・本数・compile枠を採択していない。

## 8. 全13roundへの接続

\(q_{\rm RPE}=2^m\)、\(T_m=hq_{\rm RPE}\)、\(\delta_m=T_m/q_{\rm PF}\)。qRPEとqPFは定義が異なり、δ=hを固定した選択のとき数値が一致する。qPF増加に伴うprice法則を、そのまま任意のPF分割・高次pieceへ延長しない。

将来の接続は軸別の絶対価格を用いて
\[
\widehat G_s=\sum_{m,a}N_{s,m,a}\widehat C_{s,m,a},\qquad
\Delta G_s=\sum_{m,a}N_{s,m,a}(C_{s,m,a}-\widehat C_{s,m,a})
\]
とする。同じ校正係数はroundをまたぐ相関を生むため、各roundを独立と仮定してSEを足さない。Nの条件付き十分性とpriceの校正/系統誤差も区別する。αtotal=.05はRPE失敗確率の設定であり価格の95%保証ではない。

PR期待価格へは、実際のDF10 finite-bit law・K2 application・各tailのhead resetを反映した構造特徴の期待値が別途必要。探索9本の特徴平均を全law/全qへ単純拡大しない。全13roundのN・長q価格・外挿誤差・共分散が未資格の現状では、総費用や順位保証を計算しない。DF12やTrack Dの価格寄与率も転用しない。

## 9. 回答とSTOP

**効率と必要精度の両立を狙う最初の方法は、SOURCE由来のV1構造を残し、DF10・同一full-wrapperに限定した1〜2特徴の補正を、既存の最小二乗で校正すること。** 保存データだけのS1で安価に可否を調べ、足りないq方向と独立検証だけを追加compileで埋める順序が有力である。旧pair/tripleは境界の区別・分散低減・希少orderの扱いを再利用し、係数や旧成功率を移さない。compilerの完全模倣は必要条件としない。

これは精度達成の実証ではない。今回の確認は主担当Codexによる静的技術調査で、独立科学レビューではない。成果は本書・SOURCE_INDEX・PROJECT_MAP入口のみ。V1と旧SOURCE/input/raw/grant/ledger/marker/KEEP/STOP、他worktree/processを変更せず、GitHub公開後にremote SHAと主要成果物を照合してmandatory STOP。

次の担当：**ユーザー**がS1（保存データの3 fitと既知データ評価）の実行可否を判断する。**Codex**は明示許可後にその範囲を一括実施する。追加compile・全RPE・詳細科学レビューへ自動進行しない。

[V1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eec5bdf9cc391fe49b55b5673f3e385deeb20d42/docs/research/h4_df10_price_v1_20261110/REPORT.md
[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10132af4a67ce112f79c539a9552f5ca0e83e516/docs/research/h4_df10_longtime_price_design_20261110/REPORT.md
[H46]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[QISKIT]: https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/passes/optimization/optimize_1q_decomposition.py
