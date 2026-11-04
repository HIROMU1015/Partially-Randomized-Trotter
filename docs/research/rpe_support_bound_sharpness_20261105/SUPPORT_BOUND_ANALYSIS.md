# Support上界の保守性：構造改善とIID平均転送

2026-11-05 JST / LOCAL / 技術準備。
**数学的に正当な改善を3種類構成した。推奨する次の限定評価は、保存218成分certificateを
そのまま使う「singleton policyを含む期待telescoping上界」である。**
実H4の新しいcolumn差・projector差・重み付き上界・strict判定は計算していない。
旧18/36成立・18/36未判定、RPE_tolerance_certified=false、
scientific_status=UNDETERMINED、U_virtual_IID_upper=null、U_exec_upper=nullを維持する。

## 1. 固定証拠と今回確認したこと

旧結果はb0c354ce071af71a6ce857b11f8acc0c129a05b8、
科学SOURCEは5605ea6fdb90fb7de2dac8768d392c81435217a3。
[前回報告](../rpe_controlled_connection_certificate_execution_20261105/REPORT.md)、
RESULT/DESCRIPTORS/basis_enclosures、AXIS_RESULTS.tsv、VALIDATION.jsonを参照した。
[bindings](FIXED_BINDINGS.json)はcommit/path/blob/SHA256と関連SOURCEを固定する。
前回57b479272のconnection_adapter/解析、C局所probe82b2df6af、
categorical結果d2e7a7db6・m17結果34840ce9bもGitHubの公開blobへ照合した。

固定対象はlinear H4／1.00 Å／STO-3G／保存legacy DF12、L_D3、単一δ=.02、
8 qubit・4電子sector70・保存physical stateのnormalized ray、K2、
identity抽出・cutoff1e-12、m0..17/q2^m・保存r/N/α/β/ε・signal τ/θ/law/support policy。
以下の新しい数式をこのH4へ代入することは、次の科学実行であり今回行っていない。

保存原本から確認できるのは、229個の一粒子包含・218成分の上端が取得済みであること、
最大component上端の旧表示7.0888453e-8、m17追加上端の旧表示.89198317、
boundary上端の旧表示6.5369932e-13、旧18軸成立/18軸未判定。
新しいランキング・group別最大値・確率平均・center差・判定の再集計はしていない。
bindingsのstatic_saved_inspectionには、旧最大componentとそのfull/selectedの保存radiusを
**元のrational literalのまま**引用した。新しいnormの計算ではない。

### SOURCE対応

| 対象 | SOURCEと意味 |
|---|---|
| intended support completion | research_direction_structure_pilot.support_restricted_unitary_completion。指定orbital列を浮動小数点arrayとしてcopy、残りを直交補完 |
| runtime synthesis | sequence_policy.register_support_restricted_bases。fullの再合成descriptorを照合し、selectedを別のGaussian gate列として登録 |
| preserved-column条件 | 登録時のfloat residualと、plan metadataの固定0.0は別。どちらも合成後の理想gate operator差0を証明しない |
| 96-bit basis | connection_adapter.one_particle。exact binary64 parameterのanalytic RZ/隣接XXPlusYYを評価し、vacuum-relative Gaussian GをMatrixBallとして保存 |
| 旧component bound | component_gap。各columnのcenter差と両global radiusを加え、4倍してsupportで足し、2でcap |
| 実policy | full_scope.apply_run_threshold_policy → sequence_policy.make_run_threshold_basis_plan。threshold=1、各tail occurrence内のsource-basis最大run長で決定 |
| 分布 | rte.iter_sample_rte_eventsのorder選択→n+1 component選択。参照は独立uniform53と保存CDF grid mass。固定PCG64のIID性は認定しない |

SOURCEのbasisキーは(source_basis_id, source_basis_hash)。
policyはevent境界を越えてtail内のapplication列を見るが、qのstepごとにresetする。
scalar/event phaseはこのsource-basis runに含まれない。
固定extract_identity_phaseのcomponentは非empty Z/ZZ supportで、identity applicationはない。

## 2. 旧column certificateの分解と限界

G,G'は「合成済みgateのexact parameterに対応する」真の一粒子unitary。
保存center C,C'とoperator半径ρ,ρ'について、

\[
 \|G_{:i}-G'_{:i}\|
 \le \|C_{:i}-C'_{:i}\|+\rho+\rho'.
\]

旧実装はcenter norm自体も外向き評価し、各iの上端を足す。
そのため、数学的なmodel/basis差、binary64生成parameterに由来する差、
区間幅、4ΣcolumnというCAR/三角不等式、全成分最大値、3qrという適用回数を分離する必要がある。

- intendedなexact unitary completionが同じsupport subspaceを持つなら、parity operatorは同じ。
  保存arrayのcopyだけでは、runtime G,G'にそのexact仮定を移せない。
- 今回のtargetはexact binary64 gate parameters。未丸めの化学Hamiltonianや新たな高精度basisへ
  targetを変更して「誤差を消す」ことはしない。
- 保存center/radiusは存在するが、旧component certificateにはcenter寄与/radius寄与の
  分離結果が保存されていない。定量分離を新たに計算していないため、
  H4の支配原因が区間幅かparameter/gauge差かは今回確定しない。
- 同一のqualified gate列(n、順序、class、wire、parameter)なら同一Gを証明できる。
  同じrounded centerだけではその結論にできない。共通targetのexact identityが証明できるときは、
  独立な二つのradiusを足す必要がない。ただし旧H4 certificateを0へ置換していない。

単なるprecision増加を既定としない。先に上界の構造を改善する。

## 3. 構造改善A：columnではなくsupport subspaceを比較

Sの大きさk=1/2、A=G[:,S]、A'=G'[:,S]、
Π=AA†、Π'=A'A'†とする。many-body basisを\(\mathcal U,\mathcal U'\)と書く。
fermionic parity Z_Sはsubspaceだけで決まり、
個別column phaseやS内のunitaryなframe変更には依存しない。
basis共役の中での不変性であり、Hamiltonian scalar/eventのancilla-relative phaseは保持する。

\[
 \boxed{\|\mathcal{U} Z_S\mathcal{U}^\dagger-\mathcal{U}' Z_S(\mathcal{U}')^\dagger\|
 \le \min(2,\sqrt{2k}\|\Pi-\Pi'\|_F).}
\]

証明：thin overlapのSVDでprincipal vector対u_j,v_jを選び、
<u_i,v_j>=δ_ij cos θ_jとする。1対のnumber operator差は、その2-mode Fock空間で
eigenvalue ±sin θ_j,0,0を持つのでnormはsin θ_j。
Z(u)=I−2n(u)から差は2sin θ_j。
各support frameでparityを積に書きtelescopingすると2Σ_j sin θ_j。
一方、projectionのtrace計算で||Π−Π'||_F²=2Σ_j sin² θ_j。
Cauchy–Schwarzで上式を得る。退化角0/π/2も含む。
全Fock空間での上界なのでsector70にも適用できる。
二射影の標準形の背景文献：
[Halmos, Two subspaces (1969)](https://doi.org/10.1090/S0002-9947-1969-0251519-5)。
上記有限次元の導出を証明として用い、外部文献の取得を実行前提にしない。

保存centerのthin matrixをA_c,A'_cとすると、追加のcertified Frobenius幅は

\[
 e_F=\sqrt{k}\,[\rho(2+\rho)+\rho'(2+\rho')].
\]

理由は||A−A_c||_F≤√kρ、||A||=1、||A_c||≤1+ρと積の差。
したがって、centerのprojection差のFrobenius上端にe_Fを足せばよい。
sharpness_adapter.projector_boundは両radiusとAcb積の幅を保持する。
既存ballsの再生成を要しないが、H4に適用する計算は未実行。
将来採用する場合はmin(旧component上端,新projector上端)で非悪化を保証する。
1成分のthin projector処理はO(n²k)、既存229 ballsから最大218成分へ適用できる。

これは局所operator certificateの改善で、q依存自体を変更しない。
phase/frameだけが違う人工例では大幅に鋭くなるが、H4でも同じとは主張しない。
本段階の最小科学計画にはこの新matrix評価を含めず、保留する。

## 4. 構造改善B：rotation係数とsingleton word

各component誤差上端をd_l∈[0,2]とする。
productの差はd_l、rotationの差は同じθなら|sin θ_n| d_l。
旧3はK2のproduct最大2＋rotation最大1であり、
固定wordでは実際のrole、n=0/2、選択component、policy indicatorを使える。

supportを使う位置は、そのsource-basisの最大run長が1の場合だけ。
つまり、存在する直前/直後のsource-basisと異なるsingleton位置である。
同じ最悪componentを全位置に置くと全てfullになり、support差は発生しない。
しかし、最悪basisと別basisを交互に置くwordは許される。
最悪componentが繰り返さないと仮定することはできない。

各basisのd最大値とrotation係数上端s0,s2を使う有限DPを実装した。
状態は(last source-basis, 未確定singleton寄与)、値は確定済み加法上端。
同じbasisが続けばpendingを0にし、別basisが来たらpendingを確定する。
各eventでtoken列(s0)または(1,1,s2)を試し、tail末尾でpendingを閉じる。
event末尾の状態数は高々3×basis数、r方向O(r B²)、q方向の回路展開は不要。
これは加法certificateの全word最大値であり、真のoperator normの正確な最大値とは区別する。

DPは3r max d以下、またrole情報だけでもr(2+s2)max d以下になる。
これは主に係数の改善。一般の非可換なunitary積について相殺や√q則は導けない。
局所norm情報だけの一般論では、U=I、V=exp(iε)Iに対し
||V^q−U^q||=2|sin(qε/2)|で、小さいqεではqεに比例する。
この反例はnorm情報だけからsublinear則を導けないことを示し、H4 support誤差の最適q依存を証明するものではない。
S2のF/Rと共役されたcomponentを使う人工例でも、小さい誤差のcoherentなq倍蓄積が起こり得た。
固定H4に特別な共通conjugation/収縮があれば別証明は可能だが、その証拠は未取得。
H4のbasis別最大値やDP上界は今回計算していない。

## 5. 採用候補：policyを含む期待telescoping

実装word Tの差について、unitary積のtelescopingとnorm不変性から

\[
 |z_{\rm impl,T}-z_{\rm ref,T}|
 \le\sum_j I_j(T)c_jd_{\ell_j}
\]

を得る。I_jはsingleton選択、c_jはproductなら1、rotationなら|sin θ_n|。
これをそのまま平均し、||E[D]||≤E||D||≤E[加法上端]を使う。
**word依存の実装平均を単一event平均の積へ因数分解していない。**
scalar/π64 phaseがword依存でも各因子のunitarityを用いるだけなので相関を捨てた仮定は要らない。

独立uniform53 component質量p_l、source-basis color b_lとし、

\[
 h_b=\sum_{\ell:b_\ell=b}p_\ell,\qquad
 D_j=\sum_\ell p_\ell d_\ell(1-h_{b_\ell})^j,\quad j=0,1,2.
\]

current componentを固定すると各隣接colorが異なる確率は1−h_b。
component choicesが独立なので、両隣接ならその二乗。
これはnorm差の期待値を評価する式であり、誤差operatorの符号相殺ではない。
orderを固定してcomponentを先に平均すれば、
tailの端はD1、内部はD2、長さ1だけはD0になる。

保存K2 order質量w0+w2=1、s_n≥|sin θ_n|（0≤s_n≤1）について、

\[
 \eta_1=w_0s_0D_0+w_2[(1+s_2)D_1+D_2],
\]

\[
 \eta_r=[2w_0s_0+w_2(1+s_2)]D_1+
 [(r-2)w_0s_0+w_2((2r-1)+(r-1)s_2)]D_2,\quad r\ge2.
\]

証明：n=0はrotation1、n=2はproduct2＋rotation1。
r≥2ならtail長は必ず2以上。最初の係数平均はw0s0+w2、最後はw0s0+w2s2。
残りの係数はr[w0s0+w2(2+s2)]から端を除く。
上式はその差を非負係数だけで書き、区間減算を避けた。
r=1,n=0の隣接なしの例外を別に扱う。

D2≤D1≤D0≤max d、η_r≤r[w0s0+w2(2+s2)]D0≤3r max d。
したがって旧上界より悪くならない構成が可能。
全stepへは平均の線形性を用いqη_rを転送する。
qは一般に線形のままであり、r/order/role/component/policyに関する係数を改善する。
ηは必要に応じ2でcapできる。

適用条件は、componentがorder・位置から独立な同一law、固定nごとにθはcomponent符号以外共通、
K2、非identity application、run_le_1のtail内選択、固定ψがTから独立であること。
SOURCEはこのideal IIDモデルに対応するが、PCG64固定streamを数学的IIDとは認定しない。
law/CDFの作り直し、component probability64とexact grid massの取り違えは許さない。
w2を0と近似したり、rotationだけを残してproduct項を落としたりもしない。

新しいmean専用追加上界は

\[
 \Delta^{\rm mean}_{m}
 =\min(2,\Delta^{\rm old}_{m},
       \Delta_{{\rm boundary},m}+q_m\min(2,\eta_{r_m})+\Delta_{{\rm lowering},m}).
\]

旧I_catにはπ64が既に含まれる。新式に再加算しない。
この量はIID平均専用で、任意trajectoryや固定PRNGへの新上界ではない。
旧一様上界とのminは、この意味の違いを変えない。

## 6. 人工検証・実装判断

[sharpness_adapter.py](sharpness_adapter.py)にprojector、word DP、期待上界だけを追加した。
既存ball/MatrixBall、component helper、SOURCE run長関数、classifier、byte validatorを再利用した。
旧component_gap/total_transferは上記のphase/subspace不変性、role/policy/orderの平均を扱わないため、
凍結SOURCEを変更せず新helperへ分けた。
[run_weighted_transfer.py](run_weighted_transfer.py)はdefaultをbyte/schema/literal照合に限定し、
別途承認後だけの--runで採用した期待上界を評価する。projector/DPをH4へ自動適用しない。

新26件はSOURCE run長によるr1/2/3全列挙との閉形式/DP一致、非可換signed involution、
event平均の誤因数分解と相関neighborの反例、F/R入りcoherent q累積、
phase/support-frame不変projector、非零差・radius保持、
不正law・missing/negative evidence、strict等号、部分結果STOP、人工18round end-to-endを検証した。
既存147件と合わせ173件PASS（173 passed in 7.68s）。
float matrix比較の許容差は人工testだけで、certificateの誤差を0にする根拠にはしていない。
default実行の新H4科学callは0。H4の新mean transferや36 strict比較は未実施。
旧SOURCE/raw/229 balls/218 certificates/STOPは無変更。重要な独立科学レビューを開始していない。

~~~sh
/tmp/prt-rpe-m17-enclosure-venv-20261103/bin/python -m pytest -q \
 docs/research/rpe_support_bound_sharpness_20261105/test_sharpness.py \
 docs/research/rpe_controlled_connection_runner_20261104/test_connection_runner.py \
 docs/research/rpe_categorical_circuit_connection_20261104/test_connection_adapter.py \
 docs/research/rpe_controlled_hadamard_preparation_20261104/test_circuit_semantics.py \
 docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103/test_mean_enclosure.py \
 docs/research/rpe_controlled_hadamard_runner_v2_20261104/test_local_probes.py
~~~

## 7. 判断表・次段階

| 検討項目 | 現状の保守性 | 改善可能性 | 必要な追加証拠 | 次の作業 |
|---|---|---|---|---|
| Certified column差 | center差＋両radius、phase/frameを区別しない | exact identity/subspace利用は可能 | runtime subspace差の新certificate、またはexact identity | 数値分離/再評価は保留 |
| Component operator上界 | 4Σcolumnとproduct三角不等式 | rank-k parity projector上界を証明 | 保存ballsへのprojector評価（未実施） | 人工検証済み・次期候補 |
| 最大componentの一律適用 | 全位置に同じmax | 成分別p_l d_lとsingleton確率を利用可能 | 同一の保存uniform53 law/ID順 | 期待上界を採用候補 |
| qr方向の誤差累積 | 3qr、rotationも1扱い | role/run DPと期待係数を改善 | 固定K2/policy、特別な相殺なら別証明 | qの一般的線形性は維持 |
| Categorical平均への転送 | 一様上界を平均へ移送 | 期待telescoping閉形式を証明 | IID・neighbor依存の上記条件、12 sine上端 | 固定1 passの科学評価を提案 |

推奨は保存certificateを再利用する[限定科学実行計画](SCIENTIFIC_EXECUTION_PLAN.md)。
ユーザーが公開SOURCEと範囲へ別途明示許可した後、Codexが
1回の218成分重み付け、6r/12 sine、18 transfer/36 strictだけを実行する。
全18roundの未判定解消は事前に断定しない。
今回の公開照合後はmandatory STOP。新科学実行を開始しない。
