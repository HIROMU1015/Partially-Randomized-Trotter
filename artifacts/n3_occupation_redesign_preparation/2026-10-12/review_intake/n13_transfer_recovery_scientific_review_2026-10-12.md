# N3保証生成・限定移送／N1技術復旧・費用selector：独立科学レビュー

- 作成日：2026-10-12（Asia/Tokyo）
- 研究：Hamiltonian simulationの新しいアルゴリズム改善。PR外部単独および内外共同設計を含むが、別TrackのPR内部単独改善とは区別する。
- 状態：ユーザーの明示的な開始承認に基づく科学レビュー完了。
- 対象：N3の固定12入力による生成・移送batchと、N1の固定条件による復旧batch。
- 本記録は新しい科学batchの実行認可ではない。元の固定結果・事前判定・Mandatory STOPを変更しない。

## 0. 結論

**N1の現在の有限候補を用いた改善探索はいったん区切る。N3の今回の直接生成法については、登録された移送優位基準未達を維持する。ただしN3全体を終了するのではなく、粒子数・density相互作用の情報を残した係数保証を、強い既知対照と比較する限定的な再設計を勧める。**

| 対象 | 科学的判断 | 理由 |
|---|---|---|
| N1：現有限候補・費用selector | 現構成の追加scanは終了／保留。実装・対照を保持 | 復旧後の3 context×2指標でexactを選択し、保証付き目的関数の改善0%。技術的未完了を理由に継続していた段階は終わった |
| N3：現直接生成・固定移送 | 今回の検証は基準未達で完了。現generatorの広い優位性を主張しない | 直接法5件はnorm-onlyと五点法の受理集合の和集合と一致。norm-onlyに対する追加1件はν=1 |
| N3：次の方法論 | 粒子数と正のdensity構造を保持する保証へ限定再設計 | 同じ固定T3入力で旧十分条件の情報損失を切り分けた。ただし新しい方向付き条件だけでなく、強めた外部下界＋norm-onlyでも保証可能だった |
| N2 | 補助修正完了を維持 | 今回新しい科学結果なし。charge探索を再開する根拠にはしない |
| 中心テーマ・新規性・総量子資源優位 | 未採択・未確定 | 今回の証拠はまだ特定classの小規模な保証・回路会計に限定される |

N3について、レビュー中に二つの追加結果を得た。

1. 固定粒子数の固有値和を使う、現LMIより弱い十分条件を導出し、同じT3入力で有理数の証拠を作った。同じ改善後の上界・下界を用いても、旧LMIはすべてのパラメータで不可能であることを有理数の分離証拠で確認した。
2. その一方で、外部空間の正のdensity項を残した別の安価な下界を使うと、同じ入力をnorm-onlyでも保証できた。

**したがって「新しい方向付き条件がなければこの縮約は保証できない」とは主張しない。** 新しい数式の正しさ、同じ情報での十分条件の強さ、別の情報取得法を含めた算法全体の利益を区別することが、今回の最も重要な更新である。

## 1. 研究経緯と今回の問い

前段階では、N1は保存vectorのdecode不備で科学的selectorに到達していなかった。N3は既知development小gap入力に対して、同じU・cのnorm-onlyでは拒否される縮約を方向付き条件で受理した。そのため、前回はN1の一回復旧と、N3の証拠生成・未使用入力への移送を推奨した。

今回は、その二つの未完了な検証に新しい結果が加わった。前回レビューの繰り返しではない。問うべきなのは、実装が正常に動いたかだけでなく、現構成を新しいアルゴリズム研究として継続する根拠が得られたかである。

研究目的を、既知手法のベンチマークだけで論文化する方向へ置き換えない。一方、新規算法を目指すために、数学的に正しいが安い対照を超えない構成を無期限に追う必要もない。

## 2. 固定資料・来歴・取得範囲

Repository：`HIROMU1015/Partially-Randomized-Trotter`

| 対象 | 専用branch | science source | result | publication |
|---|---|---|---|---|
| N3 | `n3-generator-transfer-execution-20261011` | `178d69a31dc7f2f1162d76294ba35a94b67d7fee` | `b3e41b76fbb21c238c25572620e967d131571458` | `64cf5ed227b7e4a74b37453cbe733db5f5d51f56` |
| N1 | `n1-decode-recovery-execution-20261011` | `3fe779f88ab6367631355cf5e5152bf32e897a4d` | `8816f9340de383d8a7efea4f083788584dc2e1af` | `4c719a72eb8caae2f77fc746bd4672d610d97610` |

共通準備commitは`ba4530678df9d665d2e6005fb08179b0f7a7921f`。実行planの固定commitはN3が`f0e5e5d254fce0458431dcc59f2ab02562ee4c39`、N1が`89a40b39e08c6c5d5ea474f7055319bcdced86b2`。

### 2.1 主要なGitHub一次資料

- [N3結果報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3e41b76fbb21c238c25572620e967d131571458/docs/research/n3_direction_generator_v1_results_20261011.md)
- [N3固定source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/178d69a31dc7f2f1162d76294ba35a94b67d7fee/src/trottertracks/representation_exploration/n3_direction_generator_v1.py)
- [N3数学・入力契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3e41b76fbb21c238c25572620e967d131571458/docs/research/n3_generator_transfer_contract_v1_20261011.md)
- [N3固定入力](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3e41b76fbb21c238c25572620e967d131571458/artifacts/n3_generator_n1_recovery/2026-10-11/preparation/n3_transfer_inputs.json)
- [N3全72 method ledger](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3e41b76fbb21c238c25572620e967d131571458/artifacts/n3_direction_generator_v1/2026-10-11/run1/certificate_ledger.json)
- [N3独立保存監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3e41b76fbb21c238c25572620e967d131571458/artifacts/n3_direction_generator_v1/2026-10-11/run1/independent_audit.json)
- [N3公開・再取得記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64cf5ed227b7e4a74b37453cbe733db5f5d51f56/docs/research/n3_direction_generator_v1_publication_20261011.md)
- [N1結果報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8816f9340de383d8a7efea4f083788584dc2e1af/docs/research/n1_decode_recovery_v1_results_20261011.md)
- [N1固定source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3fe779f88ab6367631355cf5e5152bf32e897a4d/src/trottertracks/representation_exploration/n1_decode_recovery_v1.py)
- [N1比較集約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8816f9340de383d8a7efea4f083788584dc2e1af/artifacts/n1_decode_recovery_v1/2026-10-11/run1/comparison_summary.json)
- [N1独立保存監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8816f9340de383d8a7efea4f083788584dc2e1af/artifacts/n1_decode_recovery_v1/2026-10-11/run1/independent_audit.json)
- [N1公開・再取得記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c719a72eb8caae2f77fc746bd4672d610d97610/docs/research/n1_decode_recovery_v1_publication_20261011.md)

### 2.2 何を独立に確認したか

GitHubの報告、契約、科学source、入力recipe、N3 ledger、N1比較集約、監査・STOP記録を確認した。主要資料は取得可能であり、追加pushを求める必要はない。

GPT側では、元のscience moduleをimportせずにN3の12入力を係数から再構成し、旧条件の受理集合と同じ集合を、別のproposal実装と有理数PSD判定で再現した。固定入力の真のenergy widthは別のFock実装で診断したが、certificate生成へ渡していない。

N1については、sourceのdecoder・model/PF/bias会計・selection構造、保存報告と比較集約を精査し、選択されたshot×XY費用の積を独立に照合した。**N1全1424行のargminをGPTで再監査した、594 native IRを全件再生した、全72 N3 ledgerの元witnessをそのまま再検証した、とは主張しない。** 元の保存監査と、今回の独立な確認の範囲を区別する。

追加のN3再設計検討では、同じT3入力に対する有理数証拠と反証・強化対照を構成した。保存後、NumPy/SciPyやscience moduleをimportしない標準ライブラリのみの別checkerで、その固定証拠を再検証した。これは現在のGPT環境内の独立計算であり、immutable CIや外部研究者による科学再現ではない。

## 3. 比較しているtaskを混同しない

### N3

実対称one-body＋非負density相互作用、外部一軌道を空に固定する、基底エネルギーの投影誤差保証である。active modes 4/5/7、粒子数ν=1/2/3、full modes 5/6/8、目標幅δ=0.004。geometry、化学basis、DF、L_D、時間発展、PR/RTEの量子資源は今回の評価対象ではない。

### N1

合成one-body＋signed DF rank2、weights 0.7と−0.4、JW4 modes、ν=2、sector dimension 6。決定論S2、T=0.6、q∈{1,2,4,8}、step size 0.6/0.3/0.15/0.075、model budget 0.02、complex signal error 0.04、normalization 1。

固定X0X1状態準備を含む5-qubitのX/Y測定wrapperを、Qiskit 1.3.0、optimization level 1、固定seed、rz/sx/x/cx基底、ordinary controlで評価している。評価対象は、理想gateの保証と数値compile検査を用いたshot×native wrapper proxyである。

**N3の保証件数と、N1のRZ/CX費用を同じ資源尺度として順位付けしてはならない。** また、両方とも実分子の最終的なQPE/PR総資源を確定したものではない。

## 4. N3：登録結果の評価

### 4.1 登録された判定は基準未達のまま維持する

直接法5/12、共通norm-only4/12、共通五点法4/12。norm-onlyとの差はT1のみで、事前の「少なくとも2 base」という条件を満たさなかった。正式な終端は`NO_TRANSFER_ADVANTAGE_IN_FROZEN_SCOPE`。

| base | active m | ν | norm-only | 共通五点 | 直接生成 | norm-only OR 五点 |
|---|---:|---:|---:|---:|---:|---:|
| T1 | 4 | 1 | 2/4 | 3/4 | 3/4 | 3/4 |
| T2 | 5 | 2 | 2/4 | 1/4 | 2/4 | 2/4 |
| T3 | 7 | 3 | 0/4 | 0/4 | 0/4 | 0/4 |
| 合計 | — | — | 4/12 | 4/12 | 5/12 | 5/12 |

受理数だけでなく集合を照合すると、

\[
\mathcal A_{\rm direct}
=\mathcal A_{\rm norm}\cup\mathcal A_{\rm five}.
\]

これは保存表からの集合演算で、今回独立に確認した。両方のcertificateが同じ誤差幅に対する正しい十分条件なら、そのどちらかで受理するOR対照は安全である。したがって、この固定集合において、直接generatorは既存の二つの保証を併用した場合より新しい受理を得ていない。

このOR対照は登録判定の差し替えではなく、結果の帰属を理解するための追加分析である。生成時間や証拠サイズまでORと比較し終えたわけではないので、全費用で同等とも断定しない。

### 4.2 norm-onlyに対する追加成功はν=1

直接法がnorm-onlyより追加で保証したのは`T1_amp_1_10_high_coordinates`。T1はν=1である。したがってそのsectorではすべての二体density項n_i n_jが消え、Hamiltonianはone-body行列の作用に還元される。

この成功は宣言されたclass内で正しい結果である。しかし、相互作用する多粒子系で新しい保証を得た証拠には数えない。前のdevelopment ν=2例で方向効果が確認されたことを否定するものでもないが、それと今回の未使用入力での結果は区別する。

### 4.3 T3の不受理は前提不成立

T3の全4入力では共通下界c≤Uとなり、`BOUND_PRECONDITION_FAILED`。真の誤差が小さい入力が含まれても、保証量が得られていなければ不受理のままにする運用は妥当である。

今回の12入力は3 baseに振幅・方向を組み合わせた設計であり、独立同分布の12標本ではない。サイズ、粒子数、scale等もbaseとともに変わっている。単に5/12という比率から一般化確率を推定したり、粒子数の効果だけを統計的に抽出したりしない。

### 4.4 実装・監査の支持と限界

全72 method recordsのledgerを閉じてから24 full/active診断solveを実行した構成は、truthの選択への混入を避けるうえで重要である。49,016 checks、15 synthetic tests、最大有理中間値394 bitsは保存記録として確認した。

496回の小行列固有値proposal、64件の有理候補などの取得費用も記録されている。一回約0.195秒という値はこの小規模batchの観測値で、漸近計算量、逐次実行に対するspeedup、大規模系での実用性を示すものではない。有限generatorの不受理を、証拠の数学的不存在と同一視しない。

## 5. N1：復旧は成功したが、現候補の資源改善は支持されない

### 5.1 科学結果を取得できた点は明確に更新する

今回のdecoderはreal/imag形式を正しく読み、非zeroの虚部を無条件に捨てず、schema・shape・finite・bool・ID・order等を検査している。保存packと5428個のbinary64 scalar比を保持し、元の48 numerical function/class定義とのAST一致も保存されている。

これは前回の技術停止からの復旧である。そのうえで今回、初めて費用込みselectorの科学結果が得られた。

| context | 選択q | shot/axis | X/Y RZ和 | 総RZ | X/Y CX和 | 総CX | N1/exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| planted_local_gauge | 4 | 16,448 | 2,494 | 41,021,312 | 1,664 | 27,369,472 | 1 |
| dense_group_frame | 4 | 16,933 | 2,614 | 44,262,862 | 1,696 | 28,718,368 | 1 |
| separated_spectrum | 8 | 13,631 | 5,222 | 71,181,082 | 3,392 | 46,236,352 | 1 |

全6選択がexact対照と同じIDである。有限shifted/zero cutoff対照、all_mixed有限domainの最小も同じexact。RZ/CXの保証付き目的関数改善は0%。

### 5.2 何が変わって、何が変わらないのか

basis-only selectorは最初の二つでcluster/exactを選ぶが、元Hamiltonianへのmodel/PF誤差とshotを含むselectorはexactへ戻る。したがって、basis削減そのものを否定する結果ではない。一方、その削減を最終的な保証付き費用の利益へ変えることは、現候補範囲ではできていない。

exactを候補に含めた最小化では、その同じ会計のexact対照より悪い値を選ばない。この安全なfallbackは望ましい設計上の性質だが、量子資源削減の新しい実証ではない。

### 5.3 会計と保証の範囲

shot式は概ね

\[
N(b)=\left\lceil\frac{2\log80}{(\epsilon_{\rm complex}/\sqrt2-b)^2}\right\rceil
\]

であり、sourceでは安全側の高精度・区間処理が使われる。biasにはmodel誤差、係数由来のS2交換子上界、angle/frameや微小非Hermitian成分に関する会計が入る。代表的な形は

\[
b\le T e_{\rm model}e^{T\gamma}+\frac{T^3\alpha_2}{q^2}+e_{\rm angle},
\]

\[
\alpha_2=\sum_j\left(\frac{\|[R_j,[R_j,H_j]]\|_1}{12}
+\frac{\|[H_j,[H_j,R_j]]\|_1}{24}\right)
\]

である。ここでの係数1-norm上界や浮動小数点再構成の囲いと、true operator errorは別物である。

1424 selector rowsを閉じてから1424 truth診断。594 native IRの独立保存検査26,575 checksは、元コードと異なるPauli/native作用で確認した記録である。しかし最大native差約1.3033e−12はbinary64の数値検査であり、すべてのcompile丸めを外向きに囲った証明ではない。claimは`ideal_gate_certificate_with_numerical_compile_validation`。

### 5.4 結論の限界

0%という結果だけから、実際の誤差を使っても全近似候補が必ず負けるとまでは言えない。上界の保守性、候補生成の限界、実回路費用の三者を完全に分解した新しいN1診断は、今回GPTでは実施していない。

ただし、これらの未確定性が残ることと、現構成の追加scanを推奨することは別である。新しい改善機構や構成変更を示さず、同じ有限候補を拡張して成功条件を探すだけでは、新規算法研究としての根拠が弱い。

**今回は現N1の限定探索を区切る。** frame構成、error-aware selector、exact/whole-Pauli/shifted対照は再利用可能な基盤として保持する。whole-Pauliがこの条件でnative exactより高価だったことも、N1近似の優位や一般的な表現順位を意味しない。

## 6. N3の既存保証を再整理する

以下は宣言されたclassの下での代数である。一般の二体Hamiltonian、複数外部軌道、任意の軌道回転へ無条件に流用しない。

全粒子数νを固定し、外部軌道dを空にするPと、占有するQに分ける。

\[
H=\begin{pmatrix}A&B\\B^\dagger&C\end{pmatrix},\quad
A=d\Gamma_\nu(h_A)+V_A,\quad V_A\succeq0.
\]

外部とのone-body hopping vectorをvとする。CARより

\[
BB^\dagger=d\Gamma_\nu(vv^\dagger).
\]

また

\[
C=h_{dd}I+d\Gamma_{\nu-1}(h_A+\operatorname{diag}J_{id})+V_{A,\nu-1}.
\]

外部軌道が途中のindexにあっても、適切なfermionic orderingで同じ関係が成立する。これはbasis signを無視してよいという意味ではない。

係数からC≽cIとa=λ_min(A)≤Uを取得し、δ>0、c>Uを満たすとする。D=c−U+δ。

現certificateは

\[
0\le\kappa\le D,\quad
\kappa h_A-\mu I-vv^\dagger\succeq0,\quad
\kappa U-\nu\mu\le\delta D.
\]

μを最大許容の行列側条件と整合する最小値μ₀=(κU−δD)/νへ置けば、one-variable LMI

\[
M(\kappa)=\kappa(h_A-U I/\nu)+\delta D I/\nu-vv^\dagger\succeq0
\]

を得る。Schur補元により0≤a−E₀(H)≤δを保証する。数値solverが候補を返すことと、PSD・affine条件を有理数で検証することを区別する。

この証明の型はFeshbach–Schur・相対form評価と近接する既知の議論である。未知のmany-body真値を計算せず係数から証拠を生成する具体的な手順が研究対象であり、Schur補元自体を新規性としない。[文献R1]

## 7. 粒子数情報を保持する、より弱い十分条件

### 7.1 どの情報を失っていたか

\[
\sigma_k(G)=\sum_{r=1}^{k}\lambda_r^{\uparrow}(G),\quad\sigma_0(G)=0
\]

と置く。fermionの固定k粒子sectorでは

\[
\lambda_{\min}(d\Gamma_k(G))=\sigma_k(G).
\]

最小一体固有値をk回足す下界kλ_min(G)は安全だが、同じ軌道を重複占有できない制約を捨てる。したがって、外部下界を

\[
c_{\rm occ}=h_{dd}+\underline\sigma_{\nu-1}(h_A+\operatorname{diag}J_{id})
\]

とする方法がある。下線は真のmany-body固有値ではなく、有限の係数証拠で得た一体固有値和の下界を意味する。

### 7.2 active側の条件も固有値和で評価できる

Kκ=κh_A−vv†とする。現LMIは実質的に

\[
\nu\lambda_{\min}(K_\kappa)\ge\kappa U-\delta D
\]

を求めている。これを、より弱い条件

\[
\boxed{0\le\kappa\le D,\qquad
\underline\sigma_\nu(K_\kappa)\ge\kappa U-\delta D}
\]

へ置き換えられる。

証明は、κ≥0とV_A≽0から

\[
\kappa A-BB^\dagger
=d\Gamma_\nu(K_\kappa)+\kappa V_A
\succeq(\kappa U-\delta D)I
\]

を得る。a=λ_min(A)、z=a−δとするとC−zI≽DIなので

\[
\begin{aligned}
A-zI-B(C-zI)^{-1}B^\dagger
&\succeq(1-\kappa/D)(A-aI)+\kappa(U-a)I/D\\
&\succeq0.
\end{aligned}
\]

よってH≽(a−δ)I、変分上界E₀(H)≤aと合わせて

\[
0\le a-E_0(H)\le\delta.
\]

aは証明上の記号であってselectorへ渡す入力ではない。

### 7.3 安い行列の証拠にできる

任意のHermitian G、実数τ、Zについて

\[
Z\succeq0,\qquad G-\tau I+Z\succeq0
\]

が分かれば

\[
\boxed{\sigma_k(G)\ge k\tau-\operatorname{tr}Z}
\]

である。例えばKy Fanのtrace最小化表現0≼γ≼I、trγ=kを使い、tr(Gγ)≥kτ−tr(Zγ)≥kτ−trZと示せる。これは標準的な固有値和・半正定値表現である。[文献R2]

したがって、数値固有分解はτ・Zのproposalにのみ使用し、有理数化したZと二つのPSD条件を独立に検証できる。many-body行列の全対角化を保証生成に使う必要はない。新条件もκ、τ、Zのaffineな半正定値制約として書ける。

σνはaffineなKκに対して凹関数であり、旧νλ_minより小さくならない。数学的な可行集合は旧LMI族を含む。またκ=0ではσν(−vv†)=−||v||²なので、共通norm-only条件も含まれる。ただし有限の数値generatorが必ずすべての存在証拠を発見するという意味ではない。κ=0を明示的に検査する等、実装のfallbackは別に管理する。

### 7.4 まだ主張しないこと

固有値和を使うこと、Schur補元、SDP化そのものを新規定理とはしない。小さい行列でも有理数bit長やtrial Uの生成負担は残る。既存toyのtrial構成が少数のoccupationを列挙していることを、そのまま一般の効率的なactive-space算法と呼ばない。

## 8. 同じT3入力での事後的な有理数証拠

対象は固定入力`T3_amp_1_40_high_coordinates`、active7、ν=3、外部1、full8。

入力case SHA-256：

`b34f66cbeed196090dfd77ab999cb462db821ce12e9d21408020771f47a81f16`

このhashがGitHubの固定certificate ledgerに存在することを確認した。今回の係数再構成も同じhashである。

| 量 | 値（丸め表示） |
|---|---:|
| 元と同じtrial U | −0.8 |
| 目標δ | 0.004 |
| 元のc | −1.1388673305511474 |
| 固有値和で得たc_occ | −0.6867984155405793 |
| D=c_occ−U+δ | 0.11720158445942071 |
| 新条件のκ | 0.0005652937962534192 |
| 新条件のaffine余裕 | +0.00011627668509375612 |
| 同じc_occを使うnorm-onlyのaffine余裕 | −0.0001561936621623172 |
| true width（診断専用） | 約7.043636624×10⁻⁷ |

affine余裕はσν(Kκ)−κU+δD等の値であり、エネルギー誤差幅そのものではない。true widthは証拠を作る入力に使っていない。τ、Z、κおよび検証結果は付録に全て保存する。

### 8.1 旧LMIが見つからない理由を、探索失敗と分けた

同じ改善後のU・c_occ・δについて、非zeroの有理数PSD行列Wを構成し、

\[
\operatorname{tr}[W(h_A-U I/\nu)]=0,
\]

\[
\frac{\operatorname{tr}[W(\delta D I/\nu-vv^\dagger)]}{\operatorname{tr}W}
=-4.4559099334\ldots\times10^{-5}<0
\]

を厳密に検証した。Wは二つの有理vectorの外積の凸結合なのでPSDである。

したがって任意のκでtr(WM(κ))<0となり、M(κ)≽0は不可能である。これは単に五点gridが見落としたという結果ではなく、**この共通情報に対する旧one-body LMI族全体の不可能性**である。

ただし、別の下界c、別のtrial U、別の保証族まで不可能と示したものではない。次節の強化対照が、その区別の重要性を示している。

### 8.2 登録検証との境界

この入力は今回の結果を見た後に選んだdevelopment入力である。保証自体は厳密でも、独立移送の新しい成功例ではない。元batchの5/12、2-base基準未達、STOPを変更しない。付録の12入力に対する数値proposal診断も、別途有理証拠を付けていないものを新しい保証成功数へ数えない。

## 9. 重要な追加対照：強い外部下界ならnorm-onlyでも受理できる

新しい条件だけを対照にして次へ進むと、もう一度「別の安い既知構成に勝てない」探索を繰り返す可能性がある。そこで、同じT3入力の外部下界を別の係数分解で評価した。

### 9.1 外部density項を正のまま残す

p=ν−1=2とし、Cをoccupation基底での対角成分C_diagと、one-bodyの非対角成分dΓ_p(O)へ分ける。

\[
C=C_{\rm diag}+d\Gamma_p(O),\qquad
O=h_A-\operatorname{diag}(h_A).
\]

\[
c_{\rm diag}=\min_{|S|=p}
\left[h_{dd}+\sum_{i\in S}(h_{ii}+J_{id})
+\sum_{\{i,j\}\subset S}J_{ij}\right].
\]

このT3ではp=2なので21組の係数を評価すればよく、

\[
c_{\rm diag}=-1/3.
\]

さらに同じτ・Z方式の有理PSD証拠で

\[
\underline\sigma_2(O)=-0.2569264556186681\ldots
\]

を得ると、

\[
C\succeq c_{\rm dh}I,\qquad
c_{\rm dh}=c_{\rm diag}+\underline\sigma_2(O)
=-0.5902597889520015\ldots.
\]

### 9.2 既存norm-onlyが通る

同じU=−0.8、δ=0.004、β²=||v||²=0.000625に対し、

\[
\delta(c_{\rm dh}-U+\delta)-\beta^2
=+0.0002299608441919942\ldots>0.
\]

したがって、**この入力は新しい方向付き条件を使わず、強い係数下界＋norm-onlyで保証できる。** この証拠もtruthやmany-body固有値を使わず、標準ライブラリの有理数checkerで検証した。

### 9.3 科学的な解釈

第8節の「同じc_occなら新しい条件が旧LMIより強い」という結果は依然として正しい。しかし、「この縮約を認めるためには新条件が必要」という算法全体の主張は支持されない。取得する情報の改善と、同じ情報を使う保証不等式の改善を分ける必要がある。

この対角最小化は、今回p=2なのでO(m²)で計算できる。**一般の可変粒子数pでの対角二次式の厳密最小化が安価であるとは仮定しない。** 大系では別の係数下界、LP/SDP緩和、局所分解等を検討する必要があり、その費用を含めて比較すべきである。

ここから直ちに「外部下界だけを改善すれば全入力で十分」とも結論しない。下界、trial U、結合の方向情報のどこが各入力で支配的かを切り分ける、具体的な比較課題が得られたと評価する。

## 10. 先行研究との位置づけと新規性

| 領域 | 既知の内容 | 本研究で差分を置くべき場所 |
|---|---|---|
| Feshbach–Schur・相対form評価 | block結合から固有値差を評価する理論 | 宣言したfermionic classの係数から、証拠と縮約判断を取得する具体的な算法・費用 [R1] |
| Ky Fan・trace最小化・SDP | 最小固有値和の変分表現、PSD証拠 | 粒子数・density情報を、どの安価な証拠に残すかという構成と適用範囲 [R2] |
| SQuISH等の縮約 | Hamiltonianと近似ground-state情報を用いる縮約 | many-body真値を使わない係数証拠。ただしN3も明示trialの上界Uを利用する [R3] |
| 対称性・固有値構造を利用した回路合成 | efficient synthesis、grouping等 | N1固有の誤差・frame・費用構成が既知対照を超えるか [R4] |
| RC-DF | 圧縮と資源関連のfactorization最適化 | 単なるresource-aware目的変更ではない、新しい構成・保証・資源効果 [R5] |

本レビューで確認した範囲では、現N3の「自動保証生成」や新しいσν条件を、そのまま独立論文の主要な新規定理と認定できない。同一のfermionic証拠生成手順が既報か否かを、網羅的に決着したわけでもない。

有意味な着地点は、例えば、**所定のHamiltonian classに対し、占有制約・正値性・結合方向から安価なcertificateを生成し、強い既存対照が取りこぼす縮約を、真値なしに認証できる算法**である。しかし今回のT3一件では、より強いnorm-only対照でも認証できた。新たな条件の算法固有の効果は未確定である。

一般定理や全分子での勝利は必須条件にしない。一方、標準的な行列不等式の再実装とdevelopment一件の成功だけで、新規算法が成立したとも扱わない。

## 11. 次にCodexへ渡すべき限定作業

**次の担当はCodex。研究上の目的は、N3の保証条件と、証拠となる情報の取得法のどちらに独自の改善余地があるかを判別すること。** 同じ今回結果のGPTレビューを直ちに繰り返す必要はない。

### 11.1 N3：まず本レビューの証拠を独立に確認する

- 固定sourceのclass・ordering・粒子数規約でCAR恒等式、外部C分解、σν条件、Schur証明を確認する。
- 同じT3の入力hashを照合し、有理数PSD証拠・旧LMI分離証拠・強化norm-only対照を独立に検証する。
- 元の登録結果は変更せず、review由来の事後解析として別に保存する。

### 11.2 N3：下界と判定法を公平に分離する

同じUと誤差幅で、少なくとも次の二軸を区別できる比較を構成する。

- 情報取得側：従来c、粒子数固有値和c、densityを保持した対角＋hopping等の妥当な下界。比較する各方法へ同じcを渡す。
- 判定側：norm-only、既存五点、旧直接LMI、それらの安全なOR、新しい固定粒子数条件。

全組合せの大規模総当たりを必須にするという意味ではない。どの変更が採否を変えたかが分かる最小比較を設計する。各下界の取得費用、trial Uの取得費用、数値proposal、厳密検証、bit長、未判定理由を分けて記録する。

有理PSDの検査が通ることと、生成器が証拠を漏れなく見つけることを区別する。数値optimizerの終了コードだけを証明にしない。保証不能は物理的縮約不可能ではない。

### 11.3 developmentと新しい検証を分ける

元の12入力は、今回の結果とレビューで既に見ている。再設計後に用いる場合はdevelopment・回帰検査であり、新しい独立移送ではない。新しい入力や判定基準を用いる次の確認的検証は、結果を見る前に固定する。元の2-base基準を事後的に緩めて今回の成功に変更しない。

最初は現在証明した外部1軌道・非負density classで成立を確かめる。これは研究全体の永久的な固定条件ではない。一般二体や多外部へ拡張する場合は、それに必要な数学と費用を別の研究課題として扱う。

### 11.4 N1・N2・既存Track

N1は今回のコードと結果を保持し、同型の候補追加scan、q拡大、分子追加を自動で開始しない。具体的に新しいframe/誤差/回路構成の機構が提示された場合に再開を検討する。N2は完了扱い。別Track A/Bのsource・結果・STOPを変更しない。

### 11.5 次の科学的判断点

強い下界＋norm-only等に対して、新しい保証・生成法固有の効果が残るか。残るなら、その効果が意味のある入力classと取得費用で説明できるか。すべて既知の下界改善や有限探索の修正へ還元される場合は、N3を保証primitiveとして整理し、独立主題として深追いしない選択肢を持つ。

この記録だけで次batchを自動認可せず、認可範囲を明示して技術作業をまとめて委ねる。通常の実装修正ごとにGPT承認を挟む必要はない。

## 12. 最終的な精度・資源との接続

N3のδは0≤a−E₀(H)≤δという縮約幅であり、任意状態のtime-evolution norm誤差ではない。active側のエネルギーをe_A±ε_Aで推定する成功事象上では

\[
E_0(H)\in[e_A-\epsilon_A-\delta,\ e_A+\epsilon_A].
\]

中心を補正して報告する場合の半幅はε_A+δ/2。軌道を減らしても、active solverに要求する精度、状態準備・overlap、PR/QPE等の費用が増える可能性がある。今回の保証検証から総量子資源削減を主張しない。

また、単一外部軌道を順に除くなら各段のδを総予算へ加算する必要があるが、今回はその逐次算法を実装・認証していない。

## 13. 監査上の事項と限界

- N3/N1ともattempt1、retry0、Mandatory STOP、central theme未採択という元状態を保持する。
- 元のglobal manifestの`relevant_commits[7].description`欠落FAILは既存不整合であり、今回の局所一次資料の欠損とは区別する。global PASSとは報告しない。
- GitHub再取得とlocal hash一致は公開証拠の保存性を支えるが、外部科学再現ではない。
- N1の上界目的関数0%は、固定有限domain・compiler・taskに対する結果。一般のN1構成の不可能性ではない。
- N3の新しい理論式は宣言classの下での十分条件。数値提案の生成コストと完全性、一般のtrial U・対角下界の効率は未解決。
- 今回の追加T3証拠は事後的な解析。独立検証用入力へ読み替えない。
- N3について、簡単な強化norm-onlyでも追加受理を得られることを確認したため、新しいσν条件だけを主成果として推し進めない。

## 14. 今回の最終判断

**N1は現構成の探索を区切る。N3は今回の移送基準未達を受け入れ、現direct generatorの優位性を主張しない。そのうえで、占有情報・正のdensity・結合方向を、どの係数証拠に残すべきかという限定的な再設計へ進む。**

前回からの判断変更は、N1の未取得結果が得られ、N3の未使用入力での利益が限定されたためである。新しい再設計を残す理由は、可能性があるからという一般論ではなく、同じ固定入力で情報損失と改善方法を有理数証拠まで具体化できたためである。ただしより安い対照も成功したので、独立の算法貢献はまだ未確定である。

## 15. 参考文献（一次資料）

これは今回の比較に必要な近接文献の確認であり、同一手順の不存在を保証する網羅的な新規性調査ではない。

- **R1** Geneviève Dusson, Israel Michael Sigal, Benjamin Stamm, *The Feshbach–Schur map and perturbation theory*, 2021. [arXiv:2105.02058](https://arxiv.org/abs/2105.02058), [DOI:10.4171/ECR/18-1/5](https://doi.org/10.4171/ECR/18-1/5). 相対formを用いた固有値評価に関連する。論文の射影条件と本研究のoccupation射影を同一と仮定しない。
- **R2** Xin Liang, Li Wang, Lei-Hong Zhang, Ren-Cang Li, *On Generalizing Trace Minimization*, arXiv 2021 / Linear Algebra and its Applications 656 (2023). [arXiv:2104.00257](https://arxiv.org/abs/2104.00257), [DOI:10.1016/j.laa.2022.10.012](https://doi.org/10.1016/j.laa.2022.10.012). Ky Fanのtrace最小化原理と一般化に関連する。
- **R3** Diana B. Chamaki, Stuart Hadfield, Katherine Klymko, Bryan O’Gorman, Norm M. Tubman, *Self-consistent Quantum Iteratively Sparsified Hamiltonian method (SQuISH): A new algorithm for efficient Hamiltonian simulation and compression*, 2022. [arXiv:2211.16522](https://arxiv.org/abs/2211.16522)。Hamiltonianと近似ground-state情報を用いた縮約との方法論上の比較に参照した。
- **R4** Priyanka Mukhopadhyay, Nathan Wiebe, Hong Tao Zhang, *Synthesizing efficient circuits for Hamiltonian simulation*, npj Quantum Information 9, 31 (2023). [DOI:10.1038/s41534-023-00697-6](https://doi.org/10.1038/s41534-023-00697-6). 対称性・固有値構造・grouping等を使う回路合成との比較。
- **R5** *Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization*, Quantum 8, 1371 (2024). [DOI:10.22331/q-2024-06-13-1371](https://doi.org/10.22331/q-2024-06-13-1371). DFの圧縮・資源関連最適化との比較。N1と同一算法と断定するための引用ではない。

---

# 付録A：再現方法と実行の区別

以下のPythonコードを同じdirectoryに保存する。NumPy/SciPyを使うスクリプトは係数小行列のproposalまたは診断を含むが、有理数の証拠判定はそれと区別される。最後の`verify_saved_rational_witnesses.py`は標準ライブラリのみを用いる。

```bash
python review_checks.py
python original_certificate_checks.py
python particle_number_certificate.py
python density_bound_countercheck.py
python verify_saved_rational_witnesses.py
```

新しい分子入力、元のscience runner、native compile、sampling、次batchは実行していない。最初のコードは同じ固定12入力のmany-body真値を診断として計算する。最後のcheckerは保存された有理数JSONだけを読み、数値固有値計算・optimizer・many-body solveを行わない。

proposal側の浮動小数点solverや環境が変われば、選ばれる有理数witness自体は変わり得る。固定された付録CのJSONを同名ファイルに保存し最後のcheckerを実行すれば、今回の証拠を数値proposalなしに検証できる。T3の入力hashは固定ledgerとの照合値である。

## 実行環境

```json
{
  "python": "3.13.5 (main, Jul 15 2026, 20:25:40) [GCC 14.2.0]",
  "platform": "Linux-6.18.44-x86_64-with-glibc2.41",
  "numpy": "2.3.5",
  "scipy": "1.17.0"
}
```

# 付録B：独立計算コード

## `review_checks.py`

```python
"""Independent post-hoc review calculations on the published 12-input recipe.
Not the repository runner. No new molecular input, native compile, or sampling.
NumPy spectra are diagnostics/proposals only; exact rational checks are separate.
"""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import json, hashlib, math
import numpy as np
from scipy.optimize import minimize_scalar

OUT=Path(__file__).resolve().parent
GOAL=F(1,250)

def enc(x):
    if isinstance(x,F):return [x.numerator,x.denominator]
    if isinstance(x,dict):return {k:enc(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [enc(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x

def sha(obj):return hashlib.sha256(json.dumps(enc(obj),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def recipe(m,nu,scale,d,hd_ratio,amp,high):
    active=[i for i in range(m+1) if i!=d];n=m+1
    h=[[F(0) for _ in range(n)] for _ in range(n)]
    for k,i in enumerate(active):h[i][i]=scale*(F(k,m-1)-F(2,5))
    for k in range(m-1):
        i,j=active[k:k+2];h[i][j]=h[j][i]=scale*F((-1)**k,25)
    h[active[0]][active[-1]]=h[active[-1]][active[0]]=scale/50
    h[d][d]=scale*hd_ratio
    dirs=([F(0)]*(m-2)+[F(3,5),F(4,5)]) if high else ([F(4,5),F(3,5)]+[F(0)]*(m-2))
    for i,z in zip(active,dirs):h[i][d]=h[d][i]=amp*z
    pairs=[(active[0],active[1],3*scale/10),(active[-2],active[-1],scale/10),(active[0],d,scale/20)]
    den=[{'i':i,'j':j,'coefficient':v} for i,j,v in pairs]
    base={4:'T1',5:'T2',7:'T3'}[m]
    direction='high_coordinates' if high else 'low_coordinates'
    return dict(id=f'{base}_amp_{amp.numerator}_{amp.denominator}_{direction}',base=base,modes=n,particles=nu,active=active,external=d,scale=scale,amplitude=amp,direction=direction,h=h,density=den)

def fermion_hop(state,i,j):
    if not (state>>j&1):return None
    p=(-1)**((state&((1<<j)-1)).bit_count());s=state^(1<<j)
    if s>>i&1:return None
    p*=(-1)**((s&((1<<i)-1)).bit_count())
    return s|(1<<i),p

def manybody(case):
    n=case['modes'];nu=case['particles'];h=case['h']
    states=[sum(1<<i for i in occ) for occ in combinations(range(n),nu)]
    idx={s:i for i,s in enumerate(states)};H=np.zeros((len(states),len(states)))
    for j,s in enumerate(states):
        for a in range(n):
            for b in range(n):
                z=fermion_hop(s,a,b)
                if z is not None:H[idx[z[0]],j]+=float(h[a][b])*z[1]
        for den in case['density']:H[j,j]+=float(den['coefficient'])*(s>>den['i']&1)*(s>>den['j']&1)
    p=[i for i,s in enumerate(states) if not (s>>case['external']&1)]
    q=[i for i,s in enumerate(states) if s>>case['external']&1]
    return H,H[np.ix_(p,p)],H[np.ix_(q,q)],H[np.ix_(p,q)]

def trial_U(case):
    active=case['active'];nu=case['particles'];h=case['h'];den=case['density']
    occs=list(combinations(active,nu));ss=[sum(1<<i for i in occ) for occ in occs]
    diag=[sum((h[i][i] for i in occ),F(0))+sum((r['coefficient'] for r in den if r['i'] in occ and r['j'] in occ),F(0)) for occ in occs]
    trials=list(diag)
    for k,l in combinations(range(len(ss)),2):
        diff1=[i for i in active if ss[k]>>i&1 and not(ss[l]>>i&1)]
        diff2=[i for i in active if ss[l]>>i&1 and not(ss[k]>>i&1)]
        if len(diff1)==len(diff2)==1:
            v=h[diff1[0]][diff2[0]]
            if v:trials.append((diag[k]+diag[l])/2-abs(v))
    return min(trials)

def rat_psd(M):
    a=[[F(x) for x in row] for row in M];n=len(a);pivs=[];bits=0
    assert all(a[i][j]==a[j][i] for i in range(n) for j in range(n))
    for k in range(n):
        p=a[k][k];pivs.append(p)
        if p<0:return {'ok':False,'pivots':pivs,'max_bits':bits}
        if not p:
            if any(a[k][j] for j in range(k+1,n)):return {'ok':False,'pivots':pivs,'max_bits':bits}
            continue
        for i in range(k+1,n):
            for j in range(i,n):
                a[i][j]-=a[i][k]*a[k][j]/p;a[j][i]=a[i][j]
                bits=max(bits,a[i][j].numerator.bit_length(),a[i][j].denominator.bit_length())
    return {'ok':True,'pivots':pivs,'max_bits':bits}

def shift(M,z):return [[M[i][j]-(z if i==j else 0) for j in range(len(M))] for i in range(len(M))]

def common_lower(A):
    n=len(A);r=min(A[i][i]-sum((abs(A[i][j]) for j in range(n) if i!=j),F(0)) for i in range(n))
    span=max(A[i][i] for i in range(n))-min(A[i][i] for i in range(n))
    vals=[r]
    for rho in [F(0),F(1,20),F(1,10),F(3,20),F(1,5)]:
        lam=r+rho*span;pr=rat_psd(shift(A,lam))
        if pr['ok'] and all(x>0 for x in pr['pivots']):vals.append(lam)
    proposal=F(float(np.linalg.eigvalsh(np.array(A,float))[0]));step=max([F(1)]+[abs(x) for row in A for x in row])/2**24
    floor=(proposal//step)*step
    for offset in (1,2,4,8,16,32):
        lam=floor-offset*step
        if rat_psd(shift(A,lam))['ok']:vals.append(lam)
    ell=max(vals);return ell

def propose(A,v,nu,U,c,particle_sum=False):
    D=c-U+GOAL
    if c<=U:return {'accepted_numeric':False,'precondition':False}
    arr=np.array(A,float);vv=np.outer(np.array(v,float),np.array(v,float));d=float(D)
    def margin(x):
        ev=np.linalg.eigvalsh(d*x*arr-vv)
        low=ev[:nu].sum() if particle_sum else nu*ev[0]
        return low-d*x*float(U)+float(GOAL*D)
    sol=minimize_scalar(lambda x:-margin(x),bounds=(0,1),method='bounded',options={'xatol':1e-13})
    x=max([0.,1.,float(sol.x)],key=margin)
    return {'accepted_numeric':margin(x)>=0,'precondition':True,'x':x,'margin':margin(x),'kappa':d*x}

REFERENCE_WIDTHS=[.0005674814315,3.971960119e-7,.009338985077,6.59557878e-6,.0005065015176,1.080399215e-5,.008366289461,.0001827274452,.001053313997,7.043636624e-7,.01685105753,1.147478621e-5]

def initial():
    cases=[recipe(m,nu,s,d,r,a,high) for m,nu,s,d,r in [(4,1,F(1,2),0,F(3,4)),(5,2,F(1),2,F(1,2)),(7,3,F(2),7,F(1,4))] for a in [F(1,40),F(1,10)] for high in [False,True]]
    assert sha(cases[0])=='58d1c06a8b389819b6277128cce6e9dad156ebd596ea92505b3105f222fe7da6'
    results=[]
    for case,reference in zip(cases,REFERENCE_WIDTHS):
        act=case['active'];d=case['external'];nu=case['particles'];h=case['h'];A=[[h[i][j] for j in act] for i in act];v=[h[i][d] for i in act]
        U=trial_U(case);ell=common_lower(A);c=h[d][d]+(nu-1)*ell
        H,AP,C,B=manybody(case);width=float(np.linalg.eigvalsh(AP)[0]-np.linalg.eigvalsh(H)[0])
        assert abs(width-reference)<5e-12,(case['id'],width,reference)
        extdiag=[sum((r['coefficient'] for r in case['density'] if {r['i'],r['j']}=={i,d}),F(0)) for i in act]
        Ad=[[A[i][j]+(extdiag[i] if i==j else 0) for j in range(len(act))] for i in range(len(act))]
        better=float(h[d][d])+np.linalg.eigvalsh(np.array(Ad,float))[:nu-1].sum()
        pn=propose(A,v,nu,U,F(float(better)),True)
        rec={'id':case['id'],'input_case_sha256':sha(case),'nu':nu,'U':U,'ell':ell,'c_frozen_rule':c,'width_diagnostic':width,
             'norm_frozen':c>U and GOAL*(c-U+GOAL)>=sum((x*x for x in v),F(0)),
             'old_direct_diagnostic':propose(A,v,nu,U,c),
             'particle_sum_same_c_diagnostic':propose(A,v,nu,U,c,True),
             'c_occupancy_aware_numeric':better,'old_direct_better_c_diagnostic':propose(A,v,nu,U,F(float(better))),
             'particle_sum_better_c_diagnostic':pn,
             'actual_C_min_diagnostic':float(np.linalg.eigvalsh(C)[0]),
             'beta2':sum((x*x for x in v),F(0))}
        results.append(rec)
    (OUT/'reconstructed_cases.json').write_text(json.dumps(enc(cases),indent=2))
    (OUT/'initial_diagnostics.json').write_text(json.dumps(enc(results),indent=2))
    for r in results:print(r['id'], 'U,c=',float(r['U']),round(float(r['c_frozen_rule']),7),'cnew=',round(r['c_occupancy_aware_numeric'],7),'old/newSame/newC=',r['old_direct_diagnostic']['accepted_numeric'],r['particle_sum_same_c_diagnostic']['accepted_numeric'],r['particle_sum_better_c_diagnostic']['accepted_numeric'])
    return cases,results

if __name__=='__main__':initial()
```

## `original_certificate_checks.py`

```python
"""Independent witnesses for the frozen N3 sufficient conditions.
Reconstructs coefficients from the public recipe; uses neither repository science
module nor stored ground states. Different numerical proposal routine, exact acceptance.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np
import review_checks as rc
from particle_number_certificate import eye,add,scale,outer,dyadic
OUT=Path(__file__).resolve().parent

def solve(A,b):
    m=len(A);a=[list(row)+[v] for row,v in zip(A,b)]
    for k in range(m):
        if not a[k][k]:
            p=next(i for i in range(k+1,m) if a[i][k]);a[p],a[k]=a[k],a[p]
        q=a[k][k];a[k]=[x/q for x in a[k]]
        for i in range(m):
            if i!=k:
                q=a[i][k];a[i]=[x-q*y for x,y in zip(a[i],a[k])]
    return [a[i][-1] for i in range(m)]

def witness(A,v,nu,U,c,k,mu):
    D=c-U+rc.GOAL;M=add(add(scale(A,k),scale(eye(len(A)),-mu)),scale(outer(v),-1))
    proof=rc.rat_psd(M);aff=rc.GOAL*D-k*U+nu*mu
    return {'accepted':c>U and 0<=k<=D and aff>=0 and proof['ok'],'kappa':k,'mu':mu,'affine_margin':aff,'PSD':proof}

def main():
    rows=[]
    expected_norm=[1,1,0,0,1,1,0,0,0,0,0,0]
    expected_five=[1,1,0,1,0,1,0,0,0,0,0,0]
    expected_direct=[1,1,0,1,1,1,0,0,0,0,0,0]
    for m,nu,s,d,hd in [(4,1,F(1,2),0,F(3,4)),(5,2,F(1),2,F(1,2)),(7,3,F(2),7,F(1,4))]:
        for amp in (F(1,40),F(1,10)):
            for high in (False,True):
                case=rc.recipe(m,nu,s,d,hd,amp,high);act=case['active'];h=case['h']
                A=[[h[i][j] for j in act] for i in act];v=[h[i][d] for i in act];U=rc.trial_U(case);c=h[d][d]+(nu-1)*rc.common_lower(A);D=c-U+rc.GOAL
                norm=c>U and rc.GOAL*D>=sum((x*x for x in v),F(0))
                r=min(A[i][i]-sum((abs(A[i][j]) for j in range(m) if i!=j),F(0)) for i in range(m));span=max(A[i][i] for i in range(m))-min(A[i][i] for i in range(m))
                old=[]
                for rho in (F(0),F(1,20),F(1,10),F(3,20),F(1,5)):
                    lam=r+rho*span;M=rc.shift(A,lam);pr=rc.rat_psd(M)
                    if pr['ok'] and all(x>0 for x in pr['pivots']):
                        x=solve(M,v);k=sum((a*b for a,b in zip(v,x)),F(0));old.append(witness(A,v,nu,U,c,k,k*lam))
                five=any(w['accepted'] for w in old)
                prop=rc.propose(A,v,nu,U,c);direct=False;new=None
                if prop['precondition']:
                    k=dyadic(prop['x'])*D;mu=(k*U-rc.GOAL*D)/nu;new=witness(A,v,nu,U,c,k,mu);direct=new['accepted']
                rows.append({'id':case['id'],'case_sha256':rc.sha(case),'U':U,'c':c,'norm':norm,'five':five,'direct':direct,'direct_witness':new,'five_witnesses':old})
    assert [int(x['norm']) for x in rows]==expected_norm
    assert [int(x['five']) for x in rows]==expected_five
    assert [int(x['direct']) for x in rows]==expected_direct
    direct={x['id'] for x in rows if x['direct']};union={x['id'] for x in rows if x['norm'] or x['five']}
    assert direct==union
    n1=[('planted',16448,2494,1664,41021312,27369472),('dense',16933,2614,1696,44262862,28718368),('separated',13631,5222,3392,71181082,46236352)]
    for _,n,rz,cx,grz,gcx in n1:assert n*rz==grz and n*cx==gcx
    assert 2*(9*9*2*4)+(4*4*2*4)==1424
    output={'rows':rows,'counts':{k:sum(x[k] for x in rows) for k in ('norm','five','direct')},'union_equals_direct':True,'N1_transcribed_selected_products_pass':True,'N1_finite_row_count':1424,'qualification':'No all-row N1 argmin audit, native recompilation, or 72-ledger witness replay is claimed.'}
    (OUT/'original_reproduction.json').write_text(json.dumps(rc.enc(output),indent=2))
    print(output['counts'],'union equals direct',direct==union,'N1 selected arithmetic PASS')

if __name__=='__main__':main()
```

## `particle_number_certificate.py`

```python
"""Post-hoc coefficient certificates, NOT a registered transfer experiment.
Uses the same published T3 input and retains U and delta. The external bound
and sufficient family are explicitly changed and must not replace frozen results.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np
import review_checks as rc

OUT=Path(__file__).resolve().parent

def eye(n):return [[F(i==j) for j in range(n)] for i in range(n)]
def add(A,B):return [[x+y for x,y in zip(a,b)] for a,b in zip(A,B)]
def scale(A,z):return [[z*x for x in a] for a in A]
def trace(A):return sum((A[i][i] for i in range(len(A))),F(0))
def outer(v):return [[x*y for y in v] for x in v]
def quad(A,v):return sum((v[i]*A[i][j]*v[j] for i in range(len(v)) for j in range(len(v))),F(0))
def dyadic(x,bits=40):return F(round(float(x)*2**bits),2**bits)

def spectral_lower(G,k):
    """Return exact witness sigma_k(G)>=k*tau-trace(Q).
    Acceptance uses Q>=0 and G-tau I+Q>=0, verified by rational LDL.
    """
    m=len(G)
    if k==0:return {'k':0,'lower':F(0),'trivial_empty_sum':True}
    if k==m:
        tau=max(G[i][i]+sum(abs(G[i][j]) for j in range(m) if j!=i) for i in range(m))+1
        Q=add(scale(eye(m),tau),scale(G,-1))
    else:
        w,V=np.linalg.eigh(np.array(G,float));tau=dyadic((w[k-1]+w[k])/2)
        Qn=(V*np.maximum(float(tau)-w,0))@V.T
        Q=[[dyadic((Qn[i,j]+Qn[j,i])/2) for j in range(m)] for i in range(m)]
        Q=add(Q,scale(eye(m),F(4*m,2**40)))
    R=add(add(G,scale(eye(m),-tau)),Q)
    p1,p2=rc.rat_psd(Q),rc.rat_psd(R)
    if not(p1['ok'] and p2['ok']):raise ArithmeticError('Rational spectral certificate failed')
    return {'k':k,'tau':tau,'Q':Q,'lower':k*tau-trace(Q),'Q_psd':p1,'residual_psd':p2}

def check_spectral(G,cert):
    k=cert['k']
    if k==0:assert cert['lower']==0;return
    m=len(G);Q=cert['Q'];tau=cert['tau']
    assert rc.rat_psd(Q)['ok']
    assert rc.rat_psd(add(add(G,scale(eye(m),-tau)),Q))['ok']
    assert cert['lower']==k*tau-trace(Q)

def old_family_separation(A,v,nu,U,c):
    D=c-U+rc.GOAL;m=len(A);R=add(A,scale(eye(m),-U/nu));M0=add(scale(eye(m),rc.GOAL*D/nu),scale(outer(v),-1))
    opt=rc.propose(A,v,nu,U,c);x=opt['x'];vs=[]
    for xx in [max(0,x-1e-5),min(1,x+1e-5)]:
        M=np.array(add(M0,scale(R,D*F(float(xx)))),float)
        _,V=np.linalg.eigh(M);z=[dyadic(t) for t in V[:,0]];vs.append(z)
    s1,s2=[quad(R,z) for z in vs]
    if s1==s2:raise ArithmeticError('No slope bracket')
    t=-s2/(s1-s2)
    if not 0<=t<=1:raise ArithmeticError('No positive convex separator')
    intercept=t*quad(M0,vs[0])+(1-t)*quad(M0,vs[1]);norm=t*sum(z*z for z in vs[0])+(1-t)*sum(z*z for z in vs[1])
    assert t*s1+(1-t)*s2==0 and intercept<0 and norm>0
    return {'u':vs[0],'v':vs[1],'weight':t,'slope_exact':F(0),'trace_W':norm,'intercept':intercept,'normalized_intercept':intercept/norm,'meaning':'nonzero PSD W has negative trace(W M(kappa)) for every kappa; original one-body LMI is infeasible at this common U,c,delta'}

def main():
    case=rc.recipe(7,3,F(2),7,F(1,4),F(1,40),True)
    act=case['active'];d=case['external'];nu=case['particles'];h=case['h'];A=[[h[i][j] for j in act] for i in act];v=[h[i][d] for i in act]
    ext=[sum((r['coefficient'] for r in case['density'] if {r['i'],r['j']}=={i,d}),F(0)) for i in act]
    Gext=add(A,[[ext[i] if i==j else F(0) for j in range(len(act))] for i in range(len(act))])
    extcert=spectral_lower(Gext,nu-1);check_spectral(Gext,extcert)
    U=rc.trial_U(case);c=h[d][d]+extcert['lower'];D=c-U+rc.GOAL
    assert c>U
    opt=rc.propose(A,v,nu,U,c,True);x=dyadic(opt['x']);kappa=D*x
    K=add(scale(A,kappa),scale(outer(v),-1));kcert=spectral_lower(K,nu);check_spectral(K,kcert)
    margin=kcert['lower']-kappa*U+rc.GOAL*D
    assert 0<=kappa<=D and margin>0
    normmargin=rc.GOAL*D-sum(z*z for z in v);assert normmargin<0
    sep=old_family_separation(A,v,nu,U,c)
    H,AP,C,B=rc.manybody(case)
    record={'classification':'post_hoc_review_certificate_not_frozen_transfer_result','case':case['id'],'input_case_sha256':rc.sha(case),'delta':rc.GOAL,
            'U':U,'c_frozen_rule':h[d][d]+(nu-1)*rc.common_lower(A),'c_occupancy_certificate':c,'D':D,'external_bound_witness':extcert,
            'kappa':kappa,'x':x,'active_spectral_witness':kcert,'affine_margin':margin,'norm_only_same_c_margin':normmargin,
            'original_one_body_family_infeasibility_witness':sep,'width_diagnostic':float(np.linalg.eigvalsh(AP)[0]-np.linalg.eigvalsh(H)[0]),
            'actual_C_min_diagnostic':float(np.linalg.eigvalsh(C)[0]),'certificate_uses_many_body_eigenvalues':False,
            'reconstructed_input':case}
    (OUT/'posthoc_particle_certificate.json').write_text(json.dumps(rc.enc(record),indent=2))
    print(json.dumps({k:float(record[k]) for k in ['U','c_frozen_rule','c_occupancy_certificate','D','kappa','affine_margin','norm_only_same_c_margin','width_diagnostic']},indent=2))
    print('old-family rational separator:',float(sep['normalized_intercept']))
    print('exact PSD/affine checks PASS; certificate file bytes', (OUT/'posthoc_particle_certificate.json').stat().st_size)

if __name__=='__main__':main()
```

## `density_bound_countercheck.py`

```python
"""Post-hoc stronger norm-only control for the SAME frozen T3 input.

No many-body eigensolve, ground state, or true gap is used. The external
sector has two particles, so its diagonal energy minimum is obtained by
21 coefficient-only pair evaluations. This is not asserted to scale
polynomially for arbitrary variable particle number.
"""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import json
import review_checks as rc
from particle_number_certificate import spectral_lower, check_spectral

OUT = Path(__file__).resolve().parent

def main():
    case = rc.recipe(7, 3, F(2), 7, F(1, 4), F(1, 40), True)
    assert rc.sha(case) == 'b34f66cbeed196090dfd77ab999cb462db821ce12e9d21408020771f47a81f16'
    h, d, active = case['h'], case['external'], case['active']
    p = case['particles'] - 1
    diagonal_rows = []
    for occ in combinations(active, p):
        occupied = set(occ) | {d}
        energy = sum((h[i][i] for i in occupied), F(0))
        energy += sum((r['coefficient'] for r in case['density']
                       if r['i'] in occupied and r['j'] in occupied), F(0))
        diagonal_rows.append({'occupied_active': list(occ), 'energy': energy})
    dmin = min(r['energy'] for r in diagonal_rows)
    O = [[F(0) if i == j else h[i][j] for j in active] for i in active]
    spec = spectral_lower(O, p)
    check_spectral(O, spec)
    c = dmin + spec['lower']
    U = rc.trial_U(case)
    delta = rc.GOAL
    beta2 = sum((h[i][d] ** 2 for i in active), F(0))
    margin = delta * (c - U + delta) - beta2
    assert len(diagonal_rows) == 21
    assert dmin == F(-1, 3) and c > U and margin > 0
    record = {
        'classification': 'post_hoc_stronger_control_not_frozen_transfer_result',
        'case': case['id'], 'input_case_sha256': rc.sha(case),
        'U': U, 'delta': delta, 'external_particle_number': p,
        'diagonal_candidates': diagonal_rows, 'diagonal_minimum': dmin,
        'offdiagonal_one_body_spectral_witness': spec,
        'c_diagonal_plus_hopping': c, 'beta_squared': beta2,
        'norm_only_affine_margin': margin,
        'true_energy_or_gap_used': False,
        'conclusion': 'A stronger coefficient-only external bound allows norm-only to certify this same input. The new directional spectral family is therefore not indispensable for accepting this input.',
        'limitation': 'Exact diagonal minimization is polynomial here because external particle number equals two; no polynomial method for general particle number is claimed.'
    }
    (OUT / 'density_bound_countercheck.json').write_text(json.dumps(rc.enc(record), indent=2), encoding='utf-8')
    print(json.dumps({k: float(record[k]) for k in ['U', 'delta', 'diagonal_minimum', 'c_diagonal_plus_hopping', 'beta_squared', 'norm_only_affine_margin']}, indent=2))
    print('Coefficient-only stronger norm-control exact checks PASS; no many-body truth used.')

if __name__ == '__main__':
    main()
```

## `verify_saved_rational_witnesses.py`

```python
"""Standard-library-only verification of saved post-hoc coefficient witnesses.

No numerical eigensolver, optimizer, many-body diagonalization, original science
module, or quantum circuit library is imported. The mathematical energy-error
implication is proved in the accompanying review; this checks its finite witnesses.
"""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent

def scalar(x):
    if not isinstance(x, list) or len(x) != 2 or any(type(z) is not int for z in x):
        raise ValueError('Expected [integer numerator, integer denominator]')
    return F(x[0], x[1])

def matrix(x):
    a = [[scalar(z) for z in r] for r in x]
    if not a or any(len(r) != len(a) for r in a):
        raise ValueError('Expected nonempty square matrix')
    return a

def psd(a):
    n = len(a)
    assert all(len(r) == n for r in a)
    assert all(a[i][j] == a[j][i] for i in range(n) for j in range(n))
    b = [r[:] for r in a]
    for k in range(n):
        p = b[k][k]
        assert p >= 0
        if p == 0:
            assert all(b[k][j] == 0 for j in range(k + 1, n))
        else:
            for i in range(k + 1, n):
                for j in range(i, n):
                    b[i][j] -= b[i][k] * b[k][j] / p
                    b[j][i] = b[i][j]

def add(a, b): return [[x + y for x, y in zip(r, s)] for r, s in zip(a, b)]
def scale(a, x): return [[x * z for z in r] for r in a]
def eye(n): return [[F(i == j) for j in range(n)] for i in range(n)]
def outer(v): return [[x * y for y in v] for x in v]
def quad(a, v): return sum((v[i] * a[i][j] * v[j] for i in range(len(v)) for j in range(len(v))), F(0))

def check_spectral(g, cert, k):
    assert cert['k'] == k
    if k == 0:
        assert scalar(cert['lower']) == 0
        return F(0)
    q = matrix(cert['Q']); tau = scalar(cert['tau'])
    assert len(q) == len(g)
    psd(q); psd(add(add(g, scale(eye(len(g)), -tau)), q))
    lower = k * tau - sum((q[i][i] for i in range(len(q))), F(0))
    assert lower == scalar(cert['lower'])
    return lower

def main():
    p = json.loads((ROOT / 'posthoc_particle_certificate.json').read_text())
    b = json.loads((ROOT / 'density_bound_countercheck.json').read_text())
    case = p['reconstructed_input']
    digest = hashlib.sha256(json.dumps(case, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    expected = 'b34f66cbeed196090dfd77ab999cb462db821ce12e9d21408020771f47a81f16'
    assert digest == expected == p['input_case_sha256'] == b['input_case_sha256']
    h = matrix(case['h']); active = case['active']; d = case['external']; nu = case['particles']; m = len(active)
    assert (len(h), m, nu, d) == (8, 7, 3, 7)
    assert all(h[i][j] == h[j][i] for i in range(len(h)) for j in range(len(h)))
    density = [(r['i'], r['j'], scalar(r['coefficient'])) for r in case['density']]
    assert all(i != j and x >= 0 for i, j, x in density)
    a = [[h[i][j] for j in active] for i in active]; v = [h[i][d] for i in active]
    U = scalar(p['U']); delta = scalar(p['delta'])
    assert U == scalar(b['U']) and delta == scalar(b['delta']) and delta > 0
    def diagonal(occ):
        s = set(occ)
        return sum((h[i][i] for i in s), F(0)) + sum((x for i, j, x in density if i in s and j in s), F(0))
    assert min(diagonal(occ) for occ in combinations(active, nu)) <= U
    ext = [sum((x for i, j, x in density if {i, j} == {r, d}), F(0)) for r in active]
    gext = add(a, [[ext[i] if i == j else F(0) for j in range(m)] for i in range(m)])
    c = h[d][d] + check_spectral(gext, p['external_bound_witness'], nu - 1)
    assert c == scalar(p['c_occupancy_certificate']) and c > U
    D = c - U + delta; kappa = scalar(p['kappa'])
    assert D == scalar(p['D']) and 0 <= kappa <= D
    K = add(scale(a, kappa), scale(outer(v), -1))
    low = check_spectral(K, p['active_spectral_witness'], nu)
    margin = low - kappa * U + delta * D
    assert margin == scalar(p['affine_margin']) and margin > 0
    beta2 = sum((x*x for x in v), F(0))
    assert delta * D - beta2 == scalar(p['norm_only_same_c_margin']) < 0
    sep = p['original_one_body_family_infeasibility_witness']
    z1, z2 = [scalar(x) for x in sep['u']], [scalar(x) for x in sep['v']]
    t = scalar(sep['weight']); assert len(z1) == len(z2) == m and 0 <= t <= 1
    R = add(a, scale(eye(m), -U / nu))
    M0 = add(scale(eye(m), delta * D / nu), scale(outer(v), -1))
    slope = t * quad(R, z1) + (1 - t) * quad(R, z2)
    intercept = t * quad(M0, z1) + (1 - t) * quad(M0, z2)
    trW = t * sum((x*x for x in z1), F(0)) + (1-t) * sum((x*x for x in z2), F(0))
    assert slope == scalar(sep['slope_exact']) == 0
    assert intercept == scalar(sep['intercept']) < 0 and trW == scalar(sep['trace_W']) > 0
    assert intercept / trW == scalar(sep['normalized_intercept'])
    rows = list(combinations(active, nu - 1))
    expected_rows = {tuple(r['occupied_active']): scalar(r['energy']) for r in b['diagonal_candidates']}
    actual_rows = {occ: diagonal((*occ, d)) for occ in rows}
    assert expected_rows == actual_rows and len(rows) == 21
    diag_min = min(actual_rows.values()); assert diag_min == scalar(b['diagonal_minimum'])
    O = [[F(0) if i == j else a[i][j] for j in range(m)] for i in range(m)]
    c2 = diag_min + check_spectral(O, b['offdiagonal_one_body_spectral_witness'], nu - 1)
    margin2 = delta * (c2 - U + delta) - beta2
    assert c2 == scalar(b['c_diagonal_plus_hopping']) > U
    assert beta2 == scalar(b['beta_squared'])
    assert margin2 == scalar(b['norm_only_affine_margin']) > 0
    result = {'status': 'PASS', 'case_sha256': digest,
        'coefficient_trial_U_verified': True,
        'new_particle_spectral_certificate_verified': True,
        'old_LMI_infeasible_at_same_occupancy_bound_verified': True,
        'stronger_density_bound_norm_only_verified': True,
        'numerical_eigensolver_calls': 0, 'many_body_solves': 0,
        'registered_transfer_status_changed': False,
        'qualification': 'Exact finite-witness verification, not independent external scientific reproduction.'}
    (ROOT / 'saved_witness_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
```

# 付録C：固定した有理数証拠と検証結果

分数は`[分子, 分母]`。行列の各要素も同形式。数値診断のfloatと混同しない。

## `posthoc_particle_certificate.json`

```json
{
  "classification": "post_hoc_review_certificate_not_frozen_transfer_result",
  "case": "T3_amp_1_40_high_coordinates",
  "input_case_sha256": "b34f66cbeed196090dfd77ab999cb462db821ce12e9d21408020771f47a81f16",
  "delta": [
    1,
    250
  ],
  "U": [
    -4,
    5
  ],
  "c_frozen_rule": [
    -23883779,
    20971520
  ],
  "c_occupancy_certificate": [
    -755142843825,
    1099511627776
  ],
  "D": [
    16108063113363,
    137438953472000
  ],
  "external_bound_witness": {
    "k": 2,
    "tau": [
      -2549090763,
      8589934592
    ],
    "Q": [
      [
        [
          222152988013,
          549755813888
        ],
        [
          -10812275899,
          137438953472
        ],
        [
          -2957944951,
          549755813888
        ],
        [
          7555315,
          137438953472
        ],
        [
          7284049,
          1099511627776
        ],
        [
          -470243671,
          1099511627776
        ],
        [
          -9300867601,
          1099511627776
        ]
      ],
      [
        [
          -10812275899,
          137438953472
        ],
        [
          197120335265,
          1099511627776
        ],
        [
          44614269863,
          1099511627776
        ],
        [
          -5245183171,
          1099511627776
        ],
        [
          -418523033,
          1099511627776
        ],
        [
          47146273,
          549755813888
        ],
        [
          815093019,
          549755813888
        ]
      ],
      [
        [
          -2957944951,
          549755813888
        ],
        [
          44614269863,
          1099511627776
        ],
        [
          10556837437,
          1099511627776
        ],
        [
          -1262490617,
          1099511627776
        ],
        [
          -25163439,
          274877906944
        ],
        [
          6922959,
          1099511627776
        ],
        [
          80319421,
          1099511627776
        ]
      ],
      [
        [
          7555315,
          137438953472
        ],
        [
          -5245183171,
          1099511627776
        ],
        [
          -1262490617,
          1099511627776
        ],
        [
          151930451,
          1099511627776
        ],
        [
          1513655,
          137438953472
        ],
        [
          -143649,
          1099511627776
        ],
        [
          1987573,
          549755813888
        ]
      ],
      [
        [
          7284049,
          1099511627776
        ],
        [
          -418523033,
          1099511627776
        ],
        [
          -25163439,
          274877906944
        ],
        [
          1513655,
          137438953472
        ],
        [
          965179,
          1099511627776
        ],
        [
          -3515,
          274877906944
        ],
        [
          265183,
          1099511627776
        ]
      ],
      [
        [
          -470243671,
          1099511627776
        ],
        [
          47146273,
          549755813888
        ],
        [
          6922959,
          1099511627776
        ],
        [
          -143649,
          1099511627776
        ],
        [
          -3515,
          274877906944
        ],
        [
          497765,
          1099511627776
        ],
        [
          9841085,
          1099511627776
        ]
      ],
      [
        [
          -9300867601,
          1099511627776
        ],
        [
          815093019,
          549755813888
        ],
        [
          80319421,
          1099511627776
        ],
        [
          1987573,
          549755813888
        ],
        [
          265183,
          1099511627776
        ],
        [
          9841085,
          1099511627776
        ],
        [
          97440131,
          549755813888
        ]
      ]
    ],
    "lower": [
      -1304898657713,
      1099511627776
    ],
    "Q_psd": {
      "ok": true,
      "pivots": [
        [
          222152988013,
          549755813888
        ],
        [
          40049901553529005916013,
          244259793465475845849088
        ],
        [
          66723708161957891229997,
          2446407358299401582885921531887616
        ],
        [
          8416767063286699548698437,
          330135718375822941114580603213185024
        ],
        [
          237586510063907452974525201,
          9254333254365782229360063048016986112
        ],
        [
          81072894582526143410637461,
          3225050992814640609963436750768439296
        ],
        [
          82705359685967514016879328461,
          3209061250474033388670202000964893802496
        ]
      ],
      "max_bits": 132
    },
    "residual_psd": {
      "ok": true,
      "pivots": [
        [
          2328635617,
          2748779069440
        ],
        [
          6987232224244794410519,
          960135726654315036672000
        ],
        [
          80828844307083783660968230313837,
          11523814614792472431906318100463616
        ],
        [
          145758209894411363378514301694984650633023693,
          27772579429792050556463816071434058501980160000
        ],
        [
          252271899937519634816263876960069573774339055581558271,
          267104744371200178722769491135247548454750915852008161280
        ],
        [
          1932164074732828554618051575348593555706105385289393333534029767159,
          312047873260252207169390300289504014566688311988635881983031902208000
        ],
        [
          36325764137228646718070053431951261878462614939321286508174923605070316301,
          637331060081940370917848162869515964519993099572282152694457186793349107102515200
        ]
      ],
      "max_bits": 269
    }
  },
  "kappa": [
    85424783244841250132949,
    151115727451828646838272000
  ],
  "x": [
    5303231223,
    1099511627776
  ],
  "active_spectral_witness": {
    "k": 3,
    "tau": [
      -83118057,
      1099511627776
    ],
    "Q": [
      [
        [
          414300711,
          1099511627776
        ],
        [
          -49693449,
          1099511627776
        ],
        [
          -128215,
          1099511627776
        ],
        [
          -103329,
          274877906944
        ],
        [
          429353,
          1099511627776
        ],
        [
          -3993697,
          549755813888
        ],
        [
          -8268057,
          549755813888
        ]
      ],
      [
        [
          -49693449,
          1099511627776
        ],
        [
          207153929,
          1099511627776
        ],
        [
          24070435,
          549755813888
        ],
        [
          -5687697,
          1099511627776
        ],
        [
          -94327,
          274877906944
        ],
        [
          -95599,
          274877906944
        ],
        [
          109479,
          274877906944
        ]
      ],
      [
        [
          -128215,
          1099511627776
        ],
        [
          24070435,
          549755813888
        ],
        [
          6063227,
          549755813888
        ],
        [
          -1534015,
          1099511627776
        ],
        [
          -39899,
          274877906944
        ],
        [
          203579,
          1099511627776
        ],
        [
          17701,
          137438953472
        ]
      ],
      [
        [
          -103329,
          274877906944
        ],
        [
          -5687697,
          1099511627776
        ],
        [
          -1534015,
          1099511627776
        ],
        [
          206623,
          1099511627776
        ],
        [
          30319,
          1099511627776
        ],
        [
          -45557,
          549755813888
        ],
        [
          -36321,
          549755813888
        ]
      ],
      [
        [
          429353,
          1099511627776
        ],
        [
          -94327,
          274877906944
        ],
        [
          -39899,
          274877906944
        ],
        [
          30319,
          1099511627776
        ],
        [
          11813,
          1099511627776
        ],
        [
          -18825,
          274877906944
        ],
        [
          -81763,
          1099511627776
        ]
      ],
      [
        [
          -3993697,
          549755813888
        ],
        [
          -95599,
          274877906944
        ],
        [
          203579,
          1099511627776
        ],
        [
          -45557,
          549755813888
        ],
        [
          -18825,
          274877906944
        ],
        [
          151437,
          274877906944
        ],
        [
          758689,
          1099511627776
        ]
      ],
      [
        [
          -8268057,
          549755813888
        ],
        [
          109479,
          274877906944
        ],
        [
          17701,
          137438953472
        ],
        [
          -36321,
          549755813888
        ],
        [
          -81763,
          1099511627776
        ],
        [
          758689,
          1099511627776
        ],
        [
          1088685,
          1099511627776
        ]
      ]
    ],
    "lower": [
      -442424067,
      549755813888
    ],
    "Q_psd": {
      "ok": true,
      "pivots": [
        [
          414300711,
          1099511627776
        ],
        [
          13892430199604653,
          75921408190060691456
        ],
        [
          51245679399775880809747,
          91649331255190635670161850368
        ],
        [
          193005564384868642774685,
          7043152546666826169838635617091584
        ],
        [
          1098735449928434858480961,
          42442372453326498712226753111130112
        ],
        [
          2145143533927410281044837451,
          48322896121840366158255265112590909440
        ],
        [
          42344804270005675745011561173,
          1179305129400843980167903431895168319488
        ]
      ],
      "max_bits": 130
    },
    "residual_psd": {
      "ok": true,
      "pivots": [
        [
          31110394223202987051,
          188894659314785808547840000
        ],
        [
          3141441900084404677472138947649861976809,
          16323853660946135351591843353061379538944000000
        ],
        [
          14736421016516148589978782343089121619550093743733226721827903,
          21362457509050932766556524571624151988930922436112753633693532160000
        ],
        [
          730013885036625250410825327933075287913768652993654440984526034898055074194329395151,
          278363122743406746068279097656577918119386117499881049224050166051467561008682237952000000
        ],
        [
          585088298200392921778607974065626497379611534345016729565561834635870573653740441318151761120112722569163,
          137895724109056540239546456050720191317142773197749950281965820572836205344540757948036818026380232847523840000
        ],
        [
          296251505519548592576248360938549231720470157410524504888176914338389373780220203843211886479747813700092328363140705216737237,
          5526002737881551382313711592073526367128499442426622168497212293117199869673499226761665977005465066690878901873668238549712896000000
        ],
        [
          402018348431569020287318788086299881870766418455868881598420876126462672237272618081763076733899437367230174587417684275954928170835447963419189,
          4476826176528601514168508314560225595539116321217567128647802110114298860346700923603029166224373531925708775644285375742899756441650657493585913446400000
        ]
      ],
      "max_bits": 511
    }
  },
  "affine_margin": [
    549101120425942407770061,
    4722366482869645213696000000
  ],
  "norm_only_same_c_margin": [
    -5366773366637,
    34359738368000000
  ],
  "original_one_body_family_infeasibility_witness": {
    "u": [
      [
        853483783869,
        1099511627776
      ],
      [
        -187042137769,
        1099511627776
      ],
      [
        -5389813729,
        274877906944
      ],
      [
        1824595991,
        549755813888
      ],
      [
        12961383681,
        549755813888
      ],
      [
        -441660410031,
        1099511627776
      ],
      [
        -31205747621,
        68719476736
      ]
    ],
    "v": [
      [
        453975653521,
        549755813888
      ],
      [
        -100553828477,
        549755813888
      ],
      [
        -23264395385,
        1099511627776
      ],
      [
        221972983,
        68719476736
      ],
      [
        22777301777,
        1099511627776
      ],
      [
        -193316746835,
        549755813888
      ],
      [
        -439940728213,
        1099511627776
      ]
    ],
    "weight": [
      3170465984752480124959292,
      6710136372626645792499711
    ],
    "slope_exact": [
      0,
      1
    ],
    "trace_W": [
      2704019038000566834366212393430202718139543735443,
      2704019038001200842264675318282991995444329971712
    ],
    "intercept": [
      -4139958590514423934262742632027339894688889235986594995115997,
      92909386687812310610035480544670731947705425737085161046016000000
    ],
    "normalized_intercept": [
      -4139958590514423934262742632027339894688889235986594995115997,
      92909386687790526264521048937380847464477170464714888577024000000
    ],
    "meaning": "nonzero PSD W has negative trace(W M(kappa)) for every kappa; original one-body LMI is infeasible at this common U,c,delta"
  },
  "width_diagnostic": 7.043636623960126e-07,
  "actual_C_min_diagnostic": -0.39467257538355094,
  "certificate_uses_many_body_eigenvalues": false,
  "reconstructed_input": {
    "id": "T3_amp_1_40_high_coordinates",
    "base": "T3",
    "modes": 8,
    "particles": 3,
    "active": [
      0,
      1,
      2,
      3,
      4,
      5,
      6
    ],
    "external": 7,
    "scale": [
      2,
      1
    ],
    "amplitude": [
      1,
      40
    ],
    "direction": "high_coordinates",
    "h": [
      [
        [
          -4,
          5
        ],
        [
          2,
          25
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          1,
          25
        ],
        [
          0,
          1
        ]
      ],
      [
        [
          2,
          25
        ],
        [
          -7,
          15
        ],
        [
          -2,
          25
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ]
      ],
      [
        [
          0,
          1
        ],
        [
          -2,
          25
        ],
        [
          -2,
          15
        ],
        [
          2,
          25
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ]
      ],
      [
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          2,
          25
        ],
        [
          1,
          5
        ],
        [
          -2,
          25
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ]
      ],
      [
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          -2,
          25
        ],
        [
          8,
          15
        ],
        [
          2,
          25
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ]
      ],
      [
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          2,
          25
        ],
        [
          13,
          15
        ],
        [
          -2,
          25
        ],
        [
          3,
          200
        ]
      ],
      [
        [
          1,
          25
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          -2,
          25
        ],
        [
          6,
          5
        ],
        [
          1,
          50
        ]
      ],
      [
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          0,
          1
        ],
        [
          3,
          200
        ],
        [
          1,
          50
        ],
        [
          1,
          2
        ]
      ]
    ],
    "density": [
      {
        "i": 0,
        "j": 1,
        "coefficient": [
          3,
          5
        ]
      },
      {
        "i": 5,
        "j": 6,
        "coefficient": [
          1,
          5
        ]
      },
      {
        "i": 0,
        "j": 7,
        "coefficient": [
          1,
          10
        ]
      }
    ]
  }
}
```

## `density_bound_countercheck.json`

```json
{
  "classification": "post_hoc_stronger_control_not_frozen_transfer_result",
  "case": "T3_amp_1_40_high_coordinates",
  "input_case_sha256": "b34f66cbeed196090dfd77ab999cb462db821ce12e9d21408020771f47a81f16",
  "U": [
    -4,
    5
  ],
  "delta": [
    1,
    250
  ],
  "external_particle_number": 2,
  "diagonal_candidates": [
    {
      "occupied_active": [
        0,
        1
      ],
      "energy": [
        -1,
        15
      ]
    },
    {
      "occupied_active": [
        0,
        2
      ],
      "energy": [
        -1,
        3
      ]
    },
    {
      "occupied_active": [
        0,
        3
      ],
      "energy": [
        0,
        1
      ]
    },
    {
      "occupied_active": [
        0,
        4
      ],
      "energy": [
        1,
        3
      ]
    },
    {
      "occupied_active": [
        0,
        5
      ],
      "energy": [
        2,
        3
      ]
    },
    {
      "occupied_active": [
        0,
        6
      ],
      "energy": [
        1,
        1
      ]
    },
    {
      "occupied_active": [
        1,
        2
      ],
      "energy": [
        -1,
        10
      ]
    },
    {
      "occupied_active": [
        1,
        3
      ],
      "energy": [
        7,
        30
      ]
    },
    {
      "occupied_active": [
        1,
        4
      ],
      "energy": [
        17,
        30
      ]
    },
    {
      "occupied_active": [
        1,
        5
      ],
      "energy": [
        9,
        10
      ]
    },
    {
      "occupied_active": [
        1,
        6
      ],
      "energy": [
        37,
        30
      ]
    },
    {
      "occupied_active": [
        2,
        3
      ],
      "energy": [
        17,
        30
      ]
    },
    {
      "occupied_active": [
        2,
        4
      ],
      "energy": [
        9,
        10
      ]
    },
    {
      "occupied_active": [
        2,
        5
      ],
      "energy": [
        37,
        30
      ]
    },
    {
      "occupied_active": [
        2,
        6
      ],
      "energy": [
        47,
        30
      ]
    },
    {
      "occupied_active": [
        3,
        4
      ],
      "energy": [
        37,
        30
      ]
    },
    {
      "occupied_active": [
        3,
        5
      ],
      "energy": [
        47,
        30
      ]
    },
    {
      "occupied_active": [
        3,
        6
      ],
      "energy": [
        19,
        10
      ]
    },
    {
      "occupied_active": [
        4,
        5
      ],
      "energy": [
        19,
        10
      ]
    },
    {
      "occupied_active": [
        4,
        6
      ],
      "energy": [
        67,
        30
      ]
    },
    {
      "occupied_active": [
        5,
        6
      ],
      "energy": [
        83,
        30
      ]
    }
  ],
  "diagonal_minimum": [
    -1,
    3
  ],
  "offdiagonal_one_body_spectral_witness": {
    "k": 2,
    "tau": [
      -101731171375,
      1099511627776
    ],
    "Q": [
      [
        [
          5634429401,
          1099511627776
        ],
        [
          -4170267613,
          549755813888
        ],
        [
          -4510059319,
          549755813888
        ],
        [
          7885505491,
          1099511627776
        ],
        [
          2975184453,
          549755813888
        ],
        [
          -2154824909,
          549755813888
        ],
        [
          -426403421,
          137438953472
        ]
      ],
      [
        [
          -4170267613,
          549755813888
        ],
        [
          12499723129,
          1099511627776
        ],
        [
          13250856263,
          1099511627776
        ],
        [
          -11027734799,
          1099511627776
        ],
        [
          -3842547995,
          549755813888
        ],
        [
          2595664331,
          549755813888
        ],
        [
          2154824909,
          549755813888
        ]
      ],
      [
        [
          -4510059319,
          549755813888
        ],
        [
          13250856263,
          1099511627776
        ],
        [
          7253669645,
          549755813888
        ],
        [
          -6525223381,
          549755813888
        ],
        [
          -5134347277,
          549755813888
        ],
        [
          3842547995,
          549755813888
        ],
        [
          2975184453,
          549755813888
        ]
      ],
      [
        [
          7885505491,
          1099511627776
        ],
        [
          -11027734799,
          1099511627776
        ],
        [
          -6525223381,
          549755813888
        ],
        [
          6874149523,
          549755813888
        ],
        [
          6525223381,
          549755813888
        ],
        [
          -11027734799,
          1099511627776
        ],
        [
          -7885505491,
          1099511627776
        ]
      ],
      [
        [
          2975184453,
          549755813888
        ],
        [
          -3842547995,
          549755813888
        ],
        [
          -5134347277,
          549755813888
        ],
        [
          6525223381,
          549755813888
        ],
        [
          7253669645,
          549755813888
        ],
        [
          -13250856263,
          1099511627776
        ],
        [
          -4510059319,
          549755813888
        ]
      ],
      [
        [
          -2154824909,
          549755813888
        ],
        [
          2595664331,
          549755813888
        ],
        [
          3842547995,
          549755813888
        ],
        [
          -11027734799,
          1099511627776
        ],
        [
          -13250856263,
          1099511627776
        ],
        [
          12499723129,
          1099511627776
        ],
        [
          4170267613,
          549755813888
        ]
      ],
      [
        [
          -426403421,
          137438953472
        ],
        [
          2154824909,
          549755813888
        ],
        [
          2975184453,
          549755813888
        ],
        [
          -7885505491,
          1099511627776
        ],
        [
          -4510059319,
          549755813888
        ],
        [
          4170267613,
          549755813888
        ],
        [
          5634429401,
          1099511627776
        ]
      ]
    ],
    "lower": [
      -70623406359,
      274877906944
    ],
    "Q_psd": {
      "ok": true,
      "pivots": [
        [
          5634429401,
          1099511627776
        ],
        [
          864279646250444653,
          6195120642282462642176
        ],
        [
          193639239882801476813,
          950285520702491855383825481728
        ],
        [
          78178863516129738385003,
          212908595844846391237718350757888
        ],
        [
          4341683443380580075107713,
          42979284741148773741647838958321664
        ],
        [
          451009414969820888773803601,
          9547462860238980661297587718325272576
        ],
        [
          3669985026697809548912378523,
          123972523998942306832712960500610105344
        ]
      ],
      "max_bits": 127
    },
    "residual_psd": {
      "ok": true,
      "pivots": [
        [
          13420700097,
          137438953472
        ],
        [
          308596943555731012077533,
          6148423253977496289280000
        ],
        [
          29291600094427788257798753643868735,
          1017917783206980595388916673045069824
        ],
        [
          1092624277632590084375243766188369536337632149441,
          80516137249969831936549416863985310554809958400000
        ],
        [
          6102738975387377083834606864322619068142251516701933932869,
          1201353098047385271338381457358570939938000306157383598473216
        ],
        [
          16858474910056116262780855868620868523454844949299129925808917011,
          4193770290450133361006317880845241294517914518406592643460002312355840000
        ],
        [
          310644645955984444732576840472409805053913885491340490001583201587,
          1810164959978189021835437949265407573283859199749763528815105122685576806400
        ]
      ],
      "max_bits": 251
    }
  },
  "c_diagonal_plus_hopping": [
    -486748126021,
    824633720832
  ],
  "beta_squared": [
    1,
    1600
  ],
  "norm_only_affine_margin": [
    5926045830991,
    25769803776000000
  ],
  "true_energy_or_gap_used": false,
  "conclusion": "A stronger coefficient-only external bound allows norm-only to certify this same input. The new directional spectral family is therefore not indispensable for accepting this input.",
  "limitation": "Exact diagonal minimization is polynomial here because external particle number equals two; no polynomial method for general particle number is claimed."
}
```

## `saved_witness_verification.json`

```json
{
  "status": "PASS",
  "case_sha256": "b34f66cbeed196090dfd77ab999cb462db821ce12e9d21408020771f47a81f16",
  "coefficient_trial_U_verified": true,
  "new_particle_spectral_certificate_verified": true,
  "old_LMI_infeasible_at_same_occupancy_bound_verified": true,
  "stronger_density_bound_norm_only_verified": true,
  "numerical_eigensolver_calls": 0,
  "many_body_solves": 0,
  "registered_transfer_status_changed": false,
  "qualification": "Exact finite-witness verification, not independent external scientific reproduction."
}
```

# 付録D：元条件の独立再現と事後診断

次の結果は元の保存ファイルを置換しない。特に`initial_diagnostics.json`の`accepted_numeric`は数値proposalの診断であって、有理certificateの成功とは限らない。

## `original_reproduction.json`

```json
{
  "rows": [
    {
      "id": "T1_amp_1_40_low_coordinates",
      "case_sha256": "58d1c06a8b389819b6277128cce6e9dad156ebd596ea92505b3105f222fe7da6",
      "U": [
        -1,
        5
      ],
      "c": [
        3,
        8
      ],
      "norm": true,
      "five": true,
      "direct": true,
      "direct_witness": {
        "accepted": true,
        "kappa": [
          15819178276233,
          1099511627776000
        ],
        "mu": [
          -713788073146977,
          137438953472000000
        ],
        "affine_margin": [
          0,
          1
        ],
        "PSD": {
          "ok": true,
          "pivots": [
            [
              479,
              250000
            ],
            [
              649843046121790145565222055191,
              144768866898851843671064576000000
            ],
            [
              364729886068300232815765278713146319595561,
              51418392734620302741327956928465207296000000
            ],
            [
              13142042265716941667434246473038172216644717476257582813,
              1385232299583805919505153207298907884037923319840768000000
            ]
          ],
          "max_bits": 190
        }
      },
      "five_witnesses": [
        {
          "accepted": true,
          "kappa": [
            2523179,
            187850000
          ],
          "mu": [
            -58033117,
            18785000000
          ],
          "affine_margin": [
            35936523,
            18785000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                55537,
                18785000000
              ],
              [
                652219062889,
                312978763500000
              ],
              [
                352937053016291,
                72836301525000000
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 57
          }
        },
        {
          "accepted": true,
          "kappa": [
            28123369,
            205945000
          ],
          "mu": [
            -1153058129,
            41189000000
          ],
          "affine_margin": [
            13454071,
            8237800000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                11647769,
                41189000000
              ],
              [
                41677679799871,
                17990998400287500
              ],
              [
                614916788224144,
                14306345918203125
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 59
          }
        }
      ]
    },
    {
      "id": "T1_amp_1_40_high_coordinates",
      "case_sha256": "f0668c3157e5ed6b3c3a4c0fc54bdbeb80ee57848572bd2b49d0009e6cd0d9d7",
      "U": [
        -1,
        5
      ],
      "c": [
        3,
        8
      ],
      "norm": true,
      "five": true,
      "direct": true,
      "direct_witness": {
        "accepted": true,
        "kappa": [
          962009628471,
          549755813888000
        ],
        "mu": [
          -183204548832351,
          68719476736000000
        ],
        "affine_margin": [
          0,
          1
        ],
        "PSD": {
          "ok": true,
          "pivots": [
            [
              579,
              250000
            ],
            [
              196988318961908002312521501,
              75557863725914323419136000000
            ],
            [
              20837879773989131139056616740440054173,
              7793283939071148321869634708963328000000
            ],
            [
              21113285709575735041485197721413076104561779560950431,
              7637163703233125521212023227101538416768283836416000000
            ]
          ],
          "max_bits": 183
        }
      },
      "five_witnesses": [
        {
          "accepted": true,
          "kappa": [
            861,
            650000
          ],
          "mu": [
            -19803,
            65000000
          ],
          "affine_margin": [
            147957,
            65000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                2583,
                65000000
              ],
              [
                3157,
                13000000
              ],
              [
                452929,
                1787500000
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 31
          }
        },
        {
          "accepted": true,
          "kappa": [
            306269,
            205945000
          ],
          "mu": [
            -12557029,
            41189000000
          ],
          "affine_margin": [
            19017491,
            8237800000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                306269,
                41189000000
              ],
              [
                3368959,
                24713400000
              ],
              [
                28847641,
                106190390625
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 42
          }
        }
      ]
    },
    {
      "id": "T1_amp_1_10_low_coordinates",
      "case_sha256": "58f0aafe4f72acb102857dc94dbbe9b28c9660223d1a683e2a314df32450001c",
      "U": [
        -1,
        5
      ],
      "c": [
        3,
        8
      ],
      "norm": false,
      "five": false,
      "direct": false,
      "direct_witness": {
        "accepted": false,
        "kappa": [
          253106907810921,
          1099511627776000
        ],
        "mu": [
          -6645981311514177,
          137438953472000000
        ],
        "affine_margin": [
          0,
          1
        ],
        "PSD": {
          "ok": false,
          "pivots": [
            [
              -1021,
              250000
            ]
          ],
          "max_bits": 0
        }
      },
      "five_witnesses": [
        {
          "accepted": false,
          "kappa": [
            2523179,
            11740625
          ],
          "mu": [
            -58033117,
            1174062500
          ],
          "affine_margin": [
            -19401633,
            4696250000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                55537,
                1174062500
              ],
              [
                652219062889,
                19561172718750
              ],
              [
                352937053016291,
                4552268845312500
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 53
          }
        },
        {
          "accepted": false,
          "kappa": [
            56246738,
            25743125
          ],
          "mu": [
            -1153058129,
            2574312500
          ],
          "affine_margin": [
            -17729009,
            2059450000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                11647769,
                2574312500
              ],
              [
                166710719199484,
                4497749600071875
              ],
              [
                9838668611586304,
                14306345918203125
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 55
          }
        }
      ]
    },
    {
      "id": "T1_amp_1_10_high_coordinates",
      "case_sha256": "caf8a3e9e0a2dd6313c6104b9f34b437bf87e37ebdb2c5a8a513a9b1a704fb78",
      "U": [
        -1,
        5
      ],
      "c": [
        3,
        8
      ],
      "norm": false,
      "five": true,
      "direct": true,
      "direct_witness": {
        "accepted": true,
        "kappa": [
          7696076562831,
          274877906944000
        ],
        "mu": [
          -271979068131063,
          34359738368000000
        ],
        "affine_margin": [
          0,
          1
        ],
        "PSD": {
          "ok": true,
          "pivots": [
            [
              579,
              250000
            ],
            [
              129335653063033702566924269,
              18889465931478580854784000000
            ],
            [
              184272104010263800407021546776212904029,
              23025591714509097578061388048236544000000
            ],
            [
              6838762040067799848465719177129174915735281969902389,
              896501420504573135383593947946102545600273711104000000
            ]
          ],
          "max_bits": 180
        }
      },
      "five_witnesses": [
        {
          "accepted": true,
          "kappa": [
            861,
            40625
          ],
          "mu": [
            -19803,
            4062500
          ],
          "affine_margin": [
            27303,
            16250000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                2583,
                4062500
              ],
              [
                3157,
                812500
              ],
              [
                452929,
                111718750
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 27
          }
        },
        {
          "accepted": true,
          "kappa": [
            612538,
            25743125
          ],
          "mu": [
            -12557029,
            2574312500
          ],
          "affine_margin": [
            4524671,
            2059450000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                306269,
                2574312500
              ],
              [
                3368959,
                1544587500
              ],
              [
                461562256,
                106190390625
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 38
          }
        }
      ]
    },
    {
      "id": "T2_amp_1_40_low_coordinates",
      "case_sha256": "523326d32344ec6ab47fe0f0030176bb41e286cab5cd75cbc559dbb94c88b594",
      "U": [
        -63,
        200
      ],
      "c": [
        1565181,
        16777216
      ],
      "norm": true,
      "five": false,
      "direct": true,
      "direct_witness": {
        "accepted": true,
        "kappa": [
          0,
          1
        ],
        "mu": [
          -864639113,
          1048576000000
        ],
        "affine_margin": [
          0,
          1
        ],
        "PSD": {
          "ok": true,
          "pivots": [
            [
              445208713,
              1048576000000
            ],
            [
              180950906633746769,
              466835171442688000000
            ],
            [
              864639113,
              1048576000000
            ],
            [
              864639113,
              1048576000000
            ],
            [
              864639113,
              1048576000000
            ]
          ],
          "max_bits": 69
        }
      },
      "five_witnesses": [
        {
          "accepted": false,
          "kappa": [
            3355627,
            499506800
          ],
          "mu": [
            -77179421,
            24975340000
          ],
          "affine_margin": [
            -1581231098747529,
            654713552896000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                15349,
                4995068000
              ],
              [
                2949908206311,
                1916732468300000
              ],
              [
                10195914925031,
                2744455820827500
              ],
              [
                187027091073,
                34493816704100
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 54
          }
        },
        {
          "accepted": false,
          "kappa": [
            31474657,
            337232800
          ],
          "mu": [
            -1290460937,
            33723280000
          ],
          "affine_margin": [
            -10052270925484467,
            221008887808000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                3597069,
                6744656000
              ],
              [
                73518094650059,
                37907801583225000
              ],
              [
                198958766390709,
                4923149938835000
              ],
              [
                601893926041753,
                8526913811894400
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 60
          }
        }
      ]
    },
    {
      "id": "T2_amp_1_40_high_coordinates",
      "case_sha256": "503ece76d60b79a3633807989bc798020aac83587b5a5daf08fdda443210105c",
      "U": [
        -63,
        200
      ],
      "c": [
        1565181,
        16777216
      ],
      "norm": true,
      "five": true,
      "direct": true,
      "direct_witness": {
        "accepted": true,
        "kappa": [
          106506831300019501,
          144115188075855872000
        ],
        "mu": [
          -271219841502862843487,
          288230376151711744000000
        ],
        "affine_margin": [
          0,
          1
        ],
        "PSD": {
          "ok": true,
          "pivots": [
            [
              186014376462847242687,
              288230376151711744000000
            ],
            [
              51390977781025272448753843972213,
              62008522274093071536180166656000000
            ],
            [
              17368323851429172321209010339071959065623387,
              17131356462969973009341939601605650284544000000
            ],
            [
              5638008781697851382174346084322789751488574670628342913,
              5789789568335404666533958254957806193525593793888256000000
            ],
            [
              1417906160426440596322446892629004958416271549960386533411250366323039484431,
              1625045391895425767234927783680965951952679929162941693426176767161270272000000
            ]
          ],
          "max_bits": 260
        }
      },
      "five_witnesses": [
        {
          "accepted": true,
          "kappa": [
            343753,
            499506800
          ],
          "mu": [
            -7906319,
            24975340000
          ],
          "affine_margin": [
            807141612266871,
            654713552896000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                1031259,
                24975340000
              ],
              [
                5843801,
                29970408000
              ],
              [
                202470517,
                530725975000
              ],
              [
                243049,
                735523763
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 43
          }
        },
        {
          "accepted": true,
          "kappa": [
            522191,
            674465600
          ],
          "mu": [
            -21409831,
            67446560000
          ],
          "affine_margin": [
            278069505158733,
            221008887808000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                522191,
                67446560000
              ],
              [
                522191,
                6744656000
              ],
              [
                128981177,
                337232800000
              ],
              [
                15031129,
                41648250800
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 44
          }
        }
      ]
    },
    {
      "id": "T2_amp_1_10_low_coordinates",
      "case_sha256": "5e81b6369d8aa86356406e5d20300daec618ccbe07264ac241efc3ec0ac37977",
      "U": [
        -63,
        200
      ],
      "c": [
        1565181,
        16777216
      ],
      "norm": false,
      "five": false,
      "direct": false,
      "direct_witness": {
        "accepted": false,
        "kappa": [
          0,
          1
        ],
        "mu": [
          -864639113,
          1048576000000
        ],
        "affine_margin": [
          0,
          1
        ],
        "PSD": {
          "ok": false,
          "pivots": [
            [
              -5846247287,
              1048576000000
            ]
          ],
          "max_bits": 0
        }
      },
      "five_witnesses": [
        {
          "accepted": false,
          "kappa": [
            3355627,
            31219175
          ],
          "mu": [
            -77179421,
            1560958750
          ],
          "affine_margin": [
            -41495689448315529,
            654713552896000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                15349,
                312191750
              ],
              [
                2949908206311,
                119795779268750
              ],
              [
                40783659700124,
                686113955206875
              ],
              [
                748108364292,
                8623454176025
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 50
          }
        },
        {
          "accepted": false,
          "kappa": [
            31474657,
            21077050
          ],
          "mu": [
            -1290460937,
            2107705000
          ],
          "affine_margin": [
            -166303547352748467,
            221008887808000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                3597069,
                421541000
              ],
              [
                147036189300118,
                4738475197903125
              ],
              [
                397917532781418,
                615393742354375
              ],
              [
                601893926041753,
                532932113243400
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 56
          }
        }
      ]
    },
    {
      "id": "T2_amp_1_10_high_coordinates",
      "case_sha256": "8b5674953e4d74b35d6cdfb34d73d0e2b1327aa1a103e78d646d68c07e5fa644",
      "U": [
        -63,
        200
      ],
      "c": [
        1565181,
        16777216
      ],
      "norm": false,
      "five": false,
      "direct": false,
      "direct_witness": {
        "accepted": false,
        "kappa": [
          27265748856901587019,
          2305843009213693952000
        ],
        "mu": [
          -12391433924217707121737,
          4611686018427387904000000
        ],
        "affine_margin": [
          0,
          1
        ],
        "PSD": {
          "ok": false,
          "pivots": [
            [
              -9421165161303562493463,
              4611686018427387904000000
            ]
          ],
          "max_bits": 0
        }
      },
      "five_witnesses": [
        {
          "accepted": false,
          "kappa": [
            343753,
            31219175
          ],
          "mu": [
            -7906319,
            1560958750
          ],
          "affine_margin": [
            -3281726072085129,
            654713552896000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                1031259,
                1560958750
              ],
              [
                5843801,
                1873150500
              ],
              [
                404941034,
                66340746875
              ],
              [
                3888784,
                735523763
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 39
          }
        },
        {
          "accepted": false,
          "kappa": [
            522191,
            42154100
          ],
          "mu": [
            -21409831,
            4215410000
          ],
          "affine_margin": [
            -1018100462457267,
            221008887808000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                522191,
                4215410000
              ],
              [
                522191,
                421541000
              ],
              [
                128981177,
                21077050000
              ],
              [
                15031129,
                2603015675
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 40
          }
        }
      ]
    },
    {
      "id": "T3_amp_1_40_low_coordinates",
      "case_sha256": "b57765c297874f2e3f6367d0f88c175b54e51b4463268d2afc96146010fc981f",
      "U": [
        -4,
        5
      ],
      "c": [
        -23883779,
        20971520
      ],
      "norm": false,
      "five": false,
      "direct": false,
      "direct_witness": null,
      "five_witnesses": [
        {
          "accepted": false,
          "kappa": [
            231514487,
            68885476340
          ],
          "mu": [
            -5324833201,
            1722136908500
          ],
          "affine_margin": [
            -3578517106346427091,
            451447857741824000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                7110872,
                2152671135625
              ],
              [
                2363438889918828001,
                2351211863581288704000
              ],
              [
                135649888694311914203,
                52741822717731036886500
              ],
              [
                942458904190426915418,
                252260326104767065246625
              ],
              [
                102321070890673598037953,
                21031641924647240370057000
              ],
              [
                6838547489399703384065327,
                1141683799023388792993667250
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 90
          }
        },
        {
          "accepted": false,
          "kappa": [
            1004538615911,
            2199621892640
          ],
          "mu": [
            -41186083252351,
            109981094632000
          ],
          "affine_margin": [
            -1368457566992712399167,
            1801930254450688000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                4802730890291,
                549905473160000
              ],
              [
                1070059809062406244005467,
                99039300100710765190858500
              ],
              [
                4174712660205906861273887,
                21966492985874008386469500
              ],
              [
                52425041560828507798237959,
                114266256386552164757386000
              ],
              [
                3530728961641905400235673011,
                5739713103770664369080808000
              ],
              [
                890758025300482297885309849751,
                1159676979758890363015049496000
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 100
          }
        }
      ]
    },
    {
      "id": "T3_amp_1_40_high_coordinates",
      "case_sha256": "b34f66cbeed196090dfd77ab999cb462db821ce12e9d21408020771f47a81f16",
      "U": [
        -4,
        5
      ],
      "c": [
        -23883779,
        20971520
      ],
      "norm": false,
      "five": false,
      "direct": false,
      "direct_witness": null,
      "five_witnesses": [
        {
          "accepted": false,
          "kappa": [
            454148847,
            1377709526800
          ],
          "mu": [
            -10445423481,
            34442738170000
          ],
          "affine_margin": [
            -896378924622676691,
            451447857741824000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                1362446541,
                34442738170000
              ],
              [
                454148847,
                3444273817000
              ],
              [
                43749672261,
                172213690850000
              ],
              [
                227982721194,
                622121958195625
              ],
              [
                6600447959349,
                13832203649072000
              ],
              [
                10887607627310307,
                30034756539003400000
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 65
          }
        },
        {
          "accepted": false,
          "kappa": [
            6920091083,
            10998109463200
          ],
          "mu": [
            -283723734403,
            549905473160000
          ],
          "affine_margin": [
            -4295717916833282367,
            1801930254450688000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                6920091083,
                549905473160000
              ],
              [
                6920091083,
                329943283896000
              ],
              [
                2567353791793,
                8248582097400000
              ],
              [
                129274221521523,
                204014930542360000
              ],
              [
                1741295599124207,
                2054556828820392000
              ],
              [
                27023870558235769,
                32430976009166634375
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 70
          }
        }
      ]
    },
    {
      "id": "T3_amp_1_10_low_coordinates",
      "case_sha256": "433f6df64a6c8174f723b8e798b7eb4b437c59c13881216d044d032f84ef5c0b",
      "U": [
        -4,
        5
      ],
      "c": [
        -23883779,
        20971520
      ],
      "norm": false,
      "five": false,
      "direct": false,
      "direct_witness": null,
      "five_witnesses": [
        {
          "accepted": false,
          "kappa": [
            926057948,
            17221369085
          ],
          "mu": [
            -21299332804,
            430534227125
          ],
          "affine_margin": [
            -48185765361240507091,
            451447857741824000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                113773952,
                2152671135625
              ],
              [
                2363438889918828001,
                146950741473830544000
              ],
              [
                542599554777247656812,
                13185455679432759221625
              ],
              [
                15079342467046830646688,
                252260326104767065246625
              ],
              [
                204642141781347196075906,
                2628955240580905046257125
              ],
              [
                54708379915197627072522616,
                570841899511694396496833625
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 89
          }
        },
        {
          "accepted": false,
          "kappa": [
            1004538615911,
            137476368290
          ],
          "mu": [
            -41186083252351,
            6873818414500
          ],
          "affine_margin": [
            -21859116617434563279167,
            1801930254450688000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                4802730890291,
                34369092072500
              ],
              [
                4280239236249624976021868,
                24759825025177691297714625
              ],
              [
                16698850640823627445095548,
                5491623246468502096617375
              ],
              [
                52425041560828507798237959,
                7141641024159510297336625
              ],
              [
                3530728961641905400235673011,
                358732068985666523067550500
              ],
              [
                890758025300482297885309849751,
                72479811234930647688440593500
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 100
          }
        }
      ]
    },
    {
      "id": "T3_amp_1_10_high_coordinates",
      "case_sha256": "5375b8f7b6764f1b991cb8f6601e4b301892690ba5dd1e7fa114b956edf43999",
      "U": [
        -4,
        5
      ],
      "c": [
        -23883779,
        20971520
      ],
      "norm": false,
      "five": false,
      "direct": false,
      "direct_witness": null,
      "five_witnesses": [
        {
          "accepted": false,
          "kappa": [
            454148847,
            86106845425
          ],
          "mu": [
            -10445423481,
            2152671135625
          ],
          "affine_margin": [
            -5271554453660500691,
            451447857741824000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                1362446541,
                2152671135625
              ],
              [
                908297694,
                430534227125
              ],
              [
                43749672261,
                10763355678125
              ],
              [
                3647723539104,
                622121958195625
              ],
              [
                6600447959349,
                864512728067000
              ],
              [
                10887607627310307,
                1877172283687712500
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 61
          }
        },
        {
          "accepted": false,
          "kappa": [
            6920091083,
            687381841450
          ],
          "mu": [
            -283723734403,
            34369092072500
          ],
          "affine_margin": [
            -32527032220497410367,
            1801930254450688000000
          ],
          "PSD": {
            "ok": true,
            "pivots": [
              [
                6920091083,
                34369092072500
              ],
              [
                6920091083,
                20621455243500
              ],
              [
                2567353791793,
                515536381087500
              ],
              [
                129274221521523,
                12750933158897500
              ],
              [
                1741295599124207,
                128409801801274500
              ],
              [
                432381928931772304,
                32430976009166634375
              ],
              [
                0,
                1
              ]
            ],
            "max_bits": 66
          }
        }
      ]
    }
  ],
  "counts": {
    "norm": 4,
    "five": 4,
    "direct": 5
  },
  "union_equals_direct": true,
  "N1_transcribed_selected_products_pass": true,
  "N1_finite_row_count": 1424,
  "qualification": "No all-row N1 argmin audit, native recompilation, or 72-ledger witness replay is claimed."
}
```

## `initial_diagnostics.json`

```json
[
  {
    "id": "T1_amp_1_40_low_coordinates",
    "input_case_sha256": "58d1c06a8b389819b6277128cce6e9dad156ebd596ea92505b3105f222fe7da6",
    "nu": 1,
    "U": [
      -1,
      5
    ],
    "ell": [
      -3398631,
      16777216
    ],
    "c_frozen_rule": [
      3,
      8
    ],
    "width_diagnostic": 0.0005674814314515775,
    "norm_frozen": true,
    "old_direct_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.024848806267332324,
      "margin": 0.0019132077958223938,
      "kappa": 0.014387458828785414
    },
    "particle_sum_same_c_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.024848806267332324,
      "margin": 0.0019132077958223938,
      "kappa": 0.014387458828785414
    },
    "c_occupancy_aware_numeric": 0.375,
    "old_direct_better_c_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.024848806267332324,
      "margin": 0.0019132077958223938,
      "kappa": 0.014387458828785414
    },
    "particle_sum_better_c_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.024848806267332324,
      "margin": 0.0019132077958223938,
      "kappa": 0.014387458828785414
    },
    "actual_C_min_diagnostic": 0.375,
    "beta2": [
      1,
      1600
    ]
  },
  {
    "id": "T1_amp_1_40_high_coordinates",
    "input_case_sha256": "f0668c3157e5ed6b3c3a4c0fc54bdbeb80ee57848572bd2b49d0009e6cd0d9d7",
    "nu": 1,
    "U": [
      -1,
      5
    ],
    "ell": [
      -3398631,
      16777216
    ],
    "c_frozen_rule": [
      3,
      8
    ],
    "width_diagnostic": 3.9719601188648035e-07,
    "norm_frozen": true,
    "old_direct_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.003022254438754329,
      "margin": 0.002310355656602405,
      "kappa": 0.0017498853200387564
    },
    "particle_sum_same_c_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.003022254438754329,
      "margin": 0.002310355656602405,
      "kappa": 0.0017498853200387564
    },
    "c_occupancy_aware_numeric": 0.375,
    "old_direct_better_c_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.003022254438754329,
      "margin": 0.002310355656602405,
      "kappa": 0.0017498853200387564
    },
    "particle_sum_better_c_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.003022254438754329,
      "margin": 0.002310355656602405,
      "kappa": 0.0017498853200387564
    },
    "actual_C_min_diagnostic": 0.375,
    "beta2": [
      1,
      1600
    ]
  },
  {
    "id": "T1_amp_1_10_low_coordinates",
    "input_case_sha256": "58f0aafe4f72acb102857dc94dbbe9b28c9660223d1a683e2a314df32450001c",
    "nu": 1,
    "U": [
      -1,
      5
    ],
    "ell": [
      -3398631,
      16777216
    ],
    "c_frozen_rule": [
      3,
      8
    ],
    "width_diagnostic": 0.009338985077427114,
    "norm_frozen": false,
    "old_direct_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.3975809872820308,
      "margin": -0.004128675266841691,
      "kappa": 0.2301993916362958
    },
    "particle_sum_same_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.3975809872820308,
      "margin": -0.004128675266841691,
      "kappa": 0.2301993916362958
    },
    "c_occupancy_aware_numeric": 0.375,
    "old_direct_better_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.3975809872820308,
      "margin": -0.004128675266841691,
      "kappa": 0.2301993916362958
    },
    "particle_sum_better_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.3975809872820308,
      "margin": -0.004128675266841691,
      "kappa": 0.2301993916362958
    },
    "actual_C_min_diagnostic": 0.375,
    "beta2": [
      1,
      100
    ]
  },
  {
    "id": "T1_amp_1_10_high_coordinates",
    "input_case_sha256": "caf8a3e9e0a2dd6313c6104b9f34b437bf87e37ebdb2c5a8a513a9b1a704fb78",
    "nu": 1,
    "U": [
      -1,
      5
    ],
    "ell": [
      -3398631,
      16777216
    ],
    "c_frozen_rule": [
      3,
      8
    ],
    "width_diagnostic": 6.595578779661215e-06,
    "norm_frozen": false,
    "old_direct_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.04835606810621164,
      "margin": 0.0022256905056384775,
      "kappa": 0.027998163433496537
    },
    "particle_sum_same_c_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.04835606810621164,
      "margin": 0.0022256905056384775,
      "kappa": 0.027998163433496537
    },
    "c_occupancy_aware_numeric": 0.375,
    "old_direct_better_c_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.04835606810621164,
      "margin": 0.0022256905056384775,
      "kappa": 0.027998163433496537
    },
    "particle_sum_better_c_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.04835606810621164,
      "margin": 0.0022256905056384775,
      "kappa": 0.027998163433496537
    },
    "actual_C_min_diagnostic": 0.375,
    "beta2": [
      1,
      100
    ]
  },
  {
    "id": "T2_amp_1_40_low_coordinates",
    "input_case_sha256": "523326d32344ec6ab47fe0f0030176bb41e286cab5cd75cbc559dbb94c88b594",
    "nu": 2,
    "U": [
      -63,
      200
    ],
    "ell": [
      -6823427,
      16777216
    ],
    "c_frozen_rule": [
      1565181,
      16777216
    ],
    "width_diagnostic": 0.000506501517580249,
    "norm_frozen": true,
    "old_direct_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.0,
      "margin": 0.0003991682300567626,
      "kappa": 0.0
    },
    "particle_sum_same_c_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.0,
      "margin": 0.0010241682300567625,
      "kappa": 0.0
    },
    "c_occupancy_aware_numeric": 0.14176776327854557,
    "old_direct_better_c_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.0,
      "margin": 0.0005930710531141822,
      "kappa": 0.0
    },
    "particle_sum_better_c_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.0,
      "margin": 0.0012180710531141821,
      "kappa": 0.0
    },
    "actual_C_min_diagnostic": 0.14176776327854537,
    "beta2": [
      1,
      1600
    ]
  },
  {
    "id": "T2_amp_1_40_high_coordinates",
    "input_case_sha256": "503ece76d60b79a3633807989bc798020aac83587b5a5daf08fdda443210105c",
    "nu": 2,
    "U": [
      -63,
      200
    ],
    "ell": [
      -6823427,
      16777216
    ],
    "c_frozen_rule": [
      1565181,
      16777216
    ],
    "width_diagnostic": 1.0803992153030073e-05,
    "norm_frozen": true,
    "old_direct_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.0017925147698238367,
      "margin": 0.0012676792113314374,
      "kappa": 0.0007390396025752455
    },
    "particle_sum_same_c_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.0025705690841385923,
      "margin": 0.0013861184975079014,
      "kappa": 0.001059825216681869
    },
    "c_occupancy_aware_numeric": 0.14176776327854557,
    "old_direct_better_c_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.0016039307748911625,
      "margin": 0.001461582034388857,
      "kappa": 0.0007390395956002253
    },
    "particle_sum_better_c_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.0023001288343934066,
      "margin": 0.001580021320565321,
      "kappa": 0.001059825218275938
    },
    "actual_C_min_diagnostic": 0.14176776327854537,
    "beta2": [
      1,
      1600
    ]
  },
  {
    "id": "T2_amp_1_10_low_coordinates",
    "input_case_sha256": "5e81b6369d8aa86356406e5d20300daec618ccbe07264ac241efc3ec0ac37977",
    "nu": 2,
    "U": [
      -63,
      200
    ],
    "ell": [
      -6823427,
      16777216
    ],
    "c_frozen_rule": [
      1565181,
      16777216
    ],
    "width_diagnostic": 0.008366289460985166,
    "norm_frozen": false,
    "old_direct_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.0,
      "margin": -0.01835083176994324,
      "kappa": 0.0
    },
    "particle_sum_same_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.0,
      "margin": -0.008350831769943237,
      "kappa": 0.0
    },
    "c_occupancy_aware_numeric": 0.14176776327854557,
    "old_direct_better_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.0,
      "margin": -0.01815692894688582,
      "kappa": 0.0
    },
    "particle_sum_better_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.0,
      "margin": -0.008156928946885817,
      "kappa": 0.0
    },
    "actual_C_min_diagnostic": 0.14176776327854537,
    "beta2": [
      1,
      100
    ]
  },
  {
    "id": "T2_amp_1_10_high_coordinates",
    "input_case_sha256": "8b5674953e4d74b35d6cdfb34d73d0e2b1327aa1a103e78d646d68c07e5fa644",
    "nu": 2,
    "U": [
      -63,
      200
    ],
    "ell": [
      -6823427,
      16777216
    ],
    "c_frozen_rule": [
      1565181,
      16777216
    ],
    "width_diagnostic": 0.0001827274451878802,
    "norm_frozen": false,
    "old_direct_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.02868023635796799,
      "margin": -0.004454656069548436,
      "kappa": 0.01182463365801992
    },
    "particle_sum_same_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.041129105393196176,
      "margin": -0.0025596274907250167,
      "kappa": 0.016957203486278848
    },
    "c_occupancy_aware_numeric": 0.14176776327854557,
    "old_direct_better_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.02566289259291084,
      "margin": -0.00426075324649102,
      "kappa": 0.011824633619293083
    },
    "particle_sum_better_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.0368020610887729,
      "margin": -0.0023657246676675996,
      "kappa": 0.016957203371914285
    },
    "actual_C_min_diagnostic": 0.14176776327854537,
    "beta2": [
      1,
      100
    ]
  },
  {
    "id": "T3_amp_1_40_low_coordinates",
    "input_case_sha256": "b57765c297874f2e3f6367d0f88c175b54e51b4463268d2afc96146010fc981f",
    "nu": 3,
    "U": [
      -4,
      5
    ],
    "ell": [
      -34369539,
      41943040
    ],
    "c_frozen_rule": [
      -23883779,
      20971520
    ],
    "width_diagnostic": 0.0010533139973661898,
    "norm_frozen": false,
    "old_direct_diagnostic": {
      "accepted_numeric": false,
      "precondition": false
    },
    "particle_sum_same_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": false
    },
    "c_occupancy_aware_numeric": -0.6867984153623199,
    "old_direct_better_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.0,
      "margin": -0.0014061936614492795,
      "kappa": 0.0
    },
    "particle_sum_better_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.0,
      "margin": -0.00015619366144927957,
      "kappa": 0.0
    },
    "actual_C_min_diagnostic": -0.39467257538355094,
    "beta2": [
      1,
      1600
    ]
  },
  {
    "id": "T3_amp_1_40_high_coordinates",
    "input_case_sha256": "b34f66cbeed196090dfd77ab999cb462db821ce12e9d21408020771f47a81f16",
    "nu": 3,
    "U": [
      -4,
      5
    ],
    "ell": [
      -34369539,
      41943040
    ],
    "c_frozen_rule": [
      -23883779,
      20971520
    ],
    "width_diagnostic": 7.043636623960126e-07,
    "norm_frozen": false,
    "old_direct_diagnostic": {
      "accepted_numeric": false,
      "precondition": false
    },
    "particle_sum_same_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": false
    },
    "c_occupancy_aware_numeric": -0.6867984153623199,
    "old_direct_better_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.0029951646902563866,
      "margin": -0.0001338067913289585,
      "kappa": 0.0003510380479488748
    },
    "particle_sum_better_c_diagnostic": {
      "accepted_numeric": true,
      "precondition": true,
      "x": 0.00482326069202703,
      "margin": 0.00011627686574711078,
      "kappa": 0.0005652937962262014
    },
    "actual_C_min_diagnostic": -0.39467257538355094,
    "beta2": [
      1,
      1600
    ]
  },
  {
    "id": "T3_amp_1_10_low_coordinates",
    "input_case_sha256": "433f6df64a6c8174f723b8e798b7eb4b437c59c13881216d044d032f84ef5c0b",
    "nu": 3,
    "U": [
      -4,
      5
    ],
    "ell": [
      -34369539,
      41943040
    ],
    "c_frozen_rule": [
      -23883779,
      20971520
    ],
    "width_diagnostic": 0.016851057530133495,
    "norm_frozen": false,
    "old_direct_diagnostic": {
      "accepted_numeric": false,
      "precondition": false
    },
    "particle_sum_same_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": false
    },
    "c_occupancy_aware_numeric": -0.6867984153623199,
    "old_direct_better_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.0,
      "margin": -0.029531193661449278,
      "kappa": 0.0
    },
    "particle_sum_better_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.0,
      "margin": -0.00953119366144928,
      "kappa": 0.0
    },
    "actual_C_min_diagnostic": -0.39467257538355094,
    "beta2": [
      1,
      100
    ]
  },
  {
    "id": "T3_amp_1_10_high_coordinates",
    "input_case_sha256": "5375b8f7b6764f1b991cb8f6601e4b301892690ba5dd1e7fa114b956edf43999",
    "nu": 3,
    "U": [
      -4,
      5
    ],
    "ell": [
      -34369539,
      41943040
    ],
    "c_frozen_rule": [
      -23883779,
      20971520
    ],
    "width_diagnostic": 1.1474786205778464e-05,
    "norm_frozen": false,
    "old_direct_diagnostic": {
      "accepted_numeric": false,
      "precondition": false
    },
    "particle_sum_same_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": false
    },
    "c_occupancy_aware_numeric": -0.6867984153623199,
    "old_direct_better_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.04792263524790275,
      "margin": -0.009173003739524146,
      "kappa": 0.005616608791067746
    },
    "particle_sum_better_c_diagnostic": {
      "accepted_numeric": false,
      "precondition": true,
      "x": 0.07717217071187905,
      "margin": -0.005171665226307022,
      "kappa": 0.00904470069736179
    },
    "actual_C_min_diagnostic": -0.39467257538355094,
    "beta2": [
      1,
      100
    ]
  }
]
```

# 付録E：コード・結果のSHA-256

```json
[
  {
    "file": "review_checks.py",
    "sha256": "049d6fd0e8bdb8362949008e8210e141ef6763d066d209f25c833b80c0d7c2f6",
    "bytes": 8015
  },
  {
    "file": "original_certificate_checks.py",
    "sha256": "6ce5d2da8805f4ce15ff68788fd7860c385f94a3119500ab9973e5d56c80af0c",
    "bytes": 3778
  },
  {
    "file": "particle_number_certificate.py",
    "sha256": "4f3f48cf90e1f60f9b4b6f453587e7cd9e8869e142ba14025a93055deb21dca8",
    "bytes": 5166
  },
  {
    "file": "density_bound_countercheck.py",
    "sha256": "ce0752ff17c5fd2939148a49d39bd05647e243110c49a920e421b980fc7bc1a0",
    "bytes": 2966
  },
  {
    "file": "verify_saved_rational_witnesses.py",
    "sha256": "dcc5e0414ba9884c409fa70d4936d8e9d893dcd1c1fd6974505c25f8abe8af7e",
    "bytes": 6339
  },
  {
    "file": "posthoc_particle_certificate.json",
    "sha256": "560738dbf7299a93819228f018677452522970c5c28bb36f40073609116cfa56",
    "bytes": 17480
  },
  {
    "file": "density_bound_countercheck.json",
    "sha256": "8adae20776e8b831626f1d3c3262cde255b57592f58a9346fb50296776a3fe18",
    "bytes": 8860
  },
  {
    "file": "saved_witness_verification.json",
    "sha256": "e3f4985dc801ca046935c6328e7a069822328183a797a8443bcc7d786015a138",
    "bytes": 532
  },
  {
    "file": "original_reproduction.json",
    "sha256": "405a8bc7df26a30b4761d2ad77b17518012afe3cb0f252bf7757ee277bfdb485",
    "bytes": 32632
  },
  {
    "file": "initial_diagnostics.json",
    "sha256": "8d764b663ed40f9111d407c3c0b149be92b40fa304286718880b6665bf5669f1",
    "bytes": 15109
  }
]
```
