# Return aggregation：parent係数のchild別配分の限定技術監査

2026-10-14 JST。**技術監査完了、native利益・独立新規性は未確立。比較計画は草案、実行source・authorizationは未作成、mandatory STOP。**

利用者が指定した[GPT方針レビュー原本][review]§6〜13を独立に照合した。旧G/Dの再実行、再ランキング、登録入力の係数評価、新しいHamiltonian・行列・回路・合成・sampling・LPは行っていない。数学的導出、sourceの静的読取り、登録外の形式代数controlのみを使用する。科学的採択・研究GO/STOP・priority認定をCodexが代行しない。

## 1. 監査の結果

| 項目 | 結果 | 証拠の限界 |
| --- | --- | --- |
| 同じ有限operator mean | 仮定付きで成立。正負時間、非可換な積順序、parent位相を保持する。 | 実装誤差込みのmeanは別に認証する必要がある。 |
| 既存pairingとの関係 | 共通角配分を包含し、既存点がこのfamilyの係数質量最小。 | 全辞書・signed表現の最適性ではない。 |
| 固定単価モデル | 異なる正単価では改善方向、凸KKT式とも成立。 | 新角度の実際の合成価格、range、bias込みの資源改善を証明しない。 |
| 過去の再配分／既知IS | 調べたR0/G1/G5の自由度、固定実装のproposal変更とは区別できる。 | 一般LCU・probabilistic synthesisに対する独立新規性は未認定。 |
| P5 rootへの接続 | O(L)個の追加root係数処理で可能。非rootの構成式を保持できる。 | 現sourceをそのまま実行できる状態ではない。 |
| 最小比較 | ordinaryとclosed P5へ同じroot配分・proposal候補を与える草案を準備。 | [計画案][plan]は未採択・未実行。 |

重大な数学的反例は見つからなかった。一方、既存sourceへの変更点を明示せずに、現P5へ新ratioだけを入れて実行する方法は不適切である。特にrootのchild抽出、group normalizer、endpoint位相の対応が必要となる。

## 2. 独立導出：平均、位相、端点

偶数長reduced parent uとchild集合について、\(a\ge0\)、\(b_i\ge0\)、\(s=\sum_i b_i\)、\(\zeta=(-i\sigma)^{|u|}\in\{1,-1\}\)、\(\sigma=\pm1\) とする。構成の対象は

\[
M_u=\zeta\left(aI-i\sigma\sum_i b_iQ_i\right)Q(u).
\]

\(z_i\ge0\)、\(\sum_i z_i=a\) とし、\(c_i=\sqrt{z_i^2+b_i^2}\)、\(\phi_i=\operatorname{atan2}(b_i,z_i)\)。\(c_i>0\) のeventに対し

\[
c_i e^{-i\sigma\phi_iQ_i}=z_iI-i\sigma b_iQ_i,
\quad V_i=\zeta e^{-i\sigma\phi_iQ_i}Q(u),
\quad \sum_i c_iV_i=M_u.
\]

使う条件はHermitian \(Q_i^2=I\) のみ。他labelとの可換性や一般の奇数語のinvolution性を使わない。回路時間ではQ(u)を右端から作用させ、その後に回転を作用させる。回転をparentの反対側へ移動しない。

controlled枝0はI、枝1はV_iである。\(\zeta\) をsystem global phaseとして落とすと枝間信号が変わる。以下の端点を通常の有限ratioへ丸めて代用しない。

| 条件 | 正しいunitaryと処理 |
| --- | --- |
| \(z_i=0,b_i>0\) | \(V_i=\zeta(-i\sigma)Q_iQ(u)\)。phase付きcontrolled-Qでexactに構成できる。 |
| \(b_i=0,z_i>0\) | \(V_i=\zeta Q(u)\)。純parentとして処理し、無意味な回転を合成しない。 |
| \(z_i=b_i=0\) | eventを除去し、angle・weight・確率の除算を行わない。 |
| \(s=0,a>0\) | 全体は純parent。複数の同一unitaryを異なるcostのprimitiveと見なさない。 |
| \(a=0,s>0\) | 全zが0となり配分自由度はない。 |

純childのphaseは正時間で \(-i\zeta\)、負時間で \(+i\zeta\)。ancillaのS/S†/Zの組合せとして保持できるが、現G9実装はこの新端点用adapterではない。R_Qの物理角は \(2\sigma\phi_i\)。既存controlled分解の各RZ primitiveには符号付き \(\phi_i\) が入り、二primitiveとCXによってこのcontrolled回転を表す。合成器に指数角を二重に渡さない。

## 3. 係数質量と固定単価の監査

### 3.1 元のcommon-angle点はnorm最小

\[
L(z)=\sum_i\sqrt{z_i^2+b_i^2}
\ge\sqrt{a^2+s^2}=d.
\]

これは二次元ベクトル \((z_i,b_i)\) の三角不等式。\(s>0\) なら等号は正bの全childで \(z_i/b_i=a/s\)、零bのchildにzを置かない場合に成立する。したがって既存点 \(z_i^0=ab_i/s\) を回復する。\(s=0\) の退化例を一意性の主張へ含めない。

また \(d\le L(z)\le a+s\) なので、root変更後の全質量は \(B_0\le B'(z)\le B_{\rm rest}+a+s\) と解析的に囲める。normalizationを減らす方向はこのfamily内にはない。

### 3.2 費用を固定した場合

正単価 \(C_i>0\) を配分によらず固定し、\(S(z)=\sum_i\sqrt{C_i}\sqrt{z_i^2+b_i^2}\) とする。\(a>0,b_i>0\)、\(\sum_i v_i=0\) の内点方向なら

\[
DL(z^0)[v]=0,\qquad
DS(z^0)[v]=\frac a d\sum_i\sqrt{C_i}v_i,\qquad
D^2L(z^0)[v,v]=\sum_i\frac{b_i^2v_i^2}{(c_i^0)^3}>0
\]

（非零v）。安いchildへzを移す方向でSが一次で減り、norm損失は二次から始まる。費用差がない場合の改善は導かれない。

全ensembleの既知cost-aware IS proxyの最小値は \((S_{\rm rest}+S(z))^2\)。この診断は、元配分へも同じ最適ISを与えた比較である。canonical samplingでも \(B'=B_{\rm rest}+L\)、\(A'=A_{\rm rest}+\sum_i c_iC_i\) に対するproxy \(B'A'\) は、元点で \(D(B'A')[v]=B' (a/d)\sum_iC_iv_i\) となり、固定Cが異なれば減少方向を持つ。

Sは正b・正Cでstrict convexであり、KKTから

\[
z_i=\frac{b_i\lambda}{\sqrt{C_i-\lambda^2}},\qquad
\sum_i\frac{b_i\lambda}{\sqrt{C_i-\lambda^2}}=a,
\quad 0<\lambda<\sqrt{\min_i C_i}
\]

を得る。左辺は0から無限大までstrict増加するので解は一意。等単価ではcommon-angle配分へ戻る。本作業でλを数値的に求めたり、登録costを入力して最適化したりしていない。

零bには純parentの線形項、零Cには別の境界解が必要。T=0の正係数eventがあると \(q_i\propto c_i/\sqrt{C_i}\) をそのまま使えない。正C eventもある場合、proxy infimumがあっても有限full-support proposalで達成されないことがある。十分confidenceのrange・shot capを無視した極限を実装可能な利益としない。

### 3.3 角度依存価格と有限confidenceへの移送条件

実際には \(C_i=C_i(\phi_i,\epsilon,\text{provider/control/compiler})\)。仮に滑らかなら方向微分には

\[
DS[v]=\sum_i\left[
\sqrt{C_i}\frac{z_i}{c_i}
-\frac{b_i}{2c_i\sqrt{C_i}}\frac{dC_i}{d\phi_i}
\right]v_i
\]

の価格項が加わる。離散T価格にはこの微分モデルを適用しない。endpointのexact回路削減も、そのphase・provider費用を戻した後に評価する。

実proposal \(q_e\)、digital係数 \(\widetilde c_e\) について

\[
\nu=\sum_e\widetilde c_e^2/q_e,\quad
W=\max_e\widetilde c_e/q_e,\quad
\overline C=\sum_e q_eC_e,\quad
N=\left\lceil\log(2/\alpha)\left(2\nu/\Delta^2+4W/(3\Delta)\right)\right\rceil
\]

を共通policyで戻す。判断に必要なのは \(2N(\overline C+h)\) などの実際の割当費用差であり、proxyだけではない。同bias・canonical・moment項支配の一step近似で \(\overline C_{new}/\overline C_0<(B_0/B_{new})^2\) は利益の目安になるが、ceil/range/bias込みの十分条件とはしない。

同じideal targetならTaylor remainderは不変。係数・角度・native biasは変わり得る。\(\Delta\) を旧値へ無条件固定せず、新表現の重み付きoperator誤差を戻す。r>1への一般化、拒否ありの全試行費用、Dを挟むPR、D2境界節約の移植は今回の式だけで確定しない。

## 4. 既存構成との対応

| 対照 | 共通部分 | この具体候補との区別／確認できない部分 |
| --- | --- | --- |
| [R0][r0] | Euler化、非負a/b、angleと係数の再配分、controlled phase。 | R0は隣接raw degree間で \(a_k+b_{k-1}=t_k\) と固定IID word lawを使う。本候補は集約済み一parentのchild別z、angle、lawを変える。一般的な同一parameter化を示していない。 |
| [G1][g1]・[G5][g5] | meanを保存する非負係数classと費用・precision・IS。 | 七prototype、固定angleとprototype内IID sharesのclass。任意のchild別角度を自動的に含めない。common-angle特殊点や一部重複はあり得る。class外であることは利益の証拠ではない。 |
| 既存RA/P5 | 正値性、parent/child、少数群・局所生成。 | 現rootはb_i/sと共通s/a。本候補はc_i/Lとchild別b_i/z_i。非rootを保存できるが、新しいunitary辞書・取得費用が生じる。 |
| cost-aware IS | 同じeventに対する費用×second moment、q変更。 | [IS一次本文](https://arxiv.org/html/2603.13495v1)§II Theorem 1/§III Theorem 2は固定実装のproposalとbiasを扱う。本候補はcとVも変わるので、proposal変更のみの同一実装ではない。既存側にもISを与える必要がある。 |
| PTSC/Euler | identityをimaginary atomへ配る、回転へまとめる。 | [PTSC一次本文](https://arxiv.org/html/2212.04566v2)Eq.(7), IV.2 Eqs.(74)–(75)等の共通角構成は既知。本candidateの恒等式自体に独立priorityを認定しない。return parentごとの費用依存配分・生成まで同じ実装という証拠は未確認。 |
| SPS | 安い辞書、係数最適化、sampling負担と精度の交換。 | [SPS一次本文](https://arxiv.org/html/2402.15550v2)§II.1–II.3/Eqs.(1)–(6)はprocess表現とdesign matrixを使う。一般的最適化原理との重複は強いが、coherent first moment・構造query・取得費用までの同値は示していない。 |

SPSをphase理由だけで除外しない。\(C(V)=\operatorname{diag}(I,V)\) のchannelはoff-diagonal blockを \(X\mapsto XV^\dagger\) と変換するため、重み付き和からoperator meanを読める。一方、branch1のdiagonal blockは \(\sum_e c_eV_e\rho V_e^\dagger\) に依存し、first momentだけではfull channelを指定しない。controlled processや必要blockへの写像、辞書生成と情報取得を明示して比較する必要がある。

一次文献は既済[文献照合][prior]の指定版を継承し、今回は上記三論文の関係する本文だけを再確認した。網羅的priority調査、全過去branchでの不存在証明、SPSの最適対照計算ではない。[取得範囲とsource identity][identities]を参照する。

## 5. P5 sourceへの実装接続の監査

静的対象は基点 `66c96f8033bcf8392237d820d4a4615f3cebc75a` のsource。行番号は固定ファイルに対応する。

| 箇所 | 現在の構造 | 新しいsourceで必要な処理 |
| --- | --- | --- |
| `g9_p5.py:20–23,63–65` | root a0/child0はrational閉式。 | a0、b_iを再利用できる。全word/Pauli表や登録値の新取得を必要としない。 |
| `g9_p5.py:31–45` | root一つのradicandとroot_interval、全群law。 | root質量をsum_i sqrt(z_i²+b_i²)の区間へ置換し、全群の確率を更新する。非rootのmassを変えない。 |
| `g9_p5.py:91–108` | root child_mass=b_i/s、ratio=s/a。 | rootのみc_i/L、ratio_i=b_i/z_i、digital coefficient_iと実q_iの対応へ変更。 |
| `g9_p5.py:110–123` | `sample()`もb_i/sでchildをdrawする。 | `event()`だけの修正では分布が一致しない。sampleとeventの双方を新root lawへ結ぶ。 |
| `g7_generator.py:79–90` | eventはFraction ratio、phase_i_power、weightを保持。 | interiorのchild別ratioを表せる。infinite ratioやpure-parentの明示event kindは既存形式のままでは扱えない。 |
| `g9_native.py:154–175` | 子ごとのratioを読むcontrolled回転。phaseは0/2を前提。 | interiorは既存primitive構成に接続可能。pure-childをphase付きCQへ、pure-parentをparentのみへlowerするadapterが必要。phase1/3を落とさない。 |
| `rte_development/pipeline.py:14–33` | angle kind/ratio/tool/strict phaseをkey化。 | 新ratioを旧cache価格で代理しない。endpointはexact、interiorの新keyは独立に取得・hash bindingする。 |

一般のσを現native入口が自動で扱うとはしない。形式恒等式は両符号で成立するが、既存D1/G9 providerは正時間に固定されている。将来negative-timeを実装するなら、R token signとphaseを別途検証する。今回の比較草案は既存正時間だけを候補とする。

root質量は一つのrational radicandの平方根とは一般に書けない。新adapterでは各c_iの区間を足したroot質量を一群として保持すれば、群数 \(L^2+1\) を維持できる。既存の「全群=単一平方根」という内部形式は拡張する必要がある。代わりにrootをL群へ分割すると群数は \(L^2+L\) となり、旧 \(L^2+1\) boundをそのまま主張できない。

新配分zが既に与えられる場合、追加の係数・sqrt区間・ratio準備はL個、normalizer更新もO(L)有理演算。現P5の初期化と群選択は引き続きO(L²)の部分を持つ。cost取得、配分選択、bits、cacheは無料でない。native provider自体を新設する必要はないが、実行可能sourceの完成・性能確認ではない。

このroot局所性は、全ensembleに対するcost-aware ISの安価な生成を証明するものではない。global ISでは各eventのcostと変更後proposalを取得・正規化する情報が必要になる。小さい既知fixtureの保存event/IR表を双方へ公開することは可能だが、その表の走査・proposal準備・online samplingを計上し、一般I0やO(L²)のsamplerへ無料で移送しない。

### finite-bit仕様の閉じ方

1. zを正確な有理数として持ち、非負性とsum_i z_i=aをexactに確認する。
2. c_iの区間とmidpointを個別に取得し、root digital massをその和とする。root係数とgroup/child係数で異なる丸めを混用しない。
3. groupとchildのdyadic lawを更新し、実proposal \(\widetilde q_e\) から \(w_e=\widetilde c_e/\widetilde q_e\) を構成する。zero coefficientのみsupportから除く。
4. \(\widetilde q_ew_e=\widetilde c_e\)、確率和、全stepのmoment/rangeを検証する。旧root確率や旧rangeの上界をコピーしない。
5. \(e\le\sum_e|\widetilde c_e-c_e|+\sum_e\widetilde c_e(\delta_{angle,e}+\delta_{native,e})\) に既存nonroot誤差を接続し、共通taskのbiasへ戻す。信号biasの既存保守係数も変更しない。

旧K/Hや許容誤差で区間解像・full supportが閉じなければtechnical inconclusiveとし、実行中に精度・bits・thresholdを変えない。

## 6. 登録外の独立control

[形式代数checker][checker]はPython 3.10.12、標準`fractions`と`unittest`のみで実行した。**13 tests PASS**。scriptはproductionをimportせず、登録p/xや保存costを読まない。

非可換free-involution語でのmean/unitarity、正負σとeven parent位相、誤った順序とphase脱落の検出、pure-child/pure-parent/zero endpoint、common-angle回復、norm determinant恒等式、固定cost勾配、等cost KKT、digital weightとzero proposalを確認した。平方根の一般証明とKKT一意性は上の解析的導出であり、有限テストをその代わりにしない。

実測はwall約0.022秒、child CPU約0.022秒、peak RSS約13.25 MiB。[実行receipt][validation]とlogを保存した。native matrix guard、回路lowering、synthesizer、登録機構や資源利益を検証した件数ではない。

## 7. 今回の停止点と次の判断

監査によって、mean・norm最小性・限定costモデルの解析は閉じ、rootへの実装経路も特定できた。現時点の技術判定は **`TECHNICAL_AUDIT_COMPLETE_NATIVE_VALUE_UNRESOLVED_COMPARISON_DRAFT_ONLY`**。実行sourceは未完成なのでone-shot承認待ち／実行可能とは表現しない。

次に技術準備を認可する場合は、root adapterと端点、digital会計、source/guard/登録外native validationを新しいsourceで準備する。今回はその実装を開始していない。科学実行は草案の比較範囲を結果前に確定し、別途ユーザー承認を受ける必要がある。

既知法との完全対応、誤差会計の本質的変更、情報取得負担の増大が判明した場合、またはnative比較結果を科学的に解釈する段階でGPTへ返す。コード整備ごとに新しい詳細GPTレビューを要求する結論は出していない。一方、新規性や研究継続をこの形式監査だけで確定しない。

**mandatory STOP。旧成果・source・contract・authorization・消費済みmarker・STOPを保持し、新たな科学実行とGPT科学レビューを自動開始しない。**

[review]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4d80b45693a1ea7c14d7cbf9035e7358788953bd/docs/research/track_b_algorithm_direction_review_20261014.md
[plan]: parent_allocation_minimal_comparison_draft_20261014.md
[r0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66c96f8033bcf8392237d820d4a4615f3cebc75a/docs/tracks/algorithm_codesign/rte_reallocation_r0_independent_proof_v1.md
[g1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66c96f8033bcf8392237d820d4a4615f3cebc75a/docs/tracks/algorithm_codesign/g1_one_shot_result_validation_20261009.md
[g5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66c96f8033bcf8392237d820d4a4615f3cebc75a/docs/tracks/algorithm_codesign/g5_fixed_dictionary_closure_scope_20261010.md
[prior]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md
[identities]: ../../../artifacts/track_b_parent_allocation_audit/2026-10-14/v1/source_and_primary_identity.json
[checker]: ../../../artifacts/track_b_parent_allocation_audit/2026-10-14/v1/off_domain_formal_checks.py
[validation]: ../../../artifacts/track_b_parent_allocation_audit/2026-10-14/v1/off_domain_validation_receipt.json
