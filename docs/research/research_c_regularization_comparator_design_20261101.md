# 研究C：PR-tail目的と既存正則化の対応・公平比較の準備

2026-11-01 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。
起点：354a090da388c6d9718624bb4525ff3c72a22022。

**技術分類：COMPARABLE_DISTINCT_OBJECTIVE。** 同じ一般対称J・実軌道回転・tensor lossの下、RC-DFの全frame uniform L2目的を比較対象として準備した。現在のsmooth tail目的全体は、その固定された係数L1/L2重みを選ぶだけでは一般に再現できない。ただし、非平滑化かつ符号制約付きの領域では、重み付きL1二乗へ正確に帰着する場合がある。この区別は方法の独立新規性や性能を認定するものではない。

新しいfitter・optimizerは作らず、既存ffsim目的/gradient・修正codec・SectorCAR・一般Kを再利用した。人工新22＋fit/codec回帰88＋評価回帰48＋独立一般K回帰13＝**171 PASS、加えて29 subtests PASS**。保存H4の数値decode・新fit・科学的再解析・PF/RTE・shots・実回路build/compileはすべて0。

[未承認Stage A契約](research_c_regularization_comparator_contract_unapproved_20261101.json)は、新規1 fit・最大17 logical units・retry/resume0・fresh run01だけを対象とする。Stage B/Cは未承認かつ実行範囲外。今回の作業は技術比較と準備であり、重要GPT科学レビュー・論文化GOの正式判断ではない。

## 1. 引継ぎと確認資料

[Evidence Map](research_c_evidence_synthesis_and_decision_readiness_20261031.md)と[Stage A](research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md)の固定条件を引き継ぐ。保存直鎖H4/1.00 Å/STO-3G、CDF8/sector36の同x0でtail λ34.7246%低下・両DF予算PASS。元順LD2、δ=.1/s4/r16/K0・13round、own-DF oracle/状態準備除外、P0/N32価格proxyのideal条件付き比較でA/shots改善とPR内部総RZ83.09%削減、という既存結果を保持する。一方、同じJ/Uの[強決定論M2](research_c_pr_aware_df_M2_same_JU_deterministic_execution_20261028.md)に対する[M4 PR](research_c_pr_aware_df_M4_native128_execution_20261028.md)の主RZ優位は未支持である。

包括レビューresearch_c_comprehensive_scientific_review_20261031.mdはworkspaceのローカル文書として存在し、関連箇所を確認した。GitHubに公開済みとは扱わず、原本を転載していない。今回の依頼に記載された判断、公開Evidence Map、一次論文を根拠に進めた。過去の個別NO-GOをPR全般の不可能性へ拡張しない。

正式共通運用指示2026-10-25版はworkspace原本で実際に確認し、§8.1/8.2/8.4/10/10.1/10.3に従った。承認証明・承認原文保存・新grantは作成しない。

確認対象の一次資料は次の三点に限定した。本文で使用した式と節を明示し、DF全般の網羅的レビューへは拡張しない。

| 一次資料 | 確認箇所・使用目的 |
|---|---|
| [RC-DF v3](https://arxiv.org/html/2212.07957v3) | 式5/8/17、本文のweighted L1/L2、Appendix A式18以降、Appendix Bの強度選定。係数・loss・solverの対応。 |
| [SCDF v1](https://arxiv.org/html/2403.03502v1) | §2.2–2.4、式19/20/26/27、最適化付録。一般Jとrank-one/shift付き方式の区別。 |
| [PR v2](https://arxiv.org/pdf/2503.05647v2) | 有限RTE、§VIIのDF/orbital weight、Appendix C3。目的量と最終資源の区別。 |

## 2. 同じDF表現・tensor lossを比較する

空間軌道添字p,q,r,s、frame t、軌道内添字i,jを使う。現在の既存ffsim contractionは

\[
g_{\rm DF,pqrs}=\sum_{tij}U^t_{pi}U^t_{qi}J^t_{ij}U^t_{rj}U^t_{sj},
\qquad f=\tfrac12\sum_{pqrs}(g_{pqrs}-g_{\rm DF,pqrs})^2 .
\]

RC-DFの式5/8と添字を対応させれば、**空間係数Z_RCDF=J**である。Jは一般実対称、Uは実直交。RC-DFのspecial-orthogonal parametrizationも、det=-1の列符号を変えるgaugeにより同じdensity projector・tensorを表現する。表現集合にrank-one/PSD制約を追加していない。[RC-DF式5/8](https://arxiv.org/html/2212.07957v3)

この対応はtensor contractionの対応である。Spin-orbital number行列は別にK_ab=J_(a//2,b//2)/2であり、Z_RCDFをJ/2と置かない。J/U packは既存ffsimの実形式を使い、rank8/norb4では回転48＋対称係数80＝128 parametersとなる。

物理比較では、両目的に同じ元g補正κ、nuclear scalar一回、元frame順を適用する。論文のPauli形でone-bodyへ移された項やqubitizationのλを、現在のnumber fragmentのtail係数と混同しない。論文のmeasurement/qubitization実装全体の再現とは呼ばない。

## 3. PR-tail目的の正確な線形写像

既存[一般K](../../src/trottertracks/c1_general_k_adapter.py)と[PR目的](../../src/trottertracks/c_pr_aware_df.py)は、interleaved spin順a=2i+σについて

\[
K_{ab}=J_{ij}/2,\qquad c_{Z_a}=-\tfrac12\sum_bK_{ab}
 =-\tfrac12\sum_jJ_{ij},\qquad c_{Z_aZ_b}=K_{ab}/2\quad(a<b)
\]

を使う。Identityは抽出した相対位相として保持し、massから除外する。したがって、一般対称J一frameの非平滑massは

\[
\lambda_t
 =\sum_i\left|\sum_jJ^t_{ij}\right|
  +\tfrac14\sum_i|J^t_{ii}|+\sum_{i<j}|J^t_{ij}|.
\]

最初の行和項が重要である。Coefficientの符号相殺を含み、entrywise |J|の単純和ではない。この式は既存mapからの技術的導出であり、人工Jで独立general-K展開と照合した。

L_D=2以降の各frameについて、同じ写像Tでc=TJと書けば、保存された研究Cの目的は

\[
F_{\rm PR}=f/f_*+\gamma(M_\epsilon/M_*)^2,\quad
M_\epsilon=\sum_{\ell\in{\rm tail}}\sqrt{(TJ)_\ell^2+\epsilon^2},
\quad\epsilon=10^{-8}\ {\rm Ha}.
\]

これは**tail上の線形写像後のsmooth L1型massの二乗**である。Head2 frameのJは罰しないがtensor lossの変数としては自由で、Uはpenaltyへ直接入らない。固定frameラベルのtail選択であり、frameのweight順への並べ替えを導入しない。

## 4. 同値性・非同値性と適用条件

### 4.1 RC-DFのL1/L2を分ける

RC-DF式17はf＋Σ_tij ρ_tij|Z_tij|²。本文は、weighted L1/L2の一般形(Σρ|Z|^p)^(2/p)、p=1,2も記載する。一方、Appendix A式18はΣρ|Z|^pという形でL1を扱う。**本文のL1二乗と付録の非二乗L1を同じ関数とは扱わない。** 本比較は曖昧さのない式17のuniform L2値を採用する。[RC-DF式17/18](https://arxiv.org/html/2212.07957v3)

| 対象 | 現在のPR-tail目的との関係 |
|---|---|
| Entrywise weighted L2、ΣρJ² | 一般に同値でない。Tail maskだけでなく、行和・mass二乗の交差項・smoothingが異なる。 |
| Entrywise weighted L1、Σw|J| | ε=0でも次数1と2が異なり、γ>0のPR mass二乗全体を一般に表せない。 |
| Entrywise weighted L1の二乗、(Σw|J|)² | ε=0かつ符号制約付きの領域では再現可能な場合がある。任意のsigned J全体と現在のε>0には一般に拡張できない。 |
| 線形写像TJ上のsmooth normを許す一般化正則化 | 現在の式自体をその枠組みで書ける。これをRC-DF原論文の固定entry-weight選択だけの再現と呼ばない。 |

### 4.2 L2では再現できない理由

ε=0、非負の対角J=diag(a,b)、a,b>0だけを考えても、λ=(5/4)(a+b)なのでλ²にはab交差項がある。Entrywise weighted L2はw_a a²+w_b b²で、固定重みではこの交差項を表せない。複数frame間にもmassを全体で二乗する交差項がある。

### 4.3 Weighted L1二乗へ帰着する条件

ε=0で、各row内に符号相殺がない領域（例えば全entry非負、またはframe全体が非正）では

\[
\lambda_R=\sum_{t\ge2}\left[
\tfrac54\sum_i|J^t_{ii}|+3\sum_{i<j}|J^t_{ij}|\right].
\]

よってordered添字ならtailのw_ii=5/4、w_ij=3/2（i≠j）、headのw=0とすれば、(Σw|J|)²と同じである。正規化を外したf＋α λ_R²のα=γf*/M*²を入れる場合、L1二乗内部の重みをさらに√α倍すればよい。これは条件付きの**厳密な帰着**であり、その条件の下なら重複したfitを方法差の証明として要求しない。

一般のsigned Jにはこの帰着を延長できない。人工family J(t)=[[1,t],[t,1]]ではλ(t)=2|1+t|+1/2+|t|。t<0でもt=-1に行和の折れ曲がりがあるが、entrywise weighted L1はその区間でaffineである。固定非負重みで全区間を一致させられない。J依存の重みを結果ごとに選ぶことは、固定した正則化目的との同値ではない。

現在のε>0では、Mε²の零点値を単に定数として引いても、一般に次数2の同次関数へ戻らない。異なる係数を含むsqrt積が残るためで、人工rayでR(2J)-R(0)≠4(R(J)-R(0))を確認した。特別な低次元制約での一致まで否定せず、**一般対称J全体・現在のsmooth目的**について区別する。

例えばrank8/norb4/LD2ではtail coefficient slotsは216。Tail内の一つの対角Jだけをt>0、他を0にする人工rayでは

\[
M_\epsilon(t)=2\sqrt{t^2/4+\epsilon^2}
 +\sqrt{t^2/16+\epsilon^2}+213\epsilon .
\]

その二乗は大きなtで(25/16)t²+(5/2)·213εt+O(1)となる。固定entry重みのL1二乗/L2はこのrayで純粋なt²型なので、定数を除いても同じ関数にはならない。これは数理的な相違の例であり、小さなsmoothing差自体に有用な性能差があるという主張ではない。

### 4.4 SCDFとPR論文

SCDFの式27はV^t=W^t(W^t)^Tを使い、α^tのshiftも最適化する。一般対称Jの現在の変数集合・元κ固定・同じrank/x0にそのまま移せない。SCDF原方式との比較だけを問うならCOMPARISON_REQUIRES_DIFFERENT_MODELであり、今回は第一対照に選ばない。[SCDF §2.2–2.4・式26/27](https://arxiv.org/html/2403.03502v1)

SCDF Appendix Aも、WのL-BFGS更新→αのmedian更新→UのL-BFGS更新というnested手順であり、今回のjoint128変数・同停止政策とは異なる。変数制約とsolverの両方を分けて扱う。[SCDF Appendix A](https://arxiv.org/html/2403.03502v1)

PR論文は係数weight、orbital basis、DFを扱う。今回のλ_Rは現在のtail/event辞書のmassであり、全Hamiltonianのqubitization λやmeasurement varianceではない。K0の固定h/r/nならA=(1+(λ_R h/r)²)^(-rn/2)との接続はあるが、λ低下だけではPF/RTE誤差やnative価格を決めない。[PR §VII・Appendix C3](https://arxiv.org/pdf/2503.05647v2)

これらの結果から、比較可能な目的差はあるが、独立した新アルゴリズム原理・新規性・性能GOは認定しない。

## 5. 選んだ対照と強度の公平性

第一対照は、**全8 frame・全ordered空間添字のuniform L2**とする。

\[
P(J)=\sum_{tij}(J^t_{ij})^2,\qquad
F_{\rm RC2}=f/f_*+\alpha P(J)/P_0,\quad
P_0=P(J_0),\quad\alpha=0.1.
\]

f*はStage Aで固定済みの2.1839632053132894e-9 Ha²を再利用し、J0/x0も同じqualified snapshotを使う。P0だけを将来の登録objective_setup callで評価し、raw目的f＋ρPの**ρ=α f*/P0**を記録する。今回、保存H4からP0やρを計算していない。P0=0/nonfiniteならSTOPし、任意のfloorや別ρへ置換しない。

保存PR目的はx0でpenalty寄与がγ=.1、対照もα=.1になる。これは「ρ=.1をそのまま採用」する設計ではない。両目的で損失scaleと初期罰則寄与を合わせる、結果に依存しない一回の比較である。

RC-DFの原論文はρをsystematic errorとvariance/shot budgetの交換として調整する。分子・taskが違うそのρを数値だけ移さず、今回の規則を原論文のtuning手順そのものと称さない。[RC-DF Appendix B](https://arxiv.org/html/2212.07957v3)

| 揃えるもの | 揃えたとは主張しないもの |
|---|---|
| 保存H4/1Å/STO-3G、sector36、rank8、同x0・J/U変数、元g補正κ/scalar/frame順 | 最終tensor残差、最終sector誤差の完全一致 |
| 同ffsim loss・解析gradient、f*、実Schur codec、L-BFGS-B/options=None | 目的の曲率、初期gradient norm、iteration数・停止時stationarity |
| 同じ初期penalty寄与、DF admission budget CA/8=.00019875 Ha | 同じraw γ/ρ、最適ρ、DF予算境界に沿ったPareto最適解 |
| 事前固定の1 fit、同じ有限性/表現/数値guard | 複数seed/初期値の母集団、真の大域最適解 |

原論文はU更新と係数の線形解などの交互最適化を用いる。今回はjointな既存ffsim L-BFGS-Bを両目的で共通使用するため、名称は**「共通optimizer下でのRC-DF型目的比較」**とする。原論文のsolver・初期化・測定/Toffoli評価を含むRC-DF実装との性能比較ではない。

全frame L2とtailのみのsmooth L1二乗を対比するので、差はnorm形状と対象frameの両方を含む。「tail限定」一要素だけの因果ablationとはしない。

## 6. 最小実装・既存SOURCEの再利用

| 実装 | 再利用/今回の差分 |
|---|---|
| [c_pr_aware_df.py](../../src/trottertracks/c_pr_aware_df.py) | objective baseline、係数map、components、γ0経路を変更せず再利用。 |
| [c1_ffsim_corrected_codec.py](../../src/trottertracks/c1_ffsim_corrected_codec.py)、[実直交codec](../../src/trottertracks/c1_real_orthogonal_codec.py) | 同じqualified x0をprepared_initialでそのまま渡す。再logm・新初期分解・旧codec fallbackなし。 |
| [新penalty adapter](../../src/trottertracks/c_rcdf_comparator.py) | full ordered J²/P0のJAX x64 value/gradientだけを追加。Temporary factory hookを専用process内で復元する。 |
| [新Stage A接続](../../src/trottertracks/c_rcdf_comparator_pilot.py) | 旧prepare/fit entryの再起動を避ける別entry。保存packet、既存Calls/finish、SectorCAR/rank_assessment/representation_gateを呼ぶ。 |
| [新CLI](../../scripts/run_research_c_regularization_comparator.py) | Defaultはbyte/environment preflight。明示executeとfresh outputなしには科学処理へ進まない。 |
| [新人工試験](../../tests/management/test_research_c_regularization_comparator.py)、[runner試験](../../tests/management/test_research_c_regularization_comparator_pilot.py) | 係数・勾配・実fitter dispatch・mock optimizer・17-call/STOP・非適格/nullを検証。 |

対称JのPはdiag一回＋off-diagonal二回。上三角の独立parameterに対するpenalty gradientはdiag2J、off4Jであり、JAX ADをこの独立式・有限差分と照合した。U座標への直接penalty gradientは0。Penalty強度0で旧γ0のvalue/gradientに退化する。

既存のStage A目的・runner・codec・manifestは編集していない。なお、起点354a090daの再利用moduleのうち、c1_molecule_extensionとc1_stage3_referenceはStage Aのe1085c95fからの履歴にarithmetic keyword拡張を含む。差分を静的確認し、今回は従来と同じdefault arithmeticを使用する。PF referenceはこのStage Aでは呼ばない。旧SOURCE manifestのhashを書き換えず、新SOURCEに現在版とこの来歴差を明記した。

## 7. 人工・回帰検証と環境

| 試験群 | 結果・範囲 |
|---|---|
| 新penalty/runner | 22 PASS。人工2/6軌道とrank8形状のみ。Z=J、spin係数、off二重計上、loss/gradient、γ0、scale、det±1/π branch、異常値、hook復元、実ffsim経路/mock optimizer、成功/DF未達/optimizer不成功/不一致STOP。 |
| 旧fit/codec/x0 | 88 PASS。既存PR目的、qualified x0、workers、修正codec、実直交codec。 |
| 旧Stage A/PF-RTE/RPE接続 | 48 PASS。人工入力・mock・人工scalarのみ、実H4を使用しない。 |
| 旧一般K | 独立processで13 PASS＋29 subtests。 |

[検証記録とlogs](../../artifacts/research_c_regularization_comparator/2026-11-01/technical/verification.json)に実行環境と結果を保存した。初回に旧一般K試験を他試験とまとめたところ、Qiskit/trotterlib未ロードを要求するteardownと衝突し13 FAIL＋1 teardown errorになった。元テスト/SOURCEを変更せずprocessを分離し、全てPASSを確認した。これは科学的FAILやfit retryではない。

Fit環境は既存ffsim0.0.83/NumPy1.26.4/SciPy1.14.1/JAX0.4.38を読み取り専用で利用し、評価環境は既存Qiskit1.3.0等を使用した。[fit pins](../../artifacts/research_c_regularization_comparator/2026-11-01/technical/fit-pins.txt)／[evaluation pins](../../artifacts/research_c_regularization_comparator/2026-11-01/technical/evaluation-pins.txt)。既存のffsim loss/utilの2 SOURCE hashもStage A契約と一致する。共有/既存venvへのinstallや書換えは0。

確認時affinity32 CPU、MemAvailable55,008,416 kB、load約1.19/1.14/1.10。JAX CPUを利用し、tiny matrix BLASのみ自process内1 threadでpoolの過剰競合を避けた。新しいCPU/RAM/worker/wall capは設けず、cache/tmpはtask-local。将来の1 fitも専用processが必要で、独立した追加fitを並列に増やす意味ではない。空き資源は実行時に再確認する。

実optimizer iteration・実分子decode・新科学callは0。人工PASSは実H4のDF予算・性能・方法の新規性を証明しない。

## 8. 固定Stage A契約と実行順序

| 固定物 | Identity |
|---|---|
| 実装SOURCE | [25e4c3c8e9a2a949fd4f626f502089ca2d8c6332](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/25e4c3c8e9a2a949fd4f626f502089ca2d8c6332) |
| [SOURCE36](../../artifacts/research_c_regularization_comparator/2026-11-01/technical/SOURCE.json) SHA256 | 4074644f94b8b4af64601e86a87bb5114bcd0735d3d584bbd0e8e7b9b3e901ec |
| [未承認契約](research_c_regularization_comparator_contract_unapproved_20261101.json) SHA256 | 9b57ae71e79a6c6d7eb45036b53f28307d6b2c2957bc6d5fa60599935ab58cc3 |
| 元P-H4来歴 | 8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50 |
| h/g/c snapshot SHA256 | 1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db |
| Qualified J/U/x0 snapshot SHA256 | 78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856 |
| Stage A scale metadata SHA256 | 78b337df677a622876c8bf6b4593e1f338d9de46bc7594e00fa74f8eeb68b1e8 |
| 旧fit科学SOURCE | e1085c95f5c01b68566e17d543ace5832cb94061。γ0/.1終了J/U・diagnostic/sector/optimizerは保存原本を再利用。 |
| Fresh output | artifacts/research_c_regularization_comparator/2026-11-01/run01。今回は作成していない。 |

入力15件はcontractにhash固定した。現在の比較SOURCEと、元fitのSOURCE/初期snapshot/終了exportを独立に保持する。Normやtensorを過去候補について新たに再計算せず、旧scalarを読む場合も登録比較call内に限定する。

| 登録call | 最大units |
|---|---:|
| Saved snapshots load | 1 |
| Initial tensor assessment | 1 |
| Sector context / original physical sector | 1＋1 |
| Initial/final sector assessment | 2 |
| Initial/final independent representation gate | 2 |
| Initial gate | 1 |
| Fit process input load / objective setup | 1＋1 |
| Existing optimized fitter | 1 |
| Initial/final objective-gradient diagnostic | 2 |
| Final tensor assessment | 1 |
| Final snapshot/scalar load / comparison | 1＋1 |
| **合計** | **17** |

Parent11＋専用fit process6。Fit呼出し内部の初期/終了diagnosticも独立unitとして上表に含める。JAXのiteration内loss/gradientは既存の一fitに含む。複数試行を1 unitへ隠さない。

実行順序は、byte/environment/fresh確認→保存packet/初期tensor→sector36 physical reference→初期norm/guard・generalK/CAR→初期予算gate→P0/setup→実ffsimへの同x0接続と1 fit→終了tensor/sector/表現→旧scalar比較→STOP。

保存するのは、前後のobjective/gradient、exact/smooth tail λ、tensor Frobenius/max-entry、sector norm/guard、raw x/jac、success/status/message/fun/nit/nfev/njev、SOURCE・input identity、call journal、親子の資源記録とSTOPである。初期loss正規化とobjective exportには固定数値allowance、物理DFには既存sector_operation_scale_guard8_proposal_v1を使う。これらはengineering pilotで、厳密enclosureではない。

### 判定・STOP

- Input/SOURCE/環境/非finite/codec/独立表現のcorrectness不一致：依存後続callを止め、開始済み証拠を保存する。
- 初期DF予算未達：optimizerを開始しない。
- Optimizer successと終了DF予算PASSを別に記録する。有限な不成功終了なら、登録済みの終了診断とsector評価だけを行いSTOPする。
- 終了DF予算NOT_MET：ρ変更・rank変更・追加fit・初期値fallbackなし。固定強度での結果を保存する。
- Tail λの旧Proposedとの差は1e-10 Ha以内をUNRESOLVEDとする固定engineering点比較。差の大小を資源優位・CA・新規性へ昇格させない。
- 正常完了でもSTAGE_A_ELIGIBLE_FOR_RESEARCH_DECISIONまで。B/Cの自動開始なし。Retry/resume0、公開照合後mandatory STOP。

今回確認したのは、両環境の**passive preflightのみ**である。[evaluation](../../artifacts/research_c_regularization_comparator/2026-11-01/technical/evaluation_preflight.log)／[fit](../../artifacts/research_c_regularization_comparator/2026-11-01/technical/fit_preflight.log)。数値module importやNPZ decodeをせず、SOURCE/input bytes・依存版を照合した。

## 9. Stage Aで分かることとB/Cの分離

Stage Aの問いは、同じ初期値・同じ停止政策・初期寄与を合わせたfull L2対照でも、保存PR-tail候補と同程度のtail λをDF予算内で得られるか、である。どちらが良くても、固定1試行・固定強度の結果にとどまる。

| 将来のStage A結果 | 支持できること／次の判断 |
|---|---|
| RC2が予算内でtail λも同等以下 | このpilotではtail特化のproxy追加利益は示されない。既存正則化のtask-specific評価として位置付ける根拠になるが、資源同値はまだ不明。 |
| PR-tailのλが小さく、双方予算内 | 固定目的・強度政策でのtail proxy差はある。PF/誤差/価格で消える可能性があり、Bへ進む価値をユーザー/研究Cが判断する。 |
| RC2の予算未達またはoptimizer不成功 | 当該固定手順の不適格。RC-DF方式全体の失敗や、PR目的の大域優位とはしない。 |
| 差がallowance内・証拠不足 | 未解決とし、都合のよいρ再選定や自動追試はしない。 |

**Stage Bは別の判断・別契約・別許可**とする。既存のprepare_arm、一般K/CAR、tail_gate、PF/RTE kernels/reference_roundを再利用し、新RC2 exportのJ/U/x/hash、元h/g/c、own-DF ground/alias、元順LD2、13roundのphase-room/Aを新候補として接続する。既存[Stage B1](research_c_pr_aware_df_stage_b1_execution_20261027.md)・[M4](research_c_pr_aware_df_M4_s1_r128_feasibility_breakeven_20261028.md)の役割/SOURCEへ偽装しない。旧結果から価格・shotsを移送しない。

**Stage Cも今回の契約外**。B適格で資源比較が研究判断を変える場合だけ、既存RPE、native provider/P0、finite cohort校正、serial会計へ接続する。将来の比較では保存γ0/.1とのPR内部対照に加え、新RC2と同じJ/Uの強決定論S2対照が必要になる。候補固有のPF/RTE/shots/価格を取得するまでは、今回のproxyから条件付き総RZを推定しない。全law誤差・母平均価格・strict CAは未資格を維持する。

B/Cの候補条件・call数を先回りして科学実行契約に含めていない。Stage Aだけで研究判断が付くなら、B/Cを増やす必要はない。

## 10. Q1〜Q10と次の担当

| 質問 | 回答 |
|---|---|
| **Q1 どの正則化か** | 線形map後・tail限定のsmooth L1型mass二乗。Entrywise L2そのものではない。 |
| **Q2 特定重みで厳密再現できるか** | ε=0・row内符号相殺なし等の制約下なら、tail diag5/4/off3/2 ordered重みのL1二乗で再現可能。一般signed Jと現在のε>0全体には固定entry重みで一般化できない。 |
| **Q3 数学差の意味** | Full coefficient抑制と、実tailの行和/ZZ・head/tail配分を狙う目的を区別できる。性能・一般原理・独立新規性の証明ではない。 |
| **Q4 原方式との区別** | RC-DF式17の値に対応する目的を、既存joint ffsim optimizerで比較。原論文の交互更新・測定/Toffoli実装の忠実再現とは呼ばない。SCDFは別model。 |
| **Q5 第一対照** | 一般対称Z=J、全frame・全ordered添字のuniform L2。 |
| **Q6 揃える条件** | 同x0/入力/変数/loss/f*/L-BFGS-B/options=None、初期penalty寄与.1、同DF予算。最終誤差/gradient/最適ρ/大域解は揃えたとはしない。 |
| **Q7 再利用と新規部分** | Loss/AD/codec/fit、SectorCAR/guard/generalK、保存packet/Callsを再利用。追加はL2 penalty adapterと専用Stage A接続・人工試験のみ。 |
| **Q8 1 fitが変える判断** | 現在のtail目的のproxy利益が、この既存full L2対照でも得られるかという固定pilot判断。PR資源の追加価値を確定するのは必要ならB/C後。 |
| **Q9 最小実験** | まずStage A 1 fit/17 unitsまで。適格かつ次の判断を変える場合だけB/Cを別途検討。ρ/rank探索なし。 |
| **Q10 同値/不適格なら戻すか** | 一般的に同値と確定した条件では重複fitをせず位置付けへ戻す。不適格なら方式全体のNO-GOとせず、model/強度/solverの制約を示して研究判断へ戻す。今回の一般classでは比較可能な目的差があるため限定Stage Aを準備した。 |

次は**ユーザー/研究C**が、この初期寄与一致・最終誤差非一致・共通optimizerという比較の意味を確認し、固定17-callの科学実行を許可するか判断する。**Codex**は既に許可された技術実装・人工検証・公開を完了し、remote照合後mandatory STOPする。重要GPT科学レビュー、新fit、B/C、別ρ/rank/分子探索へ自動移行しない。
