# 厳密基底状態を用いる一体項再配分：数学・方法差・実装条件の事前監査

2026-10-14 JST。ユーザー指定の一回のCodex技術監査。受領した[方針再レビュー原文](../reviews/quantum_chemistry_pf/qc_exact_ground_state_pf_direction_rereview_2026-10-14.md)の数式を検証対象として扱った。今回、新しいHamiltonian、固有値、状態期待値の数値計算、数値最適化、PF性能評価、compile、samplingは行っていない。

二分割のPSD二次形式、E0不要の二重交換子、単一方向の局所二次性とbody-rank上限は、条件を明示した上で整合する。さらに、固定S2の先頭一体fragmentと最初のdensity fragmentの再配分には、**ground期待値の変化分を1-RDMだけで生成する明示式**を確認した。一般位置の最大3-/2-bodyという上限より狭い、特定orderingの還元である。

再配分自由度、state依存PF主要誤差、線形solve、compact BCHは既知。本監査ではこの1-RDM応答生成そのものの明示記載を確認できなかったが、独立新規性・既知法への効率優位を認定していない。具体的な差分候補を対象にPhase Bの静的費用と最小pilot案まで整理した。production実装・pilot起動は行わず、公開後Mandatory STOPする。

## 1. 固定前提と証拠

- 厳密**量子**基底状態を利用可能とする。状態準備法の研究へ戻らない。E0は設計入力として与えない。
- 必要なRDM収縮の取得・処理費用は別建て。無償の全古典2-RDMや全固有分解は仮定しない。
- block-encoding/qubitization/QSVT設計は対象外。Track A/Bの科学作業、C1–C4や終了済み系列、第二候補のDF回転を再開しない。
- 元Hとは、scalar・一体normal-order補正を含めた固定Hermitian有限基底Hamiltonian。既存DF切断があればその誤差は別会計で、再配分で消えたと扱わない。

Base `5f35988d3145641739fb83da98567772183046ef`、branch `qc-exact-ground-pf-onebody-preaudit-20261014`。[受領原文SHA](../../artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/received_review_identity.json)、[旧source/本文identity](../../artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/reference_identity.json)、[scope](../../artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/scope.json)、[一次本文registry](../../artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/literature_registry.json)。原レビューをbyte保存し、GPTレビューの再実施として数えない。

## 2. Phase A：二分割の独立確認

### 2.1 符号、順序、ground条件

`S2=e^(−ihA/2)e^(−ihB)e^(−ihA/2)`、`S2=exp[−ih(H+h²D+O(h⁴))]`、H=A+Bの規約を固定する。形式logから

\[
D=\frac1{24}[A,[A,B]]+\frac1{12}[B,[A,B]],\qquad
D+\frac1{24}[A,[A,H]]=\frac1{12}[H,[A,B]].
\]

exact groundで `〈[H,K]〉0=0`、Hermitian Aについて `〈[A,[A,H]]〉0=−2〈A(H−E0)A〉0`。従って

\[
d=\langle D\rangle_0=\frac1{12}\langle A(H-E_0)A\rangle_0\ge0.
\]

groundがsector限定なら、Aと再配分方向がそのsectorを保存することが必要。任意励起状態には非負性を移せない。外側Aと中央Bを交換すると係数の対応が変わるため、順序を同一視しない。

`A_x=A+X`、`B_x=B−X`、`X=Σx_pN_p` の和はHを厳密に保つ。同じ式より

\[
12d(x)=c+2b^{\mathsf T}x+x^{\mathsf T}Mx,
\quad M_{pq}=\operatorname{Re}\langle N_p(H-E_0)N_q\rangle_0,
\quad b_p=\operatorname{Re}\langle N_p(H-E_0)A\rangle_0.
\]

Mは実Gram行列でPSD。`u∈ker M` なら `(H−E0)^(1/2)Σu_pN_p|ψ0〉=0` なのでu·b=0。よってb∈range M、`x*=−M⁺b`、`dmin=(c−bᵀM⁺b)/12`。総粒子数等のnull方向を除けるが、regularizationやboundsを追加した解を無制約最小値と表示しない。単なるpseudoinverseは既知の線形代数である。

### 2.2 E0不要の情報と、1-RDMへの精密化

\[
M_{pq}=\tfrac12\langle[N_p,[H,N_q]]\rangle_0,\quad
b_p=\tfrac12\langle[N_p,[H,A]]\rangle_0,\quad
c=\tfrac12\langle[A,[H,A]]\rangle_0.
\]

ground条件を展開すると上のenergy-weighted Gramと一致する。対称性・実部規約を保持する。Np,Aが一体、Hが二体なら最大二体。さらにBとNpが可換なら

\[
M_{pq}=\tfrac12\langle[N_p,[A,N_q]]\rangle_0,
\qquad b_p=\tfrac12\langle[A,[A,N_p]]\rangle_0.
\]

後者は `[Np,[H,A]]=[Np,[B,A]]`、Jacobi、ground上のH交換子消去から得られる。**Mだけでなくbも一体期待値で足りる**。元レビューの「1-/2-RDMで記述」は上限として正しいが、二分割classの更新係数M,bは1-RDMへ下がる。c・絶対主要誤差の取得は別である。無制約の更新を求めることと、dminを評価することを区別する。

非縮退ならd(x)=0はA_x|ψ0〉がgroundと平行であることを意味し、B_xも同じstateの固有演算子となる。そのstate上のPF位相はexactになる。これは一般分子で達成可能という予測ではない。縮退ではA_x|ψ0〉がground空間内にあることしか得られず、その単一stateから全hの正しい位相を結論しない。

## 3. Phase A：一般DFの局所応答

固定palindromeを

\[
S_2=e^{-ihF_0/2}\cdots e^{-ihF_{L-1}/2}e^{-ihF_L}
e^{-ihF_{L-1}/2}\cdots e^{-ihF_0/2},\quad F_0=A
\]

とする。`R_i=Σ_{j>i}F_j` を用いた既知の再帰BCHは

\[
D=\sum_{i=0}^{L-1}\left\{\tfrac1{24}[F_i,[F_i,R_i]]+
\tfrac1{12}[R_i,[F_i,R_i]]\right\}.
\]

F0に+tY、Fkに−tYを移すとDの三次t係数は `[Y,[Y,Y]]=0` で消える。有限hの全logの二次性ではない。二次演算子係数は、P_k=Σ_{0<i<k}F_iとして

\[
Q_k(Y)=-\frac1{24}[Y,[Y,F_0+F_k+3P_k]].
\]

DF方向で[Y,Fk]=0ならFk項が消える。この式だけで一般位置のground上QがPSDとはいえない。number-conserving k-/l-bodyの交換子上限k+l−1を用い、D≤4、線形応答≤3、二次応答≤2となる。一体との交換子はrankを増やさない。別々の非可換X_lを複数fragmentへ同時に動かすと三次項が残り得る。共通一方向の局所式をglobal convex quadraticへ一般化しない。

### 3.1 隣接する最初のdensity fragment：明示的な低費用応答

ここでk=1、B=F1、T=Σ_{j≥2}Fj、H=A+B+T、`[X,B]=0`。尾部TはXと可換とは仮定しない。再帰式のi=0/1だけが変わり、独立な形式展開で

\[
\boxed{D(X)-D(0)=
\frac1{12}[A,[A,X]]+\frac18[H,[X,A]]+\frac1{24}[X,[A,X]].}
\]

従って任意exact eigenstate上で

\[
\boxed{d(X)-d(0)=
\tfrac1{12}\langle[A,[A,X]]\rangle_0+
\tfrac1{24}\langle[X,[A,X]]\rangle_0.}
\]

この**変化分の式**にはE0・尾部の期待値・3-/4-RDMが現れない。A,Xが一体なら、両項も一体である。三体になり得るoperator応答と、ground上でH交換子を除いた等価な応答を区別する。`[H,[X,A]]` はground対角に寄与しないが、有限hのPF全体が単なる共役であるとはいわない。

### 3.2 係数表の具体的生成手順

F1の固定density frameでA=Σa_rs a_r†a_s、`γ_rs=〈a_s†a_r〉0`、`X=Σx_p n_p`。期待値は `tr(γ o)`。一体CAR閉包 ` [dΓ(a),dΓ(b)]=dΓ([a,b]) ` より

\[
d(x)=d_0+g^{\mathsf T}x+x^{\mathsf T}Qx,
\quad g_p=\tfrac1{12}\operatorname{tr}(\gamma[a,[a,n_p]]),\quad
Q_{pq}=\tfrac1{24}\operatorname{tr}(\gamma[n_p,[a,n_q]]).
\]

Qは実対称。w_rs=−Re(a_rsγ_sr)/12（r<s）からQのLaplacian表示を生成できる：Q_rr=Σ_{s≠r}w_rs、Q_rs=−w_rs。一般多fragmentではこのLaplacianはPSD保証を持たない。全N方向はnullで、Σg_p=0。

\[
g_p=\tfrac1{12}\{(\gamma a^2)_{pp}+(a^2\gamma)_{pp}-2(a\gamma a)_{pp}\}.
\]

Hermitian誤差を丸めで隠さず、実部を適切に検証する。固定γをframeへ回転し、dense matrix積でgをO(n³)、QをO(n²)、保存O(n²)で作る。F1との再配分でaのoff-diagonalが変わらないのでQは一定、g(x)=g(0)+2Qx。H・exact state・F1 frameが同じ間、γを取り直す必要はない。これは具体的な応答表の生成・再利用規則であり、本番collector/optimizerの実装ではない。

d0は、**一度の**compact BCH期待値等で別取得する。二分割の無制約PSD更新と異なり、一般DFで|d(x)|を選択するにはd0の値またはintervalが必要。1-RDMから元二体HのE0やd0まで無償に復元できるとは仮定しない。

## 4. 数学検査の範囲

[非可換語checker](../../artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/formal/check_word_identities.py)、[27検査の結果](../../artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/formal/word_identity_results.json)。stdlib Fractionを使い、抽象fragmentの指数級数とlogを三次まで直接乗算する経路を、nested-commutator再帰式と比較した。

抽象fragment数2…6、各局所位置の三次消去・閉じたQ式、最初のdensityとの可換条件下での線形応答恒等式、二分割ground交換子還元、複数異方向の三次項を検査し、27/27 PASS。数値Hamiltonian・state・PF unitary・期待値・eig・optimizerを一切作らない。rank/CAR、PSD、finite-h保証、新規性は手動証明・別の条件の問題で、word checkerのPASSに含めない。これは局所技術検査であり外部科学再現・CIではない。

## 5. Phase A：強い先行研究との対応

| 一次本文・確認箇所 | 既知内容 | 今回の具体手順との差／境界 |
| --- | --- | --- |
| [FFF, Quantum 7,889](https://quantum-journal.org/papers/q-2023-01-03-889/pdf/), 2.1 Eqs.(9)–(10), 2.2 Eq.(14), 結論 | 同じ一体再配分、固定measurement配分でのcovariance linear solve。結論にTrotter-error応用を明記 | 自由度・linear solve・Trotter応用は既知。energy-weighted応答をvarianceと同一視しない。先頭pairの1-RDM表の明示記載は未確認 |
| [Partitioning assessment v2](https://arxiv.org/pdf/2210.10189v2), 2.3 Eqs.(37)–(42) | 同じ再配分と可解一体fragment、reflection/L1 heuristic、basis融合 | 本案はstateの主要PF固有値目的。frame保持だけを無料費用や方法差としない |
| [PF eigenvalue error, 2025出版版](https://pubs.rsc.org/ja-jp/content/articlepdf/2025/dd/d5dd00185d?page=search), II/IV/AppendicesD/F | BCH主要ground期待値でpartitioning・QPE費用を予測 | exact stateに替えるだけでは新方法ではない。変化分の1-RDM生成が比較対象 |
| [Orbital transformations v1](https://arxiv.org/html/2602.18913v1), 2.3/結論 | state-specific errorとorbital/order/compositionの影響 | 第一候補はdensity frameを固定するrestricted再配分。一般orbital改善へ拡大しない |
| [Theory/practice, SPRINT v1](https://arxiv.org/html/2606.30741v1), II.1/IV/V.3 | 一体成分のfragment配置、Gaussian/density circuit、構成と誤差の交換 | 同じ構造classというだけでは新規性なし。既知の安い配置・融合にも同じ自由度を許す |
| [Practical error estimation v1](https://arxiv.org/html/2606.30738v1), III.2/IV.1 | S2をO(L)のgrouped commutator、S4をO(L²)へ圧縮、再帰/caching/state contraction | 本案の強い対照。O(L³)列挙との比較だけでspeedupを主張しない。少数のdirect評価から二次式を回復する対照も含める |
| [Corrected PF v3](https://arxiv.org/html/2409.08265v3), II–IV | 純processorと挿入corrector、補正の構成費用 | 全CPFを固有値不変と誤認しない。適用可能な真のcorrectorを費用付き対照にする |

本文7件を取得した（6件はローカルbytes/SHA、RSCはpublisher PDFのweb本文）。重要本文の欠落は0。関連節の限定照合であり、全分野の初出性証明ではない。原レビューの補助low-energy文献を新たに全証明監査したとは数えない。

**判定の強さ**：変換、state-specific目的、BCH、二次式の最小化はREDUCES_TO_KNOWN。特定orderingの1-RDM応答生成はSUPPORTEDな技術構成で、確認した本文にその明示手順を見いだしていない。これは「文献にないことが証明された」「独立新算法を採択した」という意味ではない。同じground恒等式・合法cachingを許したcompact BCH対照が同じ表を安く生成できる可能性を残す。独立性・全費用利益はUNDETERMINED。

ここでは単なる二次最小化を差分とせず、情報rankと再利用が明示された一つの限定生成構成を対象に、Phase Bの静的整理だけへ進んだ。一般位置・全fragmentを広く探索するgeneratorへは進まない。

## 6. Phase B：量子化学DF・情報・回路費用

`F1=U1†(ΣVpq n_pn_q)U1` に対し `X=U1†Σx_pn_p U1`。`F1−X` は同frameのdensity+linear、F0+XはGaussian one-bodyのまま。signed DF weightにも適用可能。scalar、n_p²=n_p、full Hの一体補正、spin/number sectorを固定する。

| 成分 | 静的条件・費用 | 未確定点 |
| --- | --- | --- |
| γ取得 | exact quantum stateから必要な1-RDM情報を測る。素朴な均等精度取得はO(n² ε_γ^-2 log(n/α)) copiesの上限型会計 | 測定回路、共分散・より良い取得法、必要精度を含む実単価UNKNOWN |
| γの誤差 | Schatten trace normで `‖γhat−γ‖₁≤η` が保証されれば各応答誤差≤η times 一体係数matrix operator norm。h²を掛けenergy予算へ戻す | exact quantum stateは古典γのexact無料取得を意味しない |
| 応答表 | γのframe変換＋g O(n³)、Q O(n²)、memory O(n²)。γを再利用 | d0の取得・誤差/bit長・失敗率は別。全RDM保存は要求しない |
| 更新・方向 | 固定Yではℓ=g·y、q=yᵀQy。d0+ℓt+qt²の絶対値は端/実根/頂点/0で候補化 | scalar minimizationは既知。有限h/回路価格の最良を保証しない |
| 新H0 frame | one-body対角化とGaussian生成は一般dense O(n³) classical/O(n²) Givens型 | 元疎H0がdenseになる費用、隣接U0/U1 fusionの変化 |
| density phase | 既存frameを保持。各linear位相は既存coefficientへ融合し得る | 新非zero位相やsquare-arithmeticの別linear実装は追加費用 |
| controlled回路 | scalarとcontrol relative phase、Gaussian adjoint/compute-uncompute、rotations synthesisを全対照で保持 | T/depth/gate実費UNKNOWN、compile未実行 |
| finite-h精度 | leading d、h4残差、PF branch/overlap、合成誤差を分離 | d低下のみで同energy精度/成功率や総費用を主張しない |

期待値取得のcopy数式はbounded observablesへの標準集中会計を表す素朴な設計上限であり、新測定法・最適性のclaimではない。state準備法を課題へ戻さず、供給されたexact stateの利用回数と必要な情報測定を記録する。完全2-RDMを無料にするとE0も収縮から分かるという原レビューの注意は保持するが、本限定応答表は1-RDMだけを要求する。

最も強い古典対照は、既知grouped BCH＋ground恒等式＋同じ合法cacheを使い同じ応答表を生成する方法。単に三点のdirect値から二次式を復元する対照も許す。取得backendが十分賢ければ同じscalingへ達し得るため、古典speedupは未認定。量子資源では再配分S2だけでなく、同情報からのscalar-corrected S2、S4、外挿、既知再分割/ordering、適用可能なCPFとの比較が必要。

既存の[PF検証source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f35988d3145641739fb83da98567772183046ef/src/trotterlib/pf_delta_validation.py)はread-onlyで確認した。`_phase_and_perturbative_energy_bias` 等はground_energyを引数に取り、truth診断として使う関数である。未来の監査側への再利用候補であって、E0未知の生成器へそのまま流用しない。一体応答collector・interval selector・独立finite-h checkerは未実装で、既存科学sourceをimport・実行・変更していない。

## 7. 最小pilot案と、実行認可境界

[最小pilot案・未固定項目](../../artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/minimal_pilot_design.json)、[GPT判断用引継ぎ](qc_exact_ground_pf_repartition_gpt_handoff_v1_20261014.md)。これは結果前の設計案であり、source/input SHA付き実行契約・実行承認ではない。

最小の情報価値は、一般DFの一つの既存固定Hamiltonianで、第一pairの1-RDM表がdirect compact BCHの変化分と一致し、取得・生成費用が本当に異なるか、その後有限hでscalar補正等を含む対照を超える余地があるかを区別すること。二分割toyだけを成功として一般分子へ広げない。

分子名・固定pack・ground供給方法・方向/bounds・h grid・target・call caps・runtimeは未固定のまま。新生成・数値expectation・optimize・eig・compileを実行しない。execution-readyと表示せず、GPTが構成と方法差を科学判断した後、必要なら具体source/input/認可を別に固定する。

## 8. 保全と停止

既存387files、旧146manifest entries/non-result fields/global description欠落FAIL、元worktree135pathsを保全する。[保存索引](../../artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/preservation_index.json)。受領レビューに記載されたC1現系列終了・C2対照保持・C3/C4現構成への投資見送り、N1 STOP/N2 HOLD/N3 CLOSED/THRIFT現template CLOSEDは新たにCodexが決定したものではない。旧source/結果/認可/STOPのbytesを変えず、受領判断へのpointerを追加する。

数値科学calls0、C1–C4再監査0、Track A/B科学作業0、production0、GPTレビュー0。[GitHub再取得receipt](../../artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/publication/receipt.json)で公開物を照合後Mandatory STOP。次はユーザーの別明示開始承認を受けたGPTの重要科学判断。同じ候補の一般文献探索や数値取得を自動延長しない。
