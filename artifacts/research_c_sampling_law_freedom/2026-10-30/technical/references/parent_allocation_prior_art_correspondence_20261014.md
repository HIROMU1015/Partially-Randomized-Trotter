# Parent allocation：近接先行研究との厳密対応

2026-10-14 JST。**技術的な同値・包含・差分の調査完了。独立新規性・科学的採否・実行価値は認定しない。科学条件7領域は未採択のまま。mandatory STOP。**

今回の配分は既知のEuler identityをchildごとに使った構成であり、固定辞書に対するcost×moment最適化も既知ISの直接特殊化である。一方、任意配分による角度/dictionaryの変更、P5のroot局所生成、finite-bit/strict native会計までを、調査した既知アルゴリズムと同じ情報・手順・費用として対応させた証拠は得られていない。**一般理論への包含を完全な実装同値とせず、具体差分を独立新規性とも呼ばない。**

## 1. 範囲と分類

入力は公開 `50c3db592041709247e5bd8f3d6849b6d20f3c10` の[Decision Packet](parent_allocation_scientific_design_decision_packet_20261014.md)、監査 `c12558d682f6aef3cfc68e8d75082269f2282ac8` の[技術報告](parent_allocation_technical_audit_20261014.md)、GPT方針 `4d80b45693a1ea7c14d7cbf9035e7358788953bd` の[原本](../../research/track_b_algorithm_direction_review_20261014.md)。3資料を指定GitHub commitから取得してbytes/SHA一致を確認した。[入力receipt](../../../artifacts/track_b_parent_allocation_prior_art_correspondence/2026-10-14/v1/fixed_input_identity.json)。

分類Aは**明示した対象範囲**で対象・自由度・手順を対応できる場合、Bは一般理論への包含または明示した特殊化で生成/情報access差が残る場合、Cは限定条件の具体差、Dは資料から確定できない事項である。一論文について、局所式がAでも全candidateはB/C/Dとなり得る。

| 先行研究／対象水準 | 技術分類 | 確認した対応と限界 |
| --- | --- | --- |
| Wan–Berta–Campbell | **B**。ordinary endpointの理想samplerは限定**A** | Euler/paired Taylorのcommon-angle点を包含。任意child配分とreturn後rootの生成までは同じ手順でない |
| PTSC | **B**。一atomのEuler式は**A** | identityの共通比率配分を回復できる。一般z、P5 return群、同じ有限target/情報accessへは追加特殊化が要る |
| SCU/CTS | **B**。local common-angle式は**A**、literal collected辞書との比較は**C** | operator分解とcoherenceを含むので比較可能。Pauli collectionと一般involution root生成は同一ではない |
| SPS | 原論文のfull-process一致問題には**C**。以下の線形埋め込みには**B** | first momentだけの自由度はそのままfull-channel制約の部分集合とはいえない。projected/signed embeddingを明示すれば同じ線形枠組みで記述できる |
| cost-aware IS | 固定辞書の内側問題は**A**。非common配分との比較は**C** | qだけの変更はc/Vを変えない。角度と係数を変える外側配分まで同一定理が最適化したとはしない |
| R0 | 固定IID・degree-local classに対し一般child配分は**C** | ordinary common点は一致し得るが、label非依存angleという制約を一般zが破る |
| G1/G5 | 固定prototype/angle/word sharesに対し**C** | 新角度の連続familyを同一固定有限辞書の係数変換で覆えない。旧closureの前提は維持 |
| 全candidateの同じ有限bit/native/source取得までの完全同値 | **D：未確認** | 全既存実装・全parameter化の不存在や独立priorityは調査していない |

全candidateに対するAは確認できていない。既知部品との局所Aを理由に、構成全体を救済・再設計したり、新しい自由度を追加したりしていない。

## 2. 一次資料と実際の取得版

日付はarXivの初回公開／選択版公開日であり、PDF typesetting日と分ける。SPS/IS/PTSCのHTMLは前回の保存bodyのSHAを確認して再利用、SCU/Wanは同じ指定版を取得し過去receiptとも一致した。HTML/PDFの**関連本文**を読んだ。abstractだけの推測、全論文精読、網羅的priority調査ではない。[取得identity](../../../artifacts/track_b_parent_allocation_prior_art_correspondence/2026-10-14/v1/primary_retrieval_identity.json)／[関連箇所索引](../../../artifacts/track_b_parent_allocation_prior_art_correspondence/2026-10-14/v1/correspondence_evidence_index.json)。全文は一時領域で参照し、リポジトリへ複製していない。

| ID | 論文・著者 | 初回／版と公開日 | 本文で確認した位置 |
| --- | --- | --- | --- |
| SPS | [Sparse Probabilistic Synthesis of Quantum Operations](https://arxiv.org/html/2402.15550v2) — Bálint Koczor | 2024-02-23／v2, 2024-10-31 | II.1–II.3、Statements 1–2、Eqs.(1)–(6)。process辞書、sampling、design matrix、L1問題 |
| SCU | [Quantum Simulation via Stochastic Combination of Unitaries](https://arxiv.org/html/2407.21095v2) — Joseph Peetz, Scott E. Smart, Prineha Narang | 2024-07-30／v2, 2026-07-06 | II.1、II.3 Theorem 1 Eqs.(4)–(7)、IV.2、Supplementary Note 5 Eqs.(S13)–(S15) |
| IS | [Resource-Optimal Importance Sampling for Randomized Quantum Algorithms](https://arxiv.org/html/2603.13495v1) — Davide Cugini, Touheed Anwar Atif, Yiğit Subaşı | 2026-03-13／v1, 同日 | II Eqs.(7)–(10) Theorem 1、III Theorem 2 Eqs.(21)–(26)。固定protocolのnet-costとbias |
| WAN | [A randomized quantum algorithm for statistical phase estimation](https://arxiv.org/pdf/2110.12071v2) — Kianna Wan, Mario Berta, Earl T. Campbell | 2021-10-22／v2, 2022-07-13 | Lemma 2、Appendix C、PDF p.15 Eqs.(C4)–(C5)、p.16 Algorithm 2、F.2 finite cutoff |
| PTSC | [Simple and high-precision Hamiltonian simulation by compensating Trotter error with linear combination of unitary operations](https://arxiv.org/html/2212.04566v2) — Pei Zeng, Jinzhao Sun, Liang Jiang, Qi Zhao | 2022-12-08／v2, 2025-03-27 | II Eq.(7)、IV.2 Eqs.(74)–(75)。K0/異なるtargetの整理は旧R0.5監査を継承 |

SCUはv2を使用する。今回の初期web取得でv3指定が失敗したことを、必要なv2本文の未取得とは扱わない。ISのweb一時エラーも、byte一致の取得済みv1本文と後続web確認から回復した。必須5本に本文未取得はない。新しい版との全差分比較は行っていない。

過去の[R0.5対応監査](rte_reallocation_r05_equivalence_novelty_audit_v1.md)、[G6先行研究照合](g6_prior_art_and_method_delta_20261010.md)、[D1〜D3のclaim/evidence](d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md)を継承した。既済の数値・先行研究全体を再監査しない。

## 3. Euler pairing：何が既知で、どこが自由度の変更か

既存監査の記号を用い、M_u=ζ(aI−iσΣ_i b_i Q_i)Q(u)、Q_i²=I、a,b_i>=0、偶数parent uとする。zero event/純parent/純childは既存sourceの端点であり、巨大ratioを導入しない。

任意配分は、最初に同一parentを線形に分割して

\[
aI-i\sigma\sum_i b_iQ_i=\sum_i(z_iI-i\sigma b_iQ_i),\qquad \sum_i z_i=a
\]

と書き、各項に既知の一involution Euler identityを適用したものと正確に対応する。**c_i=sqrt(z_i²+b_i²)、phi_i=atan2(b_i,z_i)という局所式そのものは新しいpairing原理ではない。** 前後のζとQ(u)はword順序とcontrolled相対位相を保って掛ける。

これはcommon点だけでなく、**全適格zに対する平均保存恒等式が既知identityと線形分割の直接帰結**という確認である。区別して未確認としたのは、配分の選定・情報取得・有限bit/native費用まで含む具体アルゴリズムの完全対応である。

common-angleはs=Σb_i、z_i=a b_i/sであり、a>0なら

\[
a\sum_i\frac{b_i}{s}\left(I-i\sigma\frac{s}{a}Q_i\right)
=\sqrt{a^2+s^2}\sum_i\frac{b_i}{s}e^{-i\sigma\arctan(s/a)Q_i}.
\]

この特殊点はSCU S15とPTSC Eq.(75)のnormalized identity allocationへaとsignを戻した式である。一般zでは各termのb_i/z_iが異なる。原文の共通angle・固定比率のsamplingをそのまま使う手順とは区別する。[SCU Note 5](https://arxiv.org/html/2407.21095v2#Sx3)、[PTSC Eq.(75)](https://arxiv.org/html/2212.04566v2#S4.E75)。

WanのC4/C5はeven Taylor order nにcommon angle atan(x/(n+1))とIID labelsを使う。符号・word順の対応は、branch全体のadjointとIID wordの逆順reindexによりordinary endpointで一致する。rotationを非可換wordの反対側へ交換してはいない。これは[R0.5 §2](rte_reallocation_r05_equivalence_novelty_audit_v1.md)の既済確認を継承する。return後のparent係数・条件付きchild law・任意zまでを同じAlgorithm 2とする結論ではない。[Wan Appendix C](https://arxiv.org/pdf/2110.12071v2#page=15)。

SCUのliteral CTSはTaylorのPauli係数をcollectし、real群とimaginary群を扱う。一般involutionの語はPauliに閉じるとは限らない。P5 sourceはQ_i²=Iだけのreturn構造からrootのa,bを取得するので、全Pauli collectionを必要としない。この差は具体的な入力・生成手順の差であり、理想meanの比較不能や性能優位を意味しない。Pauliアクセスがあるcontextではliteral collected対照も同じ情報条件で検討できる。[SCU IV.2](https://arxiv.org/html/2407.21095v2#S4.SS2)。

## 4. IS、S(z)、KKT：内側の既知最適化と外側の配分

固定zで全event係数をc_e、native単価をC_e>0、B=Σc_eとする。原protocolのp_e=c_e/BにIS Theorem 1を適用し、測定値の全体scale Bを戻すと

\[
\left(\sum_e q_e C_e\right)\left(\sum_e c_e^2/q_e\right)
\ge\left(\sum_e c_e\sqrt{C_e}\right)^2,\qquad
q_e^*\propto c_e/\sqrt{C_e}.
\]

**この内側の解析式は既知理論の直接特殊化A**である。原論文のnet-costと同じcost×weight-second-moment proxyであり、有限confidenceのceil/range/biasを含む最適十分Nではない。[IS Theorem 1](https://arxiv.org/html/2603.13495v1#S2)。

非rootが固定ならmin_q proxy=(S_rest+S(z))²、S(z)=Σ_i sqrt(C_i)sqrt(z_i²+b_i²)。正の固定C、正b、a>0という既存監査の仮定で、外側min_zのLagrange停留条件は

\[
\sqrt{C_i}\frac{z_i}{\sqrt{z_i^2+b_i^2}}=\lambda_K,
\quad z_i=\frac{b_i\lambda_K}{\sqrt{C_i-\lambda_K^2}}.
\]

これは**既知IS proxyを代入した後の標準凸KKT**であり、新しい一般IS定理・独立priorityを付けない。配分による辞書変更を原IS Theorem 1自体が最適化したのでもない。内点一意性とzero-cost/zero-bの仮定外処理は前回[Decision Packet §3–4](parent_allocation_scientific_design_decision_packet_20261014.md)の証拠を使用し、候補の探索を繰り返さなかった。

pure ISは固定の係数・unitary・近似実装を保持し、qだけを変える。非common zではc_iとphi_iとV_iが変わるため、元dictionaryのp/q weightだけでは同じ新ensembleを生成できない。先に全z候補を含むlibraryを用意するならISを適用できるが、そのlibrary構成・係数選択は別の操作である。IS Theorem 2のbias不変も固定実装に対する命題であり、角度を変えて実装誤差が変わる比較へ無条件に移さない。[IS Theorem 2](https://arxiv.org/html/2603.13495v1#S3)。

角度依存C_i(phi_i)では外側の目的に価格変化が入り、離散T価格へ滑らかなKKTを移せない。root-local proposalはgroup mass固定・rest込みでは一般にglobal proxy optimumでなく、finite-bit後には実qによるmoment/rangeを使う。この整理は前回Packetのまま維持する。新しいzero-cost処理・正則化・proposal最適化・精度を採択していない。

## 5. SPSとの厳密な対応：first momentをfull channelと区別する

SPSは既知のprocess columnsを並べ、固定design matrixの係数を求める。原問題のfull-process equalityとL1目的はEqs.(4)–(6)、samplingはStatements 1–2にある。一般的な辞書/係数/統計負担の交換は既知である。ただしoperator Mの一致だけを指定する今回の制約を、原論文の同じunitary-channel targetの制約と同一とはしない。[SPS II](https://arxiv.org/html/2402.15550v2#S2)。

以下は**本調査での対応式の導出**であり、論文にこのParent/P5手順が記載されているという引用ではない。C(V)=diag(I,V)、A_V=Ad_{C(V)}、入力block X=[[X00,X01],[X10,X11]]と置くと

\[
\sum_i c_i\mathcal A_{V_i}(X)=
\begin{pmatrix}
L X_{00}&X_{01}M^\dagger\\
M X_{10}&\sum_i c_iV_iX_{11}V_i^\dagger
\end{pmatrix},\quad M=\sum_i c_iV_i,\quad L=\sum_i c_i.
\]

Mが等しくてもLや右下blockは等しいとは限らない。normalized physical channelでは左上はX00、off-diagonalはM/Lになる。Hadamard first-moment推定はclassical weight Lを戻してMを読む。**同じmean、同じnormalized channel、同じweighted full processは別の命題**である。

一方、process手法から切り離す必要もない。block projection Π_offによってΠ_off Σc_i A_ViがMだけで決まる。あるいは、同じ線形process spanへの明示的なliftは

\[
\frac{\mathcal A_V-\mathcal A_{-V}}2(X)
=\begin{pmatrix}0&X_{01}V^\dagger\\VX_{10}&0\end{pmatrix},
\quad
\mathcal O_M=\sum_i\frac{c_i}{2}(\mathcal A_{V_i}-\mathcal A_{-V_i}).
\]

である。全zに対し同じfull linear map O_Mを得て、係数L1はL。**O_Mは一般にCPTP channel/desired unitary gateではない**。したがってこれはSPSの線形design-matrix手法への特殊化Bであって、元の同一physical-gate/full-process最適化との無条件Aではない。適切な投影またはtarget liftを定義すればmean-compatibleな強対照を考えられる、という技術情報である。新しい候補アルゴリズムや実験として実装・採択していない。

global phaseについてはAd_{exp(iχ)V}=Ad_Vでも、C(exp(iχ)V)のoff-diagonal blockはχを読む。C(-V)=Z_ancilla C(V)という相対位相を保持する必要がある。上のliftの二符号枝を実装する費用・測定frame処理を0としたり、今回のnative費用と等しいとしたりしていない。

### 固定dictionaryの係数問題と、連続angleの配分は同じではない

有限candidate anglesを先にlibraryへ含め、Π_offまたはO_Mをtargetにすれば、今回の有限配分候補はそのlinear feasible setの構成例になる。SPS側が同じ情報でさらに多くのangle混合を許す場合は、Parentの「childあたり一angle」classより自由度が広い可能性がある。逆に、固定有限libraryは未収録angleを自動的に取得しない。

zを動かす今回の全連続familyではdictionary column自体が変わる。これは原SPSの固定Rに対する係数LP/LASSOと同じ変数だけの問題ではない。正価格で固定candidateを評価するweighted L1 surrogateは一般凸最適化として書けるが、両符号枝の同じ費用等の仮定が必要であり、native価格・有限confidenceの最適性は未確立である。

原SPSは明示されたprocess列を使う。LARSによる疎supportや少数iterationと、全候補列を無料に取得できることを同一視しない。他方、全dense matrixの一括materializationがどの実装にも不可避という下界も主張しない。構造的列query・lazy生成は論理的に可能だが、読んだ本文から今回のP5 rootのO(L)係数取得・少数群samplerが同じ手順として与えられたとは確認できない。

## 6. R0・G1・G5との完全な変数変換の可否

既存[R0 §A1–A2](rte_reallocation_r0_independent_proof_v1.md)はa_k+b_{k−1}=t_kというdegree間の配分と、同じIID label lawを使う。各degreeのangleはchild labelに依存しない。今回のrootで一般にb_i/z_iが異なればこの不変量を破る。ordinaryのcommon配分だけは一致し得るが、**固定IID/degreesを保ったまま任意zをR0のa_k,b_kへ一対一変換することはできない**。IID条件を外す拡張は、既存の同じclassとは扱わない。

P5 return後のb_iも一般に共通scalar×p_iではない。全体meanが同じfinite P5であることは、root単体が旧IID prototypeと同じ分布であることを意味しない。

[G1](g1_one_shot_result_validation_20261009.md)/[G5](g5_fixed_dictionary_closure_scope_20261010.md)のclosureは固定prototype/angleとprototype内IID sharesが前提。旧G5はfinite P3の限定classであり、P5への全主張ではない。非退化childのzを動かすとphi_iが連続に変わるのに対し、固定有限angle dictionaryはその全familyをbranchごとに含めない。既存dictionary内の係数混合は同じmeanを作り得ても、新しいunitary ensemble・回路単価と同じとはいえない。

この構造差Cは、旧closureが破れたという実験結果や、新候補が利益を持つという証拠ではない。旧G1/G5の下界・終了判断は対象classのまま保持する。sourceを変更したり新prototypeを追加したりしていない。

現在のG9の3-qubit providerは明示され、既存sourceにPauli表現もある。一般involution生成法で比較しても、このtoyでPauli情報が取得不能とはいえない。したがってroot局所情報だけで生成できる性質と、実問題でのI0/I1取得cost優位の実証を分ける。この点は強対照を選ぶ際の入力条件となり、今回は新しいPauli collectionや価格計算をしていない。

## 7. アルゴリズム水準の照合表

| 項目 | Parent allocationの現source | 近接構成との対応／残る技術差 |
| --- | --- | --- |
| 入力情報 | parent a,b、exact z、involution/provider実装access。P5ではpの低次数momentsからrootを取得 | Euler部分は同じ情報で使える。global Pauli collection/process columns/cost tableは追加access |
| unitary辞書 | ζ exp(−iσphi_i Q_i)Q(u)。zにより変化 | commonは既知Euler。固定IS/G1/G5は辞書を保持。SPSのfinite libraryには新column取得が必要 |
| 係数・angle | sqrt(z_i²+b_i²)とatan2、root一群の複数sqrt和 | 任意identity分割＋Eulerという代数的構成。fixed proxy KKTは標準最適化。実angle価格は未取得 |
| sampling law | 既存nonroot条件付きlaw＋更新group/root dyadic law | commonは旧digital経路へ復帰。一般zのlawが旧common algorithmのままではない |
| importance weight | 実qでtilde_c/q | 線形/quasiprobability/ISの既知原理。同じqでもcが変わればweight/biasが変わる |
| normalization | B_rest+Σc_i。commonが本familyの質量最小 | 低いper-event costのためB増を許す問題。sampling-only最適化と区別 |
| finite bit | rational z、sqrt interval、actual dyadic q、coefficient/angle/native ledger | 読んだ一般定理のideal qだけではこの同じbit/source会計は確定しない。既済実装証拠を継承 |
| controlled phase | S/S†/Z、signed actual adjoint、strict guard、W保持 | channel-level phase無視を移せない。process側もcontrolled embeddingで相対位相を扱える |
| setup | root追加O(L)個の係数処理。既存group準備はO(L²)部分を持つ | 全event/process/cost列の取得・最適化をこのO(L)へ含めない。SPS全dense取得不可避という下界もない |
| online | 局所sampler、実測costはNOT_ACQUIRED | 同じ数値的生成時間・bit access・table利用費用までの同値はD |
| native費用 | 新angle key/端点/provider/strict誤差を別会計 | mean/abstract norm一致からT/CX/native1Q価格は一致しない。今回新価格は取得0 |

ordinary/P5のcommon・任意z接続、endpoint/phase、cache bridge、26+6 testsは既存Packetへ参照する。全62 testsやnative/source実装を再検証しない。関係式はsymbolic block/Euler/変数構造から確認できるため、新たな形式テストも行わなかった。

## 8. 7領域へ返す具体的情報（採択しない）

| 既存領域 | 今回の調査で明確になった判断材料 |
| --- | --- |
| 1 比較方式 | SPS/SCUをchannel理由だけで除外できない。same-targetにはprojected/lifted operator条件と情報accessを明示する。old G1/G5固定classだけへの差を全既知法への差としない |
| 2 配分候補 | common/vertex/interiorは既知Euler identity分割の異なるparameter。generic pairing原理の新規性ではなく、具体的な構造・生成・成立条件を判断する材料になる |
| 3 Proposal | global fixed-dictionary ISは明示A、辞書を変えるz最適化は別。両表現へ同じIS機会を与える公平性は保持。root-localはfull finite-confidence最適対照ではない |
| 4 正則化/zero cost | known positive-cost proxyの仮定外を明示する必要がある。KKT乗数とproposal正則化を混同しない。処理/値は未採択 |
| 5 Accuracy/confidence/budget | 先行論文のchannel shotsやCLT net-costを現taskの十分Nへ同じ数で転記しない。finite cutoff、dyadic、strict native bias、同じalpha配分を別に固定する |
| 6 Primary endpoint | 既知IS/SPSのnormやproxyを主成果として再発見扱いしない。構造情報の取得/setup/onlineと実native価格は未測定で、0としない |
| 7 科学的解釈 | 現証拠は既知部品・一般包含・限定構造差。独立priority、利益、一般化、研究継続の根拠はまだ分けて判断する必要がある |

7領域の選択値、候補集合、budget、primary、GO/STOP条件は変更していない。新しい比較実験・対照・optimizer・library取得を採択した報告ではない。

## 9. 未確認事項と停止点

- **完全なpriority/包含の網羅性D**：5本の指定版と既存プロジェクト資料の関連箇所に限定した。別版・全後続研究・全公開実装の同値を調べていない。
- **具体生成法の完全同値D**：arbitrary parent zとroot局所取得、same finite-bit/native/accountingまでを、既知手順と同じ入力/費用で実現する対応は未確認。これは不存在の証明ではない。
- **強対照の情報・実装費用D**：mean-compatible SPS/library、collected/structured CTS、global ISにどのcost/data oracleを与えるかは未採択。実費用・biasを取得しないと性能関係は分からない。
- **新角度/native/finite-confidenceの利益D**：保存登録値の新最適化・新合成・資源再ランキングをしていない。固定単価proxyでの内部解をnative最適性へ移さない。

今回の完了は上の対応と境界の明確化であり、候補の救済、新規性認定、科学的採択ではない。旧source・result・classification・authorization・marker・closure・STOP、親dirty変更を保持する。次のGPT重要科学レビューはユーザーの明示的開始承認待ち。新科学計算も別途明示承認を必要とする。**mandatory STOP。**
