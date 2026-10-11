# DF希薄三次補正：一次文献・claim単位比較

2026-10-11。タイトル／abstractだけでなく、下表の本文・式・定理・生成手順を確認した。
書誌と読んだ箇所は`artifacts/track_b_df_sparse_correction_audit/2026-10-11/phase_F_primary_sources.json`にも保存する。
以下の差分は構成の比較であり、originalityやresource advantageを認定しない。

## 比較の共通target

候補はordered DF partial-S2のK3を、positive ensembleのfirst operator momentで左右から近似補正する。
δ>0、γ固定、中央finite-RTE P5以上、control phaseを保持した複素信号をtargetとする。
各Tτはwhole Hermitian nested commutatorのままmax6 modeへ縮約してexp(iθTτ/κτ)を生成する。
κはcertified upper（普遍4が利用可能）。補正のweightは1だが有限stepの近似biasがあり、元RTE Γは残る。
ここで「random correction」「commutator」「DF」「少数支持」単独を新規claimにしない。

| 一次文献・確認箇所 | 入力／既知の構成・保証 | 候補との具体差・還元 | 未解決の比較条件・非claim |
|---|---|---|---|
| Zeng et al., [Trotter-LCU/NCC, arXiv:2212.04566](https://arxiv.org/pdf/2212.04566), PRX Quantum6,010359。Sec.III/IV, Eqs.(109)–(113), AppendixE(E1)–(E19), Proposition13 | PF順序からgeneral remainderをnested commutatorへ展開。leading termsをPauli LCUにし、I+iζPを√(1+ζ²)exp(i arctanζ P)へpairing。signed operator moment、Hadamard/controlとLCU norm会計、general Hamiltonianにも適用 | BCHを補正する上位原理とgeneral commutator導出は既知。候補はwhole Tτ回転、Bernoulli希薄化、左右対称、近似biasを許して追加norm1。Pauli QならTτ=0または±4PでPauli correctionへ還元するが、finite samplerはNCCのexact pairingと同一ではない | 同じDF backendでNCCのlocal Pauli展開・回転を構成した費用比較は未実施。一般LCU表現可能なだけで同値とはしないが、単にgeneral NCCをDFへ適用した差では不十分 |
| [HNCC, arXiv:2607.11856v2](https://arxiv.org/html/2607.11856v2), II.3, IV.3 Theorems2–4, Algorithms1–4, V.3/V.5 | k-local Pauli／g0-extensive情報。BCHをadjoint Pauliへ展開、parameter-shiftとidentity pairing、signed LCQC。複数BCH項の次数cutoff、λ^(2mν) repetitions、observable/channel target、ancilla-free | 候補のwhole Tτのpositive operator momentと、HNCC signed channel momentは異なる。controlを含む拡張observableへHNCCを適用する比較は可能で、channel方式だから対照から除外してはいけない。DF orbital局所支持はglobal qubit k-localと同じではない | 同じcoherent signalのcontrolled Hamiltonianへ変換した際のPauli access、k/g0、classical preprocessing、T/CXとconfidenceを揃える必要。candidateがHNCCより強い精度／ancilla scalingとは主張しない |
| Bagherimehrab et al., [Corrected PF, arXiv:2409.08265v3](https://arxiv.org/html/2409.08265v3), Sec.3.2 Eqs.(28)–(32), Sec.5 Eqs.(94)–(106), Table3 | A/B exponential access。symplectic＋symmetric correctorでS2三次を消してlocal O(λ5)。correctorをA/Bの有限exponential列へcompileし、その残差も会計 | 三次相殺・4次精度・対称correctorそのものは既知。候補は多block K3のprimitive tupleを希薄sampleし、DF local nonGaussian centerを使う。二partition・exact collective correctionでは既知CPF原理へ縮小 | exact collective expK3が安価という前提は置かない。known corrector compilationを同じDF inputへ適用した費用とnative precision比較は未実施 |
| Campbell, [qDRIFT, arXiv:1811.08017v2](https://arxiv.org/pdf/1811.08017), Eqs.(1),(4)–(11), AppendixD | positive h、norm-normalized H_j、exponential oracle。h/λでdrawし、common angle λt/Nを適用。平均channel diamond errorを会計。control用実装も検討 | 候補のaccepted correctionは、K3=Σcκ(T/κ)に対するone-draw qDRIFT型回転と一致する（下の代数対応）。full candidateはPF/finite-RTEとの対称接続とDFアクセスの構成を追加する | sparse corrected samplerの上位機構は独立新規と扱えない。qDRIFTのchannel保証をoperator first-momentへ転用しない。同じ原HのqDRIFT本番対照は未実施 |
| Ouyang, White, Campbell, [SparSto, arXiv:1910.06255v2](https://arxiv.org/pdf/1910.06255), Sec.3 Eqs.(2)–(7), Theorem1/5, Algorithm1 | term-wise independent BernoulliでH_j/p_jか0を選び、randomized forward/reverse PFへ接続。mean generatorを保存、diamond boundにsampling増分、expected gate budget | 稀なeventの角度を逆確率で増やす操作は既知。候補は原HではなくPFの三次K3を対象に、one accepted tupleを両側で独立使用。all p_j=1では疎化なし。candidate η=1でもone-tuple randomnessは一般に残る | positive thinning／期待頻度だけでは新手法claimにならない。tail、native center、RTE normalization込みの比較を必要とする |
| Cugini et al., [Resource-optimal IS, arXiv:2603.13495v1](https://arxiv.org/html/2603.13495v1), Sec.II Eqs.(4)–(13), Theorem1, Sec.III/IV | fixed circuit/channel ensembleをp→qで再sampleし、weight p/qで同じmeanとbiasを保持。net cost=E_q[c]E_q[w²]、q∝p/√c | candidateはsamplerのunitary角度とmean近似誤差を変えるため、fixed-ensemble ISそのものではない。ただし構築済みcandidate ensembleへのcost-aware再weightは既知の改善対照 | importance samplingを追加weightなしとして扱わない。finite-confidence/biased synthesis/zero cost/failed eventの契約を共通化する必要。今回IS実行なし |
| Jiang et al., [Gaussian circuits, arXiv:1711.05395v2](https://arxiv.org/html/1711.05395v2), Sec.III Eqs.(11)–(18), IV, AppendixA | orbital matrices、Givens/Bogoliubov分解、JWとphase/order、Slater/fermionic Gaussian circuits | Gaussian embeddingとorbital accessは既知。候補のTτはeven number-conservingだが、その指数は一般にGaussianではない。Slater-vacuumだけで不要なphaseを捨てる慣習をcontrolに持ち込まない | generic 6mode centerをGivensだけで安価に実装できるとはしない。thin isometry/source bridgeのproduction認証は未実施 |
| Motta et al., [Low-rank DF, arXiv:1808.02625v2](https://arxiv.org/pdf/1808.02625), Eq.(3), circuit discussion, Appendix Partial Basis Rotation Eq.(22)以降 | electronic tensor→double factorization→occupation diagonal terms＋Gaussian basis rotations。partial basis rotationのO(Nρ)構築、thresholdとnative costも会計 | DF展開、2-orbitalparity、O(nr)部分基底変換そのものは既知。candidate差候補は三つのprimitiveのunion spanとwhole nested-commutator center、希薄平均の接続 | current square-U APIにthin O(nr)実装があるとはしない。DF acquisition／thresholdをfreeに扱わず、低rank simulation全般に勝つとは主張しない |
| McArdle et al., [fermionic seminorm/low-rank bounds, arXiv:2107.07238](https://arxiv.org/pdf/2107.07238), Sec.III Eqs.(16)–(21), AppendicesB/C | one-body小行列のseminorm・commutator identitiesとfactorized two-body bounds。full Fockを作らないPF誤差評価 | 小行列とDF structureでcommutatorを扱うことは既知。candidateはboundだけでなく、sampleされたwhole Tτの回転を生成する点を区別 | norm評価とnative expT費用を混同しない。小行列error bound既知法の名称変更になっていないか、同じDF情報で比較が必要 |
| 追加近接確認：Kim/García-Pintos, [Randomized PF, arXiv:2608.07720v1](https://arxiv.org/html/2608.07720v1), Eqs.(2)–(13),(23)–(28), SM S3.2 Eqs.(S63)–(S96) | standard accessではuniform insertion。stronger expA/exp(A+αB_l) accessでは、dyadic rectangle上のW-block group-commutator correctionをpositive確率1/m_n²でsample、残りidentity。階層sampling、平均誤差相殺、期待costとtailを証明 | **positive sparse correction＋identity＋階層生成自体も直接既知**。candidateはK3の有限primitive tuples、whole Tτ exponential、η∝δとbounded max6mode center。既知法はinteraction-picture second-α commutatorとdyadic infinite branches、block repetitionを用いる | coherent first moment、取得oracle、fixed error/order、controlled resourceを揃えた同値性／費用比較は未実施。candidateを単にpositive sparse high-order correctionと呼ぶ差では不十分 |

## qDRIFT／疎化への具体的対応

Λ3=Σcτκτ>0なら、accepted drawの分布はq′τ=cτκτ/Λ3、generatorはH′τ=Tτ/κτ。
ηeff=ηΛ3/B、tC=δ³/2とおくと

\[
\bar C=(1-\eta_{eff})I+\eta_{eff}\sum_\tau q'_\tau
\exp\left(i\frac{\Lambda_3 t_C}{\eta_{eff}}H'_\tau\right).
\]

これは**one qDRIFT-type drawをBernoulliで間引き、角度をinverse probabilityで増やす分布**である。
κがnorm upperなら∥H′∥≤1、κがexact normならnorm1。zero eventはidentityへ戻す。
δ³/2に対するoperator一次meanを合わせ、thinningの二次欠損をO(δ6/η)=O(δ5)に許す設計である。
この代数対応は本監査の導出であり、既存qDRIFT論文が同じDF-corrected full protocolを公表したという意味ではない。
全体の完全同値は証明していないが、「補正samplerの上位原理が独立に新しい」という主張は支持できない。

## Pauli/NCCの還元とfinite差

同じglobal Pauli basisのQならTτは0か±4P。κ=4のnonzero eventで候補の平均は
(1−η+ηcosθ)I+iηsinθ Pとなる。
NCC Eq.(111)はI+iζPをexactにpairingし、√(1+ζ²)というLCU weightを払う。
候補はηθ=ζだけを合わせ、そのfinite remainderをbiasとして払う。したがってpositive norm1という数字だけでNCCのdominanceは結論できない。
DF-native Tτが全系Pauliへ展開可能でも、orbital input→全系Pauli取得と、local r≤6 Pauli展開の費用は別に会計する。

## 残る差分候補と非claim

残り得る具体差分は、(i) ordered multi-DF-block K3のhierarchical tuple access、
(ii) global Pauli tableを避けたunion orbital span上のwhole Tτ rotation、
(iii) coherent first-moment taskとfinite P5-RTEのerror/normalizationを保つ対称接続、である。
それぞれ単独の素材は既知であり、その組合せが独立の方法論／有用な資源削減になるかは本監査では未確定。

とくにrandomized THRIFTのSMはpositive identity-mixed correctionのhierarchical生成とgate-count tailまで扱う。
この近接性は「NCCはsignedなのでpositiveにしたこと自体が新しい」という読み方を排除する。
SM Eq.(S91)–(S92)はoperator first momentの独立step積も明示しているため、first-moment証明の形式自体も独立差分ではない。
候補の有限branch hard capとDF-native whole-commutator accessが具体的にどの価値を追加するかが未解決である。

同じDF inputにgeneral NCC/HNCC/CPF＋local backendを与えたときに同じ処理になるか、同じ誤差とconfidenceでどのnative資源が残るか、
certified rank/digital/synthesisの取得費用まで戻せるか、をGPTレビューで判断する必要がある。
単なるbibliography検索で同一名が見つからないこと、general LCUへ書けること、小支持やλ=1だけから新規性採択／否定をしない。
今回強いbaselineの数値比較やproduction回路、algorithm採択はしていない。mandatory STOP。
