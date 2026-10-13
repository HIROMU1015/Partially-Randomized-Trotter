# D1〜D3：一次文献照合・claimと固定証拠・GPT科学レビュー引き継ぎ

2026-10-14 JST。**科学レビュー前の技術資料整理を完了し、mandatory STOPする。**
新規性の最終認定、研究GO/STOP、アルゴリズム採択、論文化可否は判断していない。

費用とsampling負担の交換、Taylor分解の再構成、誤差配分による総費用の変化は既知の一般原理である。
D1〜D3の固有の証拠は、明示したfinite operator task、情報access、位相・finite-bit・nativeモデル、
保存された合成価格の下での比較と機構分解である。この具体性だけから独立新規性を認定しない。

今回重点的に確認した近接箇所は、SCU/CTSの補足Note 4.1の総誤差・shot・CNOT費用会計、
Sparse Probabilistic Synthesis（SPS）のprocessと複素NMR信号の定義、resource-optimal ISの最適化対象とbias不変条件である。
既済の数値監査・機構分析・G系列の同値性監査は再実施していない。

## 1. 正本、取得範囲、共通適用条件

Repository：`HIROMU1015/Partially-Randomized-Trotter`。
文書の固定取得commitは **`534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f`**。
D3科学結果commitは **`f0df9a3b36bc3540f0c83941871fa0d3110ddbf7`**、
science S2は **`dceccaa774229527a46a1386131583a4f618f8ad`**。
原source・結果・authorization・消費済みmarker・STOPは変更しない。

17研究文書を上記固定commitから取得し、bytes・SHA256をlocal/fixed Git blobと照合した。
[入力receipt](../../../artifacts/track_b_d1_d3_literature_handoff/2026-10-14/v1/fixed_document_inputs_v1.json)に固定URLとidentityを記録した。
これは指定資料の受領確認で、以前のGitHub全出力監査や2,579項目の保存値検証の再実行ではない。

D1〜D3の数値の適用範囲は、既知developmentの3-system-qubit synthetic involution：
`p=(1/5,3/10,1/2)`、`tau=5/7`、`sigma=+1`、state=`000`、
`Q0=Z0`、`V1=R_XX01(pi/4)`、`V2=R_XX12(pi/4)R_ZZ01(pi/4)`、`Qi=Vi†ZiVi`、`R=sum p_i Qi`。
六方式はordinary、partial_return_tail、closed_P3_tail、full_return、closed_P5_full、matched_CTS。
明示I1情報を全方式に公開する。分子geometry、軌道basis、DF rank、split L_Dは適用外。
独立held-out、実量子shot、分子PR総資源の結果ではない。

**固定r内では、六方式のideal first operator meanの対象が同じ `P5(-i tau R/r)^r`。**
一方r1とr2では有限多項式自体が異なり、共通なのはideal exponential `exp(-i tau R)` と精度/confidence taskである。
「全12構成が同じfinite多項式を実現」とは書かない。
`closed_P3_tail`は3次部分の表現と残りのtailを組み合わせて同じfinite P5を実現する。3次targetへの変更ではない。
ideal恒等式と、係数・角度・native実装を有限精度にしたmeanの誤差保証も分ける。

### 固定証拠の入口

以下の旧文書はすべて上記固定commitのbytesを参照する。節番号は当該版に対応する。

| ID | 文書・該当箇所 | 支える内容 |
| --- | --- | --- |
| E1 | [D3最終保存機構分析](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d3_final_saved_mechanisms_and_gpt_handoff_20261014.md)：価格寄与、G1_P1、保証/r、primary/全方式、h領域 | exact key寄与・保存N・保存比較への接続。追加369算術checks/9人工testsは前段完了の記録 |
| E2 | [D3 v2結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d3_v2_results_and_gpt_handoff_20261014.md) §1–6 | 36合成・48行・四ledger、元データ/正本監査/provenanceへの索引 |
| E3 | [D3結果前計画](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d3_guarantee_price_preregistration_v1_20261013.md) §1–5 | 同task、equal-r自由度、保証×価格二因子、固定native誤差会計、相殺の再利用規約 |
| E4 | [D3数学確認](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d3_mathematical_confirmation_20261013.md) §1–3 | contractionを要する積分剰余、非unitary有限meanの反復、digital/native bias |
| E5 | [D2 key機構・境界契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d2_key_mechanisms_and_boundary_preparation_20261013.md)：条件付きkey係数、literal逆対の規則 | 保存event/key incidence、費用分解、実sequence adjointの相殺 |
| E6 | [D2保存値統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d2_final_saved_integration_and_gpt_handoff_20261013.md) | D1 r1＋D2 r2の共通literalモデル、全方式/h/未取得費用 |
| E7 | [D1結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md) | 固定provider、12構成、finite-bit/moment/acceptance/native資源会計 |
| E8 | [G6数学監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md)、[finite-bit/access監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md) | Q_i²=Iによる形式return、ideal/digital平均と情報・生成費用の境界 |
| E9 | [G9 P5構成](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md)：P5導出・native公平性 | closed少数群、同finite targetのcollected CTS specialization、phase |
| E10 | [G6 prior-art照合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md)、[return集約ノート](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_0.md) §1–6,8–9 | 既知構成部品、具体生成法、条件付き下界とpriority未確立 |
| E11 | [D1/D2 GPT判断原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/research/track_b_d1_d2_integrated_scientific_review_20261013.md) §7–11 | 既採択のD3目的・誤差保証/価格を分離する判断。今回のGPTレビューではない |

原数値・有理数・価格寄与の正本はE1/E2がリンクする既存JSON/CSV。今回それらを再生成・再ランキング・再計算していない。

## 2. Claim-to-evidence：直接事実、条件、未確認事項

区分Iは固定証拠が支える内容、IIはそれを支える既知の一般原理、IIIは追加貢献の可能性を判断する未確立部分。
Iに属する局所結果がIIの原理を使うことは両立する。区分分けは「独自性の最終判定」ではない。

| ID / 主張候補 | 固定証拠・区分Iの内容 | 適用条件・区分IIとの対応 | 未確認事項・区分III |
| --- | --- | --- | --- |
| A1 同finite targetの複数表現 | E8/E9のideal恒等式、E3/E7の六方式契約 | 同じr、奇数cutoff P5、ordered word、必要な位相を保持。pairing/Pauli collection自体はP2/P4/P5の既知部品 | 全構成のpriority・非自明性。digital/native平均をexact ideal meanと同一視しない |
| A2 表現によりN・単価・順位が変わる | E1/E2の保存48行とfactorial/comparison。moment/range、acceptance、十分N、native単価を分離 | 同I1、同精度・confidence・native条件。費用×sampling負担の一般原理はP1/P2/P3で既知 | 最適IS、別辞書/price/providerでも残る資源差 |
| A3 fixed-rとequal-r自由度 | E3の結果前primary、E1の固定r/equal-r/全方式の保存表示 | fixed-rは同finite target。r選択は同exponential/accuracy taskへの比較で、有限meanはrで異なる | r=1,2外の最良r・一般選択則は未取得 |
| B1 強いTaylor上界を使用できる | E4の積分剰余と入力normの根拠 | 実時間、Hermitianかつnorm<=1。標準Taylor積分剰余の帰結で新定理として扱わない | 非Hermitian/複素時間/別normへの無条件移転 |
| B2 保証改善のr依存 | E1の保存整数：r1の十分Nは約29.5%減、r2は約0.93%減 | law・価格・native会計を固定。Nは十分forecastで、実shots/必要最小Nではない | 独立入力や別statistical policyで同率になるとは未確認 |
| B3 保証だけで最良rが変わる | E1/E2：G0_P0 r2 P5、G1_P0 r1 P5。P1ではr2 P3+tail→r1 P3+tail | 共通総時間、h=0、同価格。統計幅とbiasの費用交換はP2 Note4.1にもある | 普遍的r選択法、前向き予測の成立 |
| C1 保存sequence価格の変化 | E2の36 key cache、E1のT/1Q差・incidence・price-only一致 | old1e-6/new1e-7、同tool/seed/strict phase。全四ケースのnative誤差会計は1e-6固定 | 理論最小T数や別seed/provider価格。新精度の余裕はNへ戻していない |
| C2 G1_P1でP3+tailが低い | E1：r1 P5−P3+tail=598,967.339 T、P5分母で0.169653%（丸め表示） | 固定law/N、h=0、同finite P5。root 2189372/2921709の68→80と310/413の70→78が主要寄与 | 一般P3優位、微小価格差の実現可能性、別合成条件への頑健性 |
| C3 primaryと全方式は別 | E1/E2：primaryは各ケースでP5のh=0 Tが低い。全方式のP1はP3+tailが最小 | primary双方にr1/r2、全六方式にも同じr機会。1Q/CXの最小はTと一致しない | primary勝者を全方式・全資源最適とする主張 |
| C4 hと価格感度の意味 | E1の既存affine領域・key tie。G1_P1 r1 P3+tailのT最小領域はh約15.098未満 | exact境界は元JSON。hはaccepted whole callあたり。formal独立domainとphysical h_1Q>=h_T条件を区別 | tie価格が合法sequenceとして取得可能か、実際のprep/readout単価 |
| D1 D2境界相殺を含む比較 | E5/E6、E3の保存literal-only節約額、E1の再利用 | 全方式共通adjacent exact inverse、同law/価格/bias/N、actual sequence adjoint、位相保持 | whole-circuit最適化・別compilerの費用と順位 |
| D2 native加法会計の範囲 | E1/E2/E5のliteral T/CX/native1Qとh係数 | 同モデルの費用forecast。Tは1Qの部分集合で重複加算しない | 実装全体の最小T、fault-tolerant runtime、経路全体の最適回路 |
| D3 古典費用と情報access | E1/E6/E8/E9の取得・生成条件。online費用はNOT_ACQUIRED | explicit Pauli展開を取得できるsynthetic I1。setup記録とonline費用は別 | DF規模の情報取得優位、古典online/scalingを含む総資源優位 |

C2のexact rational差、keyごとの寄与、記号的tieはE1の既存artifactをそのまま参照する。
小差を一般的P3優位へ広げず、同じ物理taskへのaccuracy guaranteeを示した範囲と性能一般化を分ける。

## 3. 一次文献：書誌、本文取得、該当箇所

実際に取得した版を以下へ固定した。公開日はarXiv初回/当該版の提出日であり、PDF組版の日付やjournal刊行日とは区別する。
8論文の本文を取得し、記載した関連節を確認した。全論文の全証明を精読・再証明したclaimではない。
[取得receipt](../../../artifacts/track_b_d1_d3_literature_handoff/2026-10-14/v1/primary_retrieval_receipt_v1.json)にはURL・HTTP・bytes/SHA256、
[文献catalog](../../../artifacts/track_b_d1_d3_literature_handoff/2026-10-14/v1/primary_source_catalog_v1.json)には版、著者、task、access、指標、保証、locatorを記録した。
論文PDF/HTML全文をrepositoryへ転載しない。

| ID | 論文名・著者 | 初回公開 / 取得版の日付 | 確認した本文箇所 |
| --- | --- | --- | --- |
| P1 | [Resource-Optimal Importance Sampling for Randomized Quantum Algorithms](https://arxiv.org/html/2603.13495v1) — Davide Cugini, Touheed Anwar Atif, Yiğit Subaşı | 2026-03-13 / 2603.13495v1、同日 | §II Eqs.(1)–(13), Theorem1；§III Theorem2；§IV.1/IV.2 ZeroFill/Discard |
| P2 | [Quantum Simulation via Stochastic Combination of Unitaries](https://arxiv.org/html/2407.21095v2) — Joseph Peetz, Scott E. Smart, Prineha Narang | 2024-07-30 / 2407.21095v2、2026-07-06 | §II.1–II.3, Theorem1, Eqs.(1)–(8), Fig.3；§IV.2；Supplementary Notes3,4.1,5 |
| P3 | [Sparse Probabilistic Synthesis of Quantum Operations](https://arxiv.org/html/2402.15550v2) — Bálint Koczor | 2024-02-23 / 2402.15550v2、2024-10-31 | §II.1–II.3 Statements1–2, Eqs.(1)–(6)；§III.1 Eqs.(7)–(8), Fig.2；§III.3 Statement3；AppendixB.2/B.3 |
| P4 | [Phase estimation with partially randomized time evolution](https://arxiv.org/pdf/2503.05647v2) — Jakob Günther, Freek Witteveen, Alexander Schmidhuber, Marek Miller, Matthias Christandl, Aram W. Harrow | 2025-03-07 / 2503.05647v2、2026-07-10 | §I/IVのtask、AppendixA.2 Eqs.(A18)–(A29), LemmaA.2；A.3のPR接続 |
| P5 | [A randomized quantum algorithm for statistical phase estimation](https://arxiv.org/pdf/2110.12071v2) — Kianna Wan, Mario Berta, Earl T. Campbell | 2021-10-22 / 2110.12071v2、2022-07-13 | Lemma2、AppendixC Eqs.(C4)–(C5), Algorithm2、F.2のfinite cutoff記述 |
| P6 | [Simple and high-precision Hamiltonian simulation by compensating Trotter error with linear combination of unitary operations](https://arxiv.org/pdf/2212.04566v2) — Pei Zeng, Jinzhao Sun, Liang Jiang, Qi Zhao | 2022-12-08 / 2212.04566v2、2025-03-27 | §IV.A Eqs.(42)–(50), Proposition3。§IV.C sampling比較は既存R0.5 locatorも継承 |
| P7 | [Exploiting anticommutation in Hamiltonian simulation](https://arxiv.org/pdf/2103.07988v2) — Qi Zhao, Xiao Yuan | 2021-03-14 / 2103.07988v2、2021-08-26 | §4.1の誤差評価、§4.2 Eqs.(24)–(29)の高次寄与吸収 |
| P8 | [Optimal ancilla-free Clifford+T approximation of z-rotations](https://arxiv.org/pdf/1403.2975v3) — Neil J. Ross, Peter Selinger | 2014-03-12 / 1403.2975v3、2016-06-15 | Algorithm7.6、Propositions8.1/8.2/8.8, Hypothesis8.3；§9のphaseを除く別問題 |

PRのHTML取得は失敗したがexplicit-v2 PDFを取得して本文確認した。PDFの表示日付は2026-07-13、arXiv版日は2026-07-10。
P6でもPDF表示2025-03-31と版日2025-03-27を混同しない。指定した8論文に本文未取得はない。

[Aomoto–Kato, Green functions and spectra on free products of cyclic groups (1988)](https://www.numdam.org/item/AIF_1988__38_1_59_0.pdf)
については、E10が保持するpp.62–65、§1 Eqs.(1.6)–(1.14), Lemma1.1の一次本文照合を参照し、本回は再取得しない。
この継承を今回の独立全文取得と数えない。既知Green再帰の新規性を再審査する作業も行わない。

### Task・利用情報・資源指標・保証の比較

| ID | 物理task / 利用情報 | 資源指標と誤差・統計の扱い | D1〜D3との共通部分 / 相違 |
| --- | --- | --- | --- |
| P1 | 固定channel ensembleのobservable。original pと実行・測定cost c | expected cost×worst-case weight二次momentを最適化。Theorem1のq∝p/sqrt(c)、Theorem2のbias不変 | 費用×sampling負担は既知。D3はrepresentation/rを比較しproposalは固定。range・ceil・finite confidenceの最適性を同定理からは言わない |
| P2 | Pauli Hamiltonian/channel。Taylor実/虚部の係数収集、または層別sampling | operatorのspectral error、channel normalization、Hoeffding shots、CNOT総費用。Note4.1は総誤差を統計/biasへ配分 | 同finite targetのmatched CTSはTheorem1の特殊化。元のleft/right channel taskのnormalization/費用をHadamard operator taskへ同じ数で移さない |
| P3 | gate processとNMR/MRI。候補gate library/process design matrix | L1 overheadとprocess residual。短いClifford+T列×累積repetitionsの総費用 | 精度・sampling・Tの交換は既知。D3は各key一つのstrict-phase sequenceの価格比較で、process辞書最適化は実施していない |
| P4/P5 | phase-estimation信号、Pauli/involution係数samplerとcontrolled circuit | RTE normalizationとsampling/circuit費用、有限cutoffのbias。元論文のPR/QPE資源 | ordinary Euler pairing、IID order/label生成、phase-sensitive平均は既知。D3は有限ランダムstepの比較で、QPE/PR全体の改善ではない |
| P6 | Trotter errorをLCUで補償。Pauli展開/order係数 | LCU norm、truncation error、gate/sample complexity | leading次数のpairingは既知。K=0のfirst pairと全adjacent/return lawを同一と断定しない。独立PTSC armはD3六方式にない |
| P7 | 反交換構造を利用するcoherent LCU。Hamiltonian項と高次係数 | truncation errorとLCU実装費用 | identity/単一項への吸収は既知。再構成したexponential近似と固定finite P5内の全形式returnは同じtarget/lawと断定しない |
| P8 | ancilla-free単一qubit Rz、角度/epsilonと数論subroutine | T最小化と古典期待時間、operator誤差。optimal/near-optimalには明示条件 | 精度・角度と離散価格は既存合成問題。D3のtool/seed/停止条件下の取得価格を最小T数と認定しない |

本文の解釈とD3側への対応付けは本報告の技術的照合である。文献がD3全手順・全policyをそのまま述べている、
あるいはその組合せが新規だ、という最終判定には使わない。

## 4. 近接文献との衝突可能性と、taskをそろえる際の境界

**総誤差・shot・回路費用の同時会計自体は先行する。** P2のSupplementary Note4.1 Eqs.(S8)–(S12), TableS2は
spectral/statistical error、confidence、sample数とCNOT総費用を扱う。
したがって「biasを統計幅に戻すとrや費用が変わる」「短い回路でもshot込みで比較する」をD3が初めて示したとはしない。
D3が追加したのは、採択済み有限P5・六表現の保存law、共通finite-bit/rangeを保持し、実合成価格と保証を分離した四ledgerの具体的結果である。
P2の資源数値を別taskのD3へ直接転記しない。[P2本文](https://arxiv.org/html/2407.21095v2#Sx3.SS4.SSS1)

**ISの定理の最適化対象とD3の順位は異なる。** P1のTheorem1は固定protocolのnet-costに対する定理。
D3の十分Bernstein予算にはrange、残余幅、integer ceil、classical zero outcomes、familywise配分がある。
cost=0のatomもあるため、Tだけのcへq∝p/sqrt(c)を機械的に適用して最適性を認定しない。
P1のbias不変も「実装と重みのchange-of-measureを固定する」条件であり、representation自体を替えたbiasを不変とする主張ではない。
P1のZeroFillは検出flagの処理、D3のzero eventは量子実行前のsampler規約で、同一protocolとは呼ばない。[P1本文](https://arxiv.org/html/2603.13495v1#S2)

**CTSはcoherenceを捨てる単純なchannel mixtureではない。** P2 §II.1のcross terms、§II.3のoperator分解を保持する。
一方元のchannel実行ではleft/rightを独立samplingし、operator normalizerとchannel normalizerの冪は異なる。
D3のmatched CTSは同finite P5をPauli収集したoperator表現を共通controlled Hadamard taskへ実装した対照である。
これは同一taskへの具体的対応であり、CTS全familyへの優位ではない。
Note3は層別samplingを示すので、CTSが常に全word列挙を必要とするとも言わない。
形式Q_i²=I集約と、Pauliの交換/反交換まで用いる全collectionも区別する。[P2本文](https://arxiv.org/html/2407.21095v2#S2.SS3)

**SPSの複素信号は、そのままD3 operator meanではない。** P3 §II.1はprocess/PTM equality、
§III.3 Statement3はNMR/MRIのstate/process作用後のX/Y信号である。
「複素信号を扱う先行法がない」とも、「system process一致からcontrolled Uの相対位相も一致する」とも言わない。
joint controlled spaceと適切な辞書へそろえた適用の可能性はあるが、D1〜D3では実装・費用・保証を取得していない。
process/first-momentの区別自体を独立新規性にしない。[P3本文](https://arxiv.org/html/2402.15550v2#S3.SS3)

**root吸収と生成の既知部品は既に照合されている。** P5のadjacent pairing、P6のleading pairing、P7の高次吸収、
Aomoto–KatoのGreen再帰をE10が対応付けている。
本回はこの監査を繰り返さず、「同じtarget・law・word順序・phase・生成情報が全部一致するか」という既存の境界を維持する。
一般部品が既知であることは全結合手順の同一性の証明ではなく、未特定という記録もpriorityの証明ではない。

**合成価格には条件付きの最適性と実取得価格を区別する。** P8 Algorithm7.6/§8はfactoring oracle/数論仮定を明示し、
§9はphaseを除く別問題を扱う。D3のstrict phase、実sequence、固定tool・seed・capでの価格へ、
無条件の最小T・単調なkey価格変化・普遍的頑健性を付与しない。[P8本文](https://arxiv.org/pdf/1403.2975v3)

G10の固定辞書・固定confidence policyのsampling-only下界はE10 §8の限定証拠として保持する。
価格・保証・confidence配分が異なるD3四ケースへ、その数値certificateや全proposalの分離を無条件に移転しない。
今回はIS・SPS・CTSの新baseline評価も行わない。

## 5. 三つの区分：研究主張の境界

### I：固定証拠から直接支持される内容

同rのideal finite meanの恒等式、適用仮定を満たすTaylor保証、digital/native誤差を戻した保存会計、
四ledgerの登録順位・r選択・key寄与の説明、共通literal境界相殺を含む費用の成立領域。
これらは固定development・一provider・二r・一追加precision・一seed・加法的nativeモデルの証拠である。
科学的分類、confidenceの正本、旧結果の区切りは保持する。

### II：既知研究・標準数学に属する一般原理

unitary平均とnormalization、Taylor次数pairing、identity寄与吸収、free-product再帰、
費用と二次momentの交換、統計/algorithmic errorの配分、probabilistic synthesisとT×sampling費用、
合成精度/角度による離散価格変化、literal逆対の相殺。
少なくとも上記の一般形をD1〜D3の独自発明として扱う根拠はない。
標準積分剰余の適用を「新しいTaylor theorem」として強調することも避ける。

### III：追加貢献の可能性はあるが、未確立の内容

| 検討対象となる主張 | 現証拠の不足・追加の情報が要る理由 |
| --- | --- |
| 表現/保証/離散価格から有利な構成を予測する一般的成立条件 | 保存counterfactualと記号的tieは事後の説明。独立入力での前向き予測や普遍的境界は未実証 |
| return集約の結合生成法の独立method delta | 既知部品との対応はあるが、全構成のpriority・非自明性・実資源価値は未確立 |
| 強い既知法を超える総資源価値 | 同taskの最適IS、joint-space SPS、CTSの追加自由度等をそろえた比較はD1〜D3にない |
| 合成器/seed/precisionに依存しない資源選択 | 約0.170%の差と小さい記号価格変位は保存価格への依存を示す。合法sequenceのtie/移送は未取得 |
| 全回路または実runtimeでの優位 | literal加算/隣接相殺を超えるcompiler、fault-tolerant実装、prep/readout単価が未取得 |
| scalableな古典生成・情報取得の利益 | 条件付き算術量とsupport件数はonline時間/メモリの実測ではない。I1を取得できるtoyからDF access優位は言えない |
| 分子PR/QPE全体の改善 | 固定synthetic random stepの結果で、deterministic部分、state準備、長時間、位相推定全体を評価していない |

この表は不足の理由を整理したもので、新しい対象・条件・実行計画の採択ではない。
追加検証が研究上必要か、実行に情報価値があるかはGPT/利用者が判断する。

## 6. GPT科学レビューへ返す論点と停止

GPTでは、既存return集約ノートの限定的な構成・実装結果に対し、D1〜D3が
「表現選択の条件を説明する独立した知見」をどこまで付加したかを判断する。
特に、既知の一般原理との重複、実合成価格の小margin、primaryと全六方式の違い、
必要な強い対照のtask/phase/access/cost一致、一般化の不足を含めて評価する材料が揃っている。

論文の主張を構成ノート、限定resource/mechanism study、より一般的選択則のどこへ置くか、
追加の独立検証や実装を必要とするか、継続/縮小/終了をどう扱うかは今回Codexが決めない。
近接文献の全文未取得を理由とする保留は上記8論文にはなく、網羅的なpriority調査は未実施である。

新しい合成/compile/行列、shot予算・resource forecast、ranking、runner/pair再実行、
precision/seed/provider/r探索、アルゴリズム設計・採択、既存classificationの変更はすべて0。
今回の確認は文書identity、一次文献の該当節、引用・証拠参照の整合に限定する。
旧数値testsやデータ監査は再実行しない。

報告・文献locator・入力identity・公開監査を専用branchへ保存し、remote SHA、再取得、worktree clean、
親既存変更と旧証拠の保護を確認後、**mandatory STOP**。
GPTの重要科学レビューは、ユーザーの明示的な開始承認後のみ実施する。
