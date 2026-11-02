# G6 RAの実用タスク接続：静的事前調査

2026-10-20 JST。基点：案B監査 `f823e909cb7049a612f1bf2c247f3bbe15fc184d`。
**技術調査と将来契約の草案。科学的GO、研究方向の正式採択、実行承認ではない。**

既存SOURCEへの接続が最も具体的なのは、保存H4・DF Hamiltonianの**単一時刻coherent signal**である。ただし、実行可能なG6–DF adapter、sourceに結合したexact入力、native/provider誤差、共通の認証予算は未取得であり、`CONDITIONAL_CANDIDATE_NOT_EXECUTION_READY`とする。全round RPEや分子energy精度の改善を支持する証拠はない。候補1件の[最小契約案](minimal_validation_contract_draft.md)と[GPT・ユーザーへの引継ぎ](handoff.md)を作成した。

## 1. 調査範囲と証拠

- 方法ノートv1.1と固定G6/G7/F3 SOURCE、G9/G10、D1–D3、W2b/W2c、Pilot、既存DF/RTE/RPE libraryを読解・静的照合した。科学moduleのimport/実行、科学test、数値代入・再集計・benchmark、回路生成・合成、新証明は0件。
- 私的レビュー2原本は関連部分の参考読解のみ。原本を転載・公開しない。結論の根拠は公開固定資料と一次文献へ接続した。全資料・全証明の読了、外部独立再現、別AIによるレビューとは表示しない。
- 既存報告の数値は引用であり、今回の再解析ではない。NPZ、statevector、raw archiveを新たに開封しない。分子係数分布・return頻度・費用順位を今回計算していない。
- 運用は2026-10-17統合正式版とプロジェクトAGENTSに従う。Codexは技術証拠を整理し、科学的意義・新規性・投資判断はGPT・ユーザーへ返す。

固定commit・path・bytes/SHA256、実際の読解範囲、保全対象は[evidence_inventory.json](evidence_inventory.json)に記録する。参照先が他commitにしか存在しないものは、そのcommitを明示した。

## 2. SOURCEから応用までの対応

| 層 | 固定SOURCE・資料 | 接続できるもの／残る不足 |
|---|---|---|
| G6係数 | [return_aggregation.py][KERNEL] `ReturnKernel`, `parent`, `digital_parent` | 正有理p、奇数m、0≤x≤1、Q_i²=I。label隣接returnのみ。一般DF行列やchannelを入力しない |
| 実proposal | [g7_generator.py][GEN] `FullReturnGenerator.sample`, `budget` | raw非reduced/coin拒否を量子前zero。budgetは固定confidence・条件付きCQ・単一finite target用で、分子全taskの予算関数ではない |
| 精度と証拠 | [F1/F2][F1]・[F3][F3] | 一様bit仕様、実π・補正weight・angle enclosureは既存成果。DF adapter、物理provider、測定器、任意αの全task保証は別。F3の元canonical非identity拒否と後続Pilot監査経路を区別 |
| provider | [g7_provider.py][CQ] `controlled_event_ir` | exact controlled-Q/actual-adjointという条件付きIR。物理T/CX/1Qは未指定。helperを使うgeneric経路とdirect DF loweringを同じ価格としない |
| 低次数対照 | [g9_p5.py][P5] `P5Closed` | m=5の少数群normalizer・completion・label付き生成。全parent列挙を強制しない。exact rational/roots、群selectorとchild lawの費用は残る |
| synthetic native | [g9_native.py][NATIVE] `provider_polynomials`, `cts_events` | 3-system-qubitの固定Q・次数5専用。任意分子CTSやDF providerとして呼べる実装ではない |
| 分子tail | [df_rte_tail.py][DFTAIL] `exact_df_diagonal_coefficients` (529), `extraction_to_symbolic_rte_tail` (808) | basis付きsigned I/Z/ZZを正規化し、dense列挙を避ける入口がある。ただし関数名のexactは記号展開を指し、実装はfloat/fsum/tolerance。G6の正有理pへそのまま代入できない |
| DF event/control | [df_rte_qiskit.py][DFBUILD] `_build` (214), `_append_central_application` (472) | basisをcontrolせず中央Z/ZZと回転をcontrolし、identityはancilla相対位相。既存RTEEventの型・位相・作用順を要求し、G7 event dictの直接入力口ではない |
| 反復・測定 | [df_partial_s2_repeated.py][REPEAT]、[rpe_hadamard_interrogation.py][HAD] | DF partial-S2のordered trajectoryとX/Y wrapper。後者は既存型のcontrolled trajectoryを包み、state preparation・shots・phase decoderを含まない。直接short q_m=1,2,4の経路を一般長roundへ延長しない |
| RPE会計 | [rpe_resource_accounting.py][RPE] | finite ordinary RTEのattenuation、effective partial-S2固有state・unit-radius・alias-free・fresh IIDという条件付き。G6 zero-fillへnormalizerだけ置換して使うことはできない |
| exponential接続 | [D1][D1]・[D2][D2]・[D3][D3]、[accounting.py][DACC] | 非unitary finite meanの積、Taylor/native誤差、全trial十分N・accepted価格を接続した既存例。ただしSOURCEはp、state、m、steps等を固定して検査するため、分子用汎用runnerではない |

### 接続に必要な最小の仕様差

新samplerではなく、既存generatorと分子primitiveの間のsidecar契約が必要となる。label→(sign, basis ID/hash, support, actual adjoint)を固定し、元component IDとG7 labelの一対一対応、identity移動、時間符号、語順、controlled相対位相を記録する。G6の拒否は全pathの古典zeroとして扱う。既存RTEEventへ偽装して、orderとword lengthを混同する接続は不可。

入力については、保存binary64をexact rationalとして定義する場合でも、正規化し直したpとλが元の係数を再構成するかを固定する。floatの`abs(c)/fsum(abs(c))`を個別にFractionへ変えるだけでは和1を保証しない。元モデルとの差、threshold、DF近似、basis近似は別のbias項である。今回はこの変換も係数取得も行っていない。[KERNEL][DFTAIL][F3]

Evidenceは少なくともsource/input/policy hashes、全trial ID、raw/coin/accepted/terminal分類、実proposalとweight、ordered label/phase、angle区間、provider/native certificate、測定axis/bit、prefixと全N完了状態を結ぶ必要がある。full child配列の保存・再構築を不要とする案Aは今回使わない。現G6/F3の全child、Parent.s、hash/serialization、angle時再queryの費用も予算へ残す。[A][F3]

**費用評価側の隠れた全列挙が実際にある。** [D1 pipeline][DPIPE] `collect_pending`はproduction予算の後で`list(reference_events(generator))`等を作り、全eventと全角度keyを会計へ渡す。sampling自体が非列挙でも、この参照・native費用取得経路まで非列挙とは言えない。小さいL=3の証拠取得手順を分子へそのまま移植しない。全support列挙を回避した認証費用評価・角度coverageを既存保証で作れるかは未確認であり、経験的な少数event単価を母平均の認証値にしない。これが新しい非自明な保証を要するなら、C1はその地点で未解決として止まる。

有理演算回数ではkernel前処理O(Lm²)、級数保存O(Lm)、単純なparent/child queryのcoreはO(m³+Lm²)。現selector・sum・sort・検査・Evidenceを含めた全trial計算量ではない。closed P5のO(L²)群setupも全bit計算量やonline selectorのO(log L)保証ではない。入力bit長、rational分子/分母の成長、root/angle区間、strict合成keyの取得・保存を別に付ける。これらの算術回数から実CPU秒や量子Tへ換算しない。[A][B][F2]

## 3. 候補の事前スクリーニング

| 候補 | 既存証拠との接点 | 技術分類・追加投資の前提 |
|---|---|---|
| **C1 保存H4 DFの単一coherent signal** | 分子入力identity、tail再構成、ordinary RTE、partial-S2、controlled X/Y、局所費用の既存記録 | **条件付き候補1件**。実分子のbasis費用とreturn集約の交換を問いにできる。入力公開性、adapter、誤差・同task予算、strong baselineが揃わなければ実行しない |
| 全round PR/RPE energy | PR一次文献、既存短round accounting | **契約化は保留**。state/overlap、energy prior、全round schedule、phase/visibility margin、decode failure、全費用が未接続。C1の成功から自動進行しない |
| Pauli Hamiltonianの一般シミュレーション | 既存分子Pauli生成library、WBC/CTS、G9/G10 | 技術的適用域はあるが、分子係数を含むsource-bound package・同task native表は今回未取得。Pauli全収集・word圧縮が合法な強い対照であり、RAの優位機構は未確認。別の第2実験を作らない |
| synthetic D系列の延長 | 同exponential taskへの既存会計 | 既存結果の反復に留まるなら停止。分子接続の代わりに同P5のFull-Nを実施しない |
| Parent allocation / DF cubic correction / M1-Sとの融合 | 関連する限定成果あり | **今回候補化しない**。配分・補正mean・二枝joint lawは別task。各closure/STOPを維持 |

### C1で既にある入力と、ない情報

[PR-2 V1–V3報告][PRINPUT]にはlinear H4 1.00 Å、STO-3G、8 system qubits、保存DF rank12、N=4/Nα=Nβ=2、snapshot SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`、Hamiltonian hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`がある。生成順prefix3のtailは既存報告で324 components、λ_R=0.5832704246339983。これは数値分解の既存検査で、今回の有理同一性certificateではない。

λ_Rとcomponent数だけではpの分布・return集約効果は決まらない。全p、identity policy、label重複、basis reuse、rare order、実際のnative角度が未照合なので、効果の大きさを推測しない。**全体Pauli数とDF label数、係数型数を同じLにしない。** x=λ_R|δ_tail|≤1を各random occurrenceで確認する必要がある。総時間T≤1だけではshort-step条件を確認したことにならない。

保存数値stateはsector-ground oracleであり、物理的な準備回路や厳密ground overlapを認証していない。新規分子計算を使わず既存入力を再利用できる可能性はあるが、本調査ではsnapshot本体・state本体を取得していない。公開metadataがあることと、第三者が計算入力を再取得できることは別である。

既存[PR-2契約][PRCONTRACT]は同じdevelopment H4、T=0.8、complex ε=0.05、軸failure .025、q/r/Kの候補を定義する。ただし、そのshot設計は保存数値signalとの差を用いる。これはG6/F1による未観測母平均への認証予算や物理測定結果ではない。新しい契約では既存精度を設計候補として引用し、保証方式を揃え直す必要がある。旧PR-2のwinner・materiality・held-out結果を再分類しない。

### 利益が生じ得る費用構造と反論

DF involutionは異なるbasisのZ/ZZ共役で、productとcontrolled rotationにbasis費用が付く。同一label returnの除去で実Q呼出しを減らせる可能性はある。H4の[限定費用診断][BASIS]は、保存8列の部分回路を別compileした加法的RZ/CXでbasis寄与が大きかったと報告する。しかし全eventがorder0であり、RAが主に扱う高次数returnの頻度やT削減を測ったものではない。snapshot hashもPR-2とは異なる。同一入力の価格として流用しない。

強い対照にも同じbasis相殺、signed Pauli product簡約、同一angle cache、support限定basis、合法selectorを認める。RAだけにdirect loweringを与えてordinaryへgeneric helperを課す比較は不可。一般Fullはclosed P5と同じ理想ensembleでも、未知normalizerのzero-fill、認証U、古典棄却費用を持つ。新しい分子入力にしただけでこれらの不利は消えない。利益が出るとすれば、**同じ誤差を満たす量子価格・準備頻度・古典取得費用の交換**に具体的な証拠が必要である。[G10][W2C]

## 4. 強い先行法との比較

| 対照 | 同taskに合わせる条件／比較で残る問い |
|---|---|
| Ordinary RTE | WBC Appendix C Algorithm 2はorderとIID labelを抽出し、全語列挙を要求しない。既存DF ordinaryにもbasis cache/融合を認める。RAの非列挙性だけで優位としない。[WBC] |
| Partial return、closed P3+tail | 同じmのfinite Taylor mean、実proposal、誤差、全trial単位へ揃える。一般Full対これらのN一律優位は未証明。[W2B] |
| Closed P5 | m=5で必須。群normalizer/completionとcacheを含む。全parent列挙対照にはしない。digital moment/range/bias・予算と取得価格も揃える。[P5][B] |
| CTS / SCU | Pauli収集・cross-term処理を含めて同じcoherent signalへ接続する。既存g9 CTSはtoy専用。DFからPauliを取得する費用を両側へ対称に付ける。Supplementary Note 3は層ごとの独立samplingを用い、全展開を必須にしない。その名称だけからMCMCのmixing費用を対照へ課さない。[CTS] |
| 既存PR/qDRIFT・通常PF | 同じ分子state、全時間、誤差、準備を使う。PR Appendix Aはinvolutionとその時間発展を許すので、DF一般involutionの導入自体は追加貢献でない。qDRIFTのphase distortionとRTEの減衰を同じ量としない。[PR] |
| Trotter / qubitization | PFにはcommutator構造を使う誤差評価・合法な順序/融合、qubitizationにはPREPARE/SELECT・反射・workspace・controlの実価格を認める。query数とT数を同列に比較しない。未取得対照を「支配済み」にはしない。[TROTTER][QUBIT] |

一次文献の新規読解は次の関連箇所に限定する。PRは指定v2 PDFの解析テキスト、§IV Theorem IV.1、§V、§VI DF、Appendix A/B。**Theorem IV.1はRMS energy精度のstatementで、failure probability付き保証へ無条件に置換できない。** WBCはHTML v2のAppendix C Algorithm 2・Appendix D、CTSはHTML v2の§II.1/II.3、Methods IV.2、Supplementary Note 3。Trotterはv3本文の設定・commutator誤差枠組み、qubitizationは著者の掲載ページ/要旨のaccess条件を確認した。全証明、最適な実装、priorityの網羅調査は未実施。PR PDF画像取得は失敗し、数式の視覚照合・PDF bytes同一性は認証していない。

## 5. 負結果の移送範囲

| 固定証拠 | C1/PRへ保持する内容 | 移してはいけない結論 |
|---|---|---|
| G9/G10 | 強いclosed対照を加えるとgeneral Fullの登録T利益が消える例。m7の任意proposal下界も固定辞書/価格/保証内で保持 | 分子DF・別provider・別費用への一般no-go。単なる新geometryで旧下界を解除したという扱い |
| P_U反例 | 一般U≤partial/形式P5 benchmarkという予算優位は反証済み。[PU] | 真のB_full順位や全量子費用の逆転、G6の数学的不成立 |
| W2b/W2c | N、range、bias、ceil、confidence、取得費用、準備頻度を揃える。条件付きlogical非劣はgeneral native/CPU非劣でない | N_F≤N_Oだけから総T/CPUの非劣 |
| D1–D3 | 同じexp targetでも保証と価格が選択を変える。全方式/primary pair・r選択を区別 | toyの割合やh交点を分子へ転記 |
| DF correction closure | whole指数とweighted線形補正の有限meanは違う。3条件の固定加法planでの負結果を保持。[DF] | correction価格をRA価格とみなすこと、一般DF provider不可能性 |
| Parent / M1-S | 配分・cost-aware IS、二枝相関には別のlawと保証がある。[PARENT][M1S] | S4割合・visibility/shot witnessとG6費用の掛け算 |
| 16,896 Pilot | 固定P5の古典生成/監査/保存を動かした証拠。順序・cache・prefix共有とcompact証拠制約を保持。[PILOT] | 量子shots、energy精度、L=324のCPU予測、全N費用、IID実証 |

## 6. 投資判断に返す条件

技術的に次へ進む根拠となり得るのは、C1で「保存分子入力のどの機構が実native価格と十分N・古典費用の交換を決めるか」という問いを固定でき、**closed P5等を含む対照に同等の最適化を与えた比較が、D系列toyやPR-2の再記述以上の情報をもたらす**場合である。明瞭な適用不能・非改善も、固定入力/費用/保証の範囲を閉じる成果にはなり得る。独立原著の中心に十分かは別判断である。

追加投資を止める条件は、①exact入力・state/providerが取得不能、②共通の誤差・confidenceを閉じられない、③strong baselineを省かないと利益が出ない、④改善が弱い対照・無料の準備/取得・acceptedだけへのconditioningに依存、⑤既存P5/PR-2/D系列と同じ内容の反復、⑥新sampler・新保証を黙って追加しないと成立しない場合である。未取得費用は0ではなくUNKNOWN、精度不足・技術停止はscientific lossとは別に保存する。

大分子への一般性、実行可能なstate preparation、広い精度域でのbest-to-best比較、再現可能な実入力、既知applicationとの差が未取得である。H4一点の勝敗だけでは高インパクト原著の十分性を決められない。GPT・ユーザーがこの不足に見合う投資価値を認めるか判断する。本書では正式採択しない。

**新しい証明・adapter実装・科学計算/native取得は、それぞれ別の承認対象。公開・保全確認後mandatory STOP。**

[KERNEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/return_aggregation.py
[GEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g7_generator.py
[CQ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g7_provider.py
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g9_p5.py
[NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g9_native.py
[DFTAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_rte_tail.py
[DFBUILD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_rte_qiskit.py
[REPEAT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_partial_s2_repeated.py
[HAD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/rpe_hadamard_interrogation.py
[RPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/rpe_resource_accounting.py
[DACC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/rte_development/accounting.py
[DPIPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/rte_development/pipeline.py
[F1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/preflight_precision_contract.md
[F2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/full_bit_complexity_accounting.md
[F3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aea9b8c228c5b940f2fcba0f94493b3ffdec46e/docs/research/track_b_g6_f3_preflight_angle_20261018/handoff.md
[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/df90ed8c8e847f8ea1861d9761b9515e7a895916/docs/research/track_b_partial_sum_query_sampler_audit_20261020/README.md
[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f823e909cb7049a612f1bf2c247f3bbe15fc184d/docs/research/track_b_coefficient_type_canonical_audit_20261020/README.md
[PRINPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/research/pr2_v1_v3_result_packet_ef86868_20260928.md
[PRCONTRACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/research/pr2_matched_accuracy_resource_contract_v1.md
[BASIS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/403204edf22e5712a71ac9ac7eebdbcd18626421/docs/diagnostics/rte_basis_cost_attribution_20261013.md
[G10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fcd3ea6217bc00b667180cec149a70102d75f07e/docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md
[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/tracks/algorithm_codesign/d2_boundary_results_and_gpt_handoff_20261013.md
[D3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d3_final_saved_mechanisms_and_gpt_handoff_20261014.md
[PU]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/850ac89e1a6ea955def805e5462b8694ec0840b4/docs/research/track_b_g6_pu_analytic_proof_20261018/proof_or_counterexample_certificate.md
[W2B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed26d1c85cf7ad80625f7abbc662b70abd690a97/docs/research/track_b_g6_w2b_certified_trial_budget_20261018/handoff.md
[W2C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/864097c1285f1f06399040cf0f7c0f42334d9252/docs/research/track_b_g6_w2c_total_cost_conditions_20261018/handoff.md
[DF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/research/track_b_df_correction_et_scientific_review_20261012.md
[PARENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/tracks/algorithm_codesign/parent_allocation_s4_final_comparison_evidence_packet_20261015.md
[M1S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md
[PILOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/handoff.md
[WBC]: https://arxiv.org/html/2110.12071v2
[CTS]: https://arxiv.org/html/2407.21095v2
[PR]: https://arxiv.org/pdf/2503.05647v2
[TROTTER]: https://arxiv.org/html/1912.08854v3
[QUBIT]: https://quantum-journal.org/papers/q-2019-07-12-163/
