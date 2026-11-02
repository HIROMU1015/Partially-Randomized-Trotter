# Track B：次期アルゴリズム改善方向のDecision Packet

2026-10-15 JST。**四つの未採択の候補群**を比較するための資料。順番は実装／分布／分解／生成の区分で、科学的な優先順位ではない。具体的な新アルゴリズム・独立新規性・改善率を認定していない。

ユーザー採択済みの方針は **D＋E：Parent allocationの限定的成果を保持し、現在の数値探索を区切る**。[指示原本](inputs/track_b_de_direction_preparation_user_instruction_20261015.txt)／[成果整理](track_b_parent_allocation_research_closure_20261015.md)。次の研究テーマ、実験、実装を選ぶ権限へは拡張しない。今回の資料整理完了後は**mandatory STOP**。GPT重要科学レビューの開始にも、新科学実行にも、それぞれユーザーの明示承認が必要である。

## 1. ここまでの証拠から選定時に区別すべきこと

- mean-preservingな係数再配分と、native総費用利益は別問題。既知Euler/ISの再記述だけでは方法差を示せない。[PA-ART][PA-ART]
- cheap primitive、B²低下、T切片低下が、多gate/多step/実taskで残るとは限らない。[SP1][SP1]／[R15][R15]／[PA-REVIEW][PA-REVIEW]
- 表現・保証・精度別priceで最良候補が変わった有限証拠はあるが、providerや角度priceへ依存する。[D3][D3]／[D3-MECH][D3-MECH]
- basisに費用が掛かる固定H4診断はあるが、RZの約85%はorder0の8保存列の加法比。T割合・全RTE母平均・削減可能率ではない。[BASIS][BASIS]
- performance不利、同値性、方法差未確立、証明未取得を区別する。generic wholeとGaussianの既存終了判断は維持し、RA-D0未完了を性能negativeとしない。[DF][DF]／[GAUSS][GAUSS]／[RA-D0][RA-D0]

[資源構造調査](track_b_resource_structure_and_bottlenecks_20261015.md)に契約・入力・費用scopeをまとめた。以下の「最小検証案」は**選定のための抽象的な選択肢**で、対象、seed、precision、budget、判定閾値を採択していない。新しい検証は今回行わない。

## 2. 候補群I：固定ensembleのままnative実装を安くする

| 判断材料 | 内容 |
|---|---|
| 1 問題 | 同じ理想event lawとweightを維持したまま、basis変換・controlled lowering・event/step境界のnative費用を下げられるか |
| 2 既存法との差分 | 現時点では具体的方法差は未確立。既知Gaussian active-span/partial-basis、adjacent inverse cancellation、一体compilerが対照となる。「既知相殺を使う」だけを新アルゴリズムとしない |
| 3 改善機構の仮説 | eventを変えずに共通basisやinverseを共有し、統計lawの再設計を要さない実装余地を利用する |
| 4 改善対象 | event/accepted path当たりのCX/RZ/T、depth、場合によりworkspace |
| 5 悪化の可能性 | 追加ancilla、controlled phase処理、近似bias、classical circuit生成費用、情報取得・前処理費用 |
| 6 数学・技術条件 | ordered operatorとcontrolled relative phaseの保存。有限精度error合成。shared変換の範囲・free oracleを使わないこと。lawが同じでも誤差予算を変えるならNは自動不変ではない |
| 7 根拠 | D2の共通literal境界相殺、H4基底分解、旧hierarchical compiled-cost holdout。異なるモデルを混ぜた削減率は提示しない |
| 8 近接研究 | 既知Gaussian/reflection/active-span構成、DF/native compiler。既存Gaussian監査の既知対応をまず継承する |
| 9 不足 | rare-orderを含む実taskのT/CX内訳、D-like境界を含む回路trace、一体最適化後の由来付き費用。H4の8列はこの証明ではない |
| 10 最小検証の案 | 新しい実装差が定義できた場合、同じ保存pathとcontrolled taskで旧/新実装を対応比較する案。native取得・行列検証が要るなら別契約・別承認にする |
| 11 継続/終了の判断材料 | 既知compilerより残る追加手順があるか、weight/precision/準備費用を揃えてnative節約が残るか。差が既知簡約だけなら独立method claimではなく実装整備として扱うかをGPTが判断する |

資料：[D2][D2]、[H4基底診断][BASIS]、[hierarchical holdout][RTE-COST]、[Gaussian既知法監査][GAUSS-AUDIT]。技術的には保存IR・phase guardという部品はあるが、**新しいwhole-circuit結果はない**。D2を再実行せず、85%節約などの仮説値を置かない。

## 3. 候補群II：full-pathの平均を保つcost-aware生成

| 判断材料 | 内容 |
|---|---|
| 1 問題 | basis切替を減らすpath lawを使いながら、同じcontrolled有限時間発展のfirst momentを保ち、moment/range/likelihood費用を含めて得になるか |
| 2 既存法との差分 | cost-aware IS、PRHS、qSHIFTの近接性が強い。単なるpersistent Markov proposalやP/qの導入は方法差ではない。必要なら効率的sampler・likelihood・情報アクセスに具体差を特定する |
| 3 改善機構の仮説 | 全pathのnative費用構造を利用し、安いpathを多くする。ただし補償weightの二次momentとrangeへ戻す |
| 4 改善対象 | path当たりbasis/CX/T、既知proxy上のcost×moment、場合によりsetup/online費用 |
| 5 悪化の可能性 | 最大weight、second moment、十分N、proposal生成とP/q評価の古典費用、有限bit bias |
| 6 数学・技術条件 | 周辺分布保存では不十分。ordered-product mean、support、phase、正確/認証されたpath確率が必要。単一shot±1のmeasurement noiseを含む保証、non-IID依存がある場合のconfidence証明が必要 |
| 7 根拠 | 相関乱択の既存監査にmean反例、full-path IS条件とrangeの論点がある。S4/SP1は価格低下とmoment増加が交換になるという別の限定証拠 |
| 8 近接研究 | Resource-Optimal IS、Pathwise Random Hamiltonian Simulation、Adaptive sampling based quantum simulation/qSHIFT。channel/observableとcoherent operator meanのtask差、版差を保つ |
| 9 不足 | 既知最適lawから実行可能な低費用samplerを得る具体機構、acquisition費用、実controlled taskでの保証と費用。旧監査のqSHIFT式表示の未解消箇所も残す |
| 10 最小検証の案 | 具体的な新生成手順を記述できた時に、まず登録外の短いordered-productでmean/likelihoodを検証する案。その後のpath費用取得は別の科学契約で判断する |
| 11 継続/終了の判断材料 | 新しい手順が既知法の特殊化か、情報/生成費用を含む実質差があるか、weight増加を払ってもtask資源が改善し得るか。近接手法の単なる移植か独立研究かをGPTが判断する |

資料：[既存の厳密対応監査][CORR]、[IS/CTS等のclaim対応][D-ART]、[PA既知性監査][PA-ART]。この群は、以前閉じた単純相関案を再採択するものではない。履歴依存の平均とnative costの両方を閉じる必要がある。

## 4. 候補群III：taskを固定した分解・前処理とnative access

| 判断材料 | 内容 |
|---|---|
| 1 問題 | 同じ物理Hamiltonian/taskを保つ範囲で、分解、frame、生成順序などの選択が高価なbasisや回転辞書を変えられるか |
| 2 既存法との差分 | 具体的な新分解は未提案・未採択。既知DF/Pauli collection/low-rank/Gaussian法、BF/BM、他trackでのHamiltonian変換との重複を先に識別する |
| 3 改善機構の仮説 | 単なる合成価格の偶然を追うのでなく、native accessを含む構成を変えて頻繁な高価操作そのものを減らす |
| 4 改善対象 | basis/CX/T、必要情報量、classical setup、場合により係数norm/PF誤差 |
| 5 悪化の可能性 | norm増加、PF/Taylor bias、term数、情報取得、controlled block実装、状態準備と変換の費用 |
| 6 数学・技術条件 | 同じ物理targetへの演算子同値性または明示誤差、control phase、分解取得費用、情報access、deterministic/random splitを会計。回路単価だけで分解の優劣を決めない |
| 7 根拠 | H4の限定基底費用、R1のbasis/odd費用、DF whole補正の固定局所費用不利、BM同値性、Gaussianの既知対応 |
| 8 近接研究 | 元PR、既知DF/Gaussian生成、CTS/Pauli収集、NCC/HNCC。既存監査を再利用し、候補が定まってから直接関連差を確認する |
| 9 不足 | 既知法を超える具体的な変換、実taskの同じ誤差/情報accessでの費用。分子PR/QPE全体へ接続した整合ledgerは未取得 |
| 10 最小検証の案 | 具体差が出た場合、先に数式上の同値性・access/cost表を作る案。小入力の比較実行へ進むかは別判断で、今回Hamiltonian生成を行わない |
| 11 継続/終了の判断材料 | norm/誤差/取得費用へ費用を移しただけか、既知分解の再表現か、同じ情報で追加native利益を得る具体構造か。隣接Track A/QCの課題との分担もGPT判断へ残す |

資料：[元の系列を区切ったDF記録][DF]、[DF先行研究対応][DF-ART]、[BM同値性][BM]、[Gaussian監査][GAUSS-AUDIT]、[R1.5][R15]。generic whole補正やactive-span構成の終了を解除しない。IDEで開いている他研究を作業対象へ取り込まない。

## 5. 候補群IV：非列挙生成・normalizer・角度取得の実装可能性

| 判断材料 | 内容 |
|---|---|
| 1 問題 | 数学的に有利なensembleを、全event表や無料global情報を仮定せず、有限bitで生成・価格取得できるか |
| 2 既存法との差分 | Green再帰やclosed P5少数群式は既知/既存成果。そこを新発明として再提出せず、query、bit複雑度、selector、cold取得、error certificateに追加の手順差が必要 |
| 3 改善機構の仮説 | quantum lawを変えずにsetup/onlineを減らす、または同じ情報でnormalizer/価格情報を取得可能にする。quantum lawや保証を変える場合は別問題として扱う |
| 4 改善対象 | classical memory/CPU/online latency、angle acquisition、必要に応じてzero-fillによる無駄な量子試行 |
| 5 悪化の可能性 | 取得精度・certification費用、cache miss、巨大selector、有限bit bias、未知normalizer下の保守的N |
| 6 数学・技術条件 | 理想mean・全trial分母、zero-fill、dyadic実law・weight・moment/range、取得error、bit費用の保証。未知normalizerを既知扱いしない |
| 7 根拠 | G6のlocal query、G8のcold/warm live miss、G9のP5 setupとlinear group selector、D3/S4のnative key/cache identity管理 |
| 8 近接研究 | Green/free-product再帰、SCU/CTS、既知IS、on-demand合成・sampling data structure。量子query lowerと古典算術operationを混同しない |
| 9 不足 | 大規模online費用が実際に律速か。G8の2048 interface trialはその母集団測定ではない。新しい手順差と一般scopeのbit/メモリ保証も未取得 |
| 10 最小検証の案 | 数学的な生成・取得モデルを定義し、既存traceのどの欠測が必要かを先に列挙する案。新しい長traceや合成計測は別承認。実行対象・規模はここで固定しない |
| 11 継続/終了の判断材料 | 同一lawなら量子資源改善ではなく古典実装改善に留まる可能性を含め、独立RQの価値を判断する。実際のonline bottleneckも方法差もなければ補助実装の範囲とするかはGPTが判断する |

資料：[G6][G6]、[G8][G8]、[G9v2][G9]、[return構成ノート][RA-THEORY]。この群を残した理由は実行可能なaccessが数理性能と分離されているためで、未測定費用を「最大ボトルネック」と認定したためではない。

## 6. 近接文献と既知部分の入口

文献の詳細対応は、既に本文を読んだ[固定D1–D3監査][D-ART]、[Parent allocation監査][PA-ART]、[相関乱択監査][CORR]、[Gaussian監査][GAUSS-AUDIT]を再利用した。今回は広範な文献調査・独立科学レビューを再実施していない。

| 一次文献・資料 | このPacketでの役割 | 今回の取得範囲 |
|---|---|---|
| Cugini/Atif/Subasi, [Resource-Optimal Importance Sampling for Randomized Quantum Algorithms, arXiv:2603.13495v1](https://arxiv.org/abs/2603.13495v1), 2026-03-13 | cost×統計費用の一般原理は既知。固定辞書と全pathへの適用、実装可能なsamplingを分ける | abs metadata/abstractを確認。定理・式の対応は既存本文監査を参照 |
| [Quantum Simulation via Stochastic Combination of Unitaries, arXiv:2407.21095](https://arxiv.org/abs/2407.21095) | SCU/CTS、情報アクセス・Pauli収集、normalization/回転の強い対照 | 今回新規本文取得なし。D-ART/PA-ARTの固定本文監査に依拠 |
| [Sparse Probabilistic Synthesis of Quantum Operations, arXiv:2402.15550](https://arxiv.org/abs/2402.15550) | 線形分解・辞書とvariance/costの共同選択が既知である範囲 | 同上。operator/channel embeddingと実生成費用は別問題 |
| Cugini, [Pathwise Random Hamiltonian Simulation, arXiv:2608.29756v1](https://arxiv.org/abs/2608.29756v1), 2026-08-30 | 相関slice/quasiprobabilityの近接研究。候補IIの既知性確認 | abs metadata/abstract確認。詳しい対象・生成対応はCORRの本文監査を継承 |
| Lee/Choi, [Adaptive sampling based quantum simulation, arXiv:2604.26263v3](https://arxiv.org/abs/2604.26263v3)（旧版qSHIFT） | 履歴依存生成の近接対照。controlled coherent taskとの相違 | 今回webはcache miss。本文・v1/v3差は取得済みCORR監査に限定して引用し、新たに原文確認済みとはしない |
| Günther et al., [Phase estimation with partially randomized time evolution, arXiv:2503.05647v2](https://arxiv.org/abs/2503.05647v2), 2026-07-10版 | 元PRのphase-estimation taskと部分的乱択という位置付け | abs metadata/abstract確認。toy費用をPR全体の新forecastへ移送していない |

各論文の一般原理を今回の新規貢献として扱わない。本文未再取得の箇所をabstractから定理推測して補わない。近接法との完全な実装同値性、priority、新規性は未認定である。

## 7. GPTが次に判断すべき問い

| 科学的な問い | 既存資料で答えられる範囲 | 未決の判断 |
|---|---|---|
| 何のtaskを改善するか | finite operator mean、wrapper、分子RTE部分、PR phase estimationは別契約だった | controlled signal／channel／phase estimationのどれを主taskとし、どの情報accessを認めるか |
| どの資源が研究目的か | T/CX/N/準備費用/古典費用は交換関係を持つ | 独立座標のどれが制約で、何を主要目的にするか。Tとnative1Qを重複加算しない |
| どこが新しい機構か | Euler/IS、Green、Gaussian、相関path法との既知対応がある | 上記群のどこかに具体的・非同値・実装可能な差を設定できるか |
| 次の情報取得は必要か | H4基底割合、D3価格差、S4 h依存等の限定証拠はある | 未測定のrare-order/T/online/whole-circuit条件のうち、選定に本当に必要なものは何か |
| 継続の研究価値はあるか | 現Parent allocation探索はD＋Eで一区切り | 新しい具体的RQを採択するか、追加整理で止めるか、Track Bを休止するか |

新しい実験計画や科学的GO/STOPをこのPacketで確定しない。四群のすべてを研究として残す必要もない。特にIVは古典実装整備に留まる可能性、IIIは既知前処理または別trackの課題に留まる可能性を明示した。I/IIも具体的な方法差は未確立であり、研究価値の順位は付けていない。

## 8. 資料取得・検証・停止

固定GitHub資料46件を直接取得し、各指定commitのGit blobとbytes/SHA256を照合した。完成版レビューも取得済みである。内訳とpathは[取得索引](../../../artifacts/track_b_de_direction_preparation/2026-10-15/v1/evidence_index.json)。論文webのcache missはGitHub資料欠落と区別し、欠落を別資料の性能値で補っていない。

今回実施したのは資料読取り、既存数式/主張のscope照合、表示、リンク・identity・保護範囲の確認である。新しいN、price、NC、h-envelope、資源forecast、合成、compile、matrix/operator guard、sampling、trajectory、LP、科学runner再実行は0。過去の監査件数を今回の新規PASS数として数えていない。

**今回の完了条件は資料の公開と保全。科学実行・研究テーマの採択・GPT重要科学レビューは開始しない。mandatory STOP。**

## 固定資料の参照先

参照IDのcommit・path・bytes・SHA256は[取得索引](../../../artifacts/track_b_de_direction_preparation/2026-10-15/v1/evidence_index.json)に記録した。リンク先はbranch先端ではなく固定commitである。

[BASIS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/403204edf22e5712a71ac9ac7eebdbcd18626421/docs/diagnostics/rte_basis_cost_attribution_20261013.md
[BM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md
[CORR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md
[D-ART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d1c59b28f4370cc7be65cff60d1ec877fd1491f5/docs/tracks/algorithm_codesign/d2_boundary_results_and_gpt_handoff_20261013.md
[D3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0df9a3b36bc3540f0c83941871fa0d3110ddbf7/docs/tracks/algorithm_codesign/d3_v2_results_and_gpt_handoff_20261014.md
[D3-MECH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d3_final_saved_mechanisms_and_gpt_handoff_20261014.md
[DF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/df_correction_et_series_closure_20261012.md
[DF-ART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/df_sparse_correction_prior_art_comparison_20261011.md
[G6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_results_and_gpt_handoff_20261010.md
[G8]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md
[G9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
[GAUSS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/gaussian_product_generation_closure_20261013.md
[GAUSS-AUDIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6530c055c505712d5f1d1af604f1accf85cebba/docs/tracks/algorithm_codesign/df_rte_product_generation_audit_v1_20261013.md
[PA-ART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/parent_allocation_prior_art_correspondence_20261014.md
[PA-REVIEW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3086f48c5598c8bb31b0587e807ca6c1b3d2754/docs/research/parent_allocation_s4_scientific_review_revision_20261015.md
[R15]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md
[RA-D0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/ra_d0_one_shot_result_validation_20261007.md
[RA-THEORY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_0.md
[RTE-COST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/hierarchical_cost_validation.md
[SP1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md
