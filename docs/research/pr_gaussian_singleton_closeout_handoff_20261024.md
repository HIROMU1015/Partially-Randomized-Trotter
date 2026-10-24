# Gaussian singleton研究の一時終了と次研究への引継ぎ

2026-10-24 JST。ユーザーが、**Gaussian singleton合成・選択最適化を独立研究方向として一旦終了する方針を正式採択した**。直前の[選択規則設計調査][SELECTION]の推奨を、今回の指示で採択した記録であり、過去の推奨・未採択という記録は変更しない。

今後の優先方向は、**PRの演算子辞書・乱択分布・実回路費用・統計負担を組み合わせた資源削減**。具体的な改善機構は未選定で、新しい設計・実装・科学検証を開始していない。この区切りはGaussian合成全般の不可能性やPR研究全体のNO-GOを意味しない。

## 1. 保持する成果と未確立事項

以下は固定公開結果の引用・技術的整理で、新しい再集計ではない。対象は4 mode・2粒子、初期占有(0,1)、L_D=1、残差2fragmentの固定合成Hamiltonian。分子geometry・AO basis・分子DF rankは非該当で、rank探索は行っていない。以下の「fragment F1」は残差fragment名で、科学検証段階F1と区別する。

| 証拠 | 保持する判定 | 適用範囲・限界 |
|---|---|---|
| [混在辞書F1][F1] | SS／SP／PS／PPの登録済み科学的正しさ検証は全件PASS | T=1/5、q=2、δ=1/10の一点、r=2、h=1/20、K=2。未資格intervalはUNPRICED_FALLBACK。最適選択・認証価格・shot数の認証ではない |
| [限定F2][F2] | 4構成の登録済みoperator・signal・finite mean検証はPASS。1-shot期待費用の全7回路指標で **PP＜SP＜PS＜SS** | T=1/5、q=1、δ=1/5の一点、r=2、h=1/10、K=0。同一full wrapper・固定compiler。7指標はRZ、CX、RZ depth、CX depth、total depth、quantum-only depth、size。測定数は共通 |
| [保存F2寄与分析][ATTRIBUTION] | 公開期待費用への再集計整合を確認。残差fragment F1のGaussian singleton、特にsupport2が高費用となる構造を特定 | 保存分布・case費用の記述的な寄与。Gaussian変換やcompilerの独立した因果効果は分離していない |
| [候補A・PS48][PS48] | 6基底・raw／compiled各48-wrapperで科学的同値性PASS。support1／support2は各24-wrapper | F2と同じ条件・compiler。等case重みでは一律適用によるRZ・CX削減を支持せず、index21／25の各8caseではRZ・CX・depthが改善。全F2の確率重み付き期待改善率ではない |
| [構造選択規則の静的調査][SELECTION] | Compile前の構造情報による一般化可能な選択規則は未確立 | 観測済みcase IDの記憶や同じPS48への適合は独立検証ではない。両方式をcompileして選ぶ強い対照に対する実益も未検証 |

F2／PS48の固定compilerはQiskit 1.3.0、basis={rz,sx,x,cx}、optimization_level=1、seed_transpiler=17、free topology、approximation_degree=1.0、qubits_initially_zero=false。今回これらの条件や既存の数値許容差を変更していない。

一時終了するのは、局所Gaussian singletonの合成・compile前選択を独立研究として追う作業である。候補Aの実装と同値性の成功、負の一律適用結果、局所改善例、未検証の選択案をすべて残す。既存の独立検証案や全F2展開は採択・承認しない。

**γによる共通finite targetと有限ビット実装平均の差の保証、統計条件の正式採択、interval資格、認証N、N加重費用は未解決のまま**。SS側の小さいλ・BtotとPP側の小さい1-shot費用の交換関係だけから、同精度総資源の順位は決められない。F1／F2の固定許容差内PASSを厳密なγ保証、真の時間発展信号やenergy推定全体の優位へ格上げしない。

## 2. 次研究へ渡す論点と既存資産

新しい文献探索やアルゴリズム設計は行わず、[既存の強いbaseline整理][BASELINE]と[Phase 2Cの先行研究対応][DELTA]、Track Bの固定引継ぎを再利用した。

| 論点 | 再利用する成果 | 未解決のPR固有の改善機構・判断点 |
|---|---|---|
| Cost-aware importance samplingとwhole-trajectory費用 | [C4の既知法対応・停止理由][C4]。最適確率則やwhole-path費用の採用自体は既知ISへ還元する。基底共有・相殺を含む保存F2のcase費用も接続の材料 | 非加算的な全trajectory費用を、安い生成・確率／weight評価・normalizerアクセスで扱えるか。理想moment×平均価格の改善だけでなくrange／tailと全試行の古典・量子費用が必要。一般DFでの効率的生成差は未確立 |
| 演算子辞書とsampling lawの共同設計 | [Parent allocationの限定成果と終了記録][PARENT]、混在辞書F1／F2、既知cost-aware lexiconとの[方法差整理][DELTA]、[unitary partitioningとの対応][PAIR] | 辞書・角度変更は係数質量・law・実価格を同時に変える。common配分はそのfamilyのL1最小で、child配分はL1増加と安い回転の交換。単純な辞書選択＋既知ISを独立新規性とせず、同じ候補集合を使う既知法との差を特定する必要がある |
| 必要shot数を含む総資源最適化 | Parent allocationの固定finite-P5・native価格・confidence条件付き資源地図と[強い対照の比較条件監査][G9] | 目標・誤差予算・失敗確率範囲・bounded-IID／range・finite-bit biasを先に揃える。従来Track Bの十分Nや価格を現在のF2へ移送できない。γ等が未成立なら認証Nや総資源優位を主張できない |
| 強いbaselineとの差 | [G7／G9の既済比較と限界][G9]、ordinary／partial／P3+tail／General Full／Closed P5／matched CTS、各方式に同じcost-aware ISの機会を与える既存方針 | 既存の固定P5対照を未評価として取り直さない。coherent controlled first momentを保つ意味論と、情報取得・生成・compileを含む総費用に、既知法を超える具体的な差が残るかが未確立 |

Track Bから引き継ぐ境界も保持する。

- **C4**：既存監査で最適則・平均保存・backward DPの独立な算法差は確認されず、Phase Bへ進んでいない。未知の生成効率だけを継続理由にしない。今回その停止を解除しない。
- **Parent allocation**：3-system-qubit development synthetic、finite P5・r=1・固定native provider等に限定した成果。条件付きT利益とN／CX増加を保持するが、分子DFや全PR taskへ一般化しない。D＋Eで採択済みの同一入力探索の区切りを解除しない。
- **G9**：同じsynthetic・matched provider内ではClosed P5のT-primaryの強さとCTSのCX上の交換関係が既に確認されている。別provider・confidence・費用scopeを跨いだ単一順位ではない。
- **C1**：[H4 pilotの資源観測と未取得項目][C1]を再利用可能。Linear H4 1.00 Å／STO-3G／DF12／generation-prefix L_D=3、T=4/5、q=1、r=4、m=5、TAIL_ONLY_PLUS_SCALARの結果で、現在の合成F2とは別条件。小さいlogical差や経験的weight momentをnative認証・Nへ置換しない。一時凍結を維持する。

## 3. 比較で落とさない条件

既存レビューに従い、同じfinite target・state／sector・誤差／confidence・controlled位相を保ち、同じ辞書候補・価格情報・合法cache／selector／前処理を強いbaselineにも認める。correlated channelやVQE測定の保証を、PRのcoherent first momentへそのまま流用しない。

資源にはstate preparation、controlled evolution、readoutを含むfull wrapperと、採用・棄却・zeroを含む全試行、normalizer／確率／weight取得、辞書生成、setup／online生成・compileと償却範囲を含める。未測定費用を0で補わず、RZ・CX・depthやnative Tを無根拠に単一指標へ足さない。literal price、1-shot費用、認証Nを用いる総費用、physical同精度比較はそれぞれ区別する。

次研究の未解決点は、これらを「組み合わせる」という方針自体ではなく、**同じ情報と最適化機会を持つ既知法に対して、平均保存・生成可能性・全費用まで説明できる具体的なPR固有の方法差を得られるか**である。現時点でその機構のGOや独立新規性を認定していない。

## 4. 次の担当・承認・STOP

1. **ユーザー／GPT**：本書と固定引継ぎを確認し、次に研究する具体的機構を選ぶか、追加研究を見送るか判断する。現在採択済みなのは一時終了と優先方向まで。
2. **Codex（別指示後）**：選ばれた機構について、再利用資産・既知の強い対照・未取得保証・最小scopeを整理する。新しい詳細文献レビュー、科学的設計／重要レビューは自動開始しない。
3. **ユーザー／GPT**：必要と判断した場合だけ、新実装と、保存再解析・build／compile・sampling・simulation等の科学実行を対象限定で別途承認する。γ／統計条件の採択・資格整備や認証N比較も今回の継続作業には含めない。

今回の変更は本書、研究概要・validation状態・履歴・索引と既存validation manifestの研究状態metadataだけ。新しい科学計算・数値再集計・回路build／compile・selector／sampler実装・grant発行は0件。固定SOURCE・保存結果・旧raw・grant・ledger・marker・STOPを変更せず、非公開情報は公開対象に含めない。専用branchの公開内容と保全を照合後、**mandatory STOP**。次の研究判断はユーザー／GPTへ戻す。

[SELECTION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab3b2c30df6333ceefd408f59e48c5ae0de4563f/docs/research/pr_rectangular_selection_design_20261024.md
[F1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b0dad8cc2f7a4cb821f53185c4a65f3a42e8f381/docs/research/pr_algorithm_improvement_phase2d_f1_rerun_20261024/EXECUTION_REPORT.md
[F2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc7378ab34015287215bcf4ce52147e46d94ecd9/docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md
[ATTRIBUTION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41c4e79f1c58af0f2be634f46dbeba8dcd36acc5/docs/research/pr_algorithm_improvement_phase2d_f2_attribution_20261024.md
[PS48]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1e20458d742a8fac62ce0d0264188b1a7772ee28/docs/research/pr_rectangular_givens_ps48_execution_20261024.md
[BASELINE]: pr_algorithm_improvement_phase2a_20261023/STRONG_BASELINE_AND_OVERLAP_MATRIX.md
[DELTA]: pr_algorithm_improvement_phase2c_design_20261023/METHOD_DELTA_AND_PRIOR_ART.md
[PAIR]: pr_algorithm_improvement_phase2c_method_delta_20261023/STRONG_BASELINE_COMPARISON.md
[C4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_c4_finite_rte_gpt_handoff_v1_20261014.md
[PARENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md
[G9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6576e606d4edcc6c53f095ee5bdf750defa99a42/docs/research/track_b_g7_g9_strong_comparator_audit_20261019/handoff.md
[C1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md
