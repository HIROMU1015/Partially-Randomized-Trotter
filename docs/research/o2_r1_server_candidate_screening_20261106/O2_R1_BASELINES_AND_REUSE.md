# O2-R1：baseline、固定証拠、再利用と不足

2026-11-06 JST。[候補選別](O2_R1_CANDIDATE_SCREENING.md)の証拠対応。科学コード・テストは実行していない。SOURCE/結果の所在・版の確認と、保存済みの記述・値の引用を区別する。

## 1. 実際のサーバー環境と作業領域

担当は現セッションのLinuxサーバー、文書技術確認に利用可能な処理系はPython 3.12.3。旧集約資料のSERVER 1/2と同じ物理hostであるとは確認していない。IDEのTSV名だけからhostを同定しない。

開始時のrepositoryは `partially-randomized-trotter/`、remote originは `https://github.com/HIROMU1015/Partially-Randomized-Trotter`、mainのHEADは `0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf`。多数のstaged deletionが観測されたため、既存index/作業ファイルへcheckout/reset/clean/stageを行っていない。

専用領域は `partially-randomized-trotter-worktrees/o2-r1-server-evidence-screening-20261106/`、branchは `codex/o2-r1-server-evidence-screening-20261106`、基点はS2結果commit `dc42c3a9f77e960f415d0475da2ba03bb56252c7`。開始前に同名local branch・directory・remote branchとの衝突がないことを確認し、新設worktreeだけをsparse checkoutした。既存科学SOURCEを編集せず、今回の追加はこの調査directoryの3 Markdownのみ。

`git worktree list`で既存研究C・Track A/O1/O3・Track D等の作業領域を観測し、`git branch --list`でG6/C1等の歴史的branchを確認した。これは読取専用の所在確認で、他sessionのdirty/untrackedの整理・監査や全worktreeの凍結ではない。旧C1の専用runtime領域も所在が見えるが、本調査は固定Git内容を優先し、runtime/private rawの横断検索・再解析・archive展開を行っていない。

適用AGENTS、PROJECT_MAP、研究概要・現状の全文、研究目的の状態・半径・費用条件、VALIDATION_STATUSの関連箇所を確認した。正式共通指示2026-10-25の§8.1/8.2/8.4/10.1/10.3を実読し、[公開固定コピー][COMMON]との本文一致を確認した。旧one-shot・retry/STOP条件、SOURCE/入力/結果の保全と、新作業に承認証明を追加しない方針を併用する。

### 指定4 snapshotのアクセス

4 commitはすべてサーバーのGit objectとして利用できた。以下4入口をGitHub接続でも実取得した。それ以外の全旧rawをGitHubから再downloadした意味ではない。

| 固定commit | GitHubから取得した入口 | 取得blob |
|---|---|---|
| `dc42c3a9f77e960f415d0475da2ba03bb56252c7` | [S2 v2 REPORT][S2] | `edfc76d942dbe9fa692c5fd5533718571664f624` |
| `2b71117ae0f6727cfc9b4968bb6fd96be7265a60` | [O2-B][O2B] | `f5bc63fea151de508fe6aab3851c2533cde9754a` |
| `3571477093441ac4852166fe829f8ccc5db5ec4f` | [3環境統合 REPORT][INT] | `29597445de4ed59629fd0cca648ce336bf74bb6f` |
| `e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8` | [PR＋RPE文献調査 REPORT][LIT] | `59470b3df5a7142005ca30e179ad5d4c394e4610` |

O2-A/B、候補比較、REPORTを読み、3環境統合のRA-KERNEL/RA-NATIVE/ALLOCATION/C1/RPE-COSTとRQ-04/05の関連記述、文献調査のO2・既知法比較・費用地図を参照した。統合台帳の行数を実験数とせず、全資料の再監査・独立科学レビューを実施したとはしない。文献の既知性は固定調査の該当対応を継承し、新しい網羅的priority調査・定理証明は行っていない。

## 2. 固定SOURCEと結果の対応

サーバーにあるという事実だけで正本とはせず、実行SOURCEと結果carrierを分ける。次の固定pathはローカルGitから読み取れる。原報告のうち候補選別に必要な節と保存CSV/summaryを確認した。巨大rawの全数値再検査・再現実行はしていない。

| 系列 | 元の科学SOURCE | 保存結果／原報告・今回確認範囲 |
|---|---|---|
| G6 | `28cfabb1d47e0e1824bce1e154a9c289738fa2b9` | [handoffと既知法照合][G6]。kernel本文、support/phase/zero分母の既存test SOURCEを静読 |
| G7 | `ab2549f41b3546fb3940342a2162dd9ee93699c4` | [結果a689694][G7]。条件付き価格表・P3/P5差、SOURCEのcontract_v1.json、生成/budget本文 |
| G8 | `cb60a4a1336f1803ad49d890f4e74aafd6ad7c61` | [結果収録4301c0f][G8]。cold/warm trace、provider仮定、I2診断の限定。G8 pipelineのcache/zero処理位置も静的確認 |
| G9 v2 | `0ef2b92738750a3c0d187743dd4b7c9b927db802` | [結果78cb65b][G9]。保存resource CSV、STOP.json、v2 contract、ClosedとFull同理想ensemble、direct/adjoint本文 |
| G10 v3 | `b9ed01455351628c9073748f5ba5751aa794b789` | [結果収録4301c0f][G10]。保存degree CSV・STOP.json・contract_v3.json、ClosedP5Tail本文。旧RSS/JSON停止は保全 |
| D1 | `874ca12d347f8f35e5d8d64f65f24abc4a2df523` | 結果 `fc96571881f301435a655f057709df4d6c1678e0`。条件・後続対応は[D2統合][D2]と[D系列handoff][D123]へ戻った |
| D2 | `28fc3ca17ec51bb7656a498e3eea9b72966eb3fa` | 結果 `d1c59b28f4370cc7be65cff60d1ec877fd1491f5`。[固定統合報告][D2]で全方式境界相殺・hの限定を確認 |
| D3 v2 | `dceccaa774229527a46a1386131583a4f618f8ad` | 結果 `f0df9a3b36bc3540f0c83941871fa0d3110ddbf7`。[最終機構報告][D3]の保証/価格分離、小差、強い全方式比較を引用 |
| Parent/S4 | `4ccc1832f63cd323b23fc05623f7e0b58cf34f37` | S4結果 `fe0f3f077342bf7a557033e4dfe2734b7ead6091`。[研究区切り報告][PARENT]で70 law・旧proxyとの差・h/online留保を確認 |
| C1 R1入力 | `418a76987782ce50be670b95b8f998043b3f40f5` | [結果6b48bb3][C1R1]。新basis/324 exact-dyadic係数、旧係数との来歴差、model binding未閉鎖 |
| C1 Ordinary/Full | `900a5278c604b4c58b794ca891720d9dea293919` | [結果b022f2eの旧handoff][C15A]、Full summary原文と[凍結時の3方式引用][C1LIMIT]を確認。旧全job failureを保持 |
| C1 P5R | `2fbd83a44a0deb76c6444ed7920741c6b5e2e748` | [結果bb9059b][P5R]。summary原文、execution_contract.json、handoffを確認。native/N/D/readout UNKNOWNを保持 |
| C1凍結 | 新science SOURCEではなく方針記録 | [185ec4e][C1STOP]の凍結文書・resource limitations・research_state.jsonを確認。追加科学/native assuranceは不許可のまま |

G6 kernel、G7 generator、G9 P5/native、G10 generator、D会計の計6ファイルについて、専用checkoutの本文が表の原SOURCE（D会計は既存統合版 `243eea8e840f3c472753f7935e510391626fd325`）と同一であることをbyte比較した。これは今回読んだコードの版対応であり、旧runtime再現・全依存一致・科学PASSの再認証ではない。

## 3. 静的に確認した再利用接口

以下は利用可能な部品の説明であり、実行・新規実装の提案ではない。実読snapshotは上記専用基点と、表に示した元固定SOURCE。後日利用するときは、その時の選択taskに適合する部分だけを再利用する。

| 既存SOURCE／関数・test | 使える処理 | 適合性・不足 |
|---|---|---|
| [return_aggregation.py][KERNEL]：reduce_word、ReturnKernel.word_series/mass/parent、root_interval、dyadic_distribution、rational_event_weight | 局所return係数、parent-child、root enclosure、有限bit support/weight | 全word universeやglobal normalizerを生成しない。Hamiltonian/native価格oracleではない。原条件・符号・位相・語順を保持 |
| [g7_generator.py][GEN7]：CanonicalGenerator、FullReturnGenerator.sample/packet、budget | 非列挙生成、ordinary envelope、pre-quantum zero、M2/range十分N | raw非reduced proposalをreduced wordへ再割当しない。Fullの未知真Bを無料で予算に使わない。固定16軸の配分を全RPEへそのまま移さない |
| [g9_p5.py][P5]：P5Closed、completion、parent_coefficients、sample/reference_events | P5のroot/ordered-pair/length4群、群normalizer、completion DP | productionとreference列挙を分離。群setupはO(L²)算術、現selectorは線形scanを含む。bit/群取得費を省かず、角度群から全wordの相殺価格は決まらない |
| [g10_generator.py][GEN10]：ClosedP5Tail、generators | 同finite P_m内のP5 prefix＋ordinary tail対照 | Fullとの低次数baseline。異degree間で精度を揃えたという意味ではない |
| [g9_native.py][NATIVE]：CQ、crot、native_ir、adjoint、simplify、cost | 指定3-qubit provider、direct中央control、actual adjoint、literal inverse相殺、T/CX/1Q | 一般DF providerに偽装できない。global scalarのancilla相対位相を保持し、全方式へ同じ最適化を許す |
| [g8_pipeline.py][G8PIPE]：AcquisitionCache、NativePipeline | on-demand miss、memo/LRU、zero時にprovider/cache/readoutを呼ばない接続 | 有限traceの成功は全angle終端・全support価格保証ではない。旧cache capは歴史条件で、新taskの恒常上限ではない |
| [rte_development/accounting.py][ACCOUNT]：step_bounds、whole_budget、whole_cost | 非unitary有限meanの積誤差、独立stepのweight/moment積、全trial予算とaccepted費用 | whole_costは全path事前zero判定・未相殺加法会計、cross_step_fusion=False。D2の別境界処理を実施済みと取り違えない |
| [parent_allocation.py][PA]：RootAllocationGenerator、reweight_events、event_statistics、error_accounting/native_plan | 既存ordinary/P5との合成、実dyadic qに対するweight、moment/range、native接口 | explicit table型ISはcheap全path samplerではない。native/角度誤差は外部証拠が必要。native側の実装も既存で再実装不要 |
| [whole_path.py][WHOLE]：WholePath.assemble/circuit、AllTrials.add/evidence | ordered microstep積、全planned trial分母、presampled zeroとspent prefix、before/after hooks | 現SOURCEはSYNTHETIC_ONLY、confidence_certified=False。hooksがあることはD half実装/価格取得済みを意味しない |
| [主RPE accounting][RPE]：_hoeffding_shots、evaluate_rpe_round_candidate | 軸ごとの費用・失敗配分、半径から座標誤差へ接続する既存入口 | bounded ±1用。weighted/zero-fillへ無変更では使えない。reference_radius=1の条件付き仮定を元H groundから導かない |
| [G6 test][TEST6]：test_last_label_is_not_enough_state、test_reducing_raw_proposal_would_double_count、test_accept_only_average_changes_mean、test_digital_packet_support_weights_and_l1_budget | return/語順、proposal二重計上、zero分母、finite-bitの既存人工対照 | SOURCEを静読しただけで今回PASSとはしない |
| [G9 test][TEST9]：test_same_ideal_Green_coefficients、test_physical_CQ_direct_phase、test_phase_corruption_detected；[G10 test][TEST10]のdegree/mean/lower対照 | ideal ensemble同一、strict controlled phase、finite targetと固定辞書lowerの範囲 | 新分子・全support native・全RPEの科学性能を示すテストではない。今回実行なし |
| [S2 SOURCE][S2SRC]：pair_table、normalizer_enclosure、evaluate_law | 保存D2 binding/実sequence価格とParent会計の既存接続 | 同じS2 r/degree/time/precision/regularizer探索へ再利用しない。全pair表のsetupと未測定online費を残す |

### 必須の強いbaselineと会計単位

| 比較層 | 対照と公平性 |
|---|---|
| 同finite target | Ordinary PR、partial return、Closed P3+tail、Closed P5またはP5+tail、General Full、matched CTS。P3+tailはtarget精度を緩めない。一般CTSを全support展開実装だけに限定しない |
| 量子回路 | direct control、actual adjoint、basis/境界共有、合法なnative最適化、cacheをbaselineにも対称に認める。helperを弱い唯一の対照にしない |
| 統計 | 同補正mean/phase、bias控除、実qのM2/range、共通confidence規則。保存signal oracleや未取得normalizerを無料にしない |
| full PR＋RPE | 同じ元H/state、energy精度・成功確率。D halves、ancilla/X/Y readout、全round branch条件まで戻す。必要時の決定論symmetric/適合高次PF対照も排除しない |
| 古典側 | 入力・support/return・normalizer・角度/native価格取得、setup、selector、online sampling、cache missとmemoryを別座標で記録。理想lawだけを実装済みとしない |

zero-fillでは全planned trial Kを使う統計とaccepted quantum回数を分ける。完全な事前zeroなら当該trialの量子費用を0にできるが、既実行prefix費は残る。accepted条件付き単価を使うなら対応する受理数と組み合わせ、受理率を二重に掛けない。古典zeroの生成・判定費は全trial分を支払う。

## 4. 欠けた情報と、次段階を設計しない理由

現証拠には、強いClosed/CTSを越える具体的return構造を持ち、かつ安価な取得/生成が説明された追加入力を特定できなかった。C1のnative/共通N/D/readout不足、G8の未観測angles、D/Parentのonline費未知はそのまま保持する。それらがUNKNOWNであること自体を研究候補の支持としない。

全RPE接続の将来課題は、正当化された半径・phase/bias allowance、weighted trialの有限confidence、同情報baselineの全wrapper価格、古典取得/normalizer/online費の対応である。これは不足の所在であって、全項目を埋めるbacklogではない。具体候補がないため、今回は最小新実装差分・最小科学runの条件を固定せず、計画を発行しない。

**既存STOPとの関係**：G6〜G10、D1〜D3、Parent/S4、S2の原SOURCE・raw・契約・grant・ledger・marker・one-shot/retry・STOPと、C1凍結を変更しない。旧worker問題を解消する義務を新たに課さず、旧grant/markerを再利用しない。承認証明用の台帳・認証・新管理pipelineは作らない。次の担当はユーザー／GPTで継続方針を判断すること。Codexは公開・取得照合でmandatory STOPする。

[COMMON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/archive/server1_main_collection_20261103/b05c68a693d31e3a15652faccfc4e6f0a05f8c34/research_project_instructions_unified_2026-10-25.md.original
[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc42c3a9f77e960f415d0475da2ba03bb56252c7/docs/research/o2_s2_multi_event_validation_v2_20261106/REPORT.md
[O2B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/docs/research/o2_sampling_return_technical_survey_20261106/O2_B_RETURN_AGGREGATION.md
[INT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/REPORT.md
[LIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/REPORT.md
[G6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_results_and_gpt_handoff_20261010.md
[G7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md
[G8]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md
[G9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
[G10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d2_final_saved_integration_and_gpt_handoff_20261013.md
[D3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d3_final_saved_mechanisms_and_gpt_handoff_20261014.md
[D123]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md
[PARENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md
[C1R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6b48bb3f4f7b080baca15d6fa45711619880d64d/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/handoff.md
[C15A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b022f2ea421649015a4c21ea3c904fe619d0c198/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/handoff.md
[C1LIMIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md
[P5R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/handoff.md
[C1STOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_PAUSE_AND_RESTART_CONDITIONS.md
[KERNEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[GEN7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
[GEN10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g10_generator.py
[NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_native.py
[G8PIPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc42c3a9f77e960f415d0475da2ba03bb56252c7/src/trottertracks/algorithm_codesign/g8_pipeline.py
[ACCOUNT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/rte_development/accounting.py
[PA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc42c3a9f77e960f415d0475da2ba03bb56252c7/src/trottertracks/algorithm_codesign/parent_allocation.py
[WHOLE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/docs/research/track_b_c1_4_5_c1_5_technical_preflight_20261021/whole_path.py
[RPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/rpe_resource_accounting.py
[TEST6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc42c3a9f77e960f415d0475da2ba03bb56252c7/tests/tracks/algorithm_codesign/test_g6_return_aggregation.py
[TEST9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc42c3a9f77e960f415d0475da2ba03bb56252c7/tests/tracks/algorithm_codesign/test_g9_p5_and_native.py
[TEST10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc42c3a9f77e960f415d0475da2ba03bb56252c7/tests/tracks/algorithm_codesign/test_g10_degree_preparation.py
[S2SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c9f3fb32deef5c50fd98517ccbcd03311f0b059/src/trottertracks/algorithm_codesign/s2_multi_event.py
