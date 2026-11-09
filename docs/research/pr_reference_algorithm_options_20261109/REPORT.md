# 新A：総PR＋位相推定費用を下げるアルゴリズム候補

2026-11-09 JST／SERVER 2。**静的調査・設計の完了報告。新しい科学数値結果はない。**

主な対象は、高価な長roundの回路全体を繰り返す測定費と、RTEのbasis／deterministic Dである。追加CSWAPだけを減らす方針や、m17のClifford+T保証改善の延長を主線にはしない。比較可能な候補として、(Q) 前roundの区間で読出し方向を決める方式、(I) 最新のPR回路へReferenceを接続する方式を具体化した。Qは既知の適応位相推定の応用候補、Iは既存の組合せ提案の具体化であり、どちらも独立した新規性や資源改善を確認したものではない。

Qを最初の限定診断候補とする理由は、D・basisを含む高価なinterrogationの**反復数そのもの**へ作用し、回路価格取得なしに半径不確かさによる破綻を先に判別できるためである。未知の改善率による順位ではない。PRの未知半径を保守的に扱うと利益が消える可能性がある。Iは回路単価への別の経路として残す。

- [候補・記号的条件・重複](ALGORITHM_IMPROVEMENT_OPTIONS.md)
- [固定入力・最小診断・承認範囲・STOP](NEXT_VALIDATION_PLAN.md)

## 1. 固定成果と今回の境界

研究階層は「アルゴリズム改善 → 新A → 総費用削減の方式検討」。既存V6作業枝の継続も検討したが、固定V6-Q実装と異なる読出し／回路構成の仮説を独立に保全するため、`1cd1fd29949c2079cd74a6d25855f9f1e54ce1d2`を親に、新枝`codex/algo-improve-new-a-total-cost-20261109`を作成した。旧枝は前進させない。

適用AGENTS.md、PROJECT_MAP.md、正式共通指示2026-11-09、研究概要・目的、関連する固定報告・SOURCEを参照した。今回は本ディレクトリの3文書のみを追加する。新しい研究方針の正式採択、詳細科学レビュー、保証の正式採用は行わない。旧statusや索引を改訂する科学結果は生じていない。

新Aの数値を引用する条件は、Linear H4、1.00 Å、STO-3G、DF12、元H exact ground供給・準備費除外、Partial S2／LD3／h=.02／K2、保存18round q/r/bits、ordinary diag(I,V)、canonical independent finite-bit law、Compiler A／Qiskit1.3／rz,sx,x,cx。精度1.5936e-4 Ha、failure .05、位相cap .4 rad。m17はq131072/r32/bits88。任意の物理Hamiltonian包含・実乱数器・実機の保証ではない。

| 保存証拠 | 到達点と残す限界 |
|---|---|
| [V6-P 5f672ce6][V6P] | B1の条件下でReference m17が630→626 shots/axis。全18roundのB′比削減下界10,360,815,280.326 native RZ。約0.2901%は今回指示で与えられた既存表示で、再計算していない |
| [V6-Q producer 1cd1fd29][V6Q] | provider／caller／consumer／finite-bit producerのSOURCE接続と人工8テスト。実H4 kernel fixture、本番shot loop、任意RNGの数学的IID、総Tは未資格 |
| [D0 f2638a78][D0] | 旧Nの単価差・shot差・9 role・round寄与をexactに分解済み。今回再積算しない |

B1後の総差を、旧Reference630の不利と混ぜない。旧D0の追加shot項はV6-Pで解消した負担であり、これから再び節約できる費用ではない。旧CX／depthの条件付き成果を、今回新たに626へ再会計した結果とも呼ばない。

## 2. 費用構造：保存結果の引用とSOURCEの読解

資源jごとに、`G_j = Σ_(m,a) N_(m,a) μ_(m,a,j)`、`μ=E[C_full,no-prep]`。RZ、CX、DAG depth、size、peak qubits、将来のstrict Tは別座標である。価格MC標本数はRPE shotsではない。

D0が既に確認したことを引用する。旧B′総RZは[3,571,960,960,634.802, 3,571,960,960,634.803]、旧Referenceは[3,577,340,246,876.263, 3,577,340,249,201.294]。B′のNを基準とする単価差項は[−10,360,817,596.453, −10,360,815,280.326]、旧追加shot項は[15,740,103,837.914, 15,740,103,846.819]だった。D0では両armともm17>m16>…>m0、m17だけで前17round合計を上回ると検算済み。[D0 §2–4][D0]

| 部分 | SOURCE上の単価・出現／反復 | 既存最適化と今回の読み方 |
|---|---|---|
| Deterministic D | S2のD halves／内部sweepをqに沿って反復し、各shotで再実行 | D/D境界融合済み。D未融合価格とsigned境界補正を別gateと数えない。D0の両項合計の旧純差3,569,124,816 RZは引用のみ |
| RTE basis・inverse | q×r個のeventとproductの適用順にbasis runが現れる。run共有で減るため単純event加法ではない | 旧新Aでbasis共有済み。D0の単価差は0、旧追加shotによるbasis増額は約9,238,920,433.914 RZで、保存9 role中最大の正差。これは**総basis費用の比率**ではない |
| RTE中央rotation | 中央rotationをq×r回。B′の4-CX controlled-RZZとReferenceのuncontrolled rotationが異なる | D0の単価差項−7,463,818,296 RZ。既知control削減の効果を二重計上しない |
| Random event product／event phase | order2等でproduct・符号が出現。順序、support、相対phaseが必要 | product-Z下界／上界、event phase集約を保持。最安端点を実行可能policyと呼ばない |
| Controlled演算 | B′はbasisを無制御で共有し中央のみ効率制御。ReferenceはF/Gの干渉へ交換 | 旧14-CX対照への利益を強4-CX B′比と混ぜない。Track Cにはさらにterm/parity/network対照がある |
| Reference付加部分 | 8追加qubits、8 reset、16 CSWAP／shot、vacuum補正P | CSWAPはqに比例する主要項ではない。D0旧CSWAP純差1,514,656 RZを、全増額の主因としない。qubit9→17を保持 |
| X/Y・RPE | 各軸でfresh trajectoryと測定。旧wrapperはH、YのSdg、H、readout | 同pathでX/Y価格を校正できても測定noiseを共有できない。Qはここを変更する仮説 |

式の実装は`pr_reference_transfer_accounting.expected_rz_bounds`、`pr_reference_bprime_accounting.price_bounds/error_and_shots`、`pr_reference_rotation_v3.expected_parts/dparts`を再利用先とする。今回は呼び出していない。

**推論：** shotを減らせればD・RTE basis・中央・Reference付加部分がまとめて減る。一方、既に融合済みのDやbasisへ同じ局所処理を加えても追加利益にならない。V6-P後の絶対role順位・比率は今回取得していないので、「basisが総費用の何%」という新しい断定はしない。

## 3. 重複確認と11月9日の公開差分

入口は[横断整理654b8501][CROSS]とそのmatrix／overlap表。次の固定公開snapshotの必要箇所を読み、対応SOURCEを静的に照合した。全branch・非公開チャットの不存在証明ではない。

| 系列 | 確認した既存機構・負結果 | 今回の扱い |
|---|---|---|
| Track A／O1 | DF/Pauli辞書、partition、M1/M2、B2、移送。C0では長round半径低下が安い単価を相殺。own-DF groundと元H groundは別 | rank／split gridや表現再fitを再提案しない。既存強PFを落とさない |
| Track B／O2 | [統合d28164b5][O2]、[G6–G10所在a8299889][B]。Closed P3/P5、ReturnKernel、Parent、IS、whole-path価格、SCV、A2 LCU | 同じp/√C、return、層別CVを新機構にしない。G9 P5をK2=P3へ移さない。C1凍結・各STOP維持 |
| O2後続 | [非同値性1bf88b25][DISTINCT]。normalizer二段取得は既知法へ還元、finite-state summaryは未確定、座標observableは既知法応用 | 安いoracleや未知state情報を足して再提案しない。Qはpath law/CVではなく測定方向変更。ただし適応測定自体は既知 |
| Track C | [joint isometry2745d57b][JOINT]、[diagonal fusion94b6499e][FUSION]、[Closed／共同選択c5a3c102][COPT] | basis列共同保持、term-control、parity合算、D融合、h/rate/D/αと整数shot交換は実施済み。Iはこれらを固定した後のReference**増分**だけ |
| Full／Truncated PF | [truncation da8b2c3f][TRUNC]、[相殺分離92495a04][CANCEL] | 同じ元H ground／精度の強対照。H4ではPRより安い保存結果。新Aの約0.29%をこの差の解消とは呼ばない |
| Track D／O5 | [cut／selector設計21319176][DDESIGN]、[head-tail C1 62c7b578][DC1] | reference gadget、cut変更、境界融合は他Trackの既存機構。新Aへ移管・再実装しない |
| 新A CT／D1,D2／43-word | strict phase、actual adjoint、有限T比較、非中央guardの負結果・T増tradeoff | 局所証明の改善を今回の総費用改善候補に戻さない。全18roundTは未資格 |
| RPE等 | O4/UWC、C2-A/B、既存α/β/r/K最適化。[O2推定器調査1bcf70ba][EST]ではadaptive confidence単独は既知共通部品 | Qは単なるα配分・peeking停止ではない。固定回数の読出し方向と半径nuisanceを扱う。新規性は主張しない |

GitHubのbranch refを今回照会し、Track C統合枝のHEADは`b3c81c2677988c7b9491eeee31fbcc0f9b29b078`、Track D統合枝は`141fa7624d74aa5deacbd752d09df516cad9206f`と確認した。両tipは枝整理の記録であり新しい科学改善ではない。

その後のDについては、[q64完了fe361b20][D64]で保存8 path／32 wrapperのcorrectness・価格が完了している。旧横断表の「残7未完」を最新状態として繰り返さない。ただし有限8例で、全law・13round保証ではない。[T0 c583abba][DT0]はnative streaming／max-plusの人工29件で、P0全価格は未資格のまま。C1利益の実証や新Aの18round価格へ読み替えない。

O2統合／非同値性／M metadataのremote HEADはそれぞれd28164b5／1bf88b25／e98589aeで、取得した既存固定点と一致した。Mのrawを再解析していない。別担当の未公開作業、今回未取得の後続結果はUNKNOWNであり未実施とは断定しない。

## 4. 残す二案と除外する再提案

| 案 | 実際に異なる問い | 第一の障害・判断材料 |
|---|---|---|
| Q：区間に基づく接線方向のreadout | 同じPR／Referenceのmeanを、長roundでXとYの両方を推定する代わりに、前段のenergy区間に対して感度のある一方向から読む。未知半径も含む区間反転で同じ精度を満たせるか | 半径不確かさによる識別不能幅、PF等の位相bias、feedback角誤差。これらを最悪側に置いた有限診断でshot節約が残るか。標準適応PEの無条件移植は不可 |
| I：最新PR回路×Referenceの増分 | Closed P3／joint basis／network後にも、control解除による単価節約が追加CSWAP・cut損失・shotを払って残るか | weighted mean／vacuum／実inverse／compiler contextの接続。既存組合せ案であり、新しいSE原理ではない |

Qの既知性は[Higginsらの適応PE][HIGGINS]、[DobšíčekらのIPEA][IPEA]の該当回路を限定確認した。これらは既知unitary eigenphaseを主に扱う。PRの減衰したcoherent meanへの有限成功保証を与える資料として引用していない。round間consistency自体も既知であり、[Russoら][CONSISTENCY]の検査をshot削減保証に読み替えない。Qの式・STOP条件は今回の**提案**として分けた。

強いFull／Truncatedにも同じreadout改善の機会を与える。QがPR内で安くなっても、Reference固有利益やPR対PF勝利とは限らない。Iも、既存C改善率とReference改善率を掛け合わせて性能予測しない。

強対照の重さを具体的に残す。[Track Cの保存H4比較][TRUNC]は、同じ元H exact ground／精度1.5936e-4 Ha／failure .05／18round／準備費除外で、最適化Closed PRの総RZ1.303898609e12、強Full 5.647934021e11、Truncated 3.337772238e11と報告している。PF／h・rate・compilerの価格scopeは同報告のpolicyで、新Aのh=.02固定B′会計と同じ回路ではない。これを新Aとの新しい数値比較や削減目標へ換算せず、同条件での最終比較が必要な理由として引用する。

それ以外の「basisをさらに共有」「D融合」「round別PF」「samplingを価格で変える」「精度を上げる」「Reference cutを動かす」は、追加の実装可能な性質を今回特定できなかったため独立候補にしない。Q/Iも優越や世界的新規性を認定できる状態ではない。

## 5. 次の判断・実施記録・STOP

推奨する最小の次作業はQの**半径・位相不確かさ込みの有限識別診断**。H4回路や新価格を取得する前に、不可能な誤差床と、既存強統計対照との差を調べる。具体的な固定入力・比較・必要承認は次計画§2に記した。Iは独立の条件付き案として、対応する既存担当との重複を確認してから選べる。

新しい科学計算、保存値の新しい数値再解析・role積算／順位付け、sampling、回路build/compile/simulation、strict guard／合成、shot/RPE再計算、RZ/CX/depth/T価格取得、重要科学レビューは**全て未実施**。研究moduleをimportせず、旧testも再実行していない。静的な式・SOURCE・保存報告の比較で足りるため、新runnerや人工testを重複実装しなかった。今回の検証は文書・Git参照・公開内容の技術照合のみで、correctness PASSや科学結果に数えない。

V6-P/Q、D0、CT、D1/D2、C2、旧C-N/C-S/C-P、P-D、BF/BM、PR-3、他TrackのSOURCE／raw／契約／grant／ledger／marker／STOPを変更していない。共有設定・他process・他worktreeへ変更なし。CPU/RAM/workerの新しい上限や管理基盤も設けない。

**この3文書を公開しremote SHA・内容を照合後mandatory STOP。** 次担当は**GPT／ユーザー**：Qの限定診断に進むか、Iの接続検証を選ぶか、あるいは新しい独立機構が得られるまで保留するかを判断する。Qの保証採用を要する重要レビューは開始していない。Codexは選択・明示承認された次scope以外を実行しない。

[V6P]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f672ce6884bd49231366875a9e24a539fcaec7a/docs/research/pr_reference_v6p_results_20261109/REPORT.md
[V6Q]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1cd1fd29949c2079cd74a6d25855f9f1e54ce1d2/docs/research/pr_reference_v6q_producer_20261109/REPORT.md
[D0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/docs/research/pr_reference_cost_structure_d0_20261108/REPORT.md
[CROSS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_rpe_cost_refocus_20261108/REPORT.md
[O2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d28164b5d51e0ca5a0ade74ded8ff176b443dab8/docs/research/o2_research_results_integration_20261108/O2_RESEARCH_SUMMARY.md
[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a8299889de93c201bf21e727415d629f5e4b9f48/docs/research/track_b_existing_results_survey_20261108/REPORT.md
[DISTINCT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1bf88b251fd09caa0a227972ee0f1b56c5888e6a/docs/research/o2_distinct_mechanism_screening_20261108/REPORT.md
[JOINT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2745d57bea5c28604739bf9c01338e14e649c467/docs/research/track_c_joint_isometry_20261108/REPORT.md
[FUSION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/docs/research/track_c_diagonal_fusion_20261108/REPORT.md
[COPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md
[TRUNC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/docs/research/track_c_df_truncation_20261108/REPORT.md
[CANCEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/92495a04d52c65142ed16b3abe977345065f9745/docs/research/track_c_df_cancellation_validation_20261108/REPORT.md
[DDESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2131917656446bd22f294946d7c1064a47332e4c/docs/research/track_d_o5_algorithm_design_20261108/REPORT.md
[DC1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62c7b578b8c0bb3d0079434daf637c587f53c24c/docs/research/track_d_o5_c1_head_tail_static_20261108/REPORT.md
[EST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1bcf70ba69f91c62763a8739ec620b040b097ed9/docs/research/o2_estimator_mechanisms_survey_20261106/NEW_RESEARCH_DIRECTIONS.md
[D64]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe361b206a6572142775ac01972545d312e5fb14/docs/research/track_d_c8_1_q64_20261108/COMPLETION_RESULT.md
[DT0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c583abba520a711895d21b7ec9590688349a7bd1/docs/research/track_d_t0_native_resources_20261109/REPORT.md
[HIGGINS]: https://arxiv.org/html/0709.2996v2
[IPEA]: https://arxiv.org/html/quant-ph/0610214v3
[CONSISTENCY]: https://arxiv.org/abs/2011.13442
