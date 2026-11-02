# H8：現O1では追加科学計算を要求しない

2026-11-03 JST。必要性判断の候補。新しい実験契約・実行許可ではなく、q/T/ε/標本数/seed/実行回数は未固定。

## 現在の推奨

固定入力・候補・価格cohortに条件付けたO1には、新しいH8科学計算は不要。
T=.8での固定n48主表、5精度の強化184、追加6精度の境界、同cohortの保存説明因子と正式終了済みの希少感度を再利用できる。
H8の長時間・高次PRを検証済みとは書かず、H4/H6の結果と節・表を分ける。[MAIN_REPORT] [STRONG] [EPS] [CLOSURE]

新計算の前に残る具体的な候補はG04の既存metadata回収である。調査時点で新しい公開補完は未取得。
実行SOURCE、入力binding、固定候補/cohort/index、完了・回収とaliasの対応が公開されれば、公開可能な範囲だけを固定URLで追補できる。
private実体・資格情報・原seed値の公開や全履歴監査は要求しない。公開できない部分は限定を維持する。
これは追加科学実験の理由ではない。[OPEN] [PHASE3_GAPS]

## 主張を拡張する場合だけの最小候補

**情報価値のある問い**：同じ保存H8 DF28入力で物理時間を長くすると、高次PRが使える大きな刻み幅による利得は、
signed piece・B・必要shot・whole compiled価格の負担を補い、強いFull/B0/B0I対照に対する適用条件を変えるか。
H4/H6で高次化の価値が時間・精度に依存したことを、別の保存H8入力へ拡張して判別する役割である。
既存T=.8の高次決定論対照やPR主表だけでは、この時間移送と高次PRの競合には答えられない。[LONG] [HIGH_CONFIRM] [STRONG]

候補は、一つの判別可能な時間・精度のstratumに絞ったS2対高次PRと、同入力・同精度会計の強い決定論対照。
どの時間・精度・高次公式・prefix/q/R/Kを使うかは、既存coverageとの重複確認と研究判断後に選ぶ。
これは実行セル数や公式数を定めた正式計画ではない。
H4/H6と同じ40セル、同じn、全prefix、K6、Yoshida4/m5全組合せを無条件に要求しない。
既存T=.8の5精度・追加6精度・p7/q16固定確認・S4対照・DBA実測・S2_SINGLEを再計算対象へ加えない。

| 判断項目 | 最小候補に対する答え |
|---|---|
| 1. 中心主張のどこが強まるか | 強いPFに対する時間・次数依存の適用条件を、もう一つの保存入力で検査できる。単なる検証件数増加ではない |
| 2. 既存結果では説明できないか | 現O1の条件依存性は説明できる。H8長時間・高次PRという追加の主張だけは既存結果で代用できない |
| 3. 新条件が何を判別するか | q減少対signed piece/B/N/Cの競合。特定次数がH8でも勝つという結論を先に置かない |
| 4. 強い決定論対照に影響するか | FullだけでなくB0/B0I・高次PFを含む同条件対照が必要。弱い旧対照だけでは適用条件の新知見になりにくい |
| 5. signal/shotだけで十分か | 不適格・極端なnormalization負担の判別には使える。資源順位を主張するなら、残った比較の両側にfull-wrapper直接compiled価格が必要。T=.8 modelやH4/H6単価の移送で済ませない |
| 6. samplingで不確かさが減るか | 新候補の有限価格mean/SEを見積もるためのsamplingには役割がある。ただし元H8 rareの未知tailは解消せず、同数nの複製やS2_SINGLE増量を目的にしない。近差なら取得前の精度設計と別cohortの固定確認が必要 |
| 7. 元実装を再利用できるか | 下記の数値・native・shot基盤は再利用候補。H4/H6専用loaderと候補表はH8へそのまま使用できない |
| 8. STOP変更が必要か | 元H8 rare・主表・H4/H6契約は変更しない。新しいH8時間/次数scopeとして別許可を受ける。既存contractの変更が必要なら変更点を明示して許可を得る。旧grant/one-shotの再使用や別namespaceによるSTOP回避はしない |
| 9. 何が未保証で残るか | 真のbias/全law compiled作用、未知tail・母平均winner、選抜保証、状態準備、noise/routing/FT、一般分子・size scaling、最終energy/QPE/RPE費用 |

signalから始める場合も、これは科学計算なので新しい明示許可が先に必要。
採択する固定範囲をsignal/shot・必要価格・固定確認までまとめて許可すれば、その範囲内の技術実装・人工検査・debugごとに再承認を求める必要はない。
本提案はその許可を含まない。

## 再利用の適合性と必要な差分

| 既存資産 | 再利用できる部分 | H8で先に確認・変更すべき部分 |
|---|---|---|
| h46_long_time_signal.py / h46_new2_core | 保存入力identity、occupation/spectral作用、finite paired polynomial、時間別batch、reference照合 | PREFIX/RANKはH4/H6固定、私有origin configもH4/H6。H8保存binding/referenceへ接続が必要。sector4900でdense行列・全固有分解が適切かは未判定 |
| h46_m5_tol_native.py のrestore | 保存H/state/DFを再生成せず復元しidentityを検査する考え方 | H4分岐とH6 loader、8/12qubit条件。H8は16 modesなので、このまま呼べない。既存H8 loader/preparationを優先 |
| h46_long_time_native.py / h46_m5_plan | signed schedule、ABS_TIME_PER_OUTER、piece整数配分、native DF、actual adjoint/scalar一回、paired wrapper、task-bound seeds、compiler/fingerprint | H8 preparation・law・negative time・signed piece/normalizationの対応。旧SOURCEを編集せず新scopeのadapterへ固定 |
| 既存shot/会計・confirmation設計 | N_cos*C_cos+N_sin*C_sin、paired SE、適格性と価格の分離、nomineeを取得前に固定、no pooling/top-up | H8の元u/marginを緩めず、Rとrの意味、Kと多項式最高次数を区別。既存の確認後minを事前nomineeに改称しない |
| 既存signal/nativeテスト | 人工fixture、schedule・scalar/control・paired会計を検査する枠組み | static確認のみ。今回テスト本体は実行せず、新条件が承認された場合に必要な人工fixtureだけを使う |

静的確認の出典：[SIGNAL_CODE] [NATIVE_CODE] [LOADER_CODE] [SIGNAL_TEST] [NATIVE_TEST] [LONG_PROV]
新しい科学pipelineは今回実装していない。
将来実行する場合は、その時点のCPU/GPU/RAM・既存timingに基づいて、matrix-free/sector/batch/cacheや独立候補の並列化を選ぶ。
H4/H6のworker数や過去H8の固定capを新scopeへ一律転用しない。BLAS等との過剰並列と共有環境干渉を避け、最適化経路を必要な参照・人工検査で確認する。

## 担当・必要な別承認

- **ユーザー**：現O1に限定するか、H8の時間・次数拡張を論文に求めるかを決める。現O1なら追加科学作業なし。
- **SERVER2担当Codex（次の指示がある場合）**：既存G04補完の公開可能metadataを提示。元STOP/原本は変更しない。
- **LOCAL Codex（次の指示がある場合）**：公開補完を照合して監査欄を追補。科学計算を開始しない。
- **科学実行を担当するCodex（拡張採択後、別の明示許可がある場合）**：重複のない固定scope・input/SOURCE・比較・精度・資源構成を準備し、その範囲の新科学計算を行う。保存値の新解析や重要な別レビューも各対象の許可を必要とする。

**今回は文書公開・照合後mandatory STOP。残余sampling・compile・新しい数値解析は開始しない。**

[CLOSURE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md
[EPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_h8_epsilon_boundary_run01_results_20261018.md
[HIGH_CONFIRM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/docs/research/track_a_h246_reconstructed_confirmation_results_20261022.md
[LOADER_CODE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/scripts/h46_m5_tol_native.py
[LONG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[LONG_PROV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/scientific_provenance.json
[MAIN_REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_h468_paper_evidence_integration_20261019.md
[NATIVE_CODE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/scripts/h46_long_time_native.py
[NATIVE_TEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/tests/test_h46_long_time_native.py
[OPEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ce2b5f161e94567fec322d694fd8b01d03cd854/docs/research/pr_research_paper_evidence_synthesis_20261103/STATUS_AND_OPEN_QUESTIONS.md
[PHASE3_GAPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/235e56ee7d5cac4682775a62bc2a8262f15047dd/docs/research/pr_cross_host_coverage_audit_20261102/coverage/GAP_CANDIDATES.csv
[SIGNAL_CODE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/scripts/h46_long_time_signal.py
[SIGNAL_TEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/tests/test_h46_long_time_signal.py
[STRONG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md
