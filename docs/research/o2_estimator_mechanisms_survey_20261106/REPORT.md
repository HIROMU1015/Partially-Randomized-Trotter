# PR＋RPE：ランダム化推定器の改善機構調査

2026-11-06 JST。branch：`codex/o2-estimator-mechanisms-survey-20261106`。

**Aの条件付き最小検証候補として、CV4Quantum型の既知factorized path controlを実Hadamard測定へ適用する案を1件残す。** これは既知分散削減法のPR応用候補で、新原理・新規性・科学的GOの認定ではない。総費用改善も、PRでの有用なcovarianceも未確認である。

同じ平均を持つ非加重IID±1測定では1-shot varianceは変わらない。A1は測定値へ既知平均の古典controlを加減して非binary estimatorに変え、path間の変動を減らす余地を調べる。量子測定内のnoiseは残る。安いν取得、独立な係数学習/variance校正、強い層別対照、全pilot＋full interrogation費用が検証の条件である。

## 1. 成果物

| 文書 | 内容 |
|---|---|
| [VARIANCE_REDUCTION_LITERATURE.md](VARIANCE_REDUCTION_LITERATURE.md) | 新たに確認したCV4Quantum、MLMC-qDRIFT、QA-MLMCと、既存統計文献の版・関連節・保証・費用境界 |
| [PR_RPE_ESTIMATOR_COMPATIBILITY.md](PR_RPE_ESTIMATOR_COMPATIBILITY.md) | 実測noise、control平均/fit/range、finite confidence、可変費用、全RPE条件、既存SOURCEの再利用 |
| [NEW_RESEARCH_DIRECTIONS.md](NEW_RESEARCH_DIRECTIONS.md) | A1の具体機構・strong baseline・反証・最小検証案、A2の保留条件、O1/O3/O4との対応 |
| 本REPORT | 結論、確認範囲、未実施・保全・公開・STOP |

## 2. 確認環境・固定資料・SOURCE

現セッションのLinuxサーバーで完結した。SERVER 1/2のどちらかをhost名・directory名・IDEのTSV名から推定していない。開始時のrepositoryは `partially-randomized-trotter/`、main `0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf`、originは `https://github.com/HIROMU1015/Partially-Randomized-Trotter`。staged deletionのある既存mainに操作せず、固定R1結果 `b6305ee2e33436ed96d8aefdee15588676242000` から新しい専用branch/worktreeを作成した。

適用AGENTS、PROJECT_MAP、正式共通指示2026-10-25の§8.1/8.2/8.4/10.1/10.3を確認した。研究概要・現状は前回全文実読した固定版と今回基点で変更がないことを確認し、前提・優先順位を引き継いだ。新しい研究採択・科学結果statusは変更しないため、既存概要/manifest/索引は編集しない。

指定固定資料の[S2 dc42c3a][S2]、[R1 b6305ee][R1]、[O2調査2b71117][O2]、[文献e8b19ff][LIT]、[3環境統合3571477][INT]を再利用した。前回読んだ固定本文・SOURCE対応を保持し、今回必要な推定器/相関/統計/費用の関連箇所へ戻った。Git objectとして利用できることと、全hostのprivate raw原本の現存確認を区別する。全研究の再監査はしていない。

今回静的に確認した中心SOURCEは、`rte.py`のfinite order/label lawとevent metadata、Hadamard interrogation builderのX/Y/phase/measurement、`rpe_resource_accounting.py`の半径・axis N、Parentの実q/w/moment/errorとledger、S2 pair/normalizer/evaluate接口、既存相関監査、V1/V2のbranch/energy/状態条件である。G6〜G10/C1/Parentの原SOURCE・保存結果・STOPの対応は前回R1の固定表を再利用した。関連testはSOURCEの静読だけで、実行していない。

一次文献はCV4Quantum v1、MLMC-qDRIFT v2（v1とも区別）、QA-MLMC v2の関連本文を新たに確認し、既存O2-Aが挙げたMaurer–Pontil v1とHowardらv9を該当定理まで補足確認した。題名・著者・版・節/式と、今回取得/既存継承の区分は文献文書にある。PRHS/qSHIFT、UWC、SPRINT、TE-PAI/TETRISの既知対応は固定調査を継承し、全文の新規重要レビュー・全訂正の再調査とはしない。外部論文全文をrepositoryへ転載していない。

## 3. O2の負結果から引き継ぐ制約

S2は3-qubit synthetic D2 Closed P5、degree5、r2、τ5/7、x5/14という固定finite pair-tableのT主指標で、期待path価格低下を十分N増加が上回った負結果である。R1は強いClosed P3/P5・CTSを認めた後に具体的な追加RA候補を支持できず、`NO_SUPPORTED_R1_CANDIDATE`だった。いずれも一般no-goにはしない。

A1は同じlaw・回路を固定して実測estimatorを変える既知機構で、ISの再命名・RAの別入力探索ではない。ただし同じ負担原則を受け継ぎ、varianceだけでなくrange、bias、有限confidenceと全費用を戻す。無料control期待値/最適係数/真signalを仮定しない。S2のr/degree/time/precision/regularizer探索、C1凍結解除、Parentの旧toy再開は提案・実行していない。

## 4. A/B/Cの選別

| 方向 | 結果 |
|---|---|
| A1 control variate | 既知lawから中心化可能なlocal labelのfactorized controlを、実測outcomeへ付ける適用検証案を残す。order-only/constant-weightは単独候補にしない。within-order covarianceとstrong strata後の費用差を未知のまま検証対象とする |
| A2 multilevel | MLMC-qDRIFTは測定noiseまで扱うscaled augmented構成を持つ。ただしPR target対応、非unitary map、S補正、成功率/native費用が未適格。数値pilotはまだ設計しない |
| 相関/量子reference等 | path共有だけでは測定noise床が残る。未知reference平均やjoint measurementを無料にする案、一般相関・停止の名称だけの案は支持せず |
| B RPE協調 | UWC・O4・既存18round/energy接続の既知層。Aのconfidenceを全axis/branch/energyへ戻す用途に限り、新optimizer/scheduleを作らない |
| C 近似誤差 | O3 C-N/PF、C-S/residual合成、C-P/processing、既存PRHS/qSHIFT/TE-PAI/SPRINTとの対応。独立した重複実験を始めない |

最も強い直接対照は、同じ情報から実行できるcanonical PR＋variance-sensitive confidence、order/label層別とcost-aware配分、適合する既知IS/CVである。追加の量子回路を必要としないことはA1の実装上の利点だが、pilot/校正のfull shotsと古典moment/fit費は有料。層別と同じ効果に還元される場合は追加利益に数えない。

## 5. 次の最小検証と必要な別許可

最初はユーザーが選ぶ既存canonical PR入力・保存schedule内の一roundで、同mean/phase/精度・両axis・full価格を固定する。学習pilot→係数凍結→独立variance校正→共通有限confidenceの十分N/全費用という既知手順で、raw/強い層別とCVを比較する計画である。新しい科学条件・数値を本調査で採択していない。元H/state対応が未確定なら、その限定を解消するかmodel診断へ限定するかを先に判断する。

弱いcovariance、量子noise支配、range/pilot負担、取得できないν、層別対照への還元なら終了/縮小する。1roundで有望でも全18roundへ自動進行しない。A2は別のアクセス/測定仕様が先に必要であり、A1と並行する数値runにはしない。

**ユーザー／GPT**がA1を採用する必要性と対象・比較・有限scopeを決める。**Codex**はその後の明示指示の範囲だけ実装し、新科学計算・保存値再解析・sampling・測定simulation・N再会計・compile等は別途実行許可を要する。新しい重要科学レビューの開始許可は科学実行許可と独立である。今回は承認証明・新grant/ledger・新管理pipelineを作らない。

## 6. 文書検査・公開・保全・mandatory STOP

公開変更はこのdirectoryの4 Markdownのみ。既存Phase 2Aの文書検査helperの `git` / `blob_oid` を再利用し、UTF-8/改行、相互link/参照、21固定Git参照、非公開情報と差分範囲の技術検査がPASSした。科学test/runnerや保存値数値auditorは起動していない。専用branchにcommitして公開後、remote SHAと4文書を再取得・本文照合する。最終固定commit・公開照合実績は最終チャットに記す。自己SHAを埋める追加commitは不要とする。

今回未実施：新科学計算・保存結果の数値再解析/集計/ランキング/予測、H/state/reference生成、RNG sampling/replay、回路build/compile/native synthesis、推定器/新sampler実装、科学simulation/test、新しい非自明定理の証明、重要科学レビュー、科学的GO/新規性/方針の正式採択、C1等の凍結研究再開。AI sub-agentも使用していない。

main・旧研究branch・SOURCE/input/raw/result/grant/ledger/marker/STOP・他sessionのworktree/dirty/untrackedを変更しない。共有OS/scheduler/cgroup/quota、共有venv/package/cache/service、他ユーザーのprocess/job/filesには操作しない。科学workerを起動せず、既存計算の停止・再起動もしていない。過去の固定契約・one-shot/retry/資源条件を維持し、恣意的な低資源上限を新たに設定していない。

**限定調査・候補整理・4文書公開と取得照合の完了でmandatory STOP。次の科学検証・実装・他Trackの実験へ自動移行しない。**

[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc42c3a9f77e960f415d0475da2ba03bb56252c7/docs/research/o2_s2_multi_event_validation_v2_20261106/REPORT.md
[R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6305ee2e33436ed96d8aefdee15588676242000/docs/research/o2_r1_server_candidate_screening_20261106/REPORT.md
[O2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/docs/research/o2_sampling_return_technical_survey_20261106
[LIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106
[INT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105
