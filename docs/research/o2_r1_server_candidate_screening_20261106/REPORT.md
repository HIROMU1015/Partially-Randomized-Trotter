# O2-R1サーバー調査：既存証拠による候補選定

2026-11-06 JST。branch：`codex/o2-r1-server-evidence-screening-20261106`。

**結論：`NO_SUPPORTED_R1_CANDIDATE`。今回の既存証拠から支持できる第一候補はない。**

G6〜G10/C1/D/Parentの既存構成・限定成果を保持した上で、強いClosed/CTSを認めた後に残る具体的な「入力＋追加改善機構」を特定できなかった。RA一般の不可能性、研究の正式終了、科学的GO、新規性・最適性・論文化可能性の最終判定ではない。候補であるために全RPE改善の実証を要求したのではなく、次の最小検証で何を区別できるかの根拠が不足していた。

## 1. 成果物と今回の境界

| 文書 | 内容 |
|---|---|
| [O2_R1_CANDIDATE_SCREENING.md](O2_R1_CANDIDATE_SCREENING.md) | returnの具体構造、G6〜G10/C1/D/Parentの候補分類、既存数値の引用、強baseline・負材料・候補なしの理由 |
| [O2_R1_BASELINES_AND_REUSE.md](O2_R1_BASELINES_AND_REUSE.md) | 確認した環境、固定SOURCE/結果/STOP、GitHub取得範囲、関数/test/pipelineの再利用と不足 |
| 本REPORT | 判断、次段階の扱い、保全・公開・mandatory STOP |

対象はO2技術調査の「O2-R1：構造化Return Aggregationをfull-wrapperへ接続」である。旧Track B/C1のR1 stageと混同しない。許可された静的調査・文書作成・commit/push・公開再取得照合だけを行う。

共通の最終目的は元の物理Hの厳密基底状態を利用し、状態準備を除いた `Σ_(m,b) N_(m,b) E[C_full,no-prep^(a)]` を資源aごとに減らすこと。真のE₀/gap/完全state vector/未知path信号を無料情報にせず、H_D/近似H groundへ置換しない。PF/有限RTE/finite-bit/native bias、normalization/attenuation、測定統計、全RPE branch/energy成功条件を維持する。新しい分子・r/degree/precision・compiler・成功確率・主指標は固定していない。

## 2. 使用環境と実際の確認範囲

現セッションのLinuxサーバー。過去資料のSERVER 1/2との物理host同一性は未確認。開始時の `partially-randomized-trotter/` はmain `0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf`、staged deletionのある状態だった。既存worktreeを修復・整理せず、S2固定結果 `dc42c3a9f77e960f415d0475da2ba03bb56252c7` から専用branch/worktreeを分離した。remoteは既存originのままで変更していない。

適用AGENTSとPROJECT_MAP、研究概要・現状全文、研究目的とVALIDATION_STATUSの関連箇所、正式共通指示2026-10-25の指定節を確認した。正式共通指示はサーバーの既存文書を読み、固定公開コピーと同一本文であることも確認した。承認は今回のユーザー指示による許可として扱い、承認ID・証明packet・新grant/ledgerを作っていない。

指定の[S2 dc42c3a][S2]、[O2 2b71117][O2]、[3環境統合3571477][INT]、[文献調査e8b19ff][LIT]はサーバーGitから利用でき、各入口1文書はGitHub接続からも実取得した。その他の旧報告・SOURCE・CSV/summaryは固定Git objectから必要箇所を読んだ。全private rawの現存、別hostのruntime、全科学値の整合、全先行文献の再取得を確認したとはしない。

G6 kernel、G7 generator、G9 P5/native、G10 generator、D会計の6本文は原固定版とのbyte一致を確認した。G7/G9/G10/P5Rの固定契約、G9/G10のSTOP、C1凍結のresearch_stateも静的に確認した。これはコード版・保全条件の技術確認で、科学実行・再現・独立レビューではない。

## 3. 主要所見と強いbaseline

1. **G6の構成は既存。** Q_i²=Iによる符号付きreturn集約、even-suffix/parent-child、reduced word、child rotation、zero-fill、finite-bit support/weightがある。Green再帰/Euler/zero-fillの既知性は固定照合を継承し、新アルゴリズムとして再提案しない。
2. **G7/G8のP5利益をG9 Closed P5抜きで採点できない。** P3ではClosedが強く、G8の同情報IS診断ではFull/Closed差が消えた記録もある。on-demand/cacheは有限traceで、全support native保証・分子性能ではない。
3. **G9 Closed P5とGeneral Fullは同じ理想full-return ensemble。** Closed P5の登録T成果は有効だが、Fullの別量子改善機構と二重計上しない。CTSのCX優位を維持する。
4. **G10のFullには次数内T負結果がある。** m3はClosed P3、m7はClosed P5+tailが強い。異次数を同精度とせず、未取得の別degree/precisionで候補を救済しない。
5. **D2境界相殺は既存成果。** D1〜D3には同finite target内のweight・N・nativeと境界価格の比較がある。D3では保証/離散合成価格によりP3/P5の順序が変わり、普遍的Full優位を支持しない。D halvesとの未接続だけを新機構とは数えない。
6. **Parent/S4の既存条件付き利益は保持。** 既知ISとchild配分の組合せであり、未知h/online/多step・分子全RPEは未閉鎖。同developmentの探索区切りを維持し、jointという名称で再開しない。
7. **C1固定H4は一時凍結を維持。** 3方式各1024保存logical pathsでzero回避0、child rotations/path4.0、logical差小。strict native・認証N・D/readout/full wrapper不足を0やFAILに変えない。旧worker原因UNKNOWNの全面解消は不要。

各記述の原報告・保存CSV/summary・SOURCEへの固定リンクと科学条件は[候補選別§2–5](O2_R1_CANDIDATE_SCREENING.md)にある。C1はlinear H4 1.00 Å/STO-3G/legacy DF12/R1 generation-prefix L_D=3、T4/5/q1/r4/m5のtail-only探索。G9/G10とD/Parentは既知3-qubit synthetic providerで、分子条件は非該当。異scopeの数値を統合して新ランキングを作っていない。

## 4. S2不改善との関係

S2 v2はD2 Closed P5、保存row10、3-qubit synthetic、state000、r2、τ5/7、x5/14、degree5、finite target `[P5(-i(5/14)R)]²`。probability160/root256 bit、primitive10⁻⁶、axis誤差1/200・α49/24000等の固定条件である。分子の元H ground、D/PF halves、全RPE energy保証を持たない。

元報告のT主指標ではqB/BはqA/Bより約0.00134476%低く、元law P0/Bより約0.01296807%高い。期待path単価低下を十分N増加が上回った。これは保存報告の引用であり今回の再計算ではない。RA表現を変えない同pair-table ISの限定負結果なので、RA一般no-goにはしない。[S2]

今回の候補なしは、これとは別にG9同ensemble、G10次数内負結果、C1 sparse parent/child負担、既存D/Parentとの重複と取得条件から判断した。S2のr/degree/time/precision/regularizer等の追加探索を提案・実行していない。

## 5. 第一候補と次の最小科学検証案

**第一候補なし。支持候補がないため、ユーザー指示§10に従い最小科学検証案は作成しない。** 未知価格や統計が残るというだけで追加科学runを合理化しない。新しい入力・条件・samplerを作って候補を補わない。

再検討には、既存Closed/CTSに含まれない具体的return/child/basis構造、入手可能なsupport/normalizer/価格と生成法、同finite target・phase・N・full-wrapperの対照を説明できる材料が必要である。既存部品の再利用入口は整理済みだが、新規実装差分・実験条件を採択したわけではない。

次の担当は**ユーザー／GPT**で、本選別を踏まえてO2-R1の継続方針を判断する。別候補を提案する場合は、どの追加機構と既存入力が負材料を越える可能性を持つかを示す。重要な新科学レビューの開始許可と、新実装・科学計算・数値再解析・compile等の実行許可は独立して必要。**Codexは新しい明示指示がある場合だけ、その固定範囲を担当する。** 今回完了のための追加科学計算はない。

## 6. 文書検査・公開・保全とSTOP

公開対象はこの新directoryの3文書のみ。科学結果・正式採択・既存evidence statusの変更ではないため、研究概要・manifest・旧索引を変更しない。旧SOURCE/入力/raw/result/grant/ledger/marker/STOP、one-shot/retry制限、C1凍結を維持する。

文書技術検査には既存Phase 2A `verify_documents.py` の標準ライブラリだけの `git` / `blob_oid` を再利用した。3文書のUTF-8・改行、相互リンク/参照定義、固定commit/path、公開禁止情報の混入と差分範囲を検査し、取得済み4入口のGitHub blob identityも照合した。旧namespace専用validator全体、科学test、runner、数値auditorは起動していない。恒久validator・巨大台帳・承認管理pipelineを追加しない。

専用branchへ3文書をcommit/pushし、remote SHAを確認して公開3文書を再取得・byte照合する。最終固定commitと取得一致の実績は最終チャットで報告する。自己commit SHAを埋めるための追加入稿は行わない。

**未実施**：新科学計算・保存値の数値再解析/集計/ランキング、sampling/RNG replay、Hamiltonian/state/reference生成、回路build/compile/native synthesis、新RA/アルゴリズム実装、科学test、simulation、C1凍結解除・旧one-shot retry、新重要科学レビュー、GO/新規性/論文化の最終判断。サブエージェントは使用していない。

共有OS/scheduler/cgroup/quota、共有venv/package/cache/service、他者のprocess/job/files、他sessionのworktree/lockは変更していない。科学workerを起動せず、既存計算を停止・再起動していない。任意のCPU/RAM/worker制限も追加していない。

**mandatory STOP：候補選定・既存証拠対応・3文書公開と再取得照合を完了した時点で停止する。追加実装・別候補実験・科学検証へ自動移行しない。**

[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc42c3a9f77e960f415d0475da2ba03bb56252c7/docs/research/o2_s2_multi_event_validation_v2_20261106/REPORT.md
[O2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/docs/research/o2_sampling_return_technical_survey_20261106
[INT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105
[LIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106
