# O1 Hamiltonian表現・分割：PR＋RPE総費用改善の技術preflight

2026-11-06 JST。**既知法の応用候補を、作用経路の異なる2件として条件付きで整理した。強いUWC等に対する追加の総費用改善は未実証であり、新算法の独立新規性・候補の正式採択・科学実行GOは出していない。**

## 成果物

| 文書 | 内容 |
|---|---|
| [O1_EXISTING_METHODS.md](O1_EXISTING_METHODS.md) | Pauli-JW、DF/X-DF/RC-DF、UWC、混在辞書、prefix/非prefix、PR最適化の比較。元科学条件・SOURCE・正負の保存結果と強baseline |
| [O1_IMPROVEMENT_MECHANISMS.md](O1_IMPROVEMENT_MECHANISMS.md) | 候補2件の操作、情報条件、既知法との差、PF/RTE/normalization/shot/native/全round費用への作用経路と相殺要因 |
| [O1_MINIMUM_VALIDATION_PLAN.md](O1_MINIMUM_VALIDATION_PLAN.md) | 候補別の最小確認、既存SOURCE/関数/runner/testの再利用、科学実行が必要な境界とGO/STOP条件 |
| 本REPORT.md | 結論、未解決、作業境界、次の担当 |

## 実行環境と固定参照

- **SERVER 2：GPUA100-gleap02。** 利用者の既存repositoryのGit objectとGitHub接続を参照できたため、この環境で完結させた。SERVER 1へのSSH、LOCAL切替、他サーバーでの重複調査は不要だった。
- 調査専用branch：`codex/o1-hamiltonian-partition-preflight-20261106`。同名末尾の新規独立worktreeを作成。既存研究worktree・main・元研究branchにはcheckoutや文書更新をしていない。
- 基点／PR＋RPE文献調査：[e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8][S]。
- 3環境統合：[3571477093441ac4852166fe829f8ccc5db5ec4f][I]。指定の台帳・Gap・再利用・残検証を参照。両commitはローカルGitで存在を確認し、文献調査の固定directoryはGitHub connectorでも取得した。
- 正式共通運用指示2026-10-25、共通・repository AGENTS.md、PROJECT_MAP.md、研究概要全文、規範文書の該当節、VALIDATION_STATUS、manifest構造を確認した。資料収録commitと科学実行SOURCEを分けた。
- 公開対象はこのdirectoryの4文書だけ。既存統合台帳、SOURCE/input/raw/result/grant/ledger/marker/archive、固定科学契約とSTOPは変更しない。新管理台帳や承認証明は作成しない。

## 技術的に残る2候補

| 候補 | 費用を変える構造 | 強baseline後に未解決のこと |
|---|---|---|
| **M1：同じ物理tailの辞書を替える** | HD/HRを保ってcollected Pauli／rotated Z/ZZ等を選ぶ。既存paired構成では同じ補正meanに対してλ_R・normalization・raw半径・event価格が変わるため、N×full価格の交換を隔離できる | 同じ辞書pool・collect・error/価格情報・compiler機会を既知法へ与えても利益が残るか。F2は単回費用まででN未資格。最終元H energy task、全law・全roundでの追加利益は未確認 |
| **M2：圧縮coreの残差を補償して元Hを保つ** | 既知DF/RC-DF等の実装しやすいcoreと残差の交換。backboneを軽くする代わりに、元Hとの差のrandom負担・PF誤差を支払う | 適合する既存core、残差辞書λ、元H state上の位相/半径、native価格・取得費を払って強D/UWC/既知residual PRより安いか。現時点で新fit準備のGOはない |

M1は物理split固定、M2は物理split変更という違いで分けた。どちらもPR原著、UWC、RC-DF、既存混在辞書の知見を用いる**応用候補**である。既知要素の組合せでも有効条件が見つかれば研究知見になり得るが、その成立や独立した追加性質を今回認定していない。具体的な操作を欠く非prefix、joint optimization、rank/PF交換だけの案は新機構として追加していない。

## 文献・既存証拠から分かった境界

UWCは表現・split・RPE費用をすでに結合している。したがって「joint optimizationだから新しい」とは扱えない。[UWC原著][U] 原著の直接確認箇所と既存文献調査の帰属は[比較表§4](O1_EXISTING_METHODS.md#4-強baselineを弱くしないための文献照合)に示した。外部UWCを強baselineにすることと、repository旧grouped-UWCの来歴未結合を解消することは別である。

既存F1/F2は混在辞書のcorrectnessと単回費用を支持するが、F2のN/weighted_costsはnull。core/pair、N1、P-C/P-D/R3、研究Cのnative fit NO-GOとsingleton/PS48の正式一時終了も保持した。これらを別candidate名で再開しない。研究Cのupdate064には後続nominal model限定支持がある一方、own-DF ground条件のため、今回の元H厳密stateへ無条件移送できない。[固定統合台帳][ledger]

同じ元Hの厳密基底状態を使っても、近似DF/PFは別の固有状態・位相混入を持ち得る。SOURCE読解では、旧RPE会計関数が二次DF/未減衰半径1に特化すること、研究Cの下流がcandidate groundを選ぶこと、literal価格DPがnative oracleではないことを確認した。既存コードの全面再実装ではなく、該当入力・半径・provider境界だけを将来の適合照合対象とする。

条件付き18round成功とenergy bridgeは既に存在し、「全接続が未着手」とはしない。一方、true molecular energy、全実装・全law、長q価格移送、不確かさ込みの最終優越は別の未解決である。単回RZ、λ、局所PF誤差の減少を全RPE費用削減へ読み替えない。[既存全体費用接続][costmap]

## 残る最小の科学的判断

次に必要なのは新optimizerの作成ではなく、**どちらの既知法応用を、どの元H・共通split/core・資源指標・精度/成功条件で調べる価値があるか**という選択である。本調査では新しい固定数値を設定していない。

| 担当 | 次の作業・必要な許可 |
|---|---|
| **ユーザー** | M1/M2の検証価値と対象条件を判断する。希望する場合だけ、対象を定めた新科学計算・保存値再解析・回路build/compileの範囲を明示許可する。重要科学レビューを行う場合の開始許可は科学実行許可と独立 |
| **Codex（新しい依頼後のみ）** | 選ばれた範囲で既存資産の適合照合・最小接続を行う。科学実行や停止familyの再開は、その対象の明示許可と既存固定条件を満たす場合のみ。本報告から自動起動しない |

M1での最小確認は共通HD/HR・scalar・元H stateと、辞書別mean/B/N/full価格の接続。M2では適合する既存coreと完全残差・元H stateの接続を先に確認する。新しい残差生成・信号評価・shot再計算・価格測定・再ランキングは、すべて今回の実行範囲外である。

## 技術検査・公開・STOP

4成果物のUTF-8・参照定義・相互fileリンクと、固定repository blob参照29対象の存在を技術検査した。科学test/runnerは起動していない。commit対象はこの4ファイルに限定して差分検査する。公開は専用branchのみとし、push後にremote branch SHA、固定commitからの4ファイルの取得可能性と内容一致を確認する。**最終commit SHA・公開照合の実測結果は最終チャットに示す**（自己参照SHAを文書へ埋めるための追加commitは不要）。

新科学計算、保存値の数値再解析、Hamiltonian/state生成、simulation/sampling/RNG replay、回路生成・native compile、旧run retry/resume、新算法実装、重要科学レビューは実施していない。引用結果の再計算もしていない。

**Mandatory STOP：4成果物の作成・公開・取得内容照合が完了したら停止。既存STOPは維持。Codexが自動実行する次の研究作業はない。**

[S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106
[I]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105
[U]: https://arxiv.org/html/2603.22778v2
[ledger]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/IMPROVEMENT_VERIFICATION_LEDGER.csv
[costmap]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/END_TO_END_COST_ANALYSIS_MAP.md
