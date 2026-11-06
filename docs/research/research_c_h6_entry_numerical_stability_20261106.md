# 研究C：H6 runner完成確認とENTRY関連の人工数値安定性調査

2026-11-06 JST。**作業Bを実施。開始時点・終了時点とも `RUNNER_TECHNICALLY_COMPLETE`。既知の人工2ケースは未解決FAILで、`ENTRY_SCIENTIFIC_ELIGIBILITY_UNRESOLVED` を保持する。** 実H6/H4/H2の新科学計算、実分子の保存数値結果の新再解析、科学的GO/NO-GO・重要科学レビューは0。公開SHA/tree/ファイル/SOURCE照合後にMANDATORY STOPする。

## 1. 開始時点の判定と再利用

正式共通運用指示 `research_project_instructions_unified_2026-10-25.md`、共通・worktreeのAGENTS.mdを適用。指定された準備commit `68634ef468eb6c912de7bf569484fc454fb32548` の「未接続」を現在の状態へ転用しなかった。関連remote branchを限定検索し、runner完成branchの公開HEAD `30c6f47529d708a0dd1f67818fb455325b084fc8`、line search修正branchの公開HEAD `d7ea73b56bd2cac1e85a658de0c2848e10db658f` を `git ls-remote` で確認した。

[既存完成報告](research_c_h6_runner_completion_20261106.md)はGitHub connectorでも取得した。完成commitは最新修正commitの祖先で、[既存verification](../../artifacts/research_c_h6_runner/2026-11-06/technical/verification.json)はeval192＋fit8＝200 PASSを記録している。以下の接続、実装された親E2Eと既存13q人工検査記録を確認し、今回も親E2E17件を再実行して全PASSだった。静的な関数の存在だけによる判定ではない。

| 接続 | 確認した既存経路・検証 |
|---|---|
| Fit RPC | `c_h6_runtime`→fit transport/`run_cell`→`PhysicalContext`/`SectorCAR(6)`。人工rank14/LD4・sector400/15因子export・504変数codecを実RPCで確認。 |
| Optimizer | PR/ENTRY×3strength、64更新/16診断、共通の独立J/物理適格判定。`recipe_cell`は未適格terminalを64 checkpointへexportしない。既存の未適格終端回帰も再実行。 |
| PF/RTE | 候補別referenceとcache、r32/64/128。r32不適格が他rへ伝播しない回帰を再実行。 |
| RPE | m0..12、既存feasibility/shot式、own-DF ground/aliasを既存backendから呼ぶ。 |
| Native | 12-mode対応の既存Gaussian/StrongBuilder/phase compiler/P0。親RPCの人工小次元selection/final実経路を確認。 |
| Correctness | selection sample0最大18＋final sample0最大4＋同J/U D最大4。全800入力/full8192出力の既存列処理と13q人工記録を確認。今回の親mockでも最大26/124800作用列を確認。全8192入力の演算子等価性とは呼ばない。 |
| D | 同input/candidate/J/Uの全frame ordered S2。candidate/input/reference誤接続の拒否を再確認。 |
| Selection/Final | 独立32/128、arm/candidate/r/role cohort、同pairだけ共有、選択固定後のfinal。 |
| 会計 | 既存shots、W、kernel、shell、総費用。`W*mean_kernel+shell` とr/標本数の非二重計上。 |
| 統計 | 主2比較、各N128/97.5%近似Welch差区間、元の診断/未成立ラベル。 |
| 実行管理 | 固定SOURCE/契約、fresh output、partial、例外・close failure、retry/resume0、mandatory STOP。 |
| 人工E2E | PR/ENTRY起動からselection/final/D・出力までの親mock、実eval子process、人工PhysicalContext/fit RPC。未検査標本は `NUMERICAL_EQUIVALENCE_NOT_CHECKED` のまま。 |

したがってrunnerを再実装せずBを選択した。独立branch/worktreeは `codex/research-c-h6-parallel-technical-20261106` / `research-c-h6-parallel-technical-20261106`、基点はd7ea73b5。ENTRYの実分子収束未達はrunnerの接続完成と別の問題である。

## 2. 対象と最小差分

対象は元の `test_rank14_near_degenerate_head_and_nonzero_tail[PR/ENTRY]` の**2件**。ENTRYだけ2件ではなく、同じ人工入力を使うPR・ENTRY各1件である。元fixtureの配列生成を `near_degenerate_chart` へ抽出してテストと診断で共用した。生成式・seed・assert・閾値は変更せず、skip/xfailも付けていない。

人工6軌道・rank14/LD4、seed1229のU/J、head間角度1e-8、seed310の対称J摂動1e-5から元の方法でgを作る。元のstrength .1、epsilon1e-8、tensor scale `1.5952676862881953e-8`、PR/ENTRYそれぞれの初期mass scaleを保持。保存実H6入力はdecodeしていない。

- 既存 `c_h6_conic_diagnostic.solve_and_polish` に、Jへdecodeする前のcone座標の**copyを観測するoptional callback**を追加した。既定は何もしない。solver設定、探索、目的、epsilon、受理gateは変わらない。
- [人工診断script](../../scripts/diagnose_research_c_h6_entry_stability.py)を追加。元fixture、Clarabel0.11.1、QR Newton、`head_report`、元のbinary64/extended残差・勾配・J gateを再利用する。新solver・ridge・bound・方向切捨て・scientific runner接続は追加しない。
- Decimal参照は既存 `test_research_c_j_u_stability.decimal_physical_U` の「binary64入力を `Decimal.from_float` で固定する」規約をJ勾配評価へ拡張した。元のDF列/PR係数mapを使い、全target成分を保持する独立スカラー積和を行う。80桁で評価するだけで最適化しない。丸め前cone座標の参照では、既存longdouble Wの各値を整数比として保持する。
- [新規6テスト](../../tests/management/test_research_c_h6_entry_stability.py)は、小次元PR/ENTRYでDecimalの値/J勾配/残差を既存独立縮約と照合し、非対称なtarget丸め成分の保持、cone座標と実際のpolish開始Jの対応を確認する。

[今回の診断SOURCE](../../artifacts/research_c_h6_entry_stability/2026-11-06/SOURCE.json)のSHA256は `b57a2991e6db69608a06d74a6e2992c1ddb71b413a75cbcb295b7eab235b35fb`。これは人工診断の来歴であり、新しい実H6科学実行SOURCEの固定や承認ではない。

## 3. 再現結果

詳細は [PR](../../artifacts/research_c_h6_entry_stability/2026-11-06/PR_affine/diagnosis.json) / [ENTRY](../../artifacts/research_c_h6_entry_stability/2026-11-06/ENTRY_affine/diagnosis.json)。各folderの `arrays.npz` に人工g、U、初期J、cone座標、開始/返却/検査Jを保存した。最初の診断結果もPR/ENTRY folderに残し、後続のaffine診断と区別する。

| 指標 | PR人工ケース | ENTRY人工ケース |
|---|---:|---:|
| Clarabel status / 反復 | Solved / 14 | Solved / 18 |
| 返却J Frobenius norm | 4.56313e7 | 7.18716e7 |
| J最大binary64 spacing | 1.86265e-9 | 1.86265e-9 |
| 元binary64 J勾配∞ | .16483745 | .18637690 |
| extended J勾配∞ | .08370067 | .16948259 |
| 同じ返却JのDecimal80勾配∞ | .08370147 | .16947253 |
| binary64勾配とDecimalとの差∞ | .12252140 | .14659330 |
| extended勾配とDecimalとの差∞ | 1.35949e-4 | 8.86251e-5 |
| 丸め前の既存cone座標をDecimal評価した勾配∞ | 6.92150e-5 | 1.18542e-4 |
| extended affineモデル内の勾配∞ | 2.57552e-5 | 2.48522e-6 |
| Newton polish終了 | status2、2受理後Jが動かない | status2、Armijo失敗/受理0 |
| 元J gate、閾値1e-5 | NOT_MET | NOT_MET |

今回のBLAS設定は2 threads/process。過去の1 thread記録と終端値のbitwise一致は要求しない。旧PR約.073/ENTRY約.083のFAILをPASSへ変えたわけではなく、今回も元の判定で2件FAILである。

## 4. 原因の切り分け

### 確認できたこと

**条件不良と相殺。** 固定U tensor mapは1296×294、通常のbinary64 SVD cutoffで数値rank231/nullity63。一方、headだけは84列から既知3 gaugeを除いた最小特異値約1.39456e-9、既存extended head判定はadmittedだった。全tensorのnull spaceを目的全体の非一意性と同一視しない。tail penaltyも含むquotient metricは291列すべてを保持する。SVD数値rankは診断であり、厳密null spaceの証明でも方向削除でもない。

初期点からの係数変位はほぼunpenalized headに集中し、normは4.56e7/7.19e7に対しmapped tensor normは両方約1.7862e-4。弱いhead方向が大きな相殺係数を許している。ENTRY終端のNewton残差因子は最大特異値約1.0858e4、最小約5.2249e-11で、比は約2.08e14。規定の方向切捨ては行っていない。

**浮動小数点の影響。** binary64残差とDecimalとの差∞はPR1.85e-9/ENTRY2.60e-9、extendedでも約2.68e-12/2.15e-12残る。小さいtensor scaleで正規化した勾配では無視できない。Decimal80対100の追加評価は、返したbinary64の値・勾配が一致した（[検算記録](../../artifacts/research_c_h6_entry_stability/2026-11-06/technical/decimal_reference_convergence.json)）。これは数値参照の安定性確認で、厳密誤差enclosureではない。

**丸めだけではない。** 同じ返却JのDecimal勾配も不合格で、丸め前cone座標のDecimal評価も不合格だった。ENTRYではextended affineモデル内の勾配2.49e-6と、元のtensor縮約を使う丸め前Decimal勾配1.19e-4が異なる。大きい係数の下では、組み立てたaffine列/座標モデルの数値誤差とJのdecode丸めを分ける必要がある。厳密算術での二次錐epigraph恒等式の誤りを示す結果ではない。

**探索停止。** QR Newtonの方向方程式残差∞はPR約3.93e-16、ENTRY約8.15e-13まで小さいが、元の物理勾配の資格ではない。固定Newton rayでは、PRのfull stepは二次予測−1.65e-10に対してcentered目的差+3.27e-11。alpha=2^-32以下では実Jが不変となる。ENTRYのfull stepでは二次予測約−5.17e-9に対して実目的差約+7.08で、head変位は約1.11e9。小さいalphaでも、予測減少より大きい丸め・評価差が残る。この固定rayは人工診断であり、別optimizerの採択や厳密単調性の証明ではない。

既存trust-region報告/観測コードの「epsilon近傍の局所曲率・予測/実減少の乖離」と整合する。実H6のtrust-region診断は再実行していない。今回のhard casesでは元のZeroDivisionErrorは発生せず、修正済み失敗経路がstatus2を返した。実SciPyの例外を再現する既存回帰もPASSした。

### 否定できた説明と未確定事項

- 「ClarabelがSolvedなら元のJ gateも合格」「座標勾配/線形解残差が小さければ適格」は今回の人工結果で否定される。
- 「残差積和だけ高精度にすれば現在の返却点は合格」「最終repackだけが原因」も成立しない。今回の返却Jから元テスト検査点へのrepack変位は両方0だった。
- 元のepigraph恒等式、小次元の元目的/勾配・Hessian、独立pullback、line searchの失敗伝播の回帰に、新たな数式不具合は見つからなかった。全条件下での数式正当性の新しい証明はしていない。
- **この人工2ケースでbinary64独立gateを満たす点は構成できていない。** 高精度で最適化した解も構成していない。binary64適格点の不存在、目的全体の非一意性、唯一の原因の特定、精度を上げれば必ず解決することは、いずれも未証明である。

## 5. テストと技術改善の限界

[最終verification](../../artifacts/research_c_h6_entry_stability/2026-11-06/technical/verification.json)と対応JUnitが正本。作者自身の技術検証であり、別reviewerによる重要科学レビューではない。

| 今回の最終test群 | PASS | FAIL |
|---|---:|---:|
| H6親runner人工E2E/RPC/native/STOP | 17 | 0 |
| QR7＋conditioning15＋conic6＋新参照/telemetry6 | 32 | 2 |
| fit worker/robustness/H4 selection政策 | 23 | 0 |
| evalの既存PF/RTE/RPE/会計/STOP回帰 | 18 | 0 |
| **重複実行を除いた合計** | **90** | **2** |

ERROR/SKIP/xfail 0。FAILは元のPR/ENTRY hard casesそのもの。初回互換性投入ではfit環境にmatplotlib/openfermionがなく9件失敗した。共有環境を変更せず、fit23件とeval18件へ分けて全PASSを確認した。初回ログ/XMLも保持し、最終件数へ重複加算しない。

今回改善したのは**原因の観測可能性と人工再現性**である。収束・scientific eligibilityの改善は認定しない。元の目的、epsilon、J閾値1e-5、rank、LD、科学契約、solver失敗の意味、既知FAILは変えていない。

## 6. 保全・実H6適用・次の担当

旧runner148-file SOURCE、stablemetric153-file SOURCE、H4科学SOURCE、元のH6契約2種、旧conic診断PR/ENTRYのSTOPの7ファイルについて、今回のworktreeのbytesを基点d7ea73b5のGit blobと照合して一致した。照合hashはverificationに保存した。新source差分はconic診断の観測callbackだけで、過去の固定SOURCE/raw/STOPを改訂したものではない。旧raw・grant・ledger・markerに変更を加えていない。

H4再配置診断のworktree/branch/入力/J/U/結果/cacheを使用・変更していない。H4 1.15 Åを修復・再試行していない。専用worktree、専用artifact、専用/tmp testディレクトリのみへ書き込んだ。fit/eval venvと既存Clarabel配置はread-onlyで再利用し、新package/GPU/distributed実行は追加しない。観測CPU affinity64・load約1台相当・十分なavailable RAMの下、小行列のBLAS過剰thread競合を避ける2 threads/processを自分のprocessだけに設定した。共有設定、他利用者/Codexのprocess・ファイルは変更していない。sub-agentは使用していない。

次の担当は**ユーザー／GPT**。この人工の条件不良・モデル誤差・J表現問題を踏まえ、追加技術対策が研究上必要か、H4再配置診断の結果と合わせて次の研究方針を判断する。今回重要レビューは行っていない。

対策候補は、同じ目的を維持する精度改善/補償積和、元の独立gateで確認するsolver数値精度・座標表現の改善である。単なる長時間化、Solvedの受理代用、ridge/bound追加、epsilon/閾値緩和を採用する根拠は得ていない。これらの候補を新solverとして実装・採用する作業は今回行わない。

**Codexが実H6へ適用するには、採用する技術差分・SOURCE・入力・固定U診断かtrajectoryか等の範囲を具体化した別の科学実行許可が必要。** dtype/受理条件を変えるなら該当固定契約の具体的変更も明示承認が必要である。現状の科学実行承認だけで収束が解決するわけではない。H4診断結果からH6条件を自動変更せず、6 trajectory、PF/RTE、RPE、ground、native、H2、重要科学レビューへ進まない。

## 7. Q1〜Q10

| 問い | 回答 |
|---|---|
| Q1 開始時runner | 完成。公開30c6f475、既存200 PASS、今回の親17 PASSで確認。 |
| Q2 A/B | B。完成済みrunnerを再実装しないため。 |
| Q3 再利用 | 元fixture、Clarabel/QR Newton、conditioning、独立残差/勾配/gate、runner E2E、fit/H4 selection/eval回帰。 |
| Q4 変更 | 観測callback、人工診断/参照、新6テスト、fixture共用。科学条件変更なし。 |
| Q5 件数 | 最終90 PASS/2 FAIL、ERROR/SKIP/xfail0。初回環境依存失敗は分離記録。 |
| Q6 既知2 FAIL | PR/ENTRYとも未解決。削除・skip・閾値変更なし。 |
| Q7 資格の区別 | runnerはTECHNICALLY_COMPLETE、ENTRYはSCIENTIFIC_ELIGIBILITY_UNRESOLVED。 |
| Q8 非干渉 | 独立worktree/output、既存環境read-only、共有設定/他process変更0。 |
| Q9 保全 | 旧SOURCE/契約/STOPのbyte一致を確認。旧raw等へ書込みなし。 |
| Q10 次作業 | ユーザー/GPTが追加技術対策とH4診断後の研究方針を判断。実H6適用は別許可。Codexは公開照合後MANDATORY STOP。 |
