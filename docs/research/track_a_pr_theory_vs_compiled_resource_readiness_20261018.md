# Track A H4/H6：PR理論リソースと実compiled費用の比較準備

**PREPARED_UNAPPROVED_MANDATORY_STOP。新しい実H4/H6の理論評価・費用比・残差・順位・fit・再会計・compile・sampling・signalは0。**

利用者の最新指示「H8ではなく、H4とH6で行って」に従い、対象を変更した。原論文の分子・資源数値の再現、Toffoli/Tへの換算、新しいPR法やDF順序の最適化は対象外。既存H4/H6/H8結果・grant/ledger/marker/STOP・並行A〜Fは変更していない。

- 起点：横断証拠整理 `621d7d7bed183862db8d9f01b6fdb4406f942309`。旧V2停止記録と同じ理論counterを含み、追加の理論結果は別固定commitを参照する。
- branch：`track-a-pr-theory-compiled-h46-readiness-20261018`
- SOURCE：`106ed56b294717da12d9b64ad72fbdd7317223b7`
- [一つの資料入口](../../artifacts/resource_applicability/track_a_pr_theory_compiled_h46_readiness/2026-10-18/README.md)

## 調査結果：何が既にあり、何がないか

H4はlinear 1Å/STO-3G/legacy fixed DF12/8modes/sector36/T=.8、prefix3、固定native anchor16構成。H6はlinear 1Å/STO-3G/tol-only1e-8/cutoff0/signed order/weighted projection/DF19/12modes/sector400/T=.8、prefix5/10/15/full19の既存104候補。同じ系サイズ実験や同一DF政策とはみなさず、各系の同じ候補・入力・state・回路仕様の中で比較する。

GitHubから指定理論モデル/Qiskit接続/理論評価/V2停止/横断資料と主要入力を取得し、15固定ファイルのremote Git blobをlocal固定commitへ一致確認した。H8については旧報告に含まれる文脈だけを参照し、新しい比較対象に入れない。

| 対象 | 保存証拠・metadata照合 | 今回の判定 |
|---|---|---|
| H4 native anchor | 保存532回路のうちsymmetric-directional528回路を採用、ordinary4回路を除外。manifest登録264groupのtask/worker SOURCE/trajectory event digest/paired wrapper metadataを原ファイルへ照合 | 16候補・32 axis entries。理論表が未作成、prepared prefix/basisと全binary profileの独立結合が不足 |
| H6既存理論 | 固定結果7e416705…のH6 104候補・A/B×両軸416行、72 finite-law receiptを再利用可能。全H6/H8では208/832/144、接続10,480・うち5,736・不適格4,744という旧結果を保全 | 新counter評価・確率再正規化・shot導出・接続数値の再計算は0 |
| H6 compiled来歴 | 132 scope entries（探索/確認/B0Iを区別）・264 axis entriesを既存索引と最終V2監査へ結合 | PR25 scope entry＝50 axisがCONDITIONALLY_ALIGNED。192 axisは来歴未解決、22 axisは未測定。完全数値envelope合格は0 |
| H6旧PF | 最終監査でS4/q8の旧profile receipt結合を保持。他15測定PFのprofileは未解決、未測定4 PFは欠測 | profile既知のS4/q8にもper-wrapper prepared basis不足が残り、PF対照の同一性を認定しない |

296はcandidate/scope/axisのmetadata行数であり、独立実験数や固有candidate数ではない。H6の25は探索17・確認8のscope entryで、phase間をpoolしない。完全一致の公平なPR対PF数値比較を実行可能と認定した件数は**0**。来歴条件を明示したH6 PR単価の限定診断は別承認の候補として残る。

未解決は「証拠が全サーバーに存在しない」という断定ではない。調べた固定Git資料と宣言済みH4 manifest/task/trajectory/wrapperの範囲で結合できなかった状態である。広域検索・他worktree一括凍結・旧V2救済のための回収campaignは行っていない。

## A/B/native混合と三つの費用層

| モデル/層 | 意味 | 制約 |
|---|---|---|
| Uniform A | PF prefix・RTE tailともoptional Gaussian reuseを無効にした固定分解 | native tailのreuseとcompiler最適化との差が混ざり得る |
| Uniform B | 同一basisの隣接境界だけを相殺、PR occurrence/prefix/stepは閉じる | prefixまでBとする仮想構成。実装済みnative全体ではない |
| NATIVE_MIXED | prefix=A、closed RTE tail=B、末尾scalar一度、cos/sin wrapper | 新APIを人工データだけで検査。実H4/H6の混合モデル数値表は未取得 |
| 理論のRZ/CX | 既存explicit_rz_sx_x_cx_v1 loweringの数/期待値 | compiled RZ/CXの厳密上界ではない |
| 構築nativeのpretranspile命令 | 原回路の高水準gate列 | 全対象の保存payloadを今回取得・検証していない。理論lowered countで代用しない |
| 実compiled費用 | 原Qiskit1.3.0・rz/sx/x/cx・opt1・seed17・topologyなしの保存RZ/CX | profileとbasis variantをcandidate lineageごとに保持する |

既存counterはbyte不変。新adapterはprefix/tailの部品から構成し、uniform A/B総数の補間や差分合成をしない。PF-onlyのnativeはAに対応する。制御はsymmetric-directional、P=0、state preparation除外、full Hadamard wrapper両軸、単一scalar ancilla位相を維持する。measurementをRZへ換算しない。

既存counterは理論depthを出さないため、RZ depth/CX depth/total depthの新理論予測はnull/未実装。RZはT/Toffoli数ではない。人工検査で新しいFT合成規則は導入していない。

## 数値比較の設計と実装

`pr_theory_compiled.py`と`run_pr_theory_compiled_h46.py`はstdlibのみ。既存科学package初期化、Qiskit、SCF/DF/state、sample APIをimport/呼出ししない。SOURCE・入力byte SHA・plan・新grant・登録row/model/metric・専用outputを結合し、承認拒否を数値入力読込より前に置く。

基本量はrho=C_compiled/C_theory、e=(C_compiled−C_theory)/C_compiled。理論lower/upperをそれぞれ保持し、midpointを作らない。0分母/欠測/nonfinite/反転区間は値を生成せずSTOP。期待値対有限標本の差は`EXPECTATION_VS_FINITE_SAMPLE_NOT_COMPILER_ONLY`、原因は原則UNATTRIBUTED。保存sample数・law・seed namespace・phaseを必須にする。

同一event列を照合する経路は別estimandとして用意し、event digest不一致を拒否する。H4のPR8初回trajectoryのevent payloadとfinite distributionの存在は確認したが、full prefix/basisを含む数値envelope合格には昇格させていない。理論期待値に全Taylor次数を含めても、compiled母平均や未観測rare costの上界にはならない。

epsilon/factor/model/理論endpointで反復されたcompiled値はstable observation IDで一つと認識し、矛盾した再利用値を拒否する。一般identity位相の幅は残す。希少次数や未測定費用を0で補完しない。raw低水準列がなければGaussian cancellation/rotation merge/compiler効果の厳密な寄与分解は未確認のまま。

PR/PF総費用の比較APIは同じ系・H/state/DF・T・control/wrapper・compiler/profile・epsilon/bias policy・保存shot出典・Pを要求する。各候補固有の保存N_axisを維持し、G=sum_axis N_axis*C_axisとする。今回そのAPIを実データへ適用せず、旧winner・削減率・shot・biasは変更していない。

## 人工検証

**32 tests PASS**。新人工suiteのみをPython標準ライブラリ環境で実施し、既存counter/Qiskit接続campaignは反復していない。実H4/H6のtrajectory/native build/transpileは0、人工Qiskit transpileも0。

検査はA/B/native混合、S4負時間とscalar/wrapper、全event列の小規模列挙とmixed期待値、確率10^-18のorder4、identity phase幅、prefix/q/r/K・axis・phase・profile・law等の混合拒否、保存endpoint重複・欠測、正確な有理数の比/区間、descriptive SE、shotと単価の分離、INVALID grant拒否、人工one-shot完了/再利用拒否/最初の異常でSTOPを含む。

保存table descriptor経路も、Git読込をmockした人工8行で一回の完了・STOPまで接続確認した（実table読込0）。

主担当のlocal技術検査であり、独立別担当の科学レビュー/CI/外部再現とは呼ばない。新しい科学的判断・原稿claimの採択は行わない。

## 次の最小一回実行案（未承認）

まず次の**H6 PR2候補だけのmetadata条件付き保存点診断**を提案する。費用値の大小で選んでいない。

- H6探索B2_p10_q8_2nd_r1_K2、B2_p15_q8_2nd_r2_K4。各n8、探索phase固定、確認n32と混合しない。
- uniform A/B × cosine/sine × RZ/CX＝16比較行。元の理論numerator/denominatorと保存axis meanを固定入力からコピーし、rho/eの区間だけを一度計算する。
- 新しい理論モデル評価、mixed実データ表、sampling、compile、SE再推定、shot再会計、回帰・holdout・ranking・PR/PF winner更新は0。
- 完了又は最初の不一致でmandatory STOP、automatic retry/resume0。新SOURCE/plan/input/output/grantで実施し、旧grantと累積callを返却/resetしない。

このscopeは旧V2の共通fitとは目的・対象・estimand・namespaceが異なる。旧固定hash holdoutや必要件数を変更してV2を再開するものではない。H6 PF来歴不足をPR-only診断の成功で解消したとは扱わない。

H4の次候補はfull PF p12/q1 S2・S4とPR p3/q1/r4/K2・p3/q8/r2/K4の4構成。先にprepared prefix/basis/profileを結合し、新しい理論評価への別承認が必要。全3モデル×両軸なら24 theory wrapper行、RZ/CXで48 metric比較に相当する将来案で、今回の16行案には含めない。H6 PF p19/q8 S4もbasis bindingを満たすまで別の未解決対象とする。

追加compileが本当に必要とは現時点で確定していない。保存payloadの回収又は固定snapshotからの承認済みmetadata補完を先に検討する。旧costの構築来歴を復元できない場合に限り、PF1group両軸＋保存eventを使うPR1group両軸の計4compile等を、同じprepared payloadに結合する新しい技術対照として別提案する。その4本は母平均/rare層/PR優位の確認標本にはならず、今回の認可では起動しない。

## 固定・資源・保全・実行方法

SOURCEはsource/testsを先に固定、資料は別REVIEW commitへ保存。source closureは新library/runner/test＋既存counterの4file。解析profileはstdlib用Pythonであり、保存compiled値の旧compiler profileと混同しない。H6旧profile4e32bb9b…とH4のversion宣言をH8 profile622ce4a8…へ置換していない。

資源の現行共通指示に従い、新しいCPU/RAM/worker/wallの恣意的上限やaffinity設定を追加しない。今回の32人工検査は軽い単一processで完了。将来もまず固定入力を一度だけ読み、metadata indexとcompiled観測を再利用する。重いcounter適用が承認される場合はcase単位のcache/並列処理を空き資源に合わせる。一時的なhost PSI・他worktree更新だけを科学不一致としてSTOPさせず、実際のOOM/own異常/入力・SOURCE不一致は保護する。

1,323個の宣言済みH4入力ファイルと固定Git blobをread-onlyで照合し、読み終わりに全byte不変を確認。旧grant/ledger/marker/sourceへ書込0、他worktreeやjobを停止/再起動/設定変更0。projects全体・/tmp・agent履歴の広域走査0。NPZの科学array読込、state/GPU/共有venv変更0。

公開draftは`approved=false`、`runtime_authorization=false`、`sealed=false`、grant status INVALID、CPU[]、output/absolute command=null。選択input descriptorは元Git commit/path/SHAとmetadata caseだけを固定し、今回そのadapterを実H6表へ実行していない。

承認後の形式（**現draftは拒否される**）：

```bash
<analysis-python> -P -B <SOURCE-worktree>/scripts/run_pr_theory_compiled_h46.py \
  --execute --plan <new-sealed-plan.json> --grant <new-approved-grant.json> \
  --input <bound-selected-input-descriptor.json> --output <new-home-output>
```

REVIEW HEADから直接起動せず、SOURCE106ed56b…の別実行worktreeへ固定する。実行前に利用者の16行条件付き診断への明示承認と、source/profile/input/outputの最終bindingが必要。科学レビューや追加計算を自動開始せず、資料公開後STOPする。


## Q1〜Q7への回答

| 問い | 今回の回答 |
|---|---|
| Q1 比較はどこまで完了したか | 既存H6 A/B416行と保存compiled接続を確認。新しい数値比較は未実行。H4はtheory表なし。今回の成果は来歴台帳・比較実装・人工検査・契約草案。 |
| Q2 A/Bの適用範囲 | 固定loweringの仮想構成。PF nativeはA、PR nativeはprefix A/tail B。uniform A/Bのいずれもnative全体やcompiled費用の厳密予言/上界ではない。 |
| Q3 公平に実行できる候補件数 | 完全数値envelope確認済み0。H6 PR25scopeはmetadata条件付き候補、厳密なPR/PF比較認定0。H4は理論/preparation/profile結合待ち。 |
| Q4 不足は何か | H4の新theory表・prepared prefix/basis・binary profile、H6旧PFのper-wrapper basis/profile、完全なsame-event wrapper envelope。存在しないとの全域断定ではない。 |
| Q5 追加compile不要の最小解析 | 別途条件付き承認後のH6探索PR2候補・A/B・両軸・RZ/CXの16行。既存理論/保存meanだけで単価rho/eを計算、fit/再会計/winner更新なし。 |
| Q6 追加compileが必要か | 未確定。先に保存prepared payloadの結合を試す。復元不能な場合だけPF1paired group＋保存PR1paired group等4compileを新来歴の技術対照として別提案。今回の実行対象ではない。 |
| Q7 次の承認 | SOURCE106ed56b…・固定2候補・16行・CONDITIONALLY_ALIGNEDを明示した一回の保存点診断。新grantと未使用home outputを結合し、完了/first-error STOP後停止、retryなし。H4理論補完は別認可。 |

次の担当：ユーザーが16行の条件付き診断を実行するか判断する。承認された場合はCodexが最終binding・fresh入力検査・新grantで一度実行する。GPTの新しい科学レビューは今回開始していない。
