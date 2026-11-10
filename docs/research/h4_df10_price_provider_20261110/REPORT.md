# H4 DF10 Mq price provider：統合pipeline接続準備

2026-11-10 JST。**H4 DF10の凍結Mqを、既存成果と科学的適用範囲を保持してBのRPE総費用pipelineから呼べる部品に整備した。** Bの作業中batch interfaceへの技術接続を確認したが、B完成版への取込み・科学的な全RPE採用・新しいN×C評価は行っていない。

成果：[実装](../../../src/trottertracks/h4_df10_price_provider.py)／[入出力仕様](INTERFACE.md)／[Bへのhandoff](HANDOFF.md)／[人工・保存fixtureテスト](../../../tests/check_h4_df10_price_provider.py)／[結果](TESTS.txt)。

## 実装と再利用

共通interfaceはBの`pf_pr_rpe_pipeline`に作業中実装があったため、共通framework・N/C総会計器を追加していない。Bのloader/identity/measurement/metadata validationを引数で再利用し、H4固有のcatalog・科学binding・quoteだけを追加した。B worktreeは読取専用で調査し、変更・科学実行をしていない。

`quote`とbatch callableは固定Git JSONから保存値を選択するだけで、価格meanや残差の再集計を行わない。`predict_path`は元V1会計本体とS1のbPF=606.75/bPR=176.88888888888894を再利用する明示数値API。両者を分け、lookup missで新予測・compileを起動しない。明示数値quoteも同じB recordへ変換できる。

科学対象は既存のLinear H4/1Å/STO-3G、保存tol-only1e−8 DF10/sector(2,2)、保存H/state/snapshot・fragment順/scalar、S2/δ.1、PF B0/p7、PR p1/R4/K2、元finite RTE law、symmetric_directional、測定付きfull Hadamard wrapper、P=0。元Qiskit1.3.0/rz,sx,x,cx/opt1/seed17とcompiler hashを照合する。binding違いをMqで埋めず、unavailableと理由を返す。qRPEは別fieldのまま保持する。

価格種別はdirect/model/unavailable、単位は決定的価格/個別path/cohort平均/law期待値を区別する。PRはcohort/pathとhashを明示し、order/application情報を保持して参照する。直接価格にMqを使ったと表示せず、保存cohort平均を全law期待値へ昇格させない。新しい数値entryではsource-bound descriptorとordered K2 eventを要求し、raw payloadは出力条件へechoしない。

## 検証と限界

技術テスト**8 method PASS**。新しい実H4の予測・sampling・build/compile・fit・保存数値再解析・law平均/CI推定・N・N×C/Gは0。

- 取得前固定68軸に対するdirect/Mqの計136 lookupを既存JSON正本へ照合。counterを呼べない状態で確認し、保存値のcopyであることを区別した。
- 既知q8探索9/確認33のcohortと、それぞれの誤差証拠を分離。PR law平均はNone。path/digest/ordered payloadの不一致、H/DF/state/snapshot/prep/compiler/wrapper/recipe違いを拒否。
- artificial X/Yに異なる価格を置き、軸の取違えを検出。qRPEとqPFを異なる値にした人工/mockで価格式がqPFを使うこと、固定PR/PF係数、不確かさのNone、未検証外挿の非昇格を確認。
- Bの作業中snapshotへbatch接続し、既存`plan`までを呼んだ。`evaluate/run_missing`を禁止して新N/Gを呼ばず、未取得値を0にしないことを確認。数値APIは人工empty blockとmockのみであり、新H4条件の価格精度の証拠ではない。

B snapshotの試験SHA256は`c91d5c59f4ea7ae65b37bc2c850e782888834842c0626fccd72b5faa34c1c9ba`。snapshotをA側Python環境へloadした試験で、相対importの`h4_df10_rpe_connection`はA側版を使用した。B側版には拡張があり、今回使うscope/価格種別定数は同じだが、B環境全体の結合試験ではない。当該moduleは調査時未commitで、B全体の最終設計はB担当のまま。Bに未配置のV1/Mq依存3 moduleをhandoffで特定し、Bが自分のbranchへ取り込む範囲を明記した。AがBの完了を待つことや、そのtreeを凍結することを前提にしていない。

## 科学的資格の受渡し

[独立q完了結果][INDEPENDENT]のPF q32の0.10525%、PR q16新33本の相対RMSE1.31743%等は、既存有限条件の証拠への参照として渡す。今回それらを再推定しない。S1の既知q、有限の独立q、未検証外挿、モデル対象外を区別し、良好な一点を誤差上界へ変換しない。

path別残差の元CSV参照、cohort平均bias/SD/記述的SEの保存summary、method別shared calibration group、paired XYを返す。一方、bの不確かさ・long-q systematic・全law期待値・joint covariance・total CIはNone。directにMq共有fitの相関を割り当てず、Mqではround間独立を仮定しない。αtotal=.05を価格の95%保証へ読み替えない。

`qualified_for_candidate`は保存recordの同一性または有限model domainの経験的資格であり、price unitを伴う。PR有限cohortはBの既定qualified-expectation policyを満たさず、全law未接続である。全13round providerの正式採用・Main Claim変更は行わない。

## 出典・保全・次の担当

- 独立qの正本carrier：`1cf87201ebc6c9e0296fe647354981b6f7f8b0c1`。PF/初回予測SOURCE55eccab88、承認後PR継続SOURCE f8bb11fe6、元STOPを保持。
- S1 carrier：`cefc01622a3cd75dd16d30de7b9fb68274f3eddd`、V1 carrier：`eec5bdf9cc391fe49b55b5673f3e385deeb20d42`。校正・V1不一致・固定予測/結果は変更なし。
- 元回路/価格carrier155529e2、provider c31ef424。元データ・SOURCE・raw・grant/ledger/marker/KEEP/STOP、他worktree/process・共有設定を変更しない。

**B担当**がcomponentと固定依存を自分のbranchへ取り込み、共通API最終版で接続する。**ユーザー/B担当**が新予測・law期待値の解釈・外挿採用・N×C/Gに必要な科学条件と実行可否を判断する。**A担当Codex**の今回の技術実装・保存fixture照合は完了し、GitHub公開後remote SHAと主要成果物を照合してmandatory STOP。新しい価格検証や全RPE科学計算へ進まない。

[INDEPENDENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1cf87201ebc6c9e0296fe647354981b6f7f8b0c1/docs/research/h4_df10_price_v2_independent_q_20261110/completion01/REPORT.md
