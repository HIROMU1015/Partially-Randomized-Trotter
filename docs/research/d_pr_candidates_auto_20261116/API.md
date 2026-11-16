# B2 API：選択済みPR候補の準備から同じ探索へ戻す

公開入口は既存の `scripts/run_rpe_method_optimizer.py --round … --run-dir …`。roundの `candidate_completion` を有効にすると、B2の `METHOD_MINIMUM_PRIORITY_V2` が返した不足候補を `src/trottertracks/pr_candidate_completion.py` が準備し、同じcontextとexecutorの経路へ登録する。D2はL_D・κ・幅・shotの候補順を実装し直していない。

今回の接続検収は**各方式で最初のREADY候補1件**を対象とする有限範囲。全未準備L_Dの探索完了、全分割の最適性、実価格の期待値資格を主張しない。実取得SOURCE `194e00a8…` / `1691f5f3…` と、汎用経路を修正した現APIを区別する。現実装の固定commitはHANDOFFを参照。

## 通常CLIと今回の固定実行例

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d2-pr-candidate-autoprepare-20261116
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python -B \
  scripts/run_rpe_method_optimizer.py \
  --problem configs/rpe_method_optimizer/h4_df10_problem.json \
  --round docs/research/d_pr_candidates_auto_20261116/ROUND_RUN02.json \
  --run-dir docs/research/d_pr_candidates_auto_20261116/run02
```

これは同じrunを再開するcommandでもある。実行中の同commandを二重起動せず、完了後は保存結果を使う。新しい科学範囲・別SOURCEを試す場合は、新round IDと別run directoryを用い、旧STOPと旧rawを残す。

`ROUND_RUN02.json` の主要設定：

```json
{
  "methods": ["PR_PAULI", "PR_DF"],
  "science_approved": true,
  "max_new_q3_per_method": {"PR_PAULI": 1, "PR_DF": 1},
  "candidate_completion": {
    "source": "1691f5f34e43a971ead2f67673788678588106fb",
    "bootstrap_saved_only": true,
    "new_candidate_design": {
      "formula": "2nd", "delta_over_h": "1", "kappa": "2", "K": 2
    },
    "max_completed_per_method": 1,
    "max_attempted_per_method": 9,
    "evaluate_completed_only": true,
    "resume_preparation_from": {
      "commit": "fd5c63006c4b8c14c56b5ef7691ce3d73236ad9d",
      "path": "docs/research/d_pr_candidates_auto_20261116/run01/completion/STATE.json"
    }
  }
}
```

`new_candidate_design` は、新L_Dの要求にB2側から渡す設計条件。canonical configurationを既に持つ要求はその構成を保持する。κは絶対値であり、Pauliの元入口へ渡す倍率は `requested_kappa / base_kappa` をexactに計算する。DFは `A_input.kappa_paper` に絶対値を渡す。幅・公式・Kも要求値を渡し、幅0.1や保存LDへ置き換えない。生成後にはmapperのactual step、q/R、signed pieces・allocation・physical IDを用いる。

`evaluate_completed_only` の既定値は **false**。通常利用では、既存のready候補・保存適合点と今回準備した候補を同じ探索の比較対象に残す。今回の **true** は有限接続検収用で、今回READYとして引き渡した構成に新規Q3の対象を絞る。新候補を既存候補より優先する研究規則や、モデル最安更新を保証する設定ではない。

通常利用では、既存探索の `input_requests` を再び受け取り、同じExecutor・累積Q3予算で次の候補を準備する。`max_completed_per_method` / `max_attempted_per_method` を指定した場合はその有限上限で止まり、省略時はB2の既定探索範囲とQ3予算に従う。今回のROUNDは各方式1 READYに固定した。κ提案は絶対値を優先し、古いmultiplier座標にはB2が示すbase/sourceと座標を必須とする。曖昧な座標をκ=2へ置換しない。

## 選択済み要求を直接渡すAPI

```python
from trottertracks.pr_candidate_completion import Completion

completion = Completion(run_directory / "completion", settings)
record = completion.complete({
    "method": "PR_DF",                  # または PR_PAULI
    "selection": b2_input_request,       # B2の要求と選択理由を保持
    "design": {
        "L_D": selected_L_D,
        "formula": selected_formula,
        "delta_over_h": selected_delta_over_h,
        "kappa": selected_absolute_kappa,
        "K": selected_K,
    },
})
# designの代わりに configuration: canonical_body または固定locatorを渡せる。
# canonicalを渡す場合は、その構成固有の物理条件を使用する。

if record["status"] == "READY_FOR_SEARCH":
    routes = completion.install(context, problem, [record])
```

`Completion.install` は、26軸のexact価格、configuration/signal locator、total view/binding ID、候補固有のQ3 routeを既存contextへ追加する。返ったroute表を同じ探索executorへ渡す。初期shotは元B2規則の保存seed/V0を使い、その後の配分選択は元探索器が、その候補自身の信号・価格で行う。D2は配分生成やQ3採否の順を変更しない。

`complete` の要求に含むH/state/snapshot・精度契約はcanonical specと既存problem/route契約が正本。別Hや別精度への暗黙移送、別方式の信号・係数の代用は行わない。要求されていない科学取得を入口の存在だけで許可したものとは扱わない。

## 方式別の実行経路

|工程|PR-Pauli|PR-DF|
|---|---|---|
|構成|Aのon-demand入口。必要ならdescriptor/P0/P1。要求κを絶対値で照合|固定B mapper。未保存prefixのDF λ_R・s_Rだけを元係数helperで導出し、公開descriptorへ束縛|
|信号|A共通entryでround別保存照合、不足のみ取得、descriptor付きpacket組立|DFSignalEntryのREQUESTだけを固定providerへ。保存roundと新rawを単一carrierへstageしbundle化|
|intake|descriptor対応のB通常intake→total view|既存 `K.accept_saved`→total binding→元 `W.U.metadata`|
|価格|PriceRules→Aの実step別較正要求→取得/compile→元audit→候補固有registry→再照会|PriceRules→候補固有のfull-wrapper較正→exact fit→26価格凍結→独立holdout→registry→再照会|
|探索へ復帰|26軸・signal・routeが揃うとREADYとして同じcontextへ登録|同左|
|Q3/比較|元worker・AST guard・decoder/target/loss/τで評価、実際のshot×軸単価を保存|同左|

価格policyはMODEL。`DIRECT_FINITE_COHORT`、`MODEL_REFERENCE`、`NEEDS_PRICE_EVIDENCE`と元SOURCEを保持する。未取得はnull/NEEDSであり、0円・不適合・順位最下位にはしない。X/Yは同pathの別軸。κやΔの文字列表記、全event数、R=1の一致だけでは信号や価格を共有しない。

DFの現価格実装はS2/K2・単一piece・単一actual stepを対象にする。全R1ならm0/m6の各4 pathで `α+βq` を固定し、その後に独立m3の2 pathを記述比較。Rが変わる場合はm0/m3/m6の `[1,q,qR]` がfull-rankである場合だけ三係数を取得・固定し、独立m4を確認する。rank不足、複数actual step、変更公式/piece/Kの係数根拠不足は取得前の `df-price-design` でNEEDSとなる。LD7のβを別prefixやR>1に代用しない。

Pauliの複数actual stepはAの要求生成まで対応するが、現価格登録で複数groupを表現できない場合は `NEEDS_PRICE_RULE_MULTISTEP_REGISTRATION` を返す。保存較正があってもaudit/modelの固定locatorが足りない場合はそれを要求し、新pathで黙って置き換えない。Pauli LD7/8/9の既存P0数値gate停止は無断retryしない。

## SOURCE・registry・部分証拠と再開

`completion/STATE.json` は要求keyごとにselection・設計・各locator・READY/HELD理由を保存する。`PROGRESS.jsonl` と候補directoryの `*.STARTED.json`／`*.RETURN.json`／logは、実際の子process・終了code/signal・科学工程を識別する。完了stageは再利用し、STARTEDだけの曖昧状態や既失敗stageを再投入しない。価格HOLDに新しいregistry証拠が追加された場合は、metadata quoteだけ再照会してREADYへ復帰できる。独立した他候補は元V2の要求順に進める。

- 同一configuration/配分のQ3は元executorの同一job identity・`prior_run_dirs`で再利用する。
- 同一signalの再利用は方式固有のphysical keyで行い、DFではhead/tail・λ/scalar・law・実scheduleを含む。
- 取得済み軸・receipt・保存Q3は維持する。DF較正cellでは、4 indexを個別receiptで検証できる保存cohort/seedをINPUTS固定前にmetadata順で採用する。X済なら同じpathのYだけ取得し、PATH_INDEXが旧/新の元SOURCE/receiptを保持する。平均だけ、別cohortしかない軸、receipt欠落・不一致は該当軸を理由付きHOLDとし、独立した不足軸を進める。旧holdoutを新較正へ採用しない。
- 新raw、モデル凍結、holdout、Q3 rawを段階ごとにcommitし、実行code SHAと成果公開SHAを分ける。`completion/PRICE_RULES_REGISTRY.json` はSOURCE付きの候補固有規則を追加する所有registryで、旧正本や他担当worktreeを上書きしない。
- `resume_preparation_from` は技術SOURCEを変える新runで、旧STATEの **READY terminal証拠のみ** を取り込む。旧失敗・STOPは旧stateに残り、新SOURCEからの無断retry扱いにしない。

実DF6はrun01で、新descriptor1、信号13 round、較正＋独立holdoutの計20 compile、軸別fit2、26参照価格を取得しREADYになった。run02はその公開READYを取り込み、Pauli LD4の信号13 round・4 compile・audit・26価格を自動補完した。両者が元探索へ戻り、Q3各1件の適合と条件付き費用を保存した。通常C同期への反映も固定C commit `1d44d07397031547e402171faae268a640a2807b` で確認した。初期技術STOPとDFの保存artifact再照合はREPORTに分けて記録する。

## Cへの自動公開

元探索がQ3を保存した後、`publish_comparison_exports` が同じjobのREQUEST・CASE_INPUTS・exact OUTCOME・RESULT/API・FINAL_QUOTEを結び、既存C schemaの `C_RESULT_JOIN.json` を作る。`completion/C_EXPORTS.json` が各join locatorを列挙し、`C_SYNC_CHECK.json` で既存readerによる比較表読取を確認する。数値gate停止とQ3モデル不適合を区別し、有効な不適合結果も公開する。D2のreader検査をC自身による科学的受入や正式E[C]/CI/Gの資格と取り違えない。

## 固定依存view

|用途|固定SOURCE・配置|
|---|---|
|A候補/P0/P1・Pauli較正|`02e7cc6a127954fa7c446e7383d037f0617d50d3`を起点にした所有A workspace `d2-pauli-auto-evidence-20261116`。保存成果のcommitを個別に保持|
|Bのdescriptor通常intake|`33e324b43690caeaa353ecdb9e3d42f3139198b6`、`d2-auto-b-intake-33e324b`|
|DF共通信号entryと価格取得部品|`326feadd093920bbea29b498108aa4eae6f67a75`、`d2-pr-df-ld1-m3-acquisition-20261115`（読取専用）|
|DF数値signal provider|`fefd548ecd56f60d72031b3d385f6f5e6ee95d06`、`d2-pr-df-ld7-signal-source-fefd-20261116`|
|PriceRules|`c1f95e0806f5478a1ee5ce9ecb148ce9d10461aa`、`b2-route-view-d2-c1f95e080`。登録済み較正は元locatorで追加|
|DF Q3 route|`23160cc876807c0300be3de811f767b5810640a7`の `b2-route-view-d2-23160cc87`。元worker/kernel/ASTは継承|
|DF descriptor係数helper|`7f4c9a171a1d4caeaeaee6c443217edec761f4c9:scripts/h46_new2_core.py:matrix_only_tail_coefficients`|

viewは他担当の稼働中作業treeではなく、固定読取viewまたは今回の所有workspaceを使う。数値signal/Q3は既存Python 3.11.0rc1環境、compileは固定NumPy1.26.4/Qiskit1.3.0環境。guard・compiler・normalization・総Tでのscalar一回規則を変えず、未対応は具体的な理由付きで返す。
