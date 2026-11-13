# B/C/Aへの長時間価格HANDOFF

2026-11-13 JST。[REPORT](REPORT.md)／[評価・資格](LONG_TIME_PRICE_EVALUATION_PLAN.md)／[有限科学案](FINITE_PRICE_SCIENCE_PLAN.md)。本成果は設計と人工API差分確認である。新しい値付きpath/mean/区間を供給していない。他担当への外部メッセージ送信や作業変更もしていない。

## B：価格を必要とする比較契約

BはA固定configurationの全round identity/law、同target/decoder/claim、適格なN又はN集合、競合する配分/候補、比較margin/許容regret、価格側の同時confidence・familyを渡す。現A91349のraw signalを読むことと、十分Nを供給することは別である。DはN/decoder、候補生成、risk qualificationを再実装しない。

B23dの「m10〜m12約96.92%、m12約66.20%」は固定nominal×ideal E[S]の既存診断である。これから正式Gの順位・n・精度を固定しない。大きな価格差が粗い有効区間で解決できれば全セルを高精度化せず、候補の適格下界が実行可能incumbentの有限上界を超えるときだけ除外する。正重みの未資格セルはunknown upperで残す。

## D：選択requestと取得結果

52軸のcoverageと26 selectorはD748の既存成果を使い、再生成しない。後続の明示された有限scopeでだけ選択metadataをmaterializeする。

```python
from trottertracks import ld12_price_requests as L
req = L.request_for(root, label, m, axes=selected_axes,
                    execution_source=fixed_new_execution_source)
# この関数は受動metadata。科学取得は別途承認されたRuntime/Adapterで行う。
```

新execution_sourceでcontext/request IDは変わる。歴史的publicationのIDへ結果を付け替えない。B canonical configuration/identityとA physical IDは別のまま保持する。LD1/LD2のprefix別input/tail selectorを同じRuntimeへ上書きせず、λ/scalar・deterministic/randomized fragment順・component/basis/preparationを実contextで照合する。q_RPEとq_PF、signed piece/timeのbinary64値とallocation、K、full X/Y/measurement、lowering/profileを維持する。

| 出力 | D/C/Bでの扱い |
|---|---|
| `PR_CANDIDATE_PAIRED_PATH_PRICE` | 既存AdapterのCANDIDATE_PATH。軸別S/C/D/6指標・fingerprint/cache、event digest/seed/law、resolved context、SOURCE、conditional/cohort/index。mean/CI=null、qualified_expectation=false |
| finite-cohort mean | 別の明示producerが作る外部record。nとrole・law・不確かさを保持。Bの明示finite-cohort scenarioに限る条件付き診断で、全law Gへ昇格しない |
| 適格mean | `FULL_WRAPPER_MEAN_COMPILED_PRICE`、元Bのmeasurement/axis/units、identity/law_binding/cohort/n/source、mean.rz_count、classification/expectation、model/domain、law/rare/numerical/model不確かさ |
| mean interval sidecar | `FULL_WRAPPER_MEAN_COMPILED_RZ_INTERVAL`、exact lower/upper、domain=元row.candidate、根拠SOURCE/qualification/evidence_kind、全使用familyのcoverage、親mean SOURCEとのbinding |
| 不足・未資格 | 既存D.pending_price_recordsを保持。S/events/局所window/未検証modelの値でmean又は厳密区間を埋めない |

X/Yは同pathの二応答で、標本数を倍にせず共分散を保持する。normal/replay/forcedのrole、target ideal P対sampler Q、price estimator法、range/CI仮定、系統項の未評価nullを引き継ぐ。normal全law CIに既に含まれるrareを別途二重加算しない。未知law/rare/model影響を0とするrecordは適格meanの根拠にならない。

## C5ccとD748のAPI差分

比較したCは `5cc7c07e112ce97205948465c94441cba0042416`、D旧版は `0c6e16342b3c804fa0bd89a4028d6b5bea7b0322`、D最新は `748be6875b5007707d864e16aa7a4fc4250f3714`。CのREPORT/PRICE_INTEGRATION_HANDOFF/provider/bindingを読み、既存C統合テスト・52軸展開を再実行しなかった。

| 箇所 | 現在の動作・確認 | Cが後続で必要とする最小差分 |
|---|---|---|
| provider固定読込 | C `D_ADAPTER_SOURCE=0c6...`、4 moduleをprivate namespaceへロード | 新D adapterを使うときは対応SOURCEを748又は後続の選択commitへ固定。SOURCE extensionも実際の参照に整合させる。branch丸ごとmerge不要 |
| `from_b/from_a_job` | 新optional `expected_scalar`。省略時の旧request内容は同じ（人工照合PASS）。旧D0c6へkeywordを渡す機能はない | scalarを使う呼出しとロードSOURCEを一緒に更新。B canonical identity/lawを再定義しない |
| C `pending_requests` | preparation_hashのみ転送、scalar指定なし | A固定MENU/tail出典からtrusted scalarを受け取る最小引数又は限定selectorを追加。外部の任意scalarをsource確認なしに採用しない |
| C `_requests` 再構成 | `expected_scalar`を落とし、新D requestを不一致として拒否 | constructorで同じscalarを渡し、A固定scalar/input selectorに一致することを検査。keyword転送だけでは科学的正しさは認定できない |
| C schedule anchor | binding/split/input/execution/implementation/profile/preparation_hashを比較。expected_scalarを含まない | optional scalarをanchorへ含め、一schedule中の混在を拒否。未指定の旧経路は未指定のまま保全 |
| C `_paths` | resolved scalarは必須かつpath間一致を要求するが、requestのexpected scalarとは比較しない | expected scalarありの場合、resolved scalarと元D同様のexact binary64値を照合。不一致を拒否し、丸め直ししない |
| `ld12_price_requests.request_for` | 固定A input/tail selectorとexpected scalarを一括設定 | 外部Dで選んだrequestをCへ渡す場合もC自身の再構成と同じinput/execution/scalarにする。Cの別context表現は維持 |
| context/cache | scalar指定でD context/request keyが変わる。B identity/lawは不変 | keyを手修正せず元Dで生成。実cacheはwrapper全内容＋compiler。異候補間のevent数一致では共有しない |

上表はHANDOFF仕様で、C production codeをこのbranchで変更していない。Cのnamespaceには科学Runtime依存graphが揃っていないという既存境界も維持する。Cで新solver/価格runnerを追加せず、Dから供給された結果・外部meanだけをmappingする。

### 今回の限定人工確認

[`tests/test_long_time_price_binding_design.py`](../../../tests/test_long_time_price_binding_design.py) はC5ccの `_requests/_paths` 関数をGit objectからそのまま抜き、架空H/state/scheduleのmetadataだけで5件確認した。Runtime/Adapter構築は禁止した。

1. scalar省略のD request内容は0c6と一致し、C再構成可能。
2. scalar指定はB identity/lawを変えずD keyを変える。
3. 現C再構成はscalar付きrequestを拒否。テスト内だけのkeyword転送でAPI一致する。
4. その転送だけではround間scalar混在を拒否しない。
5. 現C path bindingはrequest scalarと異なるresolved scalarを照合しない。

4/5は**残る接続差分を再現するPASS**で、修正済みという意味ではない。C full integration・長時間性能・実preparation・平均資格を検証した結果ではなく、作者自身の技術確認である。元Cの8＋36件を重複して再実行しない。

## mean/intervalを既存B whole-shot入口へ渡す条件

Cの既存 `bind_results` は値を平均せず、外部producerが供給したmean/sidecarを元requestに対応付ける。価格profileにはcompilerだけでなく新ZZ/control/full measured wrapper/P0を含める。B configurationの歴史的profileへloweringを後付けして書き換えず、元のextension方式を使う。

各bindingには元Bのidentity/law_binding/axis/cohort、`price_source=親mean.source`、binding自体のsource、profile_extension、mean_intervalに加え、Cのphysical_context、Dの `d_request_id/d_context_key` を供給する。path由来のresolved preparation/tail/scalar・SOURCEと一致しなければ拒否する。bindingが整うことと科学的なcoverageが正しいことは別である。

B b9b9の既存interval contractは、configuration_id、source、simultaneous=true、scope=`ALL_SELECTED_ROUNDS_AXES`と各intervalの同じcoverageを照合する。候補二種・複数N配分を比較するなら、その選択family全体も覆う外部の同時契約が必要である。configurationごとのpointwise flagだけを比較全体のcoverageとしない。数値端点は既存exact表現を使い、未知上端nullを0に変換しない。

既存Bはmean record＋interval sidecarを基本とする。interval-only/部分未知による候補除外の数式はD24b0にあるが、Bの別候補全体を扱う新bound-only controllerが完成したとは主張しない。必要時だけBで最小接続し、midpointの架空meanで既存会計を通さない。meanが供給されてもN/RMSE資格が不足なら正式GはUNKNOWN。

## 次の担当とSTOP

**ユーザー**：有限T案を選ぶかB入力を待つかを判断し、科学実行する場合は具体scopeを明示承認する。**B/A**：同claimの必要N/集合・law・比較精度/familyを提示。**C**：上記scalar/source差分を必要になった時点で小さく接続し、必要なcandidateのS適合性を確認。**D**：その後、承認された有限requestだけを既存pipelineで取得する。

これらは後続提案であり未実行。今回の計画・人工確認をcommit/pushし、remote照合後 **mandatory STOP**。新しい実H4・fit・価格推定・Gへ進まない。
