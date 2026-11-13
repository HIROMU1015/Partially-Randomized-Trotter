# BのRPE総コスト評価へのlowering比較guard接続（2026-11-14）

Cの比較guard（`3a8b37f1a`の`price_implementation_guard.py`）を、Bの直接cost/compare/select経路に接続した。
範囲は技術実装と人工・回帰テストだけである。実H4/H6計算、保存価格の再解析、回路build/compile/sampling、新しいN・MSE・価格・G評価、比較policyの採択は行っていない。
B本線（`codex/rpe-decoder-statistics-20261113`、`ce0517de6`）とそのworktree・DP実行processには触れていない。

- 実装commit：`434458be2`（本体）、`540ea9e04`・`3b7b5b4f8`（テスト強化）。branch `codex/b-price-implementation-guard-20261114`（基点はI1/I2/E1修正済みの`3a27f99fe`）
- Bへの取込み手順：[HANDOFF](HANDOFF.md)、patch：[B_guard_connection.patch](B_guard_connection.patch)
- 証拠：[拒否動作](REJECTIONS.json)、[C経由とB直接経路の照合](CROSS_ROUTE.json)、[回帰テスト](REGRESSION.json)

## 1. Bの直接経路と、guardを迂回できた箇所

Bの`rpe_shot_allocation.py`と`pr_rpe_rmse_controller.py`は、Cがguardを検証した固定B `b9b9ffa6`とbyte一致である（B head `ce0517de6`で確認）。

| 入口 | 変更前の処理順 | 迂回できた点 |
|---|---|---|
| `rpe_shot_allocation.compare` | menu/契約 → `_price_metadata` → N packet束縛 → `B.assess` → `B.cost` → 点最小・区間 → `B.select`（1構成） | `_price_metadata`はextensionの自己整合とsidecarのidentity/law/price SOURCEしか見ない。旧Full-H汎用RZZの価格に新loweringのextensionを付ければ、`B.cost`まで進んでGが出る |
| `rpe_deterministic_risk.compare` | model MSE filter → 上のcompare（`**cost_options`を転送） | 同上 |
| `B.cost`（controller） | `B.assess` → exact算術 | 実loweringを一切検査しない |
| `B.select`（controller） | binding・h・RMSE・profile（compiler hash等）の共通性 → 最小値 | 同じcompiler profileなら、異なるlowering domainの評価を混ぜられる |
| `pr_rpe_rmse_evidence` → `B.select` | cost=Noneの評価だけ渡す | 数値選択がないので迂回点ではない |

C経由（`PriceProvider(comparison_policy=...)` → `I.compare` → `I.select`）は、Cの側ですでにguard済みである。

## 2. 実装した最小修正

Cのguardは複製・編集していない。B側に薄いbinding moduleを1つ追加し、`compare`に10行を足した。

- **`src/trottertracks/rpe_price_implementation_guard.py`（新規）**
  - Cのguardを固定commitのblobからそのまま実行する（関数のbytecodeがC commitと一致することをtestで確認）。
  - guardが依存するCの`baseline_signal_pipeline`だけを差し替える。差し替え先は、Bの算術を実行するlive controller、その`pf_pr_rpe_pipeline`、Cのguardと同じ固定D adapter（`e7dca5f7`）である。Cのguardのdocstringにある「B統合環境ではcontrollerを直接渡す」用法にあたる。
  - `check`：Cの`G.check`。
  - `cost`：Cの`G.cost`と同じ順序（check → 変更なしの`B.cost`）。結果にcheck記録を残す。
  - `select`：Cの`I.select`と同じ規則（全評価が同じ検査済みpolicyを持つこと、policyなしの科学costは拒否）を確認してから、変更なしの`B.select`を呼ぶ。
- **`rpe_shot_allocation.compare`（変更）**
  - 引数`root`と`comparison_policy`を追加した（省略可）。
  - policyがあれば、`_price_metadata`・`B.assess`・`B.cost`・`B.select`より前に`G.check`を実行する。不一致はValueErrorで止まる。
  - policyがなく、N packetが`SYNTHETIC_ONLY`以外なら、`MISSING_PRICE_COMPARISON_IMPLEMENTATION_POLICY`を理由にpendingとする（cost=None、`B.cost`は呼ばない）。
  - 結果とouter代表に`price_implementation_check`を記録する。
- **変更していないもの**
  - controller（`B.assess`・`B.cost`・`B.select`）の算術
  - N資格とerror contract、DIRECT/MODELの区別、期待価格の資格
  - canonical configuration ID
  - 元の価格record・SOURCE・receipt
  - `SYNTHETIC_ONLY`の既存呼出し：policyなしでも従来どおり動く。出力に`price_implementation_check: None`が増えるだけ。
- **既存recordに欄がない場合**：Cのsource-bound sidecar（`price_bindings`の`implementation`と`implementation_evidence`）をそのまま使う。元recordは変更しない。

## 3. 比較条件が一致しないときの拒否動作

[REJECTIONS.json](REJECTIONS.json)：B側の人工fixtureで31種類の変異を作り、`compare`とguard付き`cost`の2経路で確認した（62行）。これに、選択の混合と、policyなしの科学N packetの2行を加えた計64行である。すべて算術の前に止まり、`B.cost`・`B.select`の呼出しは0回だった。

| 指示書の項目 | 例 | 拒否理由（guardの文言） |
|---|---|---|
| 1 | 旧Full-H RZZの価格を新ZZのscenarioに入れる | `Comparison implementation differs: rte_rotation_lowering` |
| 1 | 1つのpolicyで旧lowering・新loweringを両方宣言する | `Mixed RTE lowering domains require separate policies` |
| 1 | 新PR（新policy）と旧Full-H（旧policy）を同じ選択に入れる | `One common SOURCE-bound implementation comparison policy required` |
| 2 | compiler profileは同一で、実loweringだけ異なる | `rte_rotation_lowering`の不一致 |
| 2 | 新ID・旧emitter | `RTE lowering contradicts emitter SOURCE` |
| 2 | extensionと固定assertionの両方が旧IDを主張し、実装事実は新ID | `Profile extension contradicts actual native/lowering` |
| 3 | implementationがない、未知のID、RTE・native SOURCEがない、SOURCEを解決できない、固定assertionが改変されている、locatorが固定commitでない、policyにSOURCEがない、価格producerが別 | 項目ごとの理由で拒否 |
| 4 | scalar（tail-only、2回適用、resolved≠expected）、wrapper、測定軸、method、物理候補（T・id・q・R_outer・method）、別roundのidentity | 項目ごとの理由で拒否 |
| 5 | Full PF・Truncated PFにRTE SOURCEを付ける／policyがRTE loweringを要求する | `Deterministic PF has no RTE lowering SOURCE`／`RTE applicability ... inconsistent` |

価格が欠測しているcellは、従来どおりUNKNOWNのままで、0価格にはならない（checkの対象は値のある3 cellだけ）。

## 4. 人工・回帰テスト

| 対象 | 結果 |
|---|---|
| 新規`tests/check_rpe_price_implementation_guard.py` | 9件PASS。PR、Full-H RTE、Full PF、Truncated PFの4方式で、項目1〜8と`rpe_deterministic_risk.compare`からの転送を確認 |
| 本branch（`3a27f99fe`＋patch） | 31 file／349件、全PASS。I1/I2/E1修正時のB回帰一式に、変更経路に触れるfileを加えた |
| B head `ce0517de6`＋patch | 29 file／326件。失敗は既存の5 file（I2のbyte/AST固定）だけで、patchの有無で失敗内容は同一。新規testはPASS |
| C `3a8b37f1a`のguard関連test | 4 file／51件PASS（Cのcodeは変更していない） |
| C経由とB直接経路の照合（[CROSS_ROUTE](CROSS_ROUTE.json)） | Cのcheckoutで、Cのfixtureを使った23 caseがすべて一致 |

C経由とB直接経路の照合の内容：

- 4方式すべてで、Gは同じ（1002、216）、check記録も同一。同じpolicyなら選択結果も同一。
- 拒否15件は、両経路とも同じ文言で止まり、`B.cost`は0回。
- 欠測価格は両経路ともUNKNOWNで、check記録は同一。policyなしの科学N packetは、両経路ともpending。
- 唯一の差は、C wrapperとB関数で省略時の`unexplored_scope`ラベルが異なることで、計算結果ではない。照合ではB側にC側と同じラベルを渡した（JSONの`aligned_scope_labels`）。

1回目の照合では、このラベル差のため5 caseが不一致と出た。原因を確認したうえでラベルを揃えて再実行した。また、Cの`check_rpe_optimization_pipeline`は、相対PYTHONPATHで子processのimportに失敗したため、絶対pathで再実行してPASSした。どちらも実行環境側の問題で、codeの不具合ではない。作者自身による確認であり、独立した査読ではない。

## 5. 未対応の経路と残るUNKNOWN

- **controllerを直接呼ぶ経路**：controllerの`B.cost`・`B.select`は、算術primitiveとして変更していない。科学packetでcontrollerを直接呼べば、今もguardを通らない。
  - 変更しなかった理由：controllerと`rpe_shot_allocation.py`は、Bの固定DP runner（`scripts/run_h4_dp_risk_fixed.py`）の保護対象である。controllerは他の固定SOURCEからもbyte照合されている。
  - 守られる入口：正式なG計算は、`compare(root=..., comparison_policy=...)`、guard付き`cost`、guard付き`select`を使う。
- **実sidecarは未作成**：実際の保存価格に対するsidecarと固定assertionは、まだない（Cも作っていない）。比較scenario（policy）も採択されていない。そのため、実データでpolicy付きのG計算に進める状態ではない。
- **guardが保証する範囲**：guardが確かめるのは、供給されたSOURCE付き実装証拠と比較条件が整合することである。producerが実際にそのemitterを実行したことは、receipt側の確認に依存する（Cの制約と同じ）。
- **controllerの版**：B binding（live controller）とC経由（固定B `437ba4acd`）のcontrollerは、現時点でbyte一致である。Bのcontrollerが今後変わったら、`cross_route_in_C_tree.py`で再照合する必要がある。
- **D adapterの版**：Full PF・Full-H RTEの物理照合には、固定D `e7dca5f7`を使う。B head上のlive D adapter（`0dc48c35`）には`physical_schedule`がない。
- **B headでの既存失敗**：B head単独では、I2の5 fileが失敗する。I1/I2/E1修正（`3a27f99fe`）とは独立に適用できるが、失敗を解消するには修正の取込みも必要である。

## 6. 保全

- 新しい科学計算、保存値の再解析、compile、sampling、N・G評価：0件。
- 隔離したscratch clone（push先は無効化）でpatch適用とtestを行い、作業後に削除した。
- B・C・Dのbranch、worktree、processは変更していない。
- merge、ref削除、prune、gc、履歴書換え：なし。
- 公開とremote照合の後はmandatory STOPとする。
