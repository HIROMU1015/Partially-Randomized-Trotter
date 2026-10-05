以下をGPUサーバー側Codexへ渡してください。

---

# Track A H4 geometry拡張：契約v2の作成・検査・公開のみ

## 今回の依頼

公開済み契約v1を基に、未固定4項目と認可順序を具体化した、レビュー用の契約v2を作成してください。

今回は契約・schema・zero-compute plan・合成検査・commit/pushまでです。source port、分子入力生成、科学計算、実行authorizationの発行は認可しません。

契約v2の公開後、このチャットでレビューしてから次段階を判断します。

## 固定起点

- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 起点branch：`track-a-h4-geometry-contract-20261006`
- 正確な起点commit：`7c1a3d43f61c5501a9e79206b7c60933f94b1077`
- 準備commit：`c2ab34fed49bb1fb104d39fe83b36858a2c92c2a`
- Handoff commit：`2a80f1d5d5e5734e51d970b2b6822cd2543fd596`

契約v1の資料入口：

```text
artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06/README.md
```

起点のidentity：

```text
zero_compute_plan_v1.json SHA-256:
805874dcbe1466adbd92f2300b90a88a2e111e8d43543759fae4b28a9eb1d758

plan fingerprint:
09ce44081a5a20f2e1f3dba2fced3bf8d8732251ff88393e8a190d0fa820bb88

公開版 artifact_manifest_v1.json SHA-256:
68cc9c28faaf6ab1b43799d5ffc18ece633d244634f5146239dbc93770c530f7
```

公開前報告のmanifest hash `145ee4fa…` は今回の起点identityに使わないでください。

最初にAGENTS.md、PROJECT_MAP.md、研究概要、契約v1、review_decisions、plan、監査資料を読み、起点commitとmanifest記載33ファイルを照合してください。

## 作業場所と旧証拠の保存

起点commitから専用branchを作成してください。推奨名：

```text
track-a-h4-geometry-contract-v2-20261006
```

既存branch/worktreeを上書きせず、dirtyな変更をreset・clean・stashしません。科学入力をmaterializeしない、現在のsparse/source-only運用を維持してください。

旧v1の契約・schema・plan・検査記録・manifestはbyte-identicalで保存し、v2は別ファイルとして追加してください。更新する索引はv2 manifestへ収録します。旧v1 manifestは起点commitのblobを対象に照合し、索引更新後のHEADとの不一致を隠すために書き換えません。

旧247 source、公開準備25ファイル、保存6 JSON、原稿・図・旧結果・Track Bは変更しません。

## 変更しない科学的scope

- H4 linear、STO-3G、requested/actual DF rank12。
- 追加距離：`0.70 / 0.80 / 0.90 / 1.10 / 1.40 / 1.60 Å`。
- 8 system qubits＋ancilla **1個**、ancilla index 8、合計9 qubits。
- `T=0.8`、二次DF-prefix PF、canonical finite-RTE、`delta=T/q`。
- 各距離で同じ218 template：B0 20、B1 4、B2 145、B3 49。
- random各32 trajectory、cosine/sineは同じtrajectoryを共有。
- 1点12,464、6点合計74,784 logical wrappers。actual science transpile invocationも最大74,784。
- 最大12 spawned CPU workers、内部process/thread各1。
- accuracy-ineligibleも監査に保持し、候補の追加・除外・置換はしない。
- 旧1.00 Å・旧1.30 Åの証拠は別identity layerとして保存する。

## 具体化する4項目

次の設定はレビュー用の採用案として明記してください。今回の指示を、科学実行条件の最終承認とは解釈しません。

### D1：SCF/DF生成規約

既存sourceとinstalled package sourceを静的に確認し、以下を具体的な値・規約で記載してください。

- RHF、charge 0、multiplicity 1。
- 初期guess、DIISの通常使用設定、収束判定、最大反復数。
- `conv_tol=1e-9`、`max_cycle=50`という既存提案との整合。
- 座標・単位、積分、MO ordering/phase、spin-orbital ordering、DF生成規約。
- rank12の取得条件、SCF非収束・rank不足時のSTOP。
- 結果を見てNewton/restart/別guess等で救済しない規則。

「通常のDIIS使用」と「不収束後のDIIS設定変更による救済」を区別してください。未確認のdefaultを固定済みと呼ばないでください。

科学的な選択が残る場合は、根拠と選択肢を記載して未解決のまま停止します。分子計算で設定を試すことは禁止です。

### D2：ordering・solver・numerical gates

- 旧S0/M1のDF生成順序を継承し、汎用関数によるFrobenius再ソートを導入しない。
- weight tie、DF eigenvectorの符号・縮退、sector basis orderingの規約を明示する。
- `Nα=Nβ=2`、sector dimension 36、dense Hermitian eigensolverという提案を具体化する。
- 参照状態のphase固定と、縮退・gate不合格時のSTOPを明示する。

既存の閾値案は変更理由がなければ維持します。

```text
state norm absolute error <= 1e-12
reference residual L2 <= 1e-9 Ha
relative Hermiticity error <= 1e-12
imaginary energy magnitude <= 1e-11 Ha
minimum sector gap <= 1e-10 HaならSTOP
```

縮退規約を、結果後に都合のよいstateやprefixを選べる形にしないでください。

### D3：master seed

採用案を `20261006` として明記してください。

既存のdomain separation、axis共有、step/occurrence別stream、重複seed時STOPを維持します。実source/input identityが未固定なので、今回実trajectory seedを生成したり、placeholder hashで科学seedを固定したりしません。

### D4：memory・wall・output

既存提案のworker 8 GiB、driver 8 GiBについて、AS制限とRSS監視を区別してください。

最大12 workerの枠はdriver込み104 GiBです。「available 64 GiBなら無条件に12 worker開始」と読める規則をなくしてください。

以下の形の開始条件を機械判定可能にします。

```text
required_available(w)
  = driver_budget + w * worker_budget + fixed_host_headroom

w <= 12
w <= 利用を許されたCPU数
available_memory >= required_available(w)
```

host headroom、memoryの取得方法、worker削減方法、最小workerでも不足した場合のSTOPを明示してください。起動後のmemory pressure時も、無断のretry/resumeにせずSTOPしてレビューへ戻します。

- wall停止上限案：72時間。
- output上限案：10 GiB。
- fixed run ID、absolute project root、absolute output rootを具体的な提案として記載する。
- 今回、その科学output directoryやregistryは作成しない。
- 観測available memoryを予約済み資源と呼ばない。
- 他ユーザーのjob、priority、affinity、共有cgroup設定を変更しない。

## 認可順序を二段階に分離する

文書とmachine-readable plan/schemaで、以下を一致させてください。

```text
契約v2レビュー・条件承認
  ↓
別指示で新science source/runner/testsを実装
  ↓
actual source commit固定
  ↓
入力生成専用plan・別authorization・review・明示launch
  ↓
承認された6距離の入力生成のみ
  ↓
H/DF/state/sector/order/coordinate bytesをfreezeしてSTOP
  ↓
input-bound signal/compile planをseal
  ↓
別のresult-prior authorization・最終review・明示launch
  ↓
signal/compile/resource map
  ↓
mandatory STOP
```

今回はこの順序を契約として記述するだけです。どちらのauthorizationも発行せず、入力生成も実行しません。

認可前にactual signal/costを取得してsemantic gateを通す、という解釈を排除してください。source実装段のsynthetic semantic testsと、将来の科学実行gateを区別します。

## 許可する検査と成果物

純synthetic JSON・一時的な架空ledger・source textによる検査だけを追加してください。

少なくとも以下を検査します。

- 6距離・218 template・74,784上限の不変性。
- 認可前の入力生成／signal／compileを拒否する段階規則。
- memory admissionとworker削減の境界。
- null input hashのままinput-bound sealedとする改変の拒否。
- v1のidentity/cache/owner/digest拒否規則の維持。

実Qiskit transpile、分子source import、全repository testsは実行しません。旧129件は旧検査記録として保存し、今回の再実行件数と混同しません。

成果物は、v2契約、schema、zero-compute plan、decision案、合成tests/log、manifest、v1→v2差分説明、レビュー依頼を揃えてください。索引・研究概要・当日の研究ノートも必要な範囲だけ更新します。

状態は例えば次にしてください。

```text
H4_GEOMETRY_CONTRACT_V2_PREPARED_AWAITING_REVIEW_SCIENCE_NOT_AUTHORIZED

science_execution_authorized=false
source_port_authorized=false
input_generation_authorized=false
execution_plan_sealed=false
next_stage_authorized=false
automatic_research_decision_authorized=false
research_decision=null
mandatory_stop=true
```

未解決項目が残った場合は、残った項目を明示し、契約完成を宣言しません。

## Commit・push・報告・停止

軽量な今回の変更だけを明示的にstageし、専用branchへcommit・non-force pushしてください。remoteが `HIROMU1015/Partially-Randomized-Trotter` であることを確認します。

分子snapshot、NPZ/NPY、pickle、matrix/state/circuit、runtime/cache/registryをcommitしません。認証・共有環境を変更せず、pushできなければその理由を報告します。

最終報告には以下を示してください。

- branch・起点commit・公開commit。
- v2資料入口、plan/manifest hash。
- D1〜D4の具体設定案と未解決事項。
- memory admission式と認可順序。
- 今回のtest件数・fail/skip。
- 旧証拠/sourceの不変性。
- 分子アクセス・生成、signal、sampling、build/compile/transpile、GPU、環境変更、他job変更がすべて0であること。

公開後に停止してください。source port、本計算、authorization発行、Track B、追加距離・trajectoryへ進まないでください。
