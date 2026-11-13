# H4：4方式・最大20セルのsignal screen実行準備

2026-11-13 JST。**Bの暫定11案に対する新20セル・保存40セルのSOURCE対応と数値入口を準備した。実H4の新mean、保存数値の再解析・比較判定、N/RMSE/価格/Gは0。** menuの正式採択や候補の優劣は決めていない。

[固定PLAN](PLAN.json)／[有限科学計画](SCREEN_PLAN.md)／[B向けHANDOFF](HANDOFF.md)／[人工テスト](TESTS.txt)／[既存回帰](REGRESSION.txt)。関連A branch `codex/pr-r-dependent-n-model-20261111`を再利用し、B/C/Dのbranch/worktree/processへ変更していない。

## 既存成果の再利用

| SOURCE | 再利用内容と今回の差分 |
|---|---|
| B `51500b724a75f9535a2c831d50bcfe09eabb9326` | MINIMAL_COMPARISON_MENU/NEXT_STEPS/BASELINE_EVIDENCE_MATRIX/EVIDENCE_METADATA。11案・明示q/Rをそのまま使用。新menuや候補管理器を作らない |
| D `e7dca5f7301c8633b17a59b6a7161331303aef2d` | Full PF/Full-H V2接続と保存receiptを確認。今回のFull PF12＋Full-H6のphysical candidateがD確定physical_scheduleと一致することをmetadataテストで照合 |
| 長時間 `155529e20a8fac39e00786c40b57830b71203c4f` | Full PF m3/m4計12セル、B0I m4の保存signalとscientific_provenanceのH/state bindingを照合。数値gate再判定・bias比較なし |
| PF common `0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c` | B0I/p7/Y4/T.8/q2の保存signal。m3のB0も旧SOURCEを保持 |
| B0 batch `89c91c692e1d35de94d2e7afbe890aabe6d80b7e` | Bのmetadataが示す全13signal locatorを保全。旧HP型Nを新RMSEへ移さない |
| PR `169d3316dd008eb98dd9be56fe4ecefc5c1a7d60`／契約 `bba150834ca30bdb186fbfd58c55a8d51177b855` | 全13round保存値とm3/m6/m12の独立reference pair。旧S0/m1/新10roundの来歴を保つ |
| Full-H `9b83b96b83f62142c9d87858dc24a94db7ac5d61` | prepare_full_h/full_oracleを調査し、公開input_and_validation_summaryからone-body込みdescriptor・λ・scalarをcopy。今回preparationを再生成しない |
| Full-H T2 `7a8bc7f132e166390c943e6b32c92cc6abe1baf4` | evaluateのs=qRによるmean共有、既存action検証を確認。旧T2 mean/価格を新dyadic候補へ流用しない |
| A science `c76ffe5dbb2a8dd1a05f1a85038f2d48b959f73e` | 保存H/state loader、spectral polynomial kernel、S0由来R-aware allowance。再利用する5数値fileが不変と確認 |
| A `97146481f7b05e9418d8fc016f133c7b78d93d08`／B S3 `2084791b1b09cff65e07fa8566bb734e1e6bff8e` | 局所screenと全energy RMSE資格の区別。旧S3の再解析や反復追加は行わない |

Full-H descriptorの保存値は368 nonidentity component、うちone-body8、deterministic blocks=0、全DF fragment0…9。λR=21.02794414280327、extracted scalar=−2.6245790607368162、constant=2.29310124732、preparation hash=`0e3989e739d0e17a5d7af0fe809bd7c10ee765f09c934fe6ebb1f0e8b5c1046d`。これらは保存値の転記で、PR tailのλRやDF B3へ置換していない。

## 準備した入口

- `h4_four_method_screen.plan`：pinned Git JSONを読み、20 physical ID・signed weights・保存locator・referenceを固定。gzip carrierは1回だけ読み、同じ物理候補の重複を排除する。H/state array・normalizer・signalは評価しない。
- `h4_four_method_screen_mean`：新20セルだけの明示的な将来の数値leaf。既存schedule/spectrum/polynomialとNumPy matrix_powerを使用。既存scalar/vector/RMSE/shot solverや旧runnerは変更しない。
- `h4_four_method_screen.execute`：承認後の20セルbatch用関数。fresh出力と科学SOURCEを照合し、SOURCE/binding/numerical gate失敗時は部分保存してSTOP、retryなし。未開始cellは止め、既に動作している有限workerは結果を保存する。
- `prepare_h4_four_method_screen.py`：**metadata-only CLI**。scientific execute flagはない。実科学実行の起動を準備CLIに混ぜない。承認後は固定SOURCEのexecute関数を一括呼出しでき、新grant・承認証明書は不要。

Full PFとB0Iの1 macro-stepを元signed順で作り、q乗はmatrix_powerへ接続。m12でもq×piece×blockの逐次展開を避ける。Full-H RTEはno-headなので、既存grouped polynomialのR乗・q乗を別々のcorrected/raw列で評価する。rawをcorrected signal/Bと事後定義しない。**q/Rは回路identityとして保持し、s一致を価格同値性へ使わない。** 新しいsampling pipeline・明示event列・回路build/compileは不要。

イベント時間は元の`(T/q)/R`を保持し、qが26/1639など非dyadicのときに演算順の異なるT/(qR)へ黙って置換しない。PFの負時間・fragment順・scalar一回会計、B0Iだけのtail identity補正も保持する。

## 数値証拠の水準

既存reference pairを3時点とも利用できるため**新referenceは0 action**。Full-Hのexact-tail値にはこの保存full-H referenceを用い、新しいexact-tail reference vectorを伝播しない。そのnorm観測はnullと明示する。旧PR用のexact-tail norm gateへ架空の0を渡さず、R-aware式の適合する算術部分だけを再利用した。PFには新powerのnorm/unitarity gate、Full-HにはHermitian spectral再構成・raw/corrected独立closure・normalizer精度のgateを適用する。

u/logB marginは従来と同じ経験的式で、Full-Hのhead無しに対しては旧logical-stage数が保守的にhead項も含む。実乗算数やrigorous誤差上界ではない。power後のnormは全eventの観測最大値とは呼ばず、boundary観測として保存する。m3/m4の既存観測norm欠測も埋めない。局所phaseは信号又はreferenceが0ならnull。局所信号の良否を全RMSEのPASS/FAILへ変換しない。

## 技術検証と完了

新9件：metadata20/40・reference/one-body、D確定18候補一致、元signed PF/B0I伝播との人工照合、元no-head RTEとのK2/K4人工照合、m12を含む人工20セル、非有限/normalizer精度/不正B3・scope拒否、mock gate失敗時の部分STOP、metadata-only CLIを確認。旧物理packet8件も回帰。**計17件の技術test PASS**。既存依存fileのinvalid-escape DeprecationWarningは旧SOURCE由来で、科学的結果ではない。

実Hamiltonianの長時間性能やscientific mean同値性を新たに検証したという意味ではない。全H RTEの実sampling law・原論文finite-K移送、ground同定、全事象loss/population RMSEは未資格。価格loweringやD P2には介入せず、価格recordを新規作成していない。

次：**ユーザー**がSCREEN_PLANの新20セル・local診断だけの有限実行を別途承認する。**A**は承認後だけ固定SOURCEで実行し、**B**はphysical identityを採択後のV2 scheduleへ結び、必要な候補選定・全13round案を返す。今回commit/push・remote SHA/tree/主要成果物照合後mandatory STOP。
