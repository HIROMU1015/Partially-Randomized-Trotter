# DF代表ラインへの選択統合実行計画（今回未実行）

移行元P1 `codex/p1-df-pipeline-h4-integration-20261027` @ `af54aeb7ebac042a3ca02379d61516ac635feef9`。
移行先 `all-r-coherent-opt2-reoptimization`、計画base/着手時remote tip `f29eafbf4412193e8e8660d9f051889c2cba684b`。
元LOCAL checkoutはこの公開baseとは別HEAD・dirtyである。既存checkout/HEAD/index/作業ファイルを切り替えず、新しい統合branchも作らない。
後続実装では本人所有の独立checkoutで、承認された代表branchへの限定差分commitを準備する。

## 採用範囲と除外

[21ファイル判定](ADOPTION_DECISION_MATRIX.csv)のADOPT_AS_IS2 + ADOPT_WITH_CHANGES12が対象。
P1全branch merge/cherry-pickを使わず、固定P1の該当file/関数だけを同checkoutへ選択適用する。
旧数値module・runnerのbody、旧SOURCE/raw/input/result/STOP/cache、CEM/validation manifestは変更しない。
`src/trotterlib/__init__.py`は移行先baseのbytesを維持する。
固定P1の履歴文書5と旧TEST_MANIFESTは固定GitHub URLへリンクし、新統合の実施記録として複写しない。

## 順序と具体的変更

1. **対象差分の限定再確認**：remote tipとbaseの一致を再照会する。進んでいれば21 pathと実core bindingの依存先・APIだけを比較し、機能重複/競合/研究条件の差を解決する。main/他研究branchを巻き戻さない。独立checkoutはobject alternates/外部worktree importに依存させない。
2. **研究入口とcomposer**：2 package入口、`pipeline.py`のcontext/StageResult/StageProviders/composeを取り込む。DF rank/partition必須と固定stage列はDF domain内に留める。mode拒否、binding、STOP/欠損、科学資格なしを保持する。
3. **限定H4 adapter**：FrozenH4Input/H4GeometryAdapterの現処理と未実行shots/wrapperを保持。次項のprepare policyの現値をadapterの明示設定として追加する。H4 B2/rank12/生成順/S2/q1/sector/距離allowlist/P=0/estimand/metricを共通coreへ移さない。固定SOURCE/input変換の科学妥当性は未検証と明記する。
4. **既存core接続を選択修正**：`bind_existing_core`はsignal_shots/signal_wrapperを明示callbackで受け、H4を内部から自動選択しない。H4 callerがadapterの既存callableを渡す。`prepare`が固定していたidentity_policy/coefficient_atol/diagonal_sort/basis_construction_policy/partition_policyは研究側が明示する小さな引数群へ出し、H4の現値は一切変えない。phase-RPE callbackはfixed-signal binding時に解決せず、明示した別estimandでのみ必要とする。新registry/frameworkは作らない。
5. **event/会計/compileの境界**：同じmake_rte_config、request型、build_step、compiler APIを維持。caller supplied eventsを保持し、sampling便利関数は使わない。fixed-signalとphase-RPEを別扱いにし、N/G/priceを補完しない。scientific_providersは常に拒否する。
6. **出力とCLI**：既存atomic writer再利用を維持し、依存解決のModuleNotFoundErrorを明確なPipelineErrorへ変換する。依存解決前にmkdirしない。CLIはその技術errorを整形し、旧eager packageに必要な依存不足を説明する。plan/no-output mockは軽量、optional出力は既存core stack依存。scientific modeはimport/出力より前に拒否する。
7. **fixture/test同期**：人工fixtureは現科学条件を模したshapeだけで、物理Hamiltonian/stateではない。新hook/policy引数だけ同期する。既存49ケース中、46ケースの論点を維持。root no-site import/export identity/atomic publicationの3ケースを、旧facade維持の前提で分離・再構成する。binding missing-hook、H4 kwargs、phase未接続、CLI依存拒否/無作成を追加する。
8. **最小案内・記録**：関連案内6の入口追記だけを調整する。global lazy成功文言を除き、固定P1の62 PASSを過去証拠としてリンクする。新統合の実装/差分test結果を新code SHA付きで最小記録に残す。科学synthesis/CEM/validation statusを実装だけで更新しない。
9. **通常commit/push**：代表branchを更新する明示許可の後に、許可範囲のpathだけstage/commitする。remoteが進んでいれば上書き/force/rebaseせず新tipから限定差分を作り直す。通常fast-forward push、local/remote SHA・公開内容を確認してSTOP。

## 技術受入条件

- 元initializer/旧numerical module/旧runner/科学artifactの差分0。trotterlibからH4やPauliへの逆依存を増やさない。
- 同じcheckoutだけでplan→adapter検査→record、人工入力→全stage→recordが成立する。旧package optional stack不足でもno-output mockを利用できる。
- optional出力が使えない場合は理由付き拒否・directory未作成。利用可能なときは既存atomic writer、衝突拒否、元出力不変。完全依存環境での後者は今回未確認であり後続で実測する。
- context/SOURCE/cohort/input/compiler/metricの異なる出力を拒否、missing/STOP/failed/UNQUALIFIEDを保持、retryなし、実科学qualificationなし。
- scientific/real/resume等はinput/provider/出力より先に拒否。旧STOPを解除しない。
- 49新規ケースを盲目的に新合計へ転記しない。既存13ケース（整数6・executor人工7）は実行可能な依存環境の限定sliceで確認し、未実施なら未確認とする。resource capの旧testを新経路の資源政策に転用しない。
- 追加import/hook/policy/error差分を検証し、実施PASS/FAIL/SKIPを新コード版で記録する。whole science suiteは実行しない。
- main/他研究branch/dirty元checkoutに変更なし、固定refとSOURCE/result/STOP参照を保持、B3アクセス0。

## Rollback

機能単位の小さな選択差分commitにし、元P1公開refと統合前baseを保持する。旧runner/科学出力は新経路と独立なので、問題時は新入口を使用停止し旧固定runnerをそのまま参照する。
未公開の独立作業を修正してから公開し、公開後の戻しが必要なら対象差分を戻す通常commitを別指示の下で作る。reset --hard/force push/branch削除/科学SOURCE置換をrollbackと呼ばない。
元dirty checkoutにはrollback操作自体を行わない。

## 別承認が必要な後続

上記は技術移植だけ。real saved-input loader/identityと科学provider、state/sector/bit順、scalar/relative phase、generated-orderとcore partitionの数値同等性、sampling/mean/qualified shots/G、実build/compile、q>1/RPE全体は含めない。
必要ならユーザーがSOURCE/input/cohort/条件/call範囲/STOPを定め別途科学実行を許可する。新しい承認証明やgrant発行を前提にしない。
P2コード実装も別の明示範囲。今回この計画による移植commitは0。
