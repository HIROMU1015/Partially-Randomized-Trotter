# Lazy importの互換性評価

比較はbase `f29eafbf4412193e8e8660d9f051889c2cba684b` とP1 `af54aeb7ebac042a3ca02379d61516ac635feef9` の `src/trotterlib/__init__.py`。
**最終推奨：選択肢4。今回の選択統合では旧eager初期化を維持し、P1のglobal lazy差分を除外する。**
pipelineの採用をこの全研究共通package変更に従属させない。将来lazy化を独立に検討する余地は残す。

## 静的一致と適用限界

|項目|確認結果|
|---|---|
|`__all__`|413件、名称/重複なし/順序が同一|
|export target|413件すべて同じprovider module/attribute|
|旧eager import|48 ImportFrom、47固有provider module|
|数値moduleと旧runner|P1固定差分による既存本体変更0|
|局所依存graph|47 providerから辿れる65 core moduleをAST確認|
|core→研究package / root exportへの逆import|この65 moduleの静的Import/ImportFromに検出なし|
|循環の可能性|df_rte_circuit↔df_rte_tailの構文上SCC1。逆向きはTYPE_CHECKING/関数内で、実import cycleの証明ではない|

graphは条件付/関数内importも含む保守的検査。dynamic import、外部ライブラリ、全研究consumerを網羅するものではない。
旧root initializerはImportFromと`__all__`中心だが、providerの副作用を実行していた。例えば同じbaseの
`chemistry_hamiltonian.py:17`はNUMBA_THREADING_LAYERの既定設定、`df_gpu_statevector.py:88–90`はCUDA library path準備/preloadを持つ。
configもmodule読込時に環境設定を参照する。これらは**source静的確認のみ**で実行していない。具体的環境値・内部保存先は記録しない。
lazy化でこれらの発生が後へ移り、別importや外部SDK初期化との順序が変わり得る。

## 技術動作の比較

|操作|旧eager|P1 lazy|判定の範囲|
|---|---|---|---|
|`import trotterlib`|siteなしでnumpy不足、task venvではmatplotlib不足|どちらもroot読込成功、optional importなし|実source/隔離process|
|`from trotterlib import X` / `trotterlib.X`|root import時に全providerが必要|Xのproviderを必要時解決し同attributeをcache|413件人工、rte代表3件実module|
|`from trotterlib import *`|root開始で全provider依存|`__all__`順に全exportへアクセスし全provider依存|人工413件成功、実task環境はopenfermion不足|
|`dir()` / `vars()`|exportは既にglobals|dirには未取得exportが出るがvarsには未存在|人工provider|
|未定義属性|AttributeError|AttributeError（KeyErrorを非表示）|人工provider、P1過去test|
|未定義`from`名|ImportError|ImportError|人工provider、存在しないsubmodule fixtureは生成しない|
|欠損provider|root importで失敗|取得時に元ModuleNotFoundError、失敗値はcacheされない|人工/実代表|
|直接submodule→root取得|読込時にrootの全依存が必要|submoduleの同属性へ解決|人工、rte実代表|
|monkeypatchとreload|再reloadでproviderからexportを再束縛|既にglobalsにcacheしたexportはreload後も残る|人工provider、互換差を検出|
|自己exportへの再入|循環import条件に依存|未完成providerの属性取得でAttributeErrorになり得る|人工反例。実cycleが発生したとの主張ではない|

`__getattr__`はmappingを引き、`import_module('.'+module,__name__)`、`getattr`、globals cacheの順。
未知名だけAttributeErrorへ変え、依存不足やprovider側errorをfake fallbackへ置換しない。
exportのcacheはmodule monkeypatchへの自動追随ではない（旧eagerも通常取得後は追随しない）。
**reloadで旧eagerが再束縛しlazyは残す差**を今回新たに確認した。
`__all__`一致はimport副作用、providerの実identity全件、旧runner互換性、科学数値同等性の証明ではない。

## Pipelineを採用する際の依存境界

新しいtrottertracksのimport/planとno-output syntheticは、旧facadeを使う一時fixtureでも成立した。
ただし結果保存は `trotterlib.parallel_validation_executor.atomic_write_json` 経由で旧rootを初期化する。
原facade + task venvではmatplotlib不足で停止し、directoryは作られなかった。
別のpackage名でsourceを動的実行してこの依存を迂回することは統合方針にしない。

今回の選択は、旧initializerを保持し、optional outputの依存不足をPipelineError/明示CLI拒否にすること。
writerが利用可能な既存依存環境では従来atomic writerを再利用する。最低環境ではstdoutでplan/mock recordを取得できる。
lightweight outputまで独立化する将来要件が生じたら、既存writerの配置/依存を限定設計する。今回は新helper整理やglobal facade修正へ拡張しない。

## 未確認のまま残す事項

実optional stack全体、全47 providerの実動的identity、Pauli/Track B/B2-D/Research C/旧runnerのimport順序、
CUDA preload/NUMBA設定とSDKの相互作用、科学関数/回路/数値同等性は未検証。
このため選択肢1/2による全研究共通変更の採用を推奨しない。選択肢3の部分lazy化も根拠不足で今回は行わない。
新規19件の内訳と再現fixture範囲は[技術test](TECHNICAL_TEST_RESULTS.md)。
