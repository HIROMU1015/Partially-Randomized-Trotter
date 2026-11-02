# D3 v2：精度型変換の限定修正・実行準備

日付：2026-10-14 JST。**READY_FOR_SEPARATE_D3_V2_ONE_SHOT_AUTHORIZATION**。
今回はsource修正・登録外技術検証・公開だけ。登録比較、実合成、authorization-only子commit、新markerは作成していない。mandatory STOP。

## 1. 出発点と修正範囲

準備branchは`track-b-d3-v2-source-preparation-20261014`、基点はv1公開確認commit
`dfec99d887b55a0e3cb94c6f0e5e4da5ea6a1ba8`。
旧science source `c7d2c826433dcf3b1bb30c98fc05a3bf13970ae1`、
旧authorization-only A `db584d9fb3f80c07ae55d5e1fe477bff2e386af2`、
旧結果commit `45a7c8ec89a904b1985dea16c4857294871f69d9`を保持した。
[旧停止報告](d3_failed_execution_and_gpt_handoff_20261014.md)の原因は、合成helperの`mp.mpf(epsilon)`がPython `Fraction`を受け取れないことだった。
最初の二つの予約だけで停止し、完成sequence0・比較0/48行。v1の科学的なpositive/negative判断はない。

利用者が認可した次の作業は、型adapter、API接続テスト、新source/manifest/result namespaceの準備である。
この指示をv2の登録実行承認として扱わない。

## 2. 型adapter：精度とidentityを保持

[共通numeric helper](../../../src/trottertracks/algorithm_codesign/rte_reallocation/numeric.py)の`synthesize()`だけを限定修正した。
`Fraction`の場合はmpmathの固定precisionでnumerator/denominatorを明示変換し、backendへその値の1/4を渡す。
既存stringなどの経路は従来どおり`mp.mpf(epsilon)/4`である。

元の`epsilon`変数は置き換えない。このため、

- `epsilon='1/10000000'`、angle key、cache keyの有理数表記を維持する。
- 保存されたstrict operator errorを元のexact `Fraction`閾値と比較する。
- requestは従来契約の`1/40000000`、100 dps、seed0、`up_to_phase=false`を維持する。
- sequence/count/W phase記録、strict guard、signed actual adjointの規則を変更しない。

変換は予約callbackより前に行う。引数の変換だけで失敗した場合はbackend tokenを消費しない。
予約成功は依然として合成完了と区別する。backendが失敗しても二回目を呼ばない。
sourceの他のnumeric関数は旧SとのAST照合で一致した。
`src/trotterlib`、Track A、model/pipeline、law、provider、境界相殺の方式は変更していない。

このshared helperは新sourceで改訂した版であり、旧sourceが新helperを使用したとは解釈しない。
旧版は凍結Git commitと元の実行worktreeにbyte一致で保存されている。

## 3. v2の契約・起動・保護

[execution contract v2](../../../artifacts/track_b_d3_source/2026-10-14/v2/execution_contract_v2.json)
SHA256：`d017559c4dddee510be2caa40097592e93d0ae40303572508f8f6889842313cf`。
kindと準備状態metadata、実行用pathだけを変更した。
元design SHA256 `76b3e4d95df62a00f6e66bd9dacd3230a09025167d29de838c917de4722d726e`への科学field projectionは同じ。

| Live v2項目 | path／状態 |
| --- | --- |
| authorization | `artifacts/track_b_d3_source/2026-10-14/v2/authorization.json`、pending、runs0、science_execution_authorized=false、source_commit=null |
| source manifest | `artifacts/track_b_d3_source/2026-10-14/v2/source_manifest_v2.json` |
| protection ledger | `artifacts/track_b_d3_source/2026-10-14/v2/protected_hashes_v2.json` |
| result namespace | `artifacts/track_b_d3_result/2026-10-14/v2`、未存在 |
| 将来の任意receipt | `docs/tracks/algorithm_codesign/d3_v2_execution_authorization_receipt.md`、未作成 |

契約内の元`planned_launch`・date・base_commit等は採択済みdesignの履歴snapshotをそのまま保持する。
v2の実際の起動pathは上記`execution` blockで規範的に固定し、launchで完全一致を要求する。
旧v1の承認や消費済みmarkerはv2を認可しない。

新source manifestは**472 current hashes**、protection ledgerは**437 unchanged current files**と
**3 historical frozen source blobs**を保護する。
意図的に改訂したnumeric helper、launch、read-only verifierの旧bytesは旧Sから読み、bytes/SHA256を照合する。
改訂版のcurrent hashesはv2 manifestへ固定し、旧hashを上書きして保存したように扱わない。
v1 source artifact、authorization、marker、failure結果・監査のbytesを変えず新ledgerへ追加した。
新sourceの読み取り専用verifierも、pending authority／科学projection／current・historical identity／新namespace未存在を確認する。

将来は固定source S2のauthorization-only直接子A2、remote=A2、clean worktree、新exclusive marker、
job/parent/source identity、内外guard、失敗時prefix不使用を要求する。成功・失敗の両方でretry0、mandatory STOP。
今回S2の自己SHAをmanifestへ埋め込まない。freeze後のfull S2はpublication receiptと最終報告で示し、将来のA2だけがS2へbindする。

## 4. 科学条件・資源上限は旧契約と同じ

明示synthetic 3-system-qubit involution、`p=(1/5,3/10,1/2)`、`tau=5/7`、初期状態`000`、
degree5、`r=1,2`、六方式、G0/G1×P0/P1四ケース48行。
分子geometry、軌道basis、DF rank、deterministic split L_Dはこのdomainには適用しない。
既知developmentであり、独立held-out validationや分子PR全体の資源優位にはしない。

新実合成は将来認可された場合にだけ固定36 positive atan keysをstrict1e-7で各一回。
新precisionのidentity、tool/options、inventory、負符号のactual adjoint、旧1e-6価格の再利用範囲は同じ。
**四ケースともnative誤差会計1e-6固定**。価格を取得してもnative精度をshot数削減へcreditしない。
旧D1 law/IRと旧D2 literal境界相殺額を再利用し、219,303 pairは再評価しない。

workers2、numeric threads1、per-key CPU20秒、total CPU1800秒。
parent/worker RSS512 MiB、aggregate RSS1536 MiB、virtual address1536 MiB/process。
output16 MiB、log64 KiB/key、cache4 MiB/36 entries、sequence20,000 chars、review chunks512 KiB、shot forecast cap1e8/axis。
total/phase/per-key wall capsはnull。
provider/signal matrix、circuit build、native compile、sampling、LP、DF、分子、GPUへ進まない。
実合成時のstrict2×2 guardは契約された別処理であり、「全matrix演算0」とは表現しない。

## 5. 登録外focused validation

**最終81 PASS、FAIL0**：

- [新接続・v2保護テスト18件](../../../tests/tracks/algorithm_codesign/test_d3_precision_and_v2_protection.py)。helper自体をmockせず、pygridsynth backend本体をmodule objectでmockする。角度は登録外`atan(2/1009)`。
- [既存guard/会計63件](../../../tests/tracks/algorithm_codesign/test_d3_robustness_source.py)。人工scalar/IR、temp jobs、mock owner/worker、namespace/cap/STOP/phase/storageの回帰確認。

新テストでは、Fractionと既存decimal/fraction string、100 dps request、key・exact threshold、
phase/W/count、callback順序、bad argumentのtoken0、backend失敗でretryなし、
実worker→実helper→mock backendの接続、再起動拒否を確認した。
旧helperの型変換failureは旧Sのfunctionを登録外fixtureで確認し、backendは呼んでいない。
旧Aのv2への流用、旧namespace流用、historical blob改ざんも拒否した。

初回の新テストはmockのdotted targetがpackageのexported functionへ解決され、14 errors／4 PASSだった。
`importlib.import_module`＋`patch.object`へ訂正し、最終18 PASS。初回raw logを削除せず保持する。
この初回はhelper呼出し前のtest harness failureで、実backend呼出し0。

[最終focused記録](../../../artifacts/track_b_d3_source/2026-10-14/v2/focused_tests_v2.json)、
[static verifier結果](../../../artifacts/track_b_d3_source/2026-10-14/v2/static_source_verification_v2.json)：
10 sourcesを実行せずcompile/AST確認、647 callsを調査、pending runner拒否、登録budget/price/pair0。
実合成0、登録行0。旧63 testsの人工strict2×2 checksは2回で、provider/signal matrix0と区別する。
mock出力は実際の合成価格・operator accuracyの証拠ではなく、registered end-to-end runは未実施。
63 tests log内のtechnical inconclusive表示はtemp/mock失敗controlであり、v2実行結果ではない。

固定runtime Python3.10.12、pygridsynth2.0.0、mpmath1.3.0、executable/package treesはv1 prelaunchと同じ。
[runtime/親worktree保護](../../../artifacts/track_b_d3_source/2026-10-14/v2/runtime_and_protection_v2.json)：
旧実行worktree clean、親50 dirty files・index・HEAD・status不変。
CPU32 logical、GPUはquery/useなし。固定two workersとthread1を維持し、速度倍率は主張しない。
local technical validationであり、CI・外部再現・科学的優位を主張しない。

## 6. 次の条件とSTOP

この限定修正では主要な比較条件・誤差保証・研究仮説を変更していない。
追加のGPT研究方針レビューをroutine source修正だけで要求しない。
実backend・48-row comparisonの結果解釈や研究継続は結果取得後の科学判断である。

次は利用者の**新S2に対する別途明示one-shot実行承認**。
承認後にS2から別execution branch/worktreeとauthorization-only直接子A2を作成・pushし、
live authority・remote/clean・tool・critical/protected/historical hashes・新namespace/marker未存在を照合して、
固定runtimeの`python scripts/tracks/algorithm_codesign/run_d3_one_shot.py --source <full S2>`を一回だけ起動する。
内部ledger/key CLIsは独立起動しない。

旧v1のresult/marker/STOP、旧source・authorization、D1/D2、他研究を保持する。
今回新authorization-only child、marker、登録再会計、実合成を作成しない。
**公開・remote再取得確認後mandatory STOP。READYは実行承認ではない。**
