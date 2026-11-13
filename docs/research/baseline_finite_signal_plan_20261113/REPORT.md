# ベースライン3系列の全RPE signal：初回有限実行案

2026-11-13。**3 schedule・39roundのcanonical/physical対応を固定した。推奨scopeは、保存Full PF m6の受入再資格1件と、そのPASS後だけ新規38候補mean評価。** 無条件に再利用済みと認定したmeanは0件、受入成功時の保存再利用は1件。今回は計画・metadata確認だけで、科学計算/再解析/旧人工test再実施は0。

[固定manifest](BASELINE_FINITE_MANIFEST.json)／[coverageと再利用](COVERAGE_AND_REUSE.md)／[有限科学実行計画](FINITE_SCIENCE_PLAN.md)／[A/B/D HANDOFF](HANDOFF.md)。生成は既存B V2とC configure/prepareを使う受動 [BUILD_MANIFEST.py](BUILD_MANIFEST.py)。新pipeline・solver・PF公式・科学runnerは作っていない。

## 初回menu

|系列|固定seed|新規mean最大|保存受入再資格|理由|
|---|---|---:|---:|---|
|Full PF|S2、ΔPF=.1、one-body＋全10 fragment、qPF=2^m|12|1（m6）|既存短時間・m6の同物理recipeがあり、数値部品・価格receiptも利用可能。粗いΔ.4の既存長時間診断を精度改善の証拠にせず、元fine seedを維持|
|Truncated PF|B0I、S2、ΔPF=.1、prefix7、qPF=2^m|13|0|保存prefix7 scalarと短時間SOURCEを利用できる。identityを残すB0Iを一つの代表とし、B0→B0Iの新しい位相変換解析を追加しない|
|Full-H pure RTE|finite K2、q grouping=2^m、各round R448、head0|13|0|元full-H preparation descriptorと既存finite polynomial経路で扱える単純な接続seed。PR tail/B3へ置換せず、q/Rを価格identityとして保持|

S2・p7・B0I・R448を最適又は必要十分としない。同じPF幅を初期に使うことはPRとbaselineの探索範囲を今後同一に制限することではない。Yoshida4/M5、別幅/prefix/B0/K/R等は既存最適化設計の未探索域として残す。今回新signal/価格で候補を順位付けしていない。

## 共通taskとSOURCE

Linear H4/1Å/STO-3G、保存tol-only DF10/weighted-Hermitian completion、sector(2,2)/36、同H/state/snapshot、元full DF-Hの同sector ground energy、h=.1/M12、元prior/decoder/branch・energy RMSE予算、P0。A `91349e703` のtargetと、`a9abf8de7` LD1の共通binding/h/policyをそのまま転記した。h/Δは元binary64のexact有理数を維持し、decimal 1/10へ替えていない。

Aのsolver proxyとstateは保存数値証拠で、厳密ground/元分子H誤差保証に昇格しない。target内 `truth_proxy_adopted_for_new_simulation=false`、B policyのPROPOSEDを保持。共通energy taskのcontextをcopyしたことは、PR用の全誤差移送をbaselineにも認定したことではない。Bの正式risk/loss/claimは別途保留する。

|用途|固定SOURCE|
|---|---|
|計画・pipeline|C `5cc7c07e112ce97205948465c94441cba0042416`。baseline数値2moduleは`86f37d25f`から不変、B controllerは`b9b9ffa62`参照|
|PR比較|A menu `a9abf8de75bf3432fcd714e868e5c14c8884b8f8`、完了結果 `91349e703e0d09ef702ccd108745dfaf1acc29f7`|
|旧signal/STOP|Track A `155529e20a`、pf_common `0b1886f53`、screen SOURCE `256caad34`・run02 carrier `b72ea6b30`|
|入力・scalar|INDEX `7a8bc7f132e166390c943e6b32c92cc6abe1baf4`。snapshot/receipt/contextの公開path・hash descriptorをmanifestにcopy|
|全13reference pair|`bba150834ca30bdb186fbfd58c55a8d51177b855:PHYSICAL_INPUTS.json`。metadata上の全m/T/H/state対応とfield存在だけ確認|
|Full-H preparation|`9b83b96b83f62142c9d87858dc24a94db7ac5d61` の保存descriptorを元D helperで転記。再生成なし|
|B/D設計|B予備分析 `23d74855135e718554d824a269427de9943213d1`、D requests `748be6875`、Full PF receipts `e7dca5f73` を参照。再集計/再監査なし|

Full-H保存preparationのphysical binding全fieldは共通A bindingと同一だった。consumer descriptorにはAのcanonical provenanceであるsource/decompositionの2fieldだけを追加し、元bindingをmanifestに別保存した。preparation hash/係数/λ/scalar/順序は不変。元ファイルは書き換えていない。

## 実行可能性と未確認部分

3つのB configuration・全39 physical ID、T/q/R/signed allocation、reference descriptorを既存C prepareで生成・照合した。NumPy/SciPy/Qiskit/PySCFをimportせず、snapshot/receipt/contextのローカルファイルを開いていない。既存人工44件の証拠を参照し、再実行していない。

実入力のbytes/hash/sector/state norm、保存prefix7 scalarの読み取り、reference pairの数値gate、実macro-step/power/normalizerの成立は**承認後のscope内で確認する事項**。数値SOURCE・gateは変更しない。unknownを0やPASSで埋めない。

保存m6の受入は、元spectral SOURCE/backend・既存値/診断/allowanceを保つ科学的受入再資格であり、今回未実施。新direct-expm点と比較する検証や新uのfitではない。受入不可なら38件へ進まずSTOPし、別点や新計算で自動代用しない。

実行は一つの承認範囲・一つの論理batchとし、受入→共通入力復元→Full PF→B0I→Full-Hを既存関数で進める。共有36次元context/cacheを再利用するための順序で、CPU/worker上限を設けるものではない。新reference/DF/state/scalar抽出・sampling/compile/N/Gはscopeに含めない。

既存SOURCE/input/raw/結果/科学契約/grant/ledger/marker/KEEP/STOPを保全。計画公開・remote照合後 **mandatory STOP**。ユーザーの明示承認まで、38件の新meanも1件の受入再資格も開始しない。
