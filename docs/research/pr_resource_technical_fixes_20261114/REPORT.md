# PR resource I1・I2・E1 技術修正結果

2026-11-14 JST。監査SOURCEは `51d324ea9c83bc12f1e866da0fa409cef926f74a`。
今回の範囲は実装、人工テスト、受動的なSOURCE/input metadata検査と技術性能測定である。
実H4/H6等の新科学計算、保存科学数値の再解析、実sampling、実回路build/compile、正式G評価は0件。
主agent自身の実装・確認であり、別agentによる独立レビューではない。

## 結果

| 項目 | 修正 | 科学条件の扱い |
|---|---|---|
| I1 | 固定SOURCEを読む既存loader、実API、保存JSON/bytes、configuration、plan/request/scalarの配線preflightを追加 | 定義ロード・引数束縛・保存metadataまで。行列復元、preparation、数値kernel、sampling、回路処理に進まない |
| I2 | S0時点のpipelineは固定commit blobと保存SOURCE indexで照合。現行APIは別の互換テストで確認 | 旧SOURCE・index・結果・科学契約は変更しない |
| E1 | live pipelineはSOURCE完全bytes＋絶対pathでparse済みcid callableをcache。Cのprivate固定Bはnamespace内のparserだけをcache | 入力ごとに元関数を評価し、ID規則・例外・configuration/source/law検査を保持 |

主実装SOURCEは `df71b4857d6cebee8688ab5c8cad5e2cfcd00787`。
C固有実装SOURCEは `e9a67e65ea77270984b5a0caab92f3be6f839f75`。
後続の報告commitはこの実装を変更しない。

## I1: 過去7件と検査の対応

原STOPを各commitのblobから確認した。locatorと原因の抜粋は [STOP_CAUSES.json](STOP_CAUSES.json)。

| STOP commit | 原因 | 今回の検査 |
|---|---|---|
| `3f2a696e` | exec scopeにastがない | 実S-only loaderを固定contextの定義生成まで進め、元bytecodeの未束縛globalを検査 |
| `eeccdf18` | descriptorのseed=Noneが整数入力APIに不適合 | 実call siteを既存APIへbindし、seedを既存require_integer_countで検査。旧diagnostic定義そのものも生成前に拒否 |
| `19b95d8b` | B configurationのcommit/path違い | Cの既存check_inputsで固定blobと公開intakeのSOURCE/configurationを照合 |
| `20fe1194` | 固定Backendが読むplan['source']欠落 | 既存backend_plan_keysとexecution_planを再利用。固定Backendのmetadata constructorとmethod signatureへbind |
| `37e30194` | h46_new2_coreをimportできない | 実scriptを対象SOURCEのscriptsからimportし、bytesとAPIを照合 |
| `69917830` | 固定norm/unitarity gate失敗 | **今回対象外**。新数値診断なしに判定・緩和しない |
| `752b9747` | tail-only scalarをtotalとして渡した | 既存constant reader/LD requestを使い、quantity・SOURCE・totalを照合。hashを付け直した誤scalarも拒否 |

正常経路はA/B/C/DでPASS。異常系は実定義・実APIを使い、scope、seed型、参照commit、import、plan key、scalar意味の拒否を確認した。
[WIRING.json](WIRING.json) と [FINAL_WIRING.json](FINAL_WIRING.json) に対象SOURCEと結果を保存した。
呼出し監視で、検査対象の科学処理境界に到達していないことも確認した。
旧D P1 runnerの文字列SOURCEと新adapterのcommit object形式は両方受け付ける。

runnerは [preflight_pr_resource_wiring.py](../../../scripts/preflight_pr_resource_wiring.py)、
実装は [pr_resource_wiring.py](../../../src/trottertracks/pr_resource_wiring.py)。
使い方と停止位置は [README](README.md)。既存runnerや固定実行契約へ自動挿入せず、外部から対象treeを選んで検査できる。
この検査はdeepなruntime値の全検証や科学資格を保証せず、過去のSTOPやone-shot条件を解除しない。

## I2: 再現、修正、出力の保全

着手時の最新remoteはB `ce0517de6`、C `d7c09ef2e`、D `01e068af7`。
I2/E1の修正済み成果は見つからなかった。
B/Dの5失敗fileを再現した：[BEFORE.json](BEFORE.json)。

- `check_h4_s0_v2_intake` はproductionの `S0_H4_PIPELINE_SOURCE_CHANGED` でsetup時に失敗。
- `check_pr_r_dependent_n`、`check_h4_df10_rpe_batch`、`check_h6_price_pipeline_coverage`、
  `check_r_variable_compiled_price` は変更済み共有moduleへの旧bytes/AST固定で失敗。
- 修正後は5fileすべてB/DでPASS。旧テストの歴史SOURCE照合と、現在のfactory・plan・provider経路の機能検査を分離した。
  H4 batchは実結果のscientific_sourceを使い、C providerだけの部分snapshotを共有pipelineのSOURCEと誤認しない。

S0全13roundのconfiguration、26軸C_REQUESTS、identity map、元N handoffは
`030f3f548a35d98a88e62e92a35f132e0243a293` の公開JSONと完全一致した。
保存nominal、nullの十分N/C/G、UNQUALIFIED、SOURCE/law/identityは保全。
固定blobのhash不整合、外国configuration/SOURCE/profile、nominalの十分Nへの昇格は引き続き拒否する。
新APIで新しい科学結果を生成して同値と判定したものではない。

## E1: 出力同値性と性能

人工入力36種類を旧関数そのものと比較した。追加testsでは複数system/method/formula/q、
候補field変更、不正入力、同size/mtimeでのSOURCE変更、不正SOURCE、cache境界を検査した。
Cの実published APIでは元candidate_idのcode/SOURCE、標準ast、別namespaceへの非干渉も確認した。

同一process内で旧→新を各200回×5回測定した中央値：

| 経路 | 変更前/回 | 変更後/回 | 高速化 |
|---|---:|---:|---:|
| A live | 6.888 ms | 0.083 ms | 83.2倍 |
| B live | 6.770 ms | 0.084 ms | 80.2倍 |
| C live | 6.931 ms | 0.088 ms | 78.5倍 |
| D live | 6.921 ms | 0.086 ms | 80.8倍 |
| C private固定B | 6.931 ms | 0.198 ms | 35.0倍 |

[PERFORMANCE.json](PERFORMANCE.json)、[C_PRIVATE_PERFORMANCE.json](C_PRIVATE_PERFORMANCE.json)、
[再現用script](bench_candidate_id.py)。C privateと同じ測定のlive値は0.097 ms/71.7倍であり、実行時の競合による幅もある。
auditの約1500倍という見積もりは採用しない。毎回SOURCEを読み直す安全な実装での実測値を示した。
候補の結果自体はcacheせず、configurationの違いをID照合で隠さない。

代表回帰のwall time：B `check_rpe_r_price_intake` 129.46→7.35秒、
A `check_pr_rpe_rmse_s0_v2` 54.10→5.58秒、C `check_h4_s0_price_plan` 81.39→6.07秒。
回帰は並列度・実行時負荷が同一ではないため、これらは参考値。上のmicrobenchmarkを直接の性能比較とする。

## 横断回帰

既存専用環境 `/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B` を再利用した。
環境・共有packageの変更なし。無関係な全repository testは実行していない。

| 系列 | 隔離cloneの基点 | 対象file数 | test数 | 結果 |
|---|---|---:|---:|---|
| A | `3ae055fa7` | 13 | 124 | PASS |
| B | `ce0517de6` | 19 | 198 | PASS |
| C | `d7c09ef2e` | 14 | 123 | PASS |
| D | `01e068af7` | 17 | 168 | PASS |
| 合計 | 3種類のpipelineへ今回差分を適用した自分のclone | 63 | 613 | 全PASS |

[REGRESSION.json](REGRESSION.json)。上記cloneは科学SOURCEそのものではなく、技術差分を適用した検査用treeである。
Cの追加private cache後と最終Bでは11file/91件を必要範囲で再確認し全PASS
（[FINAL_REGRESSION.json](FINAL_REGRESSION.json)）。
最終cache helperを含むA/Dの追加10件もPASS（[FINAL_CACHE_AD.json](FINAL_CACHE_AD.json)）。
seed入力検査追加後はwiring unit 6件と実C profileがPASS。
これらの追加分には先の検査の再実行が含まれ、distinctな総test数として加算しない。
合成行列の数値同値性・metadata互換の検査であり、実H4/H6の再計算や科学性能の再認定ではない。

## 適用状況と残る作業

| 適用先 | 状況 |
|---|---|
| `codex/pr-resource-technical-fixes-20261114` | B基点。I1共通runner、I2 production/test修正、E1 live/cache helper、統合検証記録 |
| `codex/pr-resource-c-parser-cache-20261114` | C基点。E1 live/private loader/fixtureと関係する旧test版固定を修正 |
| A | 元branch非変更。隔離適用・回帰済み。[A_technical.patch](A_technical.patch)を提供 |
| D | 元branch非変更。隔離適用・回帰済み。[D_technical.patch](D_technical.patch)を提供 |

A/D patchはそれぞれ上表の固定基点に対応する。新cache testsと共通preflightは主branchにある。
既存のA/B/C/Dやreview担当のworktree・branch・processは変更しなかった。
無断merge、ref削除/復元、prune、gc、履歴書換えは実施していない。

最終remote観測ではAは `71f04be7b` へ進み、追加のFull PF follow-up関連だけが変更されていた。
Dは `3d39f8c0c` へ進んだがsrc/scripts/testsの変更は0。
検証対象と重ならないため再計算はしない。差分は [LATEST_HEADS.json](LATEST_HEADS.json)。

採用先へ適用する際は対象commitでpreflightと関連技術回帰を確認する。
固定科学契約が旧SOURCEを指定する実行は、この新treeで直接再開できない。
unitarity gate等の科学的な追加検証、新SOURCEでの実H4/H6実行、保存数値再解析、sampling/compile/Gは
別の明示許可が必要。今回の技術修正完了後はmandatory STOP。
