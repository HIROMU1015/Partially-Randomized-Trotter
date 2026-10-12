# N3 保存証拠の事後保証幅評価：準備完了

2026-10-12 JST。専用branch `n3-saved-width-v1-preparation-20261012`、基点 `86f57f05a36e7e6639e827486d300bd049bf2438`。
完了済みGPT科学レビューを受領し、証拠整理・数式の技術確認・比較実装/仕様・synthetic検査をまとめて完了した。
**実8入力の新しい保証幅と改善率は未算出。数値的事後評価には別の明示認可を必要とし、Mandatory STOPを維持する。**
同じ結果・方針のGPT詳細科学レビューを繰り返すための資料ではない。

## 1. 今回の認可と旧結果の保持

[レビュー原文](../../artifacts/n3_saved_width_v1/2026-10-12/preparation/review_intake.md)、
[運用指示の受領copy](../../artifacts/n3_saved_width_v1/2026-10-12/preparation/agent_workflow_intake.md)、
[scope intake](../../artifacts/n3_saved_width_v1/2026-10-12/preparation/review_and_scope_intake.json)。
今回の利用者指示は①②と③④の仕様準備。新しい数値的事後評価は別承認と明記されている。
元one-shot認可を流用せず、旧source8/pack/run1/result/marker/STOPを保持する。
N1追加探索停止、N2保留、中心テーマnull、他Track不変。

対象は既存固定8入力、実対称one-body＋非負unordered density、一外部mode、active6/7/8/9、ν2/3/3/4、full modes7/8/9/10。
元δgoal=.004/.006/.003/.005。geometry・化学basis・DF rank・L_D・PF/time windowは適用なし。
元NO_ADDITIONAL_ACCEPTANCE_OBSERVEDと2-base未達を変えない。新しい独立入力・移送成功に数えない。

## 2. 保存certificateの整理

[閲覧index](../../artifacts/n3_saved_width_v1/2026-10-12/saved_evidence/README.md)、
[catalog](../../artifacts/n3_saved_width_v1/2026-10-12/saved_evidence/catalog.json)。
raw method96、保存raw attempts371、保存spectral proposals142、proof pool719、operational views24を抽出。
valid spectral142は元に保存されたflagを数えたもので、今回PSDを再実行した結果ではない。
全raw method/attempt、途中spectral、元の採否、継承/fallback、係数、U/cの証拠、proof L/pivots、費用参照をlosslessに保持する。

source JSON pointer・canonical value hash・元blob identityを結び付け、**1250 pointer roundtrips PASS**。
各JSONは8192 bytes以内。852 JSON files / 5,007,569 bytes（README除く）、最大8191 bytes。
元ledgerの5,831,111 bytesを別の大きいJSONとして再pushせず、元objectを保持して小さいviewを追加する。
これは表示・取得性の整備であり、新しい保証幅・PSD・採否の再評価ではない。

## 3. 数学と比較仕様

[数学契約・独立技術確認](n3_saved_width_v1_mathematical_contract_20261012.md)、
[比較・実行仕様](n3_saved_width_v1_comparison_specification_20261012.md)。
g=c−U>0、q=κU−sから固定証拠の下端
δ_w=max(0,κ−g,(sqrt(g²+4max(q,0))−g)/2)を確認した。
NORMはs=−||v||²、旧法はs=νμ、OCCは保存τ/Zまたはfull trace由来のs。
κを固定し、二次不等式とboxの両方を厳密検査する。

比較先は保存U,c_star。BASEはNORM＋FIVE＋OLD-DIRECTの全PSD-valid raw候補、OCCは独自raw候補の全保存valid spectral pair。
途中pairを含む主保存集合とgenerator-selected-onlyの補助集合を区別する。fallback・継承コピーはOCC利益へ混ぜない。
区間は固定128-bit外向き計算、poly-only独立checker。区間が重なる場合は未判定とし、結果後の精度増・再探索をしない。
E_forcedの式と理想族の境界条件も技術確認したが、現8入力への固有値計算・数値適用は0。
先行原理との関係・判断材料を仕様へ整理した。網羅的新規性監査・研究採択は未実施。

## 4. 実装・synthetic・実行ゲート

- `n3_saved_evidence_v1.py` / `prepare_n3_saved_width_v1.py`: 抽出・byte/pointer照合だけ。
- `n3_saved_width_v1.py`: exact scalar root enclosure、poly endpoint check、有限集合envelope。
- `n3_saved_width_analysis_v1.py` / `run_n3_saved_width_v1.py`: 別authorization後の保存証拠評価。現在dormant。
- `verify_n3_saved_width_v1.py`: 元証拠集合の完全性・PSD・q・poly endpoints・比較labelの独立保存監査。
- `tests/test_n3_saved_width_v1.py`: syntheticだけの24 tests PASS。

正/負q、zero infimum、box優先、irrational/rational root、κ固定、energy shift/scale、候補minimum、interval overlap、改変拒否、bit capを検査した。
旧3synthetic fixture（active2/3、ν1/2、full3/4、一外部mode、one-body＋非負density）を使い、2 spawn workersの評価と独立checkerを確認した。
geometry・化学basis・DF・L_D・PF windowは適用なし。real8の評価は呼んでいない。
認可なしはplan/pack/ledger load前にexit2、旧科学batch認可はschemaで拒否。evaluation1未作成。

synthetic integrationの初回に、新しい呼出側がsecondary監査後のproof pool切替を見落としてmissing proofとなった。
初回失敗logをdevelopment記録として保持し、primary pool/cacheを復元する修正で24 tests PASS。
元の凍結checkerは本来の段階別動作を保ち、変更していない。これは新sourceのtechnical developmentで、元科学batchのretryではない。

## 5. 性能・資源・限界

CPU32/affinity32、memory inventory、GPU driver unavailableを記録。Python3.12.3、stdlibのみ。
支配費用は保存有理LDL/元行列再構成。case内cache再利用と2独立case workers、全numeric thread1を準備した。
rootの候補生成器、NumPy/eigensolver、κ最適化、compileを使わない。
抽出はCPU1でwall0.92s、max RSS54,528KiBというresource記録。24 synthetic suiteの別resource記録も保存。
これは実8入力の評価runtimeや速度倍率の測定ではない。

将来scopeはwidth候補512/case、bit128固定、有理4096・explicit scaled integer8192bits、worker AS4GiB、sampled aggregate8GiB。
output shard32MiB/全64MiB、log8MiB、attempt1/retry0、wall/CPU time capなし。
Fraction内部temporary integer・RSS sample間の瞬間peakは完全監視したとは主張しない。

## 6. Freeze・公開・次の手順

source11ファイルをcommit `3f295df0263059ce16d866f84c1132f8bace0670`へ固定した。fingerprint `b190ea508afb9a10fad3d3dbfeb92d50f46ad5813baadd70965aa6fc5ad044d1`。
[source-bound plan](n3_saved_width_v1_execution_plan_20261012.json)のSHAは`bd3f5f4a21deafb227a6794d3277ca3eab1d9a30c610236d42294103e91e53dd`、別実行認可はfalse。
[固定source preflight](../../artifacts/n3_saved_width_v1/2026-10-12/preparation/source_bound_preflight.json)はlossless1250 pointer checksだけ、幅/PSD/eig計算0。
既存pack/ledger/source/STOPとroot dirty内容の保全を監査し、必要な新しい小view/source/docsだけをHIROMU1015へ公開・remote再取得する。
旧global manifestのdescription欠落FAILは残し、今回の局所entry/保存view検査と区別する。

[Mandatory STOP](../../artifacts/n3_saved_width_v1/2026-10-12/preparation/MANDATORY_STOP.json)。
実8入力の幅比較は未認可、計算0。新しいsource/input/planに結び付く一回の事後評価認可が次の段階。
新結果後にGPTへ幅差の科学的意味・旧法coverage・費用対効果・継続/縮小/終了の判断を渡す。
新input、E_forced数値、新κ/τ/Z、元batch再実行、実分子・多外部・一般二体、N1/N2の再開は含めない。
