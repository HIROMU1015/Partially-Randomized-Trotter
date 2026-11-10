# R可変N provider・H6価格providerのB統合

2026-11-11 JST。**AのR依存N provider、CのH6固定R価格provider/coverageをBのR可変branchへ取り込み、保存入力→不足batch→exact費用/R選択の技術接続を完了した。** 新Rの価格がなくても欠測・未資格を保持して接続検査を完了できる。実H4/H6の新signal/N、保存科学値の数値再解析、価格予測/取得/fit/sampling/compile、実N×C/G・実R最適化は全て0。

成果：[統合module](../../../src/trottertracks/rpe_provider_integration.py)／[HANDOFF・呼出し仕様](HANDOFF.md)／[最小科学案](MINIMUM_SCIENTIFIC_PLAN.json)／[結合テスト](../../../tests/check_rpe_provider_integration.py)。

## 取り込んだ固定成果と差分

| 対象 | 固定SOURCE | 今回の扱い |
|---|---|---|
| B有限R pipeline | `7bca8f9642d1c42ca3fede4d31b2cdd8dcef46ae` | 既存branch/worktreeを再利用。元exact会計・R選択を再利用 |
| A R依存N | [`5ca2dce3234f3e4a971f4a8c852da676f050e315`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5ca2dce3234f3e4a971f4a8c852da676f050e315/docs/research/pr_r_dependent_n_model_20261111/REPORT.md) | `pr_r_dependent_n.py`、metadata CLI、元testを取込み。producer module/CLIはbyte一致 |
| C固定H6価格/coverage | [`4cb61ef4633467446122954ed6cc43462e0306a9`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cb61ef4633467446122954ed6cc43462e0306a9/docs/research/h6_provider_integration_nc_coverage_20261111/REPORT.md) | H6 provider、coverage module/CLI、元testを取込み。provider/coverage/CLIはbyte一致。H6 factory6行だけを最新Bへ追加 |
| H4固定全N | `89c91c692e1d35de94d2e7afbe890aabe6d80b7e:N_HANDOFF.json` | Cの既存loaderで補正済み整数をcopy。raw resultの小さい整数へ戻さない |
| H6固定全N | `f4c0bdabdf2afdc5bc4f20573bb10677917f4c1a` | 保存m0..7＋m8..12を既存loaderでcopy |

upstream testのSOURCE保存assertだけは意図された最新B差分へ合わせた。A testは共通moduleからH6 factoryだけを除いて元ASTと照合、C testは会計moduleをBのexact拡張7bca8f964へ照合する。他のdomain/資格/null/coverage/数値チェックは保持。旧H4 provider・N kernel・会計は再実装していない。

## 接続と資格の保全

Aの`RBatch`は1 roundにRを一つ持つtaskを受け、Bは同roundに複数R候補を持つ。新`RMenuBatch`が明示要求をA task群へ分け、candidate ID/recipe/identityを照合する。contextと時刻別referenceは同一呼出し内で共有し、各Rのaction/Nは独立したcandidateとして保持。既存H4/H6専用loaderへの遅延factoryも用意した。context/referenceを実際にload/計算する呼出しは将来の科学実行範囲で、今回mockだけを使った。

NのFAILはAのstatus/diagnostics/nullを保持。最初のgroupが不成立なら後続groupを`NOT_EXECUTED_AFTER_N_FAILURE`として区別し、価格段へ進まず再試行しない。context/reference例外はpartial recordsとAのSTOP詳細を保持して伝播する。PF、H6 m0..3、明示R対象外の不足NをAへ無断で渡さない。

**N比較policyを追加照合する。** Aの新Nは`R_AWARE_EMPIRICAL_ACTION_COUNT_V1`であり、旧固定R Nのallowanceと同一とは限らない。旧Nは保存coverage/既存用途に残すが、R変更roundの比較で同じpolicyの証拠がなければ`MISSING_MATCHED_N_ALLOWANCE_POLICY`を返す。baseline signalの同policy scalar再解析は新しい科学範囲であり、今回行っていない。PF・変更しないroundへ不要な再評価を要求しない。

`FixedSavedPrices`は既存H4/H6の保存lookupをBのlaw_binding/full candidate domainへ接続するだけ。H4 R4/H6 R16、元δ/prefix/K/compiler/full wrapperが一致した既存資格を保持する。別Rはnull/対象外、H6 qPF>4は未検証外挿のまま。direct/Affine/Mq、cohort/path/law期待値、校正q2と独立q4を混同しない。full domainの付与は凍結providerのmetadata検査を通った保存recordだけで、qualified flagをtrueへ変更しない。

`run_missing`はN/C batchとR coverageまで、`run_evaluation`は将来の明示数値入口として既存R選択・exact会計まで接続する。今回の後者の検査は全N/Cが人工/mock。現実のmissing価格を仮値や固定R係数で補わない。normalizationや古典価格標本数をNへ再乗算せず、Python巨大整数とexact有理数の積和を維持する。

## 保存coverageと不足

[固定52行coverage](fixed_coverage/COVERAGE.json)・[CSV](fixed_coverage/ROUND_COVERAGE.csv)はCの元CLIで再現した。両系の保存13round/PF/PRのNをcopyし、actual Nはnull。H4 m10の補正整数もN_HANDOFFと全桁一致。価格値・不確かさは保存値の転記で、実G/ランキングはnull。

[R候補coverage](R_MENU_COVERAGE.json)はH4 m12 R4/8、H6 m12 R16/32という**metadata例**で、54候補・保存N52 record・新RのN欠測2を保持する。scenarioをまだ選んでいないこととC自体の欠測をprice_contexts等で区別し、baseline allowanceの比較資格不足も表示した。これらのRは科学採択ではない。

| 項目 | 再利用できる範囲 | 新しい科学実行等が必要な範囲 |
|---|---|---|
| 固定R N | H4/H6各全13roundの条件付きN。H4は補正済みhandoffが正本 | 新Rのfinite mean/phase/radius/N、比較baselineの同policy scalar評価 |
| 保存直接C | H4 m3/m4両方式・m5 PF、H6 m3..6両方式の固定候補 | 新R、その他q/早期別domain、欠測候補 |
| H6 Affine | R16/p7/δ1.6、保存q2校正・q4独立検証 | qPF>4の実予測/外挿採用、R依存model/validation。早期p5/R32は別domain |
| PR期待価格 | 既存有限cohortを明示policyのscenarioとして受渡し | 全law期待値・rare orderの価格/誤差。有限cohortから自動昇格しない |
| 不確かさ | 保存nominal量・shared fit/paired XY・SOURCE | 完全な係数/systematic、cross-R/round共分散、選択後CI。未知はnull |
| 全RPE | 保存のround条件付きPASSと独立した成功packetの受渡し | prior/alias/decoder/全round/failure/implementation-lawの科学資格。NのPASSだけでは不足 |

## 検証

**83 test method PASS**。Bの現branch・実module/依存・既存専用venv（Python3.11.0rc1、NumPy1.26.4、Qiskit1.3.0）を使用。H6元golden testの旧B snapshotと、最新Bへの結合検査を区別した。

| 試験 | 件数 | 記録 |
|---|---:|---|
| 新統合 | 8 | [INTEGRATION_TESTS.txt](INTEGRATION_TESTS.txt) |
| A N provider元試験 | 12 | [A_N_PROVIDER_TESTS.txt](A_N_PROVIDER_TESTS.txt) |
| C H6価格元試験 | 10 | [C_PRICE_TESTS.txt](C_PRICE_TESTS.txt) |
| C coverage元試験 | 7 | [C_COVERAGE_TESTS.txt](C_COVERAGE_TESTS.txt) |
| B R選択 | 12 | [R_SELECTION_TESTS.txt](R_SELECTION_TESTS.txt) |
| B共通pipeline | 12 | [B_PIPELINE_TESTS.txt](B_PIPELINE_TESTS.txt) |
| 元会計 | 15 | [ACCOUNTING_TESTS.txt](ACCOUNTING_TESTS.txt) |
| 既存H4結合 | 7 | [H4_INTEGRATION_TESTS.txt](H4_INTEGRATION_TESTS.txt) |

新試験は実A RBatchの下位action/解析をmock化し、複数R→common N→mock C→exact積和/R選択、reference共有、FAIL/未実行/再呼出し抑止、保存整数・新R流用拒否・law/domain/allowance資格を確認した。新N recordを通る`2**1200+7`も整数のままJSON/費用計算を通過する。実Nとmock価格を混ぜて費用を算出していない。人工テストPASSは科学的性能・独立科学レビューではない。

[CHECKS.txt](CHECKS.txt)に固定部品・旧結果・整数正本・試験SOURCE・coverage件数の照合を記録した。

## 次の有限範囲とSTOP

[MINIMUM_SCIENTIFIC_PLAN.json](MINIMUM_SCIENTIFIC_PLAN.json)はAの最小案を受けた未承認・未実行の提案。H4 m12 R4→8とH6 m12 R16→32の新finite mean2/内部対照2、新R scalar2＋保存baselineの同policy scalar2（XY計8軸）。同H/state/Tの保存2経路referenceを資格して再利用できれば新reference0。満たさなければ無断fallbackせず、別範囲を判断する。価格/GはこのN案に含まない。

**ユーザー**が次の科学範囲を判断し、**A/B**が別途許可内でNとbaseline比較policyを扱う。**Codex C**は現在のR依存価格model設計をそのまま継続し、HANDOFFの共通C形式で後から差し込める。**Codex B**の実費用/R比較も、データ・資格と別の実行許可が揃ってから行う。

SOURCE/input/raw/結果/grant/ledger/marker/KEEP/過去STOPを保全し、他担当branch/worktree/process・共有環境を変更していない。commit/push・remote SHA/主要成果物照合後**mandatory STOP**。新しい科学計算・価格予測・実最適化へ自動進行しない。
