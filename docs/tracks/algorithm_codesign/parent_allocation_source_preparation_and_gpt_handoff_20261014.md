# Parent allocation：技術実装準備とGPT引き継ぎ

2026-10-14 JST。判定：**`TECHNICALLY_READY_FOR_SCIENTIFIC_DESIGN_REVIEW`**。

汎用parent配分adapter、ordinary/closed P5接続、controlled位相を保持するnative記述、登録外検証、比較案の技術整理を完了した。**これは科学的GO・新規性認定・登録計算の実行可能判定ではない。** 比較の科学条件、本番合成worker/runtime、旧cache bridgeと独立strict guard接続は未確定・未実装である。新科学実行・authorization・markerは0。mandatory STOP。

## 入力と旧成果の保護

| 原資料 | 固定commit / path |
| --- | --- |
| 直前の技術監査 | `c12558d682f6aef3cfc68e8d75082269f2282ac8`：[報告](parent_allocation_technical_audit_20261014.md)、[未採択草案](parent_allocation_minimal_comparison_draft_20261014.md) |
| GPT研究方針原本 | `4d80b45693a1ea7c14d7cbf9035e7358788953bd`：[原本](../../research/track_b_algorithm_direction_review_20261014.md)、44,878 bytes、SHA256 `16af1fa58ec3d0b12dba5e2e1f8fda574a5723eeff3875e16688ef4ebe445e51` |
| 今回の明示指示 | [byte一致の原本](inputs/parent_allocation_source_preparation_user_instruction_20261014.md)、10,970 bytes、SHA256 `3fa9a99b41a51672bb0b4b61a97c1283ff96eefb412275d67314f3a5a4939695`、input-only commit `b23fec7f82697893df577460ac81b8b6970f3566` |
| 共通運用指示 | [2026-10-12版](../../research/research_project_instructions_gpt_codex_review_approval_2026-10-12.md) |

上の監査・草案・GPT原本3資料は指定commitのGitHub rawから実際に取得し、Git blobとlocalのbytes/SHA一致を確認した。[取得receipt](../../../artifacts/track_b_parent_allocation_source_preparation/2026-10-14/v1/fixed_input_identity.json)。

専用branch/worktreeを監査commitから作成した。旧algorithm sourceは変更せず、旧result/contract/authorization/marker/STOPのpathは作業対象外。親worktreeは既存dirty状態を保持する。Track A/N3を取り込んだりstageしたりしていない。保護確認は[evidence manifest](../../../artifacts/track_b_parent_allocation_source_preparation/2026-10-14/v1/evidence_manifest.json)を参照。

## 実装と平均の意味

新sourceは[parent_allocation.py](../../../src/trottertracks/algorithm_codesign/parent_allocation.py)。有理数入力のみを受け、`z_i >= 0`、`sum z_i = a`をexactに検査する。任意配分に対応し、頂点限定の探索コードではない。

偶数長のreduced parent word `u`と`zeta=(-i)^|u|`について、理想係数の恒等式は

\[
\sum_i \sqrt{z_i^2+b_i^2}\,\zeta e^{-i\sigma\operatorname{atan2}(b_i,z_i)Q_i}Q(u)
=\zeta\left(aI-i\sigma\sum_i b_iQ_i\right)Q(u).
\]

sourceは複数平方根の区間を足してroot massを更新する。digital係数はmidpoint、proposalはdyadic、weightは`tilde_c/q`。したがってdigital平均はdigital係数の和に正確に対応し、理想係数との差を別の区間誤差として残す。理想meanがfinite-bitで無誤差になるとは主張しない。

common-angle `z_i=a b_i/s`では既存のfactorized区間`(b_i/s)*sqrt(a²+s²)`を使う。childごとに別々に平方根丸めする実装と区別し、元のdigital係数・group law・event metadata・sampling bit消費へ正確に復帰する。正時間canonicalのcommon点では旧samplerへ直接delegateする。

| 機能 | 実装と検証範囲 |
| --- | --- |
| `ParentAllocation` | 任意有理数配分、zero除外、pure-child/pure-parent、正負時間、固定bit失敗の拒否 |
| `RootAllocationGenerator.ordinary / closed_p5` | 同じroot adapterを双方へ接続。非root係数/角度と条件付きsamplingを保持し、変更したgroup qでweightを補正 |
| `digital_proposal / reweight_events` | canonical、正cost解析式、明示lambda正則化。root-localと外部全表global経路を分離 |
| `event_statistics / error_accounting` | 実proposalからmoment/range/acceptance。係数・angle・native誤差は別入力、missingを0にしない。十分Nは生成しない |
| `native_plan / adjoint_plan` | ancilla相対位相、演算子順序と逆の回路時間順、全sequenceのadjoint。provider callbacksを外部入力 |
| `native_ir_g9` | 既存G9のdirect/helper記述へ接続。登録providerのmatrix・回路・compileは未評価。テストではproviderをmock |
| `rotation_key / check_cache_metadata` | 正atan RZ primitiveのtool/precision/seed/spec/sequence identity。endpointはkey不要。actual operator guardの再実行ではない |

pure-childは`z=0,b>0`で`zeta*(-i sigma)*Qi*Q(u)`、pure-parentは`b=0,z>0`。ancillaへS/S†/Zを戻す。`z=b=0`は消し、`s=0`は同一pure-parentへ合併、`a=0`は全child端点、全係数0はsampling不可とする。巨大ratioによる近似はない。

root係数取得はL個の有理数・平方根処理であり、samplingは全event列挙を使わない。既存group処理とdyadic丸めのsort費用、bit演算費用までO(L)と主張しない。`reference_events`は明示上限付きの検証専用である。

## 登録外の独立検証

[test_parent_allocation.py](../../../tests/tracks/algorithm_codesign/test_parent_allocation.py)：**62 tests PASS**。最終実行は一つのfocused suite。旧科学suite、D1〜D3評価、pair再取得は実行しなかった。

独立したstdlibの2×2 involutionと4×4 controlled行列でmean/unitarity/phase/非可換順序/adjointを確認した。generatorの登録外入力は`p=(2/7,5/7), x=1/3`、単一label controlは`x=1/5`。任意配分・全端点・両sign・全groupへの強制分岐・旧bit経路再現、dyadic full support、weight/error、不正入力、key/sequence/spec改ざん、preflight拒否を確認した。

production evaluator、NumPy、Qiskit、mpmathを使わない。G9 binding checksはCQ/V/crotをmockし、実providerのQ行列を取得していない。deterministic bitを使う分岐検査であり、量子shot・trajectory・乱数品質の検証ではない。テスト件数は一般証明や実provider validationの代替にならない。

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/python3 -B tests/tracks/algorithm_codesign/test_parent_allocation.py
```

Python 3.10.12、1 worker / numeric threads 1。wall 0.128 s、child CPU 0.1225 s、peak RSS 25,804 KiB。これは局所focused実行であり、immutable CIや外部再現ではない。[実行receipt](../../../artifacts/track_b_parent_allocation_source_preparation/2026-10-14/v1/focused_validation_receipt.json)／[stdout/stderr](../../../artifacts/track_b_parent_allocation_source_preparation/2026-10-14/v1/focused_tests.txt)／[runtime identity](../../../artifacts/track_b_parent_allocation_source_preparation/2026-10-14/v1/runtime_identity.json)。

科学入力・資源/shot rows・runner・native合成/compile・LP/GPU・trajectoryは0。**登録外の技術mock行列とgate記述は使用した。** 「すべての行列/回路計算0」とは表現しない。

## 比較計画とpreflight

[技術整理v2](parent_allocation_comparison_technical_options_v2_20261014.md)と[機械可読案](../../../artifacts/track_b_parent_allocation_source_preparation/2026-10-14/v1/comparison_options_v2.json)は未採択のoptionsである。任意配分の合成key上限を旧頂点案と区別した。zero T eventは解析IS式から有限lawを作れず拒否し、lambdaのhidden defaultは置かなかった。

[zero-compute preflight](../../../scripts/tracks/algorithm_codesign/preflight_parent_allocation.py)はsource/input 9 pathのbytes/SHAを確認しPASS。科学判断7項目はpending。algorithm sourceをimportせず、登録constructor・sampling・workerを起動しない。常にexecution不可を返し、authorization/markerは作らない。[receipt](../../../artifacts/track_b_parent_allocation_source_preparation/2026-10-14/v1/zero_compute_preflight.json)。

## 未実装・未検証と判断の境界

| 残っている項目 | 重要度と担当 |
| --- | --- |
| 比較方式、配分候補、proposal候補、正則化、accuracy/confidence/budget、primary、科学解釈の7項目 | 科学設計の採択。GPT判断が必要。旧案を自動採択しない |
| 新角度・inventory・endpoint/native価格、cost-aware全表の情報取得費用 | 対象採択後の取得。今回未取得で、資源優位を認定できない |
| production synth/runtime identity、旧cache bridge、strict phase operator guardの接続 | 実行sourceの追加技術準備。metadataだけでreuse可とはしない |
| 採択providerでのcontrolled native validation、atan区間/angle誤差certificate | 現在のmock検証を超える範囲。対象と実行範囲を固定して別途認可が必要 |
| 科学worker、confidence ledger、hard resource guards、partial結果/分割保存、launch gate | 科学契約の確定後に実装する。今回のpreflightはlaunch gateではない |
| 新source・contract、authorization-only child、fresh namespace/exclusive marker、outer process receipts | 別途明示実行承認を要する。旧source/results/authority/markerを再利用しない |

技術的に解消できたのは、任意配分・両root接続・全端点・digital復帰・phase/order・proposal入力の整合である。新構成の独立新規性、有限confidenceでのproposal選択、公平な強対照、native資源利益、研究GO/STOPはCodexで採択していない。今回、新たな文献調査・登録再採点・原稿作成も行っていない。

次は、ユーザーが開始を明示承認した場合にGPTが科学設計を判断する。技術準備完了だけで計算を開始しない。**mandatory STOP。**
