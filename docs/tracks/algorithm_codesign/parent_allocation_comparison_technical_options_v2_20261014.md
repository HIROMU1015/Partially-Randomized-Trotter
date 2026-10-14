# Parent allocation：比較計画の技術整理 v2

2026-10-14 JST。**科学契約ではない。比較候補・精度・confidence・上限・primary・GO条件は未採択。登録評価・新合成は行っていない。mandatory STOP。**

[旧草案](parent_allocation_minimal_comparison_draft_20261014.md)を保持し、今回の汎用sourceで可能なことと実行前の不足を整理する。旧草案のr=1、頂点候補、lambda=1、最大24 law、最大6合成等を今回採択したものとは扱わない。機械可読版は[comparison_options_v2.json](../../../artifacts/track_b_parent_allocation_source_preparation/2026-10-14/v1/comparison_options_v2.json)。

## 比較自由度の分離

ordinaryとclosed P5の両rootに同じ`ParentAllocation`を接続した。任意の有理数配分を受け付け、頂点に限定しない。非rootは既存構成を保持する。表現、配分、proposal、合成価格を別々の入力として記録できる。

| 技術的な選択肢 | 実装状況と条件 | 科学設計で決めること |
| --- | --- | --- |
| common-angle / 任意配分 | 両表現で同じadapter。非負性と総和をexactに確認 | 比較対象、配分候補集合、各方式に与える自由度 |
| canonical | digital係数比例のdyadic law。実proposalでweightを補正 | 比較に含めるlaw |
| root-local cost-aware | rootの費用を外部入力し、root内だけproposalを変更 | この限定最適化の意味、必要な費用情報の取得条件 |
| global cost-aware | 明示的な有限event・cost表に対する`reweight_events` | 全表の情報access、表取得・初期化・online費用の会計 |
| regularized | 明示的な正のlambdaが必須。実費用にlambdaを加算しない | 正則化の採否と値、候補選定の公平性 |

局所root lawとglobal ISは異なる。後者の全表を無料に取得できること、rootだけから全表の最適proposalを生成できることを仮定しない。未知の古典online費用は0にしない。

## zero-costとfinite-bit

正の費用に対する既知式は`q_i ∝ c_i/sqrt(C_i)`。これはcost×second momentの解析式であり、range・整数shotを含む有限confidence問題の最適解とは扱わない。

T費用0のpositive-coefficient eventがある場合、解析式をそのまま有限lawへ変換できないため拒否する。全費用0の場合も式は定義されない。hidden floor、lambdaの自動採択、supportの削除は行わない。正則化、別の有限confidence proposal、canonicalのみの比較などの採否は科学設計へ戻す。

係数の平方根は固定bitの区間とmidpointで扱う。実際のdyadic proposalはpositive supportを維持し、`w_i = tilde_c_i/q_i`を使用する。実proposalからmoment/range/acceptanceを算出する仕組みはあるが、登録lawや十分Nは今回生成していない。平方根・確率bitが不足すれば例外となり、bits/toleranceを自動調整しない。

## 新角度・cache・端点

頂点のみで二表現にL候補ずつ与える旧案なら、新positive primitive keyは重複除去前に最大2L。L=3の「6件」はこの条件に限る。**任意M配分では最大2MLが静的上限であり、汎用familyを6件に制限できない。** common点のcacheがない場合や、他のprovider primitiveが必要な場合は別のcoverageが必要である。今回は登録角度・inventoryを生成していない。

`rotation_key`は正の`atan(ratio)`をラジアンで扱うRZ primitiveを識別し、precision、tool名/version/source SHA/seed、strict phaseを含む。負回転は実際のsequenceのadjointを用いる。同じ角度に見えることだけをreuse条件にしない。

`check_cache_metadata`は正規化されたmetadataのkey/spec/sequence SHA/保存guard値を照合する。旧cache schemaからのbridgeは未実装であり、metadata照合は独立したoperator guardの再計算ではない。将来のworkerではraw radians adapter、tool/runtime identity、strict operator guard、費用coverageをbindする必要がある。

端点はexact構成を使う。pure-childは`(-i sigma) Q_i Q(u)`、pure-parentは`Q(u)`にparentの位相を付ける。controlled位相はancilla上のS/S†/Zに戻し、巨大有限ratioで代用しない。zero eventは取得・samplingから除外する。新しい端点native費用はまだ取得していない。

## 科学会計へ接続する前提

係数誤差と、外部認証されたangle/native誤差を別々に入力する。後二者のmissingを0とせず、全event coverageを要求する。十分N・統計残余幅のpolicyは未実装であり、採択されたaccuracy/confidence ledgerへの接続が必要である。

新ledgerは旧D1〜D3のN、alpha、分類、結果へ上書きしない。native価格は各方式へ同じtool・精度・取得機会を与えた新inventoryへ対応付ける。T/CX/native1Qの重複加算を避け、whole-circuit最適化の実測と加法的価格を区別する。どのmodelを比較に用いるかは未採択である。

## 依存関係と実行準備の残り

今回のsourceはstdlibと固定済み`return_aggregation.py`、`g7_generator.py`、`g9_p5.py`に依存する。G9接続は`g9_native.py`を呼ぶが、登録providerの実回路評価は行っていない。技術テストruntimeはPython 3.10.12、1 worker、numeric threads 1。production合成tool/runtimeを今回確認・採択したとは扱わない。

| 実行前に必要な処理 | 現在の状態 |
| --- | --- |
| 比較方式・配分/proposal集合・lambda・accuracy/confidence/budget・primary・解釈規則 | 7項目とも機械可読版でnull、GPTの科学判断が必要 |
| 新角度・合成key・費用表のcoverage確定 | 科学候補の採択後。登録表を今回作成していない |
| 旧cache bridge、tool/runtime/source identity、strict guardと誤差入力の接続 | 次の実行sourceで実装・検証が必要 |
| 科学worker、resource guard、partial failure処理・保存分割 | 未実装。許可された対象と資源上限の確定後に準備 |
| 新source/contract、authorization-only child、exclusive marker、outer receipts | 未作成。旧authority/markerの再利用は禁止 |

将来の性能設計では既存timingとCPU/RSSを確認し、keyの独立性・cache共有・aggregate memoryを考慮する。新規契約のphase/total wall capは共通指示に従いデフォルトなし。CPU/RSS/output/log/call上限の具体値は今回採択しない。

技術的な失敗候補は、不正配分、bit不足、zero-cost解析式、native/provider coverage不足、cache identity/strict guard不一致、誤差予算不足、資源超過、部分取得。新sequenceを取得しなければ分からない価格をproxyで成功扱いしない。

## 実行を伴わない確認

[zero-compute preflight](../../../scripts/tracks/algorithm_codesign/preflight_parent_allocation.py)はJSONとbytes/SHAだけを確認する。未決定の7項目を表示し、常に`science_execution_allowed=false`を返す。実行flag、worker、marker、登録constructorsを持たず、execution launch gateの代替ではない。

今回の完了点は科学設計へ戻せる汎用sourceと技術証拠の公開である。次の研究判断・重要科学レビューの開始には明示承認を必要とし、計算実行には別の明示承認を必要とする。**mandatory STOP。**
