# Codex C：限定検証準備の完了・mandatory STOP

2026-11-04 JST / LOCAL。
**回路SOURCE対応・保証範囲の分離・人工34件PASS・次の12局所probe案を作成した。
新しい実H4のbuild/state-action/compile/数値計算/samplingは0。公開照合後mandatory STOP。**

- [CIRCUIT_SOURCE_MAP.md](CIRCUIT_SOURCE_MAP.md)：理論PF、trajectory条件付き平均、categorical平均とcontrol/phase/order/wrapperの対応。
- [VALIDATION_GAPS.md](VALIDATION_GAPS.md)：Bの6 request PASSと全18roundの間に残る不足。
- [SCIENTIFIC_VALIDATION_PLAN.md](SCIENTIFIC_VALIDATION_PLAN.md)：6r×2局所probe、入力/SOURCE・call/precision/判定/STOPの提案。
- [ARTIFICIAL_TEST_RESULTS.md](ARTIFICIAL_TEST_RESULTS.md)、[人工SOURCE](test_circuit_semantics.py)、[固定byte対応](SOURCE_BINDINGS.json)。

## 結論の範囲

Bの実H4証拠はPartial r1/q1,8の指定4 profileとFullの2 requestの条件付き平均である。
その結果を全18roundのmarginalや任意trajectoryの保証に拡張しない。
今回、production builderの合成則を2-mode人工fixtureでr1,2,4,8,16,32/K2、r3/K4、
非可換順序、負sign、scalar/global/relative phase、control0/1、Re/+Im wrapperへ接続した。
実H4の保存1e-12 preparation・signal側θへの接続とr≥2の局所回路照合が未実行である。

特にB runnerはcoefficient_atol既定0、今回の固定scopeは1e-12。
signal側τとcost model側τも別literalである。既存関数を使うだけではidentityを満たしたことにならない。
これらはSOURCE上の対応差であり、今回新H4の不一致を観測したという意味ではない。
boundary最適化のscalar再結合は保存Fのphase評価とbinary64の式が異なり、
既存event π64 boundだけで長qの回路policy全体を覆うとは言えない。この転送も未確認として残した。

別Codexが担当する全18round categorical包含を待たずに、そのSOURCE・結果・worktreeは変更しない。
このbranchはf490固定点から分離した準備成果であり、別作業の最新状態や完了を証明するものではない。
新しいscientific結果・evidence statusの変更がないため、旧synthesis/manifest/rawは更新しない。
discoverabilityはPROJECT_MAPとdocs索引の最小追記だけとする。

## 次の担当と必要な許可

| 担当 | 次の具体的行動 | 状態 |
|---|---|---|
| ユーザー | 本計画の12局所probe（親の全q直接構築とは別）を次の検証候補として選ぶ | 提案。科学実行許可は未取得 |
| Codex | 次の指示があれば、signal側θへの最小adapterと局所science runnerを実装し、人工検証・入力照合・commitで実行SOURCEを固定する | 提案する技術準備。今回そのH4 runnerは作成していない |
| ユーザー | 固定SOURCE SHA・12 probe・call範囲・STOPを確認し、実H4科学実行をチャットで別途明示許可する | 必須。承認証明書・保存承認文・grant新設は不要 |
| Codex | 上の実行許可後だけ、固定batchを1回実施し報告・公開照合後STOP | 未承認・未実行 |

重要な独立科学レビューを追加する場合は、そのレビューにも独立した許可が必要。
今回は主エージェント自身の技術調査・人工テストであり、独立詳細科学レビューではない。

`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`。
旧V1/V2/V3/R1、B、m17の条件付き判定、INCONSISTENT/u_num=null、
既存grant/ledger/marker、one-shot/retry/STOPを保全する。
状態準備・noise・総費用・全18round実回路へ自動移行しない。
