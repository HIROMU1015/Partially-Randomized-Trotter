# ユーザー・GPT向け：実装監査の引継ぎ

**G6/G7の局所有理生成と有限bit event記述、DF側の成分・basis取得とcontrolled Z/ZZ primitive、ordinary partial-S2とHadamard回路builderは存在する。G6の入力誤差・実π/W・事前zero・signed timeを保持したEnd-to-End測定経路は未完成である。**

## 最重要の不足

1. **入力binding**：元raw DF、float抽出係数/basis、有理G6 targetを区別する証拠が必要。Fraction化は元Hの誤差認証ではない。D/R・identity/constant・label↔component/hashを固定する。
2. **既存requestの不互換**：`DFPartialS2StepRequest`はordinaryのangle/probability/normalizationを明示的に検証し、repeated requestも同validatorを使う。G6 eventへ同じfield名を付けても互換性はない。既存guardを弱めていない。
3. **有限bitとsigned-time**：自動H/K・angle enclosureのSOURCE接続はない。G7 eventのσは外部文脈、G8 bindはdefault+1なので、汎用負時間のbindingが不足する。旧正時間契約の失敗を認定したものではない。
4. **測定・全trial**：Hadamard builderは測定命令を作るが、実backend/shot/IIDは含まず、G8もquantum measurements=0。実W、zero、全trial分母、backend failureを結ぶestimator APIが必要。
5. **誤差・費用**：U/digital moment/rangeは再利用できるが、元DF・basis/native・statistical/family・PF/multi-stepを同targetへ戻した契約はない。未知費用・誤差は0としない。

詳細は最重要の[Interface Contract Matrix](interface_contract_matrix.md)、[SOURCE trace](end_to_end_source_trace.md)、[誤差](precision_and_error_contract.md)、[測定・費用](measurement_and_cost_contract.md)。S/Tの固定commit、実読範囲、関数位置とhashをmetadataへ記録した。今回の静的対応を数値correctnessや外部独立再現の証拠にしない。

## 残る判断と次の担当

**ユーザー**：保証target、coverage、precision/confidence、providerとwrapper接続方式、single/multi-stepの範囲を固定し、実装へ進めるか判断する。重要な新しい数学・科学判断が必要になった場合だけ、GPTレビューの開始を別途承認する。現時点でCodex Aの理論研究順位を変える判断はしていない。

**Codex（別途実装承認後）**：上記が固定された範囲で、input binding、lossless event/trial記録、既存controlled primitiveへのbinding、weighted estimator、受入テストV1〜V6へ進める。G0〜G7の[依存順計画](minimal_implementation_gap_plan.md)は提案であり、実装・test承認ではない。

**ユーザー（新規数値実行前）**：分子decode/係数評価、回路build/native合成、sampling/測定等を含む具体的scopeには、別途の明示実行承認が必要。今回、実行契約・authorization・grant・marker・claimを発行していない。

## 独立性、保全、STOP

基点は `5f17a7548b7eef1976f45f5d09bdc979e2e0a81b`。非Pauli資料は固定 `128a0576…` をread-onlyで参照し、他branch merge・Codex A成果の自動統合はしていない。Aの未commit/index/lock/process/作業ファイル・結果は読まない。保全確認は監視した範囲を明示し、Aの全状態を監視・比較したとは主張しない。

D＋E、Parent限定成果、G6〜G10、DF/Gaussian closure、旧STOP、他研究状態を維持。新科学計算、保存数値再解析、科学SOURCE変更/import/実行、formal test作成/実行、回路build/compile/合成、新保証・重要科学レビュー、実行契約/authorization/grant/marker/claimは0件。

[管理検査](verification.json)と[公開receipt](publication_receipt.json)は文書・hash・provenance・保全の証拠。公開・最終HEAD/bytes照合後 **mandatory STOP**。実装・科学実行・レビューへ自動移行しない。
