# SOURCE差分とCodex Aへの将来統合ハンドオフ

[入口](README.md) / [SOURCE inventory](source_inventory.json) / [未採択binding候補](source_binding_candidate.json)。このbranchをCodex Aへ自動統合しない。

## 変更した範囲

| 宣言・ファイル | 差分 |
|---|---|
| Driver `run_pair` | axis/trial/stageの初期化、candidate hash/counterと完了prefixの分離、last completed、現在inflight、typed failureのprovenance |
| Driver `Stream` | 失敗stage、write返却bytesとfile-hash coverageの区別。正常envelope・chain式・flush policyは不変 |
| Driver `execute` | child failure logへtrial provenanceを転送。secondary log/cleanup失敗が原trial例外を置換しない |
| 新tests/人工fixture | F01 26件、旧SOURCEで編集前に固定した正常8ケース |
| 専用文書・metadata | 原因/保存scope/test結果/新SOURCE pinと将来binding |

上記以外のdriver top-level宣言・定数はAST同一。`LegacyEndpoint`、`Boundary`、`Meter`、`FunctionObserver`、`load_saved`、`schedule`、angle/seed設定・FROZEN_PINSも不変。science源・helper・既存testsは変更なし。旧契約候補、SOURCE inventory、publication receipt、共通索引を上書きしていない。

[source_diff.patch](source_diff.patch)は固定旧版に対するzero-context差分記録であり、自動適用・統合用の承認ではない。

## 影響する将来のbinding

旧driver SHA256は`d7947af16baccbd3e0707fa9b4ffcd26650a6569eb6606ee3c08ec443d1dd7ee`。新SOURCEのbytes/SHA256・関数行位置は[新inventory](source_inventory.json)、完全SOURCE commitは[publication receipt](publication_receipt.json)の`technical_payload_commit`で固定する。

将来統合では、Aが公開したSOURCEと本branchの差分を確認し、F01の3宣言内の修正を取り込んだ**新しい正式SOURCE**を固定する。その最終driver SHAを正式契約の`source_pins`、SOURCE inventory/release digest、外部admissionが参照する採択契約digestへ反映する。旧candidate/receiptを修正版に書き換えず、別versionを作る。

新しいfailure field・child log内のcanonical文字列・null hash/partial-tailの意味を、将来の消費側guard/保存検査に合わせる。旧stream envelopeは不変なので全schemaを書き直す必要はない。成功系の出力・call schedule・trace比較則も維持する。

本branchの[未採択binding候補](source_binding_candidate.json)はdriver pin差分だけで、τ/Kφ/J、seed、schedule、admissionの採択を代行しない。Codex Aの未公開成果・現在の候補値・将来commitは参照も推測もしていない。

## まだ必要な承認・作業

1. **ユーザー**：Codex A公開成果との統合・差分確認を別途指示するか判断する。
2. **Codex（指示後）**：両固定SOURCEを照合し、F01とAの採択準備を両立したSOURCE/pinsを新versionで固定する。実G7計算は行わない。
3. **ユーザー**：正式条件・SOURCE/契約を採択し、必要な外部atomic admissionと一束科学実行を別途承認する。

外部admissionのatomic永続claim/source-release bindingと、出力作成前の失敗保全は[既存F01監査の別要件][F01]として未実装のまま。この課題を今回実装・有効化したとはしない。

本修正の範囲では、F01以外の**新たな**必須技術blockerは確認していない。これは全driver再監査の結論ではなく、限定修正・人工検査のscopeに限る。実入力correctnessは未検証、C1/G9/P2/P3は対象外。**mandatory STOP。**

<!-- FIXED_REFERENCES -->

[F01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6576e606d4edcc6c53f095ee5bdf750defa99a42/docs/research/track_b_g7_g9_p1_final_driver_delta_20261019/technical_findings.json
