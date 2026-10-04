# Track A PM-1最終承認・authorization確定

2026-10-05 JST。利用者から提示された外部review `APPROVE_PM1_EXECUTION` を受領した。
現在の運用statusは `PM1_FINAL_REVIEW_APPROVED_AUTHORIZATION_FINALIZED_AWAITING_USER_LAUNCH`。
**PM-1はまだ実行していない。明示launch指示を待つ。**

## 確定した変更

review対象bundleは `234369f45e3fea7c825a1b4af85fba6c1edf98c3`、
authorization draftは `b6175a0f4fdeb1d2f0cd61c09dce37675624b73a`。
[authorization JSON](../../artifacts/resource_applicability/pr2_pm1_discard_authorization/2026-10-04/execution_authorization_v1.json)の
`final_review_approved`だけをfalseからtrueにし、独立commit
`bf9eaeec868361df0c8e05d06e9570a2bfc5a7a4`へ固定した。commitは1 file、1行追加・1行削除であり、JSONの他fieldも全て不変。

確定後authorization SHA-256は
`2113978b360ca763071b860c7ea2d14b83eccc68a471510a98da5ea5c2282530`。
actual source `fd7552edc0334ccf57ecf501a128c85c8d22822a`、134 source blobs、
sealed plan SHA-256 `cae692bee2be748ddbf17bace2a5652613537a244fe94238cdf669f5d2ca5624`、
fingerprint `144824b70264dd3d7d1d22898afbb1f9e1c248f105ecac8096f31a2c8f78ec9e`、
8 candidate fingerprints、16 wrapper keys、compiler/environment/caps/permissions/root/outputは不変。

## 今回の検査と監査履歴

科学データ禁止guardを先に入れ、`validate_execution_gate`だけを照合した。
committed true authorizationで実gateは `PASS_NO_SCIENCE`。backendやscience runnerは呼んでいない。
限定guard付き7-file suiteは201 passed、fail/skip0。tiny synthetic circuit testsを含むlocal検査で、
H4 science、immutable CI、外部再現ではない。NPZ stat/hash/load、runtime/cache access、H4 signal/build/compile、
random sampling、quantum shot、GPU操作は今回全て0。PM-1 output/registryは未作成。

照合harnessの初回にはPM-0 manifestのmappingをlistと扱う読み取り側エラーがあり、
科学コードを変えず形式の解釈だけ直して照合を完了した。この診断失敗もauditに保存する。
以前のNPZ4件stat/hash（load0）違反も消去せず、今回のaccess0と区別する。

[今回のfinalization audit/receipt/tests/manifest](../../artifacts/resource_applicability/pr2_pm1_finalization/2026-10-05/)を追加し、旧draft audit/manifestは書き換えない。
旧draft manifestのauthorization hashは**review bundle commitのfalse blob**に対する履歴であり、
現在のtrue fileを検査するmanifestではない。確定後は今回のmanifestを使う。

## 次の停止位置

将来の実行条件はH4 linear 1.00 Å、STO-3G、DF rank12、8 system qubits、T=0.8、
B0 discard rank4/5 × q=1/2/4/8、delta=0.8/0.4/0.2/0.1、r=K=0の8構成のみ。
signal8、full deterministic wrappers16、CPU1、BLAS1。既存development比較5件は保存値を使い、
random追加、held-out、GPU、retry/resumeは0のまま。

利用者の明示launch後だけ[固定command](pr2_pm1_discard_execution_authorization_v1.md#一回制限と将来の固定command)を一度実行する。
固定output `artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04`は日付を含め変更しない。
成功 `PM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW`、failure `IMPLEMENTATION_GATE_FAILED`のどちらもmandatory STOP、
next-stage=false、research_decision=null。PM-2、ε/P sweep、新geometry、strong synthesis、
higher-order PF、energy/RPE接続、Track B統合を自動認可しない。

承認receiptは利用者がこのチャットで提示したreviewの記録であり、runnerによる独立認証とは主張しない。
今回の承認反映・local commitも利用者のlaunch指示ではない。
