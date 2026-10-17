# M1-S 保存γ境界とπ/3：正式採択・契約固定

2026-10-17。公開候補commit `e3828965dec292e9c66f6e7d8480e46059701183`に対し、直前の「次は限定gate・契約候補の正式採択」という案内を受けたユーザーの「次の作業に進んで大丈夫」を、限定gateと契約候補の採択および正式記録固定・公開への承認として保存した。引用文を「正式採択します」等へ書き換えていない。科学実行の承認には拡張しない。

[元候補](m1s_gamma_pi3_interval_gate_contract_candidate_20261017.md)と[準備引継ぎ](m1s_gamma_pi3_preparation_handoff_20261017.md)は履歴として保全する。[正式machine契約](../../../artifacts/track_b_m1s_gamma_pi3_formal_adoption/2026-10-17/v1/execution_contract_adopted.json)は独立した新規ファイルで、元候補から変更した項目はstatusと3つの採択flagだけである。SOURCE・入力・閾値・判定規則・resource上限・scopeは同じである。

## 採択した条件

linear H4 1.00 Å／STO-3G／8 modes／保存DF rank12、G3/G6/G9、保存δ=1/8,1/16 Ha^-1、R=2,4、T=1,4 Ha^-1を対象とする。保存V1の48対照を全件報告し、G3/G6の32対照だけを閾値照合対象、G9の16対照は絶対gain未解像のまま照合せず保持する。元108方式行・36zero control、signal、state、Hamiltonianは再計算しない。

保存γ*区間[L,U]、保存π区間[p_lower,p_upper]から、`3*U<p_lower`ならSATISFIED、`3*L>=p_upper`ならNOT_SATISFIED、どちらでもなければUNRESOLVEDとする。厳密なγ*<π/3という限定decision gateであり、shared有利側が根の上側であるという保存意味論を要求する。π/3は保守的nearest-consistent-branch routeの一round角誤差上限候補であり、完全なdecoderの採択・実装やRPE全体の成功保証ではない。machine内の`PROPOSED_CONSERVATIVE...`／decoder falseは、この限定と整合する記述として変更しない。

浮動小数点閾値・丸め許容幅・区間細分は導入しない。保存区間を使う有理数の比較だけである。条件付きcoverageを採択しても、元signal・保存oracle・IEEE／BLAS／trig・SOURCE correctness・platformを認証したことにはならない。`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`を維持し、科学的GO、RPE成功、整数shot削減、native費用、総資源改善を自動認定しない。

CPU1 worker、GPUなし、memory2GiB、input1MiB、result/log各16MiB、最大48対照、fraction field1024、rational65536 bits、文字列表現25000字を固定した。現行ユーザーpolicyに従い、この新契約はCPU／wall capなしである。旧V1のCPU160秒／wall320秒や既存契約は保全し、変更しない。実入力runtimeや数値correctnessは未検証。

one-shot、technical failure時の消費・部分結果保全・自動retryなし・mandatory STOPを維持する。不正なinput identity／schema／区間意味論／資源・保存処理の失敗は科学的不成立と混同しない。研究の継続／縮小／方向転換は結果取得後のユーザー／GPT判断へ返す。

## 固定identityと権限

SOURCE S：`2f800f179c7b29589b2f9a7e532408b16b8fe6d2`、8ファイルのSHA256は正式machine契約で固定した。候補SHA256：`7989546fe64e644c01cb0080bd4623cc1b23504a5f421f446078742646db4c78`。入力はcommit `b34572770e7c2105e16da003af050d789f490662`の146764 bytes、SHA256 `bb2d09e6963fdc8b8871170ba28e98c36f4e353f452e50870103a7e97bf2e55d`。正式契約SHA256は採択記録とexecution readinessに記録する。

`decision_gate_adopted=true`、`adopted=true`、`coverage_model_adopted=true`はユーザーの上記採択を示す。**`contract_grants_execution=false`、`scientific_execution_authorized=false`**を維持する。今回、新authorization／grant／markerを発行しない。既存のV1承認・消費済markerは新照合に流用できない。

SOURCE・input・candidateと正式契約の差分、schema、SOURCE／commit順序、権限未付与・guard閉鎖は静的／metadata検査に限る。科学JSONのdecode、境界とπ/3の数値照合、toy numerical correctness test、追加科学レビュー、native compile、shot forecastは実施しない。今回の採択・SOURCE／guard静的検査は85件PASS、保全監査は12件PASS。旧非index7742ファイル、親の追跡1284ファイル・index・HEAD、既存210worktreeのHEADを保全した。これらはローカルのmetadata／byte監査であり、数値correctness検証や外部再現ではない。静的checkerは`scripts/tracks/algorithm_codesign/check_m1s_gamma_pi3_adoption_static.py`である。

## 次の段階

[採択記録](../../../artifacts/track_b_m1s_gamma_pi3_formal_adoption/2026-10-17/v1/user_adoption_record.json)、[実行準備状態](../../../artifacts/track_b_m1s_gamma_pi3_formal_adoption/2026-10-17/v1/execution_readiness.json)、[検査・証拠索引](../../../artifacts/track_b_m1s_gamma_pi3_formal_adoption/2026-10-17/v1/evidence_index.json)、[公開照合receipt](../../../artifacts/track_b_m1s_gamma_pi3_formal_adoption/2026-10-17/v1/publication_receipt.json)を参照する。

次は固定SOURCE Sとこの正式契約Cを指定する、ユーザーの別途明示的なone-shot数値照合実施承認である。承認後に限り専用authority-only子Aを固定し、初回数値照合・保存監査へ進む。現在は科学計算0件、authorization／grant／marker0件のまま、mandatory STOP。
