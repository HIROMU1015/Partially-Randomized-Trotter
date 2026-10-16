# M1-S有限step複素信号pilot v2：正式契約・採択記録（2026-10-16）

**ADOPTED / CONDITIONAL_ENGINEERING_NOT_CERTIFIED / SCIENTIFIC_EXECUTION_NOT_AUTHORIZED / mandatory STOP。**
ユーザーがcommit `68fa558043bb7b18f40bdc02f45d2c0bba95f8fa` のv2候補を正式採択した。本書と新しいmachine契約は採択の記録であり、SOURCEや科学条件を改訂しない。
科学計算は今回明示的に承認されていない。新しいexecution authorization／grant／markerを作成せず、guardを閉じたまま停止する。

[採択済みmachine契約](../../../artifacts/track_b_m1s_finite_signal_v2_formal_adoption/2026-10-16/v1/finite_signal_contract_v2_adopted.json) SHA256 `36fcbad56a6f9a241f4ce2b796b45dc39887a931b0b6b388360f631f8382bb10`。
[ユーザー採択記録](../../../artifacts/track_b_m1s_finite_signal_v2_formal_adoption/2026-10-16/v1/user_final_contract_adoption_record.json)／[今回の発言](../../../artifacts/track_b_m1s_finite_signal_v2_formal_adoption/2026-10-16/v1/user_adoption_statement.txt)／[直前応答との文脈](../../../artifacts/track_b_m1s_finite_signal_v2_formal_adoption/2026-10-16/v1/conversation_adoption_context.json)。
会話の本文を転記したものであり、IDE情報や時刻・署名をユーザー承認へ補わない。以前の採択推奨や文例自体は採択承認と扱わず、今回の明示発言を根拠とする。

## 固定する条件

|対象|正式採択した条件|
|---|---|
|モデル・状態|linear H4 1.00 Å／STO-3G／8 modes／DF rank12、生成順G3/G6/G9、保存数値oracleを元norm²で正規化した期待値、全256次元|
|grid|delta=1/8,1/16 Ha^-1、R=2,4、T=1,4 Ha^-1、zero control込み108行（非零72＋zero36）|
|比較|shared ±／ordinary／independent ±、同一Tと概念的独立label数sR、負枝は同順sign reversal。native回路費用の公平性は保証しない|
|精度|signal radius 2^-24、principal phase radius 2^-18 rad、visibility floor 1/4。閾値・grid・分割を結果に応じて変更しない|
|モデル差|raw保存bitのH_fullとH_tilde_j=HD_j+HR_saved_jを区別。exact Hermiticity条件下のB_jとeta_j(T)=abs(T)B_jを別台帳に保存|
|原因帰属|raw-H基準の総複素差を純アルゴリズム誤差と呼ばない。純アルゴリズム差は三角不等式の大きさ区間のみ、point_value=null、厳密な分離測定ではない|
|位相|モデル位相上界はvisibility下界・zeroとprincipal branch cutからの分離条件成立時だけ。成立しなければUNDETERMINED。unwrapしない|
|coverage|IEEE RN／BLAS／norm／hypot／trig／log／underflow等の条件付きモデルを採用。未検証でplatform certificateではない|
|resource|CPU1 worker、GPUなし、5 thread変数各1、4 GiB、result/log各128 MiB、context4 MiB、wall/CPU time capなし。その他のcall/event/bit/cache制限も候補どおり|
|correctness・失敗|将来の別承認one-shot内に初回correctnessを含む。failure時は完了値・予算・残行未評価を保存、no retry、mandatory STOP|
|解釈|複素coherent平均信号の診断。rhoのsmall-step仮説と有限step実測を区別し、lambda_R²を含む絶対尺度を保持。精度以下のG9差を相対率で成果へ格上げしない|

[採択した候補本文](m1s_finite_step_complex_signal_pilot_contract_v2_candidate_20261016.md)と[意味論・精度設計](m1s_finite_signal_precision_and_semantics_v2_design_20261016.md)を、指定candidate commitの固定内容として参照する。候補原本は履歴として未採択表示のまま保全する。
新旧machine契約の差は `status`、`adopted`、`coverage_model_adopted`、`adoption_record` の4項目のみ。
`adopted=true`と`coverage_model_adopted=true`はユーザーによる条件付き契約・算術モデルの採用を示す。数値correctness、platform coverage、真の厳密固有状態、科学的GOを認証するflagではない。
`contract_grants_execution=false`、実行authorization未発行、実行許可falseを維持する。

## 来歴・保全・次の承認

SOURCE S=`fbc2edea3816bf8f68508f62c6da595631df03b6` は9ファイルとも変更しない。[hash台帳](../../../artifacts/track_b_m1s_finite_signal_v2_formal_adoption/2026-10-16/v1/source_input_contract_identity.json)で、SOURCE／入力／law／旧独立結果／両レビュー／結果schema／候補／採択契約／採択記録を結ぶ。
SOURCEより後の独立した正式契約commit Cとして固定する。将来実行する場合もさらに別の明示的な科学実施承認と専用authority A、S<C<A、clean HEAD=Aが必要である。
追加の新しいGPTレビューは開始していない。SOURCEの静的検査は数値correctnessやruntime完遂の実証ではない。
旧v4 TECHNICAL_STOP/null、v5・独立one-shot結果、消費済marker、旧SOURCE／契約／レビュー／親・他worktreeを保全する。未保存旧full-H／HD直接比較のUNDETERMINEDは解消しない。
[実行準備状態](../../../artifacts/track_b_m1s_finite_signal_v2_formal_adoption/2026-10-16/v1/execution_readiness.json)／[採択済み事項と未検証事項](../../../artifacts/track_b_m1s_finite_signal_v2_formal_adoption/2026-10-16/v1/adoption_status_and_remaining_conditions.json)。
今回の新科学計算・算定式実入力評価・Pauli生成・数値test・finite signal・simulation／compile・authorization／grant／markerは全て0件。
**採択記録固定後はmandatory STOP。次の科学計算はユーザーの別途明示承認まで開始しない。**
