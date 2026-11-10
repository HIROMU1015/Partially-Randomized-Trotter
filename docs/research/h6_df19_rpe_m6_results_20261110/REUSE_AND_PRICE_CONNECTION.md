# m6再送指示の照合・価格設計への接続

2026-11-10 JST。**今回のCodex B指示は、既に公開したm6限定campaignと同じ科学条件・2候補を指定している。既存成果を再利用し、新しい科学計算・数値再解析・人工test・価格処理は行わなかった。** 本メモは保存metadataの照合と出典の接続だけを追加する。旧SOURCE・結果・STOPは保持する。

## 再利用する完了結果

- 科学SOURCE：`c9aa89548a3ab337e1a8dbc5cf219a31860820d6`。
- 結果公開commit：`7903e3633cba233eeea60f94ddf8a720198a14b8`。[REPORT][RESULT]、[数値正本][RAW]、[m0..6 coverageと価格handoff][COVERAGE]。
- Linear H6/1.00Å/STO-3G、保存tol-only1e−8 DF19/completion、12 mode・Nα=Nβ=3のsector400、保存DF-H ground target、保存数値state条件付き、P0。CA=.00159360010199040 Ha、h=.1、βtotal/stat=.4/.2、各方式αtotal=.05・αaxis=.05/26、13round・無shift・固定priorは今回の指定と一致する。

| arm | candidate ID | m6/T6.4/q_RPE64の構成 | 保存N_X / N_Y | 保存資格 |
|---|---|---|---:|---|
| PF | `201a8e365ab6f4af6ebc` | Full19/m5_best/q_PF4/δ1.6 | 705 / 705 | ideal条件付きround PASS |
| PR | `ef41b65942d56977d6cd` | prefix7/m5_best/R_outer16/K2/q_PF4/δ1.6 | 772 / 772 | ideal条件付きround PASS |

reference2経路、corrected/rawの独立伝播、bias分解、disk・phase・radius・N、人工差分3件、旧m0..5との対応、残るm7..12と次の最小案は既存REPORTに揃っている。actual実装/law・厳密ground同定・全13round成功・総費用の未資格も保持する。保存Nを再計算したり、既存campaignを独立な新標本として数えたりしない。

## 指定価格設計との静的照合

対象は利用者指定の固定commit **`4c965ff69e3619a7b2f8313a2093cde679f542b8`** の[価格設計REPORT][PRICE]と同directoryの[COVERAGE.json][PRICE_COVERAGE]。価格REPORTのGitHub取得全文とローカル同commitのblob bytesが一致した。

照合した保存欄は次のとおり。

- `configuration`全体はm6結果の保存設定と一致する。価格側`N_metadata`の旧12行はm6正本の`previous_rounds`と同じである。
- `m6_planned_only`のT6.4、formula、q_PF4、δ1.6、PF prefix19、PR prefix7/R16/K2は完了候補と一致する。
- `signed_weights_copied`の11 piece順序は両候補の保存weightsと一致し、`allocations_copied["16"]`はPRの保存整数配分と一致する。
- 価格側`binding_reference`が指すD1結果のH/state/snapshot/rank-policy bindingはm6正本と一致する。compiler期待hashも同じ`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`である。

価格設計の固定版はD2公開版95d01b6aを基準にm6科学資格を`PENDING_OTHER_AGENT`としていた。そのm6欄へ接続できるN側の完了参照が上記7903e3633である。価格側の固定REPORT・coverage・branch・worktreeを上書きせず、こちらの[COVERAGE.json](COVERAGE.json)の`m6_price_handoff`を受渡し元とする。

handoffにはcandidate、T/m/q_RPE/q_PF/δ、prefix/R/K、signed weights、配分・event数、H/state/snapshot binding、X/Y別N/α、law model、保存tail定数、compiler/control/full-wrapper期待条件、P0がある。歴史的binding.T=.8と実発展時間candidate.T=6.4を区別する。BはN側のnormalizationへ一回反映しており、Cへ追加乗算しない。

この一致はmetadataの互換性であり、compiled価格、モデル精度、native/postcompilerの移送、期待価格lawの資格を新たに検証した結果ではない。価格設計V1やq2/q4の価格pilotは実行せず、C/Gはnullのまま保持する。全RPE成功・PF/PR総費用順位も判定しない。

新規実装・検証が必要な差分を確認できなかったため、共通AGENTS.mdの既存成果再利用規約に従い同じm6の再実行を保留した。今回追加したのは本メモと直接の文書リンク・研究ノートのみ。科学call、科学的retry、追加test、価格予測/fit、sampling、build/compile、N×Cは全て0。主要成果物の公開照合後**mandatory STOP**。

次の担当：**ユーザー**がm7以降の新しい科学範囲を判断する。**価格担当Codex**は別途許可された範囲で既存m6 handoffを参照する。**本担当Codex**はm6を重複実行せず、追加許可まで次roundや価格評価へ進まない。

[RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7903e3633cba233eeea60f94ddf8a720198a14b8/docs/research/h6_df19_rpe_m6_results_20261110/REPORT.md
[RAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7903e3633cba233eeea60f94ddf8a720198a14b8/docs/research/h6_df19_rpe_m6_results_20261110/run01/result.json
[COVERAGE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7903e3633cba233eeea60f94ddf8a720198a14b8/docs/research/h6_df19_rpe_m6_results_20261110/COVERAGE.json
[PRICE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c965ff69e3619a7b2f8313a2093cde679f542b8/docs/research/h6_df19_price_design_20261110/REPORT.md
[PRICE_COVERAGE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c965ff69e3619a7b2f8313a2093cde679f542b8/docs/research/h6_df19_price_design_20261110/COVERAGE.json
