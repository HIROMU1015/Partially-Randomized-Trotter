# B/C HANDOFF：固定24候補normalization Phase 0

実装SOURCE `6ed8c6873311085bb3b39799b268105b0bc97211`。対象候補/SOURCEは`f98982118324bcc8402ed513b5c720b37bb03e87`のPhase 0を不変で使う。今回は実normalizationを評価していない。

## 入力と起動

`PLAN.json`に24のrequest、保存tail値（転記）、input/context hash7件、SOURCE9件、law/identityとR配分を固定した。保存入力の絶対pathは公開PLANに含めず、元SOURCE索引の既知pathを実行時だけ参照する。

- H4 p1、S2/δ=.1/K2、m0/6/12。
- H6 p7、4th(m5_best)/δ=1.6/K2、m4/8/12。
- law/metadataは既存B `7bca8f9642d1c42ca3fede4d31b2cdd8dcef46ae`のrequest_contextと一致。H/state/snapshot・completion・RPE条件・full measurement wrapper/expected compiler/P0を維持する。
- 過去receiptのstatus/ground資格は保全する。preflightは保存値のbinding確認であり、新しいnormの導出・保証や現在のユーザー許可の証明ではない。

既定CLIはpreflightのみ。`--execute`はユーザーが固定Phase 0を明示承認した後だけ使用する。R/system/thresholdを変更する引数や次段階の自動起動はない。

## 実行後のfield

| field | 意味・単位 |
|---|---|
| system/m/T/q_RPE/q_PF/delta | 元のround/発展時間(a.u.)/反復数。qRPEとqPFを別に保持 |
| candidate_id/formula/prefix/K/R_outer | 元のcanonical candidate。Phase 0のIDを変更しない |
| structure | signed_weights、全piece allocation、event_count=qPF×R、scalar/law規則 |
| request | B/C共通identityとtarget law_binding。価格/Nの資格を生成しない |
| normalization.log_B | 既存finite K2 normalizerの全interrogation対数（無次元） |
| normalization.optimistic_shot_inflation_log | 2log B。actual Nは計算しない |
| normalization.one_ulp_accumulation_log_sensitivity | binary64丸めの感度診断。rigorous error boundではない |
| normalization.leading_small_tau_term/max_abs_event_tau | 既存screenの診断。採択や小τ適用窓の新保証に使わない |
| attenuation.value/status | exp(−log B)と表示品質。0へのunderflowは専用status |
| shot_inflation.value/status | exp(2log B)と表示品質。overflowはnull、logを保持 |
| phase_radius_N_status / N,C,G | NOT_EVALUATED / null。十分Nや費用は含まない |
| recommended_R / scientific_candidate_PASS_FAIL | 常にnull。Rの科学的採否・最適値は返さない |

値が未資格なら`normalization.status=NUMERICAL_NORMALIZATION_UNRESOLVED`、指数表示に元値がなければUNAVAILABLE_NORMALIZATION。FINITE_BINARY64は計算値の表示statusであり、物理的RPE成功やactual法則の資格ではない。Infinity/NaNを出力せず、null/0に付随するstatusとlogを必ず保持する。

## ファイルと消費条件

- `result.json`：scientific_source、24 records、errors、入力/SOURCE保全、実行時間、scope。
- `HANDOFF.json`：同records、source、normalization_evidence_valid、normalization-onlyの制約。
- `NORMALIZATION.csv`：固定順序の24行。ID/R/q/event/allocation、log B、attenuation/係数と表示status、丸め感度。
- `progress.json`：開始時のscopeのみ。完了結果と混同しない。

root statusがPHASE0_NORMALIZATION_RECORDED、SOURCE/input保全、HANDOFF.normalization_evidence_validが全て成立した記録を使う。STOP・精度未資格・部分完了は元記録として保持し、欠測を0やPASSで補完しない。baseline flag等はcandidate_idでPLAN.requestsへjoinできる。

BはR順/round順にnormalizationを比較できるが、Rの費用優越や推奨値は決定しない。Cは同じR/qPF/δ/prefix/K/法則・compiler/wrapperで価格domainを検討する。固定R価格やNを新Rへ付け替えず、B/attenuationをCへ再乗算しない。詳細N/C/Gは別の承認範囲であり、Phase 0完了をその実行許可に読み替えない。
