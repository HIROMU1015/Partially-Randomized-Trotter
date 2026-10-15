# Track A 科学レビュー判断の共有索引

## 2026-10-16 R10後続実行の完了記録（新reviewではない）

最新の共有判断索引は[6b78e5f3固定版](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6b78e5f39917d92d6c60694d3e0b0d06394d7eb5/docs/research/track_a_scientific_review_decisions.md)。既存R01–R09を保持し、[R10の原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md)の判断と後続実行を区別する。

|原判断|採択済み方針|今回追加する事実|未認可・未保証|
|---|---|---|---|
|R10 / 2026-10-15|H8 D0、H6/H8 native B0+I対照coverage、H5→H7中性ROHF、H6固定δ=.2のT比較、5精度・6metric/P|H8 D0とH6登録対照は固定公開済み。[今回のH8対照](track_a_h8_native_b0i_full_family_results_20261016.md)も必要16軸を完了し、保存union6/32とのP=0点比較を更新|母平均・希少層・厳密u、確認外候補の主要claim、新H5/H7/T実行認可、原稿化|

H8でB0+Iを加えるとε=.05の分母が強くなりPR点低減率は約39.48%→28.77%へ縮小したが、登録内P=0/RZの全5精度ではPR点費用が低かった。これは既存方針に沿った結果整理であり、一般的PR優位の新採択ではない。新しいR11や詳細GPTレビューを作成していない。
中心的解釈・reference/allowanceの意味が変わる場合、確認union外候補を主要claimにする場合等はR10§14.3へ戻す。通常の公開・hash照合だけで同じ科学reviewを反復しない。H5/H7/Tは別の実行承認が必要。

以下は各時点の履歴として保全する。

2026-10-14 JST。既存GPT科学レビューの判断を記録する。Codexの新しい科学レビュー・論文改訂・実行grantではない。
[正式共通ノート](track_a_shared_research_status.md) / [機械可読review registry](../../artifacts/resource_applicability/track_a_cross_environment_canonical_status/2026-10-14/scientific_review_registry_v1.json)。
採択欄は各原reviewの対象scopeに限る。「採択しない/対象外」と、未検証ゆえ保留した事項を区別する。

|ID・判断日（JST）|対象scope|採択方針|採択しない／必須にしない方針|保留事項|原本|
|---|---|---|---|---|---|
| R01 / 2026-10-09 | AX-1b保存H4モデル照合 | RQ-R主・RQ-P1補助、強い対照・数値headroom・DF政策・標本不確かさ優先 | RQ-P2を必須完成条件とする案、追加fitだけの拡張 | 母平均保証・未知分子選抜・H8凍結モデル評価 | [原review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_ax1b_post_scientific_review_2026-10-09.md) |
| R02 / 2026-10-10 | AX-2B H4 v5技術pilot | 登録scopeの技術証拠、限定H4接続検証とH6準備 | technical PASSによる精度一致winner・厳密数値認定 | H6入力/pilotの明示launch認可 | [原review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md) |
| R03 / 2026-10-10 | H6 DF Hermitization診断 | 19fragment/tol-only/order/cutoff0保持、構造検査＋重み付き変更予算、新identity | fragment削除/rank fallback、旧STOPを遡及PASS | 丸めcertificate、state/input/pilot実行認可 | [原review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_h6_df_hermitization_independent_review_2026-10-10.md) |
| R04 / 2026-10-11 | H6技術pilot部分証拠 | 7cell/32wrapper保持、B2対full S2/S4を同じsignal精度で比較 | 欠測B3四つの補完/旧pilot再実行を必須とする案、直ちにH8実行 | 母平均・数値certificate、新matched runのgrant | [原review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_h6_technical_pilot_post_review_2026-10-11.md) |
| R05 / 2026-10-13 | H4 geometry元4方式 | 限定geometry/精度map、scalar保持対照の診断、native事例と分離 | 一般的PR優位、1812独立成功、統制size-scalingという説明 | 未実測強化対照、sampling/tail保証 | [原review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c81f7326dde3f40686b3caefe05e2f544d8d1826/artifacts/resource_applicability/track_a_h4_geometry_scalar_diagnostic/2026-10-13/supplied_gpt_scientific_review_20261013.md) |
| R06 / 2026-10-13 | H4 B0+I採択・実測scope | 追加対照B0+I、直近6軸pilotと全面240軸完成案 | 元B0置換/削除、6/10軸で全面map認定、一律PR増量 | 当時未知の実費用/winner、CPU・launch認可 | [原review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47dc17a83e2fc9216e22b59317969952ef79028a/artifacts/resource_applicability/track_a_h4_b0i_manuscript_integration/2026-10-14/review_context/previous_scientific_review_20261013.md) |
| R07 / 2026-10-13 | H6固定8構成の独立確認 | 探索8とfresh32を分離し、既知希少次数寄与の診断 | population winner、128/1024通常標本への自動増量 | 未観測order/component費用、H8 launch認可 | [原review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/docs/research/track_a_h6_independent_confirmation_scientific_review_20261013.md) |
| R08 / 2026-10-13 | H6条件付きs2 pilot後 | 留保付きbenchmark統合、追加H6 samplingを必須工程から一旦外しサイズ比較準備 | 全support上界、局所補正の他系移送、pilot scope自動拡張 | 他希少層・外部bit再現・H8最終結果 | [原review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38d386c8732295e6fab5d6fcbd47fbe586dc207d/docs/research/track_a_h6_s2_pilot_followup_scientific_review_20261013.md) |
| R09 / 2026-10-14 | H4全面240軸・v0.4後 | claim B主・A補助・C別native事例。経験的資源研究として科学編集へ | PR全勝/new algorithm、追加32/全support/全面高次PFを必須化、H8待ちでH4編集停止 | 母平均・最良PF一般・連続境界・operational predictor・energy総費用・H8全体結論 | [原review](track_a_h4_b0i_v04_scientific_review_20261014.md) |

## 追補・更新関係

- **R01**：Early priority decision retained; later H6/H4 reviews refine benchmark scope.
- **R02**：Followed by H6 input-policy and pilot reviews; original H4 v3/v4 STOP retained.
- **R03**：Later accepted input and H6 pilot use this policy; original STOP and diagnosis remain distinct.
- **R04**：Subsequent exploration/independent confirmation are new executions; original pilot remains STOP 32/36.
- **R05**：R06 adopts B0+I; R09 uses completed 240-axis evidence to refine main claims.
- **R06**：Costs later completed at 98cdb35c; R09 supersedes only the then-unknown-cost/claim status.
- **R07**：R08 follows after conditional s2 evidence and changes additional-sampling priority.
- **R08**：Revises R07 next-work priority after new evidence. Later H8 has separately authorized launch records; old not-started text remains historical.
- **R09**：Follows R06 completed-cost evidence, refines R05 main claim. Scientific editing recommended but this information-only task does not revise any manuscript.

## 10月14日 H4全面結果後レビューの取得

利用者が指定したローカル原本を全文取得。37,312 bytes、SHA-256 `1c4fb797dbaed0580ab4bf188f8ef4adb4fa042a5ec42731e8a56ff5bd129703`、Git blob `62110a5c92dfcf2c22050de6de08e54ee3f90cd8`。
[公開copy](track_a_h4_b0i_v04_scientific_review_20261014.md)は**原bytesと同一**。原本はローカルで保全し、review本文の科学的主張や数値を編集していない。
[provenance](../../artifacts/resource_applicability/track_a_cross_environment_canonical_status/2026-10-14/h4_post_result_review_provenance_v1.json)で原本とcopyを結び付けた。前回共有初版の「レビュー判断待ち」はその時点の記録として保持し、本正式ノートの取得状況だけ更新する。
付録の算術script/ZIPは今回取得していない。reviewに記された算術を独立に実行したことにはしない。

R09は主claim B（対照を強化した有限時間signal資源の適用領域）を主、A（件数）を補助、C（異なるnative H6/H8事例）を構成として採択。
最良決定論とのPR比を述べる際は元B0/B1/B0+Iの最小を分母にし、B0+I単独対B2の比と混同しない。これは原reviewの採択の記録であり、保存費用/shot/主指標を変更しない。
同一構成B0→B0+Iのmatched比較と、各familyで候補を選び直したB0+I↔B2比較も分ける。
H4追加sampling・全support certificate・全面高次PF追加を現時点の必須工程にしない。一般的PR優位や論文採択保証は未認定。
原reviewは科学編集を推奨するが、**今回の利用者指示は情報統合のみ**。原稿v0.1/v0.4・図表・参考文献を改訂しない。

## 権限と次の判断

科学レビューの完了、technical PASS、計算の明示認可は別事項。旧grantやSTOPに遡及変更しない。
H8は既存認可範囲で継続する方針を記録する。今回serverのlive状態は確認せず、停止/再起動/source/grant操作は0。
H8最終結果の統合、新baseline/estimator/claim scopeの実質変更、重大な数値不整合が生じる場合はGPT判断へ戻す。
通常の情報同期・hash・リンク検査だけで同じ科学レビューを再要求しない。新科学計算は今回未認可、公開後mandatory STOP。
