# GPTへの最終source review依頼：RA-D0 v2

**`READY_FOR_SEPARATE_RA_D0_ONE_SHOT_REVIEW`。**
branch `track-b-ra-d0-source-review-v2-20261007`の固定commitで、
[source report](ra_d0_source_review_v2_20261007.md)、[利用者指示](inputs/ra_d0_source_review_v2_user_instruction_20261007.md)、
[execution contract](../../../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/execution_contract_v2.json)、
[evidence manifest](../../../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/evidence_manifest_v2.json)を確認してください。
source Sは公開後に報告されるfull Git SHAで固定してください。

v1の未確定事項を、指示どおりB0_saved/ideal分離、B1_num⊂B2_num⊂B3_num、固定membership幅、
B2 outer lower、profile-paired budget、freeze-before-B3、anchor-first、bounded query、resource/launch guardへ閉じました。
旧35＋追加45＝80 focused tests PASS。各x21 columns、18 sign pairs、保存126 sequence identityは不変です。
旧R1 result/marker/source/authorization、R1.5、Track A、旧STOPも維持しています。

registered optimization=0、B2 minima/実budget/B3/witness取得=0、synthesis/science/circuit/matrix/trajectory/GPU=0。
authorizationとmarkerは作成していません。資料公開やREADYは実行承認ではありません。

最終reviewでは特に次を確認してください。

1. numerical inclusionとinterval-aware membership/mean、B2_outerがB2_numを包含する根拠。
2. **batch単位freeze**：P1全18 anchorsを先にfreeze→anchor比較→固定gateで必要xを決定→
   P2全必要coverageを別freeze→coverage比較。P1のbudgetを編集せず、P2で候補/grid/settingsを増やしません。
3. primaryはcertified U_B3<L_B2_outerだけ。Farkas失敗／budget生成失敗／cap／未完了は
   prefix witnessがあってもtechnical STOP。coverage-onlyはSTRONGへ昇格しません。
4. main 55,275、recipe total 110,550、hard total 111,000、1 process/thread、retry0、
   per-LP2 s、wall3600 s、CPU3300 s、RSS1536 MiB、AS4096 MiB、output128 MiBの接続。
5. per-LP capはHiGHS time_limit＋periodic Python check＋post-checkです。長いC呼出しの瞬時強制killではありません。
   nominal double誤差とτの幅、coverage低nでbudget certificateを取得できない可能性も認識してください。
   sourceではdenominator/tolerance/settings/second-bestの救済を行いません。

比較は既存R1の2-qubit distinct-basis controlled、finite P3、p=(3/4,1/4)、x={1/8,1/4}と三既存precisionだけ。
分子/DF-scale、I0 access advantage、actual full-circuit optimum、held-out、publication noveltyは主張していません。

review通過後にだけ、**Sの直接子のauthorization-only A → 明示one-shot実行指示 →
RA-D0 saved-table development一回 → mandatory STOP**へ進むか判断してください。
今回の引き継ぎ後、Codexはそれ以上実行しません。
