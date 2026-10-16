# V1成果整理：実測確認とモデル推定の区別

[成果整理・候補選定の判断表](../../../../docs/research/track_a_v1_results_synthesis_20261016.md)。基準は公開f89c40af…（開始時remote一致）。linear H6/1Å/STO-3G/保存tol-only DF19/T=.8、P=0、通常signed signal biasと経験的allowance。

- [5精度×倍率1/10の証拠表](precision_evidence.json) / [CSV](precision_evidence.csv)：既存値を転記し、出典commit/path/JSON pointerを行ごとに保持。
- [主張と未解決事項](claims_and_open_items.json)：局所検証・モデル条件付き・未認定を区別。
- [追加実測候補](additional_measurement_candidates.json)：利用者判断用。全候補未選定・未承認、一括campaign登録なし。
- [転記・リンク・保全の整理監査](organization_audit.json)。新しい科学実行receiptやgrantではない。

ε=.05はp7両軸実測単価×保存shotで356097424 RZ、保存PR403423200 RZ、PR削減率−13.29012029%。全候補を実測再選択していない。残る4精度は固定model条件付きの30.89/34.97/31.17/10.03%。PR側は保存確認union8/n32の標本点で母平均保証ではない。

追加実測は、約10%の厳しい側利益を強い主張に使うならB0I_p11_q4_4thが第一候補。中間精度は論文の主張に応じて利用者が選ぶ。直接compileだけでは厳密bias・母平均・全grid順位を認定しない。

旧result/raw/SOURCE/grant/ledgerは不変。新compile/signal/費用再会計/sampling/shot/fit/候補探索/新grant0。V2保留、成果整理後STOP。pushは利用者が行う。
