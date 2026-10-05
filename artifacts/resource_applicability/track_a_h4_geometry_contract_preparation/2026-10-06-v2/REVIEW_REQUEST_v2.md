# 契約v2レビュー依頼・公開後STOP

入口は[契約案](CONTRACT_DRAFT_v2.md)。設定値は[review_decisions_v2.json](review_decisions_v2.json)、
scopeとnull identityは[zero_compute_plan_v2.json](zero_compute_plan_v2.json)、認可順序は[stage_contract_v2.json](stage_contract_v2.json)。

次の4項目は具体値を埋めた**採用案**で、最終承認は未解決。

1. D1：明示RHF/minao/通常CDIISとstrict最終gradient gateを承認するか。
   installed PySCFのextra-cycle緩和を独立gateで補う案はlegacy implicit convergenceの無変更主張ではない。
2. D2：生成順、返却tie順、sign固定、有意縮退/prefix tie STOPとnull eigenspace一回freezeを承認するか。
   追加MO/DF relative1e-12閾値、rank12はfragment countという扱い、cross-host basis再現未保証を確認する。
   sector36/eighと元のnorm/residual/Hermiticity/imaginary energy/gap閾値は維持する。
3. D3：master seed20261006を採用するか。今回actual seed/hashはnullで、実trajectory seed未生成。
4. D4：worker/driver AS各8GiB＋headroom16GiB、RSS別監視、admission8+8w+16GiB、累積72h/総10GiBと固定run/root案を承認するか。
   64GiBで12 workersは開始できず、最大5。観測RAMを予約と解釈しない。

承認後も、別指示によるscience source/runner/tests実装とactual commit固定が先。
入力生成専用plan・別authorization・review・明示launchで6距離だけ生成し、全input bytesをfreezeしてSTOP。
その後input-bound signal/compile planをsealし、さらに別result-prior authorization・最終review・明示launchを要する。
生成認可をmapへ流用せず、認可前のactual signal/cost取得をsemantic gateと呼ばない。

source-stageのsynthetic physical-semantic tests、実SCF/DF/sector gate、production serializer、owned memory/atomic IOは後続で別実装・検査する。
今回320件のJSON passはそれらの科学実装や独立再現性を証明しない。旧129件は再実行していない。
準備時の2失敗ログを保存し、最終suiteはfail/skip0と区別した。

現在のfalse/null/mandatory STOPを維持し、契約完成や次段認可を宣言しない。
公開後に停止する。本計算、source port、authorization発行、Track B、距離/trajectory追加へ進まない。
