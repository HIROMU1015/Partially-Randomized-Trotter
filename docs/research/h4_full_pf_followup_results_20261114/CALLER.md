# 承認済みFull PF追加5点のcaller

固定計画は `d48d8732ee7ef521aab5d67a0945439364f5a2f8` の `docs/research/h4_full_pf_followup_plan_20261114/SCIENCE_PLAN.md`。同PLANの新actual幅5cellだけを実行し、保留M5/32hを除外する。保存M5/8hのwindow整合検査1件を最初に行う。

`scripts/run_h4_full_pf_followup.py` は既存loader/context/cache/metadataと `run_h4_full_pf_heff_scan.diagnose_cell` を再利用する。後者への差分はoptional `branch` 引数だけで、未指定の旧PRINCIPAL規約を保つ。PF式、direct-expm、多項式、Hamiltonian、数値gate、評価器は変更していない。

保存M5/.8検査はSOURCE/identity/scalar/referenceを照合し、保存unitary固有値から同じwindowでエネルギーを表現できること、energy group・重み・対応が不変であること、保存数値gateを確認する。旧signal再解析・新operator/Schurは行わない。診断と失敗理由を独立JSONへ保存し、失敗時は新5cellへ進まない。

実行は fresh `run01` に対し `--execute --source <caller固定commit> --out <run directory>` を1回だけ行う。科学SOURCEはcaller/source同定であり承認証明書ではない。元固有系cacheを使用し、新元H eigh/reference0。新step/Schur各最大5、直接・全spectral mean各最大10、窓検査1。N/MC/compiled価格/G/追加幅0。科学的な予算超過と数値gate失敗を区別し、自動retry/窓変更/閾値緩和は行わない。

新caller5＋既存plan4＋旧scan caller4の計13人工/metadata test PASS。実H4計算結果は別途run01に保存する。callerはこの文書と同じcommitで固定し、実行後も数値SOURCEを変更しない。
