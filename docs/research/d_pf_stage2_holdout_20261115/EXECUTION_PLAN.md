# 決定論PF Stage 2：m6独立holdout取得・開封・判定

対象はH4 linear 1.00 Å、STO-3G、保存tol-only DF10。Stage 1 branch `codex/d-price-records-c-bridge-20261114` を継続し、既存Full PF/p7 stage callerを使用する。

凍結正本はC `12372fa6b53c878dc5dd124e036572ba0cd82061:docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json`、SHA-256 `845c705082565a2b44d198bddb3034497620d991a00604454394ae5acda9651a`。HANDOFFはf0fc0564、Stage 1索引はb190bd815。policyは91365dcfの `COMPARISON_POLICY` を変更しない。

開封前にJSON hash、各routeのStage 1 C_INPUTSを渡した既存 `pf_price_stages.frozen_models`、全12 groupのconfiguration/physical ID・PF公式・actual幅・q・axis・予測SOURCEを確認した。[FREEZE_PREOPEN_V2.json](FREEZE_PREOPEN_V2.json) はm6観測を読まない検査結果。JSONの `predicted_RZ` を予測の正本として使い、再fit・予測再計算をしない。

| route | m6 PF | q | physical ID | axes |
|---|---|---:|---|---|
| FULL_PF | S2 16h/7 | 28 | 4f3984c6d381fa0ca848 | X/Y |
| FULL_PF | Y4 16h/3 | 12 | 7727a3fec25e033629f1 | X/Y |
| FULL_PF | M5 16h | 4 | 8db0bd3c1a5e1eeb541f | X/Y |
| TRUNCATED_PF p7 | S2 16h/7 | 28 | fb512f2d5762df64bfae | X/Y |
| TRUNCATED_PF p7 | Y4 16h/3 | 12 | 5e19b5ded7386572684f | X/Y |
| TRUNCATED_PF p7 | M5 16h | 4 | 4bc967fe8dbc6f814146 | X/Y |

全cellでT=6.4。X/Y別にcompileし、各route3 cellの資源に応じた既存worker並列を使う。最新の既知table/catalog、元trial・別名schedule、各stage出力、関連jobを差分確認した範囲では再利用可能receipt/重複候補は0。新規は最大12 wrapper。未列挙archive/別サーバーを網羅した主張ではない。[DUPLICATE_AND_INPUT_PREFLIGHT.json](DUPLICATE_AND_INPUT_PREFLIGHT.json)。実行直前も既存runnerが重複・source・凍結model・Stage 1不変を再確認する。

実行入口は各既存callerの `--execute-approved-stage HOLDOUT --frozen-models 12372fa6b53c878dc5dd124e036572ba0cd82061:docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json`。Full PFを一回取得・公開・bind/guard確認後、p7を一回取得する。codeはStage 1と同じで、実行HEADと出力namespaceはroute別に記録する。両routeの取得・実装guard後に一括で個別判定し、途中のモデル誤差に応じて取得条件を変えない。取得/source/数値gate/公開に問題が出た場合は、未開始routeを投入せず部分結果とUNKNOWNを保全する。

Full scalarはc_H、p7はc_H+s_tailをtotal Tで一回。fragment順・signed pieces・actual幅・H/state/snapshot、P0除外、symmetric_directional、測定付きfull Hadamard wrapper、native/loweringを維持。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、元7作用gateを緩和しない。native/wrapper/provider/compilerはStage 1の固定SOURCEと同じ。科学回路処理は変更しない。

出力は既存各routeの `holdout/run01`（public/privateとも未使用）。Stage 1/calibrationおよび凍結正本は書き換えない。結果公開後、既存bindで新sidecar/C guardを公開する。無断retry、別axis/seed/roundでの補完をしない。

判定は各保存予測の分子/分母をFractionへ読み、`abs(predicted-observed)/observed <= 1/50`。観測分母を使い、丸め前に判定する。数値的2%超過はFAIL、取得/出所/実装不明はUNKNOWN。campaignは12件全PASSのみ成功、FAILがあれば失敗、FAILなしでUNKNOWNがあれば未確定。丸め表示や軸平均で判定しない。

新しい証拠helperは取得を起動せず、凍結metadata照合と公開receiptの厳密比較だけを行う。人工4 testで境界一致・丸め反例・UNKNOWN・identity拒否を確認した。最初のmetadata検査はCの価格式fieldとPF公式の解釈を取り違えて停止し、検査側を修正した。正本・回路条件に不一致はなく、その時点の科学call/観測開封は0。空出力と経緯は [PREOPEN_SCHEMA_CHECK_STOP.json](PREOPEN_SCHEMA_CHECK_STOP.json) に保全した。

12行のexact結果・receipt・SOURCE・資源を公開し、remote照合後mandatory STOP。全PASSでもm6の12モデルの独立holdoutまでの結論に限る。m7〜m12、追加compile、refit、MSE/N/G、winner判定はしない。ロードマップ本体は編集せず影響をREPORTで統合担当へ渡す。
