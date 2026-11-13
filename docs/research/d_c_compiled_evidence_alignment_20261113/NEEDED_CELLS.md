# C向け必要セル・既存receiptの再利用可否

正本Cは `1914d0764115d82ec860e08810dafeb50f8d9a42` の `rpe_mean_price_model_preparation_20261113`。本表はその未資格model/coverage条件とD固定SOURCEを照合した取得候補表で、価格数値の再解析・mean生成・科学的受入は行っていない。軸は全行でX/Y個別。同一pathの二応答を2独立pathと数えない。

## 固定PR-LD1/LD2（52軸）

| 候補・round | q / R、既存証拠 | 既存receiptの扱い | 新compiled証拠の必要性・取得案 |
|---|---|---|---|
| LD1 m0 | 1 / 2、P1 development各軸2 path、P2 fresh各軸4 path＋replay各軸1 | 新ZZの有限path参照として再利用。P2 replayはP1の同一pathなので新独立標本に数えない。新A identity/requestへのbridgeは未資格 | 同じ旧証拠を再compileしない。model開発に使うなら旧roleを保持し、bridge/再解析の必要scopeを別途明示する |
| LD1 m6 | 64 / 128、P1/P2の同じrole構成 | 上と同じ。既存中規模directと性能記録を参照できる | 追加n・精度・holdoutはB/C判断後。m7へ順番に進まない |
| LD1 m1–5, m7–12 | A固定q/R selectorは存在、今回の新ZZ固定schedule directは未同定 | 旧26 E[S]は構造平均のみ。S/旧C1/未証明windowをCや厳密界にしない | 全round compileを選択しない。比較を左右するセルとmodel検証方向を後で選ぶ |
| LD2 m0 | 1 / 1、compiled receiptなし | LD1価格は使わない | **E0：normal 1 path、index0、cohort `D_C_PRICE_BRIDGE_LD2_M0_V1`、X/Y、最大2 compile**を初回候補。接続診断用 |
| LD2 m4 | 16 / 10、compiled receiptなし | LD1価格・tail期待値は使わない | **E1：normal 1 path、index0、cohort `D_C_PRICE_CHALLENGE_LD2_M4_V1`、X/Y、最大2 compile**。複数outer接続用の別保留案 |
| LD2 m1–3, m5–11 | A固定q/R selectorは存在、compiled receiptなし | 未資格modelをreceiptとして扱わない | 現時点で科学実行を選択しない。R-only/混合方向の識別設計やBの必要精度が不足 |
| LD2 m12 | 4096 / 2559、旧T2はsampling前scalar STOP、後の診断のみ | direct C/S/Dなし。STOPや診断をprice receiptとしない | **初回対象外**。最大q/Rの直接検証が不可欠で、代替証拠では必要誤差を制御できないと判断した場合だけ新たに提案 |

全roundのq/R・piece allocation・exactκ・law/identityはA固定configurationと旧D748 coverageを参照する。旧D748のtail-only expected scalarは訂正して読み替える用途のrecordではない。必要cellを**e827以降のselectorで新SOURCE付き再生成**し、過去IDは保全する。E0/E1の完全な選択metadata・seedは [FINITE_CELLS.json](FINITE_CELLS.json)。

## 他方式とCの検証方向

| Domain / Cが挙げた論点 | 既存証拠と再利用条件 | 追加取得が必要になる条件・未指定情報 |
|---|---|---|
| Full PFの固定＋q項 | D e7dの両軸EXACT 73候補（H4/H6合計、49 long-time direct＋24 pf_common）を元domain/profileで再利用。H4 S2 T=.8/q8と1.6/q16は既存EXACT。保存grid外・欠測63は別扱い | A/Bが選ぶFull schedule、formula/PF幅・T/q・profileに一致するreceiptがない場合だけ新取得。全q affineは別qの独立検証と使用最大domainへの根拠が必要。今回Full PF候補/seed/cohort/nは未選択 |
| Truncated B0/B0I | Full PF receiptは代用不可。retained H・scalarの相違も保持。今回固定SOURCEでこの新modelに受入済みdirectは未同定 | retained split/identity、元scalar契約、schedule、profileとB候補familyが先に必要。B0とB0Iも同じ価格quantityと仮定しない |
| Full-H pure RTE | one-body込み全non-scalar、head0、独自λ/scalar/qR grouping。旧固定receipt/providerは元domainのみ | PR tail-only係数やLD2 scalarを移植しない。選択候補・K/grouping/law・新lowering profileに一致するdirectが必要な場合のみ別有限案 |
| PR R-only | 旧C1はq8/16×R4/8等・旧loweringの有限cohortで、今回A scheduleと不一致 | q/T/PF/prep/K固定の合法R対照が必要。R変更でevent角とorder lawも変わる。A fixed menuを黙って変更しない。E0/E1はqとRが共に変わるので識別実験にならない |
| PR q/R同時増大 | LD1のP1/P2は開発参照。未証明window tailのまま | 仮説・係数・誤差の定義を固定した後、未使用domainと比較を左右する代表点を選ぶ。m7→8→9の逐次実行を前提にしない |
| S＋full-wrapper残差 | C9d95の26理想law E[S]は固定LD1。P1/P2は実samplerのpaired path | law bridge、finite-cohort/rare/共有誤差を含む検証が必要。構造＋event単価や小規模比率を正式Cへ昇格しない |

Full PF receiptの再利用判断はD e7dの `HANDOFF_C.md` / `COVERAGE_V2.csv` / `receipt_trace`に従い、同じsource監査を重複しない。cost policy、H/state/snapshot、scalar、formula/time/fragment順、control、compiler、測定・P0とV2 traceまで一致した**元domainのdirect**として扱う。別lowering/profileへの同値bridgeを省略してrecordを付け替えない。RTE専用の新ZZがFull PF経路では呼ばれないという静的事実だけを、未確認の全pipeline同一性へ拡張しない。

## cohort・科学的資格

P1 development、P2 replay、P2 fresh `D_P2_DIRECT_PILOT_V1` index0–3は保存どおりに維持する。旧C1のcalibration/holdout/rare roleも変更しない。新E0/E1はそれぞれ別fresh cohortのraw path候補で、旧標本へ混合しない。乱数streamが異なることだけで独立性やideal law一致を証明しない。

C1914はモデルの係数・実検証点・n・cohort・必要精度を採択していない。E0/E1も平均モデルのcalibration/holdoutとは呼ばない。Cが既存標本を開発に参照した場合、それを後から独立検証へ変更できない。今の正式価格coverageは未資格/UNKNOWNのまま。本表の不足を全件実行する指示として扱わない。
