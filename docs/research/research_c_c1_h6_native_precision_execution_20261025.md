# C1：H6 Gaussian精度修正版・compiled/raw FAIL

固定SOURCE `5309006f12e3621b578f7deca2a07d974258fc8f`、[契約](research_c_c1_h6_native_precision_contract_20261025.json)に従い、別fresh run01を一度実行した。旧SOURCE・旧162-call STOPは保全。ユーザーの修正・再実行許可に基づく。

**162/298 units（154 COMPLETE／8 FAIL）でSTOP。Gaussian精度修正後、局所30・shell3・kernel8のraw/reference検査はPASSしたが、全8 kernelのcompiled/rawが固定1e-10を超過した。** raw/referenceの個別数値reportは後続チェック例外のため未保存であり、PASSは処理順と全8件の例外位置による判定である。

[実行要約](../../artifacts/research_c_c1_h6_native_precision/2026-10-25/execution_summary.json)／[8経路](../../artifacts/research_c_c1_h6_native_precision/2026-10-25/candidate_status.csv)／[call](../../artifacts/research_c_c1_h6_native_precision/2026-10-25/calls_summary.csv)／[98 raw files・52 QPY・全8件の例外詳細・STOP](../../artifacts/research_c_c1_h6_native_precision/2026-10-25/run01/)／[postflight](../../artifacts/research_c_c1_h6_native_precision/2026-10-25/postflight.json)。

条件は直鎖H6・1.00 Å・STO-3G、Nα=Nβ3／sector400、修正fit最終CDF14／保存XDF14、全frame ordered S2 s1/s4・δ0.1・m0..12、literal/shared8経路、own-DF ground oracle／状態準備除外、P0 Qiskit1.3／opt1／seed17／rz,sx,x,cx。κ・scalar・frame順、CA0.00159／DF reserve CA/8・allowance・保存DF/PFは不変。新fit・optimizer・DF/PF再計算0。

| arm | s | route | compiled/raw最大成分差 | 判定 |
|---|---:|---|---:|---|
| CDF | 1 | literal | 5.519860650214071e-10 | FAIL |
| CDF | 1 | shared | 5.519860650214071e-10 | FAIL |
| CDF | 4 | literal | 5.523752907522645e-10 | FAIL |
| CDF | 4 | shared | 5.523752907522645e-10 | FAIL |
| XDF | 1 | literal | 5.261384624567407e-10 | FAIL |
| XDF | 1 | shared | 5.261384624567407e-10 | FAIL |
| XDF | 4 | literal | 5.264860537664491e-10 | FAIL |
| XDF | 4 | shared | 5.264860537664491e-10 | FAIL |

Gaussian単粒子再構成30件の最大差7.782677365833025e-16、局所CAR state-action30件の最大差2.011381761234907e-16。旧局所max1.558e-11から改善し、旧kernel raw/reference FAILを今回の固定probe gateでは解消した。全operatorの厳密保証や、CDFの資源優位を示すものではない。

最初のFAILで新規callを止め、開始済み8 bundleを回収した。全162 attemptに終端eventがありinflight0、run内retry/resume0。11 compileと52 QPYを保存したが、compiled/reference・adjoint・X/Y・kernel metrics・104 profile・比較は未到達。未知合成誤差／CA／shots／総費用はnullのまま。今回のfailed runを再起動しない。

wall272.15584188327193 s、CPU280.492299579 s、peak RSS3,199,368 KiB。SOURCE44／保存入力25／旧証拠593 filesを実行後hash照合しPASS。旧runも含むnative累積は324 units。

compiled/raw差がcompiler変換で生じたのか、simulation変換／実行で生じたのかは、この結果だけでは未確定。小さい人工回路では既存P0とAerが一致し、このH6差は再現しなかった。ユーザーの継続修正依頼の範囲で、保存済みの1経路だけを独立Qiskit Statevectorと比較する最小の切り分けを準備する。新規分子計算・PF再計算・shotsや総費用へは拡張しない。科学的閾値を緩和しない。

このattemptはFAILで完了しSTOPを保持する。追加診断・修正再実行では、そのSOURCE・保存入力・call範囲・fresh outputを先に固定する。
