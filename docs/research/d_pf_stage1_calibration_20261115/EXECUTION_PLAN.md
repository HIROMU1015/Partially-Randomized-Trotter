# 決定論PF Stage 1：有限実行条件

対象はH4、linear 1.00 Å、STO-3G、保存tol-only DF10。Stage 0 `acea1118a4df4e727e13398ce0f8ae11511c7396` のbranchを継続する。C新版CELL_PLAN `43c15bfa34a22f313e55b824223b200332b7f23f`、保存4価格の正本 `855573064b2c17cfe20eb94a01ece1a554141b0b` と元admissionを使用する。

| route | 新規cell | q（m4/m5） | 新規wrapper |
|---|---|---|---:|
| Full PF | S2 m4/m5、Yoshida4 m4/m5、M5 m5 | S2 7/14、Y4 3/6、M5 m5=2 | 10 |
| B0I p7 | 同上 | 同上 | 10 |

X/Yを別々に取得する。Tはm4=1.6、m5=3.2。actual幅とsigned weights、configuration/physical IDは [PREFLIGHT_METADATA.json](PREFLIGHT_METADATA.json) の元configurationに従う。input bindingのT=0.8は保存input来歴の参照時間であり、新cellの発展時間ではない。

保存M5 m4（q1、T1.6）はFull `0cd1527c80f7b9841a83` のX/Y各85960、p7 `13d1cd4869951cdf1e2c` のX/Y各60950 RZ。元EXPLORATION cohort・元SOURCE・receipt・sidecarを保持し、再compileしない。Stage 1は新規20＋保存4＝24較正価格、route×公式×幅×軸の12 groupへ渡す。

回路価格は測定付きfull controlled Hadamard wrapperのcompiled RZ数、状態準備P0除外。controlはsymmetric_directional、nativeはH46_LONG_TIME_NATIVE_COMPOSE、RTE loweringはNOT_APPLICABLE_NO_RTE。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17を維持する。Full scalarはc_H、p7はc_H+s_tailをそれぞれtotal Tで一回。head/fragment順、signed piece順、low/high control・leakage・independent occupation vector・X/Y・compiled-vectorの既存gateを維持する。

採択済みH-bは観測価格分母のRELATIVE、厳密有理数1/50、12モデル個別に全PASSを要求する。m6は較正/モデル選択に使用しない。失敗時の自動refit・feature追加・閾値緩和は禁止。`pf_price_stages.COMPARISON_POLICY` を科学条件として記録する。Stage 1ではholdout価格・予測・判定を取得しない。guardの受入はモデル資格を意味しない。

実行入口は既存callerの `--execute-approved-stage CALIBRATION`。Full PFを先に一回、成功し公開・guard束縛が完了した場合にp7を一回実行する。route間の重複guardを弱めず、route内は既存のidle CPU/available RAMに応じた独立cell workerを使用する。恣意的なCPU/RAM制限や共有環境変更はしない。

出力は既存の各route `calibration/run01`。public/privateは未使用であることを確認した。既存stage runnerは科学call直前にsource_gate/admission_gate/select/duplicate_preflightを再実行する。途中FAIL/不一致/重複疑い/資源不足/公開不能では新規投入を止め、部分結果と原因を保全する。無断retry・別seed/round・namespace変更はしない。

新しいHOLDOUT m6 compile、sampling、実fit・係数凍結、価格予測/外挿、MSE/N/G、他方式取得は範囲外。実行結果を公開し、既存bind-stage-evidenceで新bound sidecar/C guardを保存・公開後、mandatory STOP。ロードマップ本体は変更せず、影響をREPORTへ記す。

## 技術的確認

保存4価格はC正本と同じrecordである。最新の既知table/catalog、旧trial/alias、当route stage結果、関連user jobとの重複照合を更新し、新規10 physical cellに追加重複は検出しなかった。未列挙private archive/別サーバーを網羅した主張ではない。実行直前に各routeで再照合する。

inputの実bytes/hash、保存H/state/snapshot/receipt、provider-rootのimport-only経路を確認した。references/をこのworktreeに展開し、restore/prepare自体はまだ実行していない。native/wrapper/providerは元固定SOURCE、科学計算関数は変更していない。変更は採択済み比較policyのmetadataと人工testの期待値だけ。

人工・metadata parent suite18件PASS（実行ログ [TECHNICAL_TESTS.txt](TECHNICAL_TESTS.txt)）。suite内のHOLDOUT/fitは架空価格fixtureだけで、実m6取得・実fitではない。この計画とcodeを次のcommitで固定してから科学実行する。
