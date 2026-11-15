# p7/S2・幅h：固定4軸の取得と保存6軸の受入

利用者が固定計画486fa84ccのm1/m2 X/Yを明示承認した。B bec032fcの結果を実行triggerとして参照し、Q3を再計算しない。元configurationは `e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc`、保存C546 descriptorから既存plannerへ渡す。実幅は元binary64 h=.1、m1 T=.2/q2/physical7d1659503f9d99381006、m2 T=.4/q4/physical7fb12f906ab2f8cd8083、X/Y各別。科学上限は2 evolution cell・4 measured wrapper compile、新sampling・mean・fit・予測・Q3/N/G=0、retryなし。

## 最小接続と同等性

`scripts/run_pf_p7_h_prices.py` は固定configuration selector・保存receipt binding・公開だけの薄いcaller。回路pipelineを新設せず、`run_pf_m0_prices.child`→元p7 checked_trial / native cost_trial、fast context、fork worker、7作用gateを呼ぶ。共通callerへの変更はmetadata preflightに外部の固定rowsを渡す引数と、明示cohort時だけdeterministic labelを選べる点。既定m0/m1m2のscopeや結果は変更しない。

新規run labelは `D_P7_S2_H_DIRECT_V1`（samplingなし）。保存m0は元D cohort、m4はEXPLORATIONを保持。m3の元PF common trialにはcohort fieldがないため、既存carrierの表示label `SAVED_T0.8_COMMON` を接続用に使い、`original_cohort=null` / `CARRIER_LABEL_NOT_A_NEW_SAMPLE` を明示する。新標本を作ったとは扱わない。元SOURCE・trial・旧cohortを上書きしない。m3の価格dictは元metricsを既存builderへ渡す表示viewに変換するだけで、原本locatorを保つ。

保存linear H4 / 1Å / STO-3G / tol-only DF10、元H/state/snapshot、P0除外、一体項＋fragment0..6の保存順、S2、scalar c_H＋tail identityをtotal Tに一回。Qiskit1.3.0 / rz,sx,x,cx / opt1 / seed17、symmetric_directional、測定付きfull controlled Hadamard wrapper。元SOURCE af284d853890155bacd19e61850c196bd1d5c660 / provider c31ef424、元native<1e-9 / compiled<1e-7・位相合わせなしを保持する。contextでscalar/prep/fragment順を保存Stage2へ完全一致で照合し、新空cacheを使用する。

保存6軸は計画の固定PRICE_COVERAGE行を使い、m0原D receipt、m3公開PF common trial、m4既知旧result JSONを再利用する。scalar・元gate・保存値・SHAをmetadata確認し、original producer / trialを今回configurationへ結合する。新規4軸と同じ既存T3 record builder→source-bound sidecar→C guard経路を使い、guardが実際に確認した軸数を受入数とする。原SOURCEとbinding SOURCEは分ける。

## §10.5・SOURCE固定・実行

[PREFLIGHT.json](PREFLIGHT.json) は486fa84ccの照合以後の差分。既存5 carrier / catalog、現行Dのm0/m1m2/m6/m9結果、exact private trial path、関連running jobを確認した。対象4軸の追加receipt・同等job・未解決候補0、保存6軸の元記録は保全。別サーバー・未列挙archiveはUNKNOWN。実行入口で再確認し、重複候補を再compileしない。保存が片軸だけに見つかった場合は、そのpairを保留し、別の確定pairだけを進める。必要な軸subset拡張を無断の両軸再取得で置き換えない。

人工fixtureで4軸のconfig/physical/m/T/q/幅、誤ったcohort・scopeの拒否、既存C guardのsource/quantity/axis検査、保存6軸の元producer/cohortを確認する。別provider processでtoy DFによる元worker/fast context/7 gate・段階計測・異常終了の部分保存を確認する。既存guardは欠測をcallerへ返すため、fixtureでは例外だけでなく確認軸数も検査した。実H4前に今回code・test・入力参照をcommitで固定する。

```bash
python -B tests/check_pf_p7_h_prices.py
python -B scripts/run_pf_p7_h_prices.py --metadata-output <fresh.json>
# SOURCE commit後、一回のみ：
python -B scripts/run_pf_p7_h_prices.py --execute-approved
# raw公開commitに束縛：
python -B scripts/run_pf_p7_h_prices.py --bind-publication <raw-commit>
```

publicは本studyの新`run01`、privateは既存p7 private rootの新`p7_s2_h_m1_m2/run01`。資源観測に応じて2 cellを並列化し、route contextとcell内evolutionを共有する。X/Y compileは省略しない。元協調stage log・process exit/signal観測を使用し、非同期stack読取はしない。

失敗時は該当軸をUNKNOWN、部分raw/cache/logを保存。科学retry・片軸補完・gate緩和なし。成功しても当候補は10/26軸、m5〜m12の16軸が不足。新fit・長round予測・QPY監査・shot配分・全額比較は対象外。435,481,090 RZは16h/7の保存達成費用であり、hに付け替えない。結果commit/push・remote照合後mandatory STOP。
