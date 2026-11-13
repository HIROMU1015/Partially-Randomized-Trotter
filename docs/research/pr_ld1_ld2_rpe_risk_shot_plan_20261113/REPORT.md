# B：PR-LD1/LD2の全RPE risk・shot評価計画

2026-11-13 JST。**新LD1/LD2の保存meanは既存Bのmetadata入口へ接続できる。全RPE risk・必要Nの科学的資格はまだない。** 今回は固定SOURCEの確認・6 metadataチェック・技術/科学計画まで。新MC、保存科学数値の再解析、RMSE/N/G算出、P0〜P4再実行は0。production codeも変更していない。

[RISK_AND_N_EVALUATION_PLAN](RISK_AND_N_EVALUATION_PLAN.md)／[FINITE_SCIENCE_PLAN](FINITE_SCIENCE_PLAN.md)／[HANDOFF](HANDOFF.md)／[SOURCEとreadiness](SOURCE_AND_READINESS.json)／[確認記録](METADATA_CHECKS.txt)。

## 接続確認と重要な不足

| 項目 | 確認結果・意味 |
|---|---|
| 新LD1 | canonical `a4dda8f7e3fc07666ca2aabf634917e8bced14eb62151eb629ef23c0cce21d3d`、全13mean。旧S3の`587d20…`とは別identity/sourceで、旧N/RMSEを移送しない |
| 新LD2 | canonical `1fa33556c17303f51ba03c4eae11189d35e59b3f57faad0e69d9fd08bf9245ae`、全13mean。LD1のラベル変更ではなく別prefix/λ/R/split/law binding |
| 共通task | H4 linear1Å/STO-3G/保存DF10、sector(2,2)、同H/state/snapshot/full DF-H target、h=.1/M12、同prior/decoder/shift0、S2/.1/K2/共通exactκ/P0、元Minkowski契約はmetadata一致 |
| mean受入 | Bの`_validate`と`prepare(shots=[])`で両bundleを確認。blockerはMISSING_N_X/YとPHYSICAL_MEAN_UNQUALIFIED、統計実行false。shot未指定を0やnominalで埋めない |
| 評価target | Aのtargetは`reference_solver_energy_ha`を保存するが、**`energy_ha`がなく、今回proxy採用もfalse**。現`W.diagnose`は前者を自動参照せず`target['energy_ha']`を読むため、直接呼出しは不可 |
| C packet形状 | A bundleにCのtop-level `source` envelopeとmean `provider_binding`がない。Cの`signal_input`は実際にriskが使ったmean/target/sourceを要求する。source名の付替えだけで正式compareへ通さない |
| 正式N資格 | 元policyはPROPOSED_NOT_ADOPTED、meanはUNQUALIFIED。局所数値gate通過やMC点RMSEでQUALIFIEDへ書換えない。Cの正式会計gateも維持する |

εE=.00159360010199040 Ha、元εqpe=εtrot=.00079680005099520 HaとMinkowski/finite_B_diagnosticは保存定義のまま。quantityがORIGINAL_DF_H_TARGET_ENERGYでも、DF前の分子Hや厳密groundが認証された意味ではない。

## 固定SOURCEと実際に調査したもの

| SOURCE | 読んだ資料・再利用する実装 |
|---|---|
| A結果 `91349e703e0d09ef702ccd108745dfaf1acc29f7` | REPORT/HANDOFF/B_MEANS_LD1/2。configは`a9abf8de75bf3432fcd714e868e5c14c8884b8f8`、raw carrierは`16210932e5dabe7d12d41f35fea0e0a0d7a9b271` |
| A数値 `fefd548ecd56f60d72031b3d385f6f5e6ee95d06` | provider/source/元数値契約の位置を確認。signal・normalizerを呼ばない |
| B設計 `437ba4acdb82708b6fc98e3cd119a2a734f7cf64` | controller、statistics/decoder、policy/error contract。現Bの統計/controllerは当該版とbyte一致 |
| B S3 `2084791b1b09cff65e07fa8566bb734e1e6bff8e` | 固定一配分の条件付き結果・loss感度の資格を参照。trial archive・RMSE/CIを再集計しない |
| B配分 `b9b9ffa6233db9dd2b413dc07b27e7039324c666` | W.prepare/diagnose/compare、SOURCE/配分/契約binding、欠測・未資格・exact会計の既存tests |
| B予備分析 `23d74855135e718554d824a269427de9943213d1` | 固定nominal構造proxyの結果を参照。正式N/Gでなく、新LD2への重みとして採用しない。P0〜P4 callerは未呼出し |
| C統合 `2f76b5750dd4c4823c122ab04e7629f731fd03a3` | REPORT/INTEGRATION_HANDOFF/NEXT_SCIENCE_PLAN、provider接続・signal_input・外側選択 |
| C価格 `5cc7c07e112ce97205948465c94441cba0042416` | REPORT/PRICE_INTEGRATION_HANDOFF、I.compareとD request/context/mean binding。今回C moduleを統合/実行していない |
| A全risk調査 `97146481f7b05e9418d8fc016f133c7b78d93d08` | REPORTの全事象loss、target/raw/law移送、直接riskと個別Minkowski予算の違いを再利用 |

元decoderの逐次branch、zero/tie未解決、非circular energy loss、priorをclipに使わない規則を維持する。既存statisticsのbinomial/count/summary/CP/Hoeffding/R3、Wのwhole-vector、Cのsignal/price binding、B exact会計を再利用し、新しい共通frameworkは作らない。

## 提案の要点

評価水準A（経験的条件付き診断）、B（モデル内riskの統計的不確かさ）、C（元energy契約への資格）を別出力にする。**今回どれも正式採択していない。** Bで区間が出てもCへ自動昇格しない。

最小のLD対照は2構成×共通whole-vector1本で2 diagnostic call。shot感度も同時に調べる場合は2構成×2本で4 callという案を提示した。既存「1構成×最大3本」を理由なく採用しない。base vector・拡大係数・target/loss/claim・confidence/precision/試行数/seedはユーザー判断として残す。

最小追加は、原A mean/targetを保全したまま**別の採択済みevaluation_target**をsummaryへ渡す有限callerと、Cのsource/physical-context受入mapping。既存S3と同様にS.simulate_prepared/S.summarizeを直接使えば、Wや元decoderの改変は必須ではない。反復MCを保証Nへ変えるproducerは別の科学的根拠・実装が必要で、今回は設計だけ。

6 metadataチェックは実際にPASSし、source/identity/target一致と不足を確認した。既存の人工/回帰testsはSOURCEとして調査し、今回の新科学テストや独立科学レビューと数えない。新しい数値回帰suiteも重複実行していない。

## 次の担当・STOP

**ユーザー＋A/B**が評価水準、proxy/targetの意味、未解決loss、whole-vector menu、必要統計精度/有限反復数を採択。**B**が最小caller/受入mappingを人工確認し、別途許可後だけ科学実行する。**C**はそのsignal_input/N証拠を既存統合APIへ接続、**D**は同candidate/law/profile/loweringの価格を供給する。

今回はREPORT等をcommit/pushしremote SHA/tree/主要成果物を照合して**mandatory STOP**。meanが揃ったことや計画採択だけを新MC・N探索・G計算の許可にしない。
