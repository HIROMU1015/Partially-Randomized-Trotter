# Full-H κ2 m0：採択zero-ZZ規則による再判定と固定配分の総額

採択された `FULL_H_KAPPA2_NORMAL_PATH_ZERO_ZZ_ADMISSION_PROPOSAL_V1` を、元normal pathの経験的1 shot参照単価の受入に適用する。**観測後に採択された規則による別の再判定**であり、元parity戻り値・runtime lowering markerをPASSへ書き換えない。

m0のX/Yを受け入れられた場合に限り、保存24モデル軸と既存会計処理を使い、一つの固定1.5倍配分の総額を評価する。科学実行はSOURCE固定後のrun01一回だけとし、結果とexact値を追記する。

## 固定SOURCE・物理条件

Linear H4 1.00Å/STO-3G/保存tol-only DF10・H/state/snapshot、Full-H pure RTE（one-body込み全非scalar、headなし、L_D0）、finiteK2・κ2、P0。m0 physical ID `FULL_H_2a9a68a11d08af234a8c`、T=.1/q1/R9。

cohort `D_FULL_H_KAPPA2_M0_REFERENCE_N1_20261115_V1`、index0、seed15612141344395026302。同じnormal pathのX/Yであり、独立path数は1。preparation hash `92325333cfe8c5e6c5f9bf69cdfc0ea6538f8eaa66327a3820d390dc2e8aa100`、scalar −0.33147781341681615を総Tへ一回。

new ZZ `CONTROLLED_ZZ_CX_CRZ_CX_V1`、symmetric-directional、測定付きfull controlled Hadamard wrapper、Qiskit1.3.0・rz/sx/x/cx・opt1・seed17。費用単位はwrapper全体のcompiled RZ/shot。normalizationを価格へ再乗算しない。

| 役割 | 固定commit・path |
|---|---|
| 採択規則の原案 | 481c4b7abffbf070347f697980f2ec051e6a6538、full_h_m0_zero_zz_rule_20261116/PROPOSAL.json |
| D1保存証拠 | 64624ea073dfc9afc32e2f09e5cd6b2eaadcc1d3、d_full_h_m0_zz_evidence_20261115/EVIDENCE.json・REPORT.md |
| 元軸別receipt索引 | 74cd9c2625a6338793f1beef7b434c765a3a7e67、d_full_h_m0_reference_20261115/HANDOFF_C.json#axis_rows |
| 元producer・観測 | producer7f85f20e38af4df1a649ac239e57bfe76d0899ee、結果b25d7da6dedebc4cd22e8d6f00e6de75d7072292 |
| C旧保留・保存24軸接続 | 426b8c4f14b4a00869c73f7b2ae4ecdc638a0107、full_h_m0_reference_total_20261115/run01/RESULT.json |
| F1原価格 | a1aa38df46c92e2489095891f6082a5be1a277f5、full_h_f1_model_diagnostic_20261115/run01/RESULT.json |
| B固定Q3配分 | 6c5569c3479013b080b2ba53842585f1d670d3c9、full_h_kappa2_scale3o2_q3_results_20261115/ |
| PF比較正本 | b4296b110b129d226dbef2743c14f725a27c3bf6、pf6_cost_budget_q3_results_20261115/RESULTS.csv |

上表の短いpathはすべて `docs/research/` 下。機械可読の完全locatorはrunner定数・RESULTのsources・各軸の元recordへ保持する。SOURCE名や現在のIDで元の生成時IDを付け替えない。

## 受入条件と証拠の区別

1. D1が全生成範囲から独立に確認した置換前対象0件を用いる。9 applicationはいずれもnonidentity rotation/support1、head0、basisはsystemのみ。置換counterの0から推定しない。対象数不明・対象あり・範囲欠測は今回の適用外化で受け入れない。
2. 固定 `01e068af...:efficient_controlled_zz` が変更するのはcontrolled/support2分岐のみで、その他は元builderへ委譲される。D1のSOURCE調査で元requests/preparation・basis/scalar、memoizationのnew/legacy両経路、wrapperまでのactivation範囲を確認済み。今回はその固定証拠を再利用し、回路の再構築やlegacy/new比較を実行しない。
3. 保存QPYに対するD1のread-only scalar/event/同一U・X/Y/global phase/測定軸の照合を再利用する。元numeric7検査、preparation/scalar/profile、cohort/seed、軸別compiler record/fingerprintを元receiptへ結ぶ。2 receiptと各wrapper/compiled QPYは元bytesのhashだけを確認し、再parse・価格再計数・simulationはしない。
4. **元parity戻り値・scalar/event戻り値・XY戻り値・runtime markerは未保存**。現行受入根拠は「独立対象0件・固定SOURCE非ZZ分岐同一性・D1の保存QPY読取証拠」であり、元runのparity PASSという記録ではない。

原案のPROPOSED_NOT_ADOPTEDという歴史状態や、D1/C旧0/2受入・null・STOPは変更しない。今回のユーザー採択に基づく新しい再判定として記録する。価格の大小やPF順位は受入基準に使わない。

## 会計・価格tier

[reassess_full_h_m0_zero_zz.py](../../../scripts/reassess_full_h_m0_zero_zz.py) は受入の差分だけを追加し、既存 `account_full_h_reference_scenario.account` を無変更で使用する。既接続のC旧RESULTから24 model quoteと比較正本を転記し、モデルpredict・fitやBのQ3を再実行しない。

**m0は1 pathの軸別実測、残り24軸はF1のモデル参照価格。** 両軸受入時のm0分類はDIRECT_COMPILED_FINITE_SAMPLE / FINITE_COHORT_ONLY、値は各receiptから引用する。X/Yは別々のfingerprint・測定・receiptを保持する。

配分はX/Yとも [96,90,84,78,71,65,57,51,45,38,32,26,18]。集計は元整数・exact有理数で行い、m0部分額、全13round T、m0部分額/T、T−615304502、T−435481090を保存する。表示丸め値から再構成しない。片軸でも保留なら既存会計が実加算前に全額・割合・比較差をnullにする。

有効な総額もMODEL_SCENARIO_ONLY_WITH_EMPIRICAL_M0であり、正式E[C]/CI/Gや5方式winnerではない。m1〜m12の残差model外挿・有限cohort・ideal lawとsampler lawの差・共有不確かさは従来どおり未資格/null。D7係数・qualified flag・F1原負値/全額nullは保全する。

## §10.5・検証・性能

C既存branch、固定426bの接続結果、D1の74cd/64624と規則案481c、その後の関連公開ref・履歴・今回path・同名の自分のPython job名を差分確認。同じ採択後再判定/全額の公開結果・実行中jobは確認範囲で0。未知archive/他ホストを網羅せず、他担当private領域へはアクセスしない。

新しい人工test4件は、受入readerから既存会計までを人工価格で通し、対象不明/正数/counterのみ、他条件欠測、片軸保留、代替証拠未成立を拒否することを確認する。旧18 testの証拠は426bを再利用し、無変更のfit/decoder/compilerを再監査しない。これは作者自身の技術確認であり独立科学レビューではない。

小さいJSONと元QPYの圧縮bytesだけを読む。巨大event列の生成・展開はなく、24値は既接続記録から再利用する。プロセス並列化を追加しても効率上の必要がない短い保存値処理であり、CPU/RAM/workerへの制限を設けない。新しい高速化方式や未実測の速度倍率は主張しない。

sampling/build/compile/re-simulation/価格再計数/再fit/Q3はすべて0。結果公開・remote照合後mandatory STOP。ロードマップ本体は統合担当に委ねる。
