# p7/S2・要求幅h：m1/m2直接取得と保存6軸の受入

**新規4 wrapper compileがすべて成功。保存6軸と合わせて、元configurationで10/26軸をC guardへ通した。対象UNKNOWN/受入失敗/保留0、残る16軸は未取得。mandatory STOP。** q別価格とB接続の正本は[HANDOFF_B](HANDOFF_B.md)の冒頭、[26軸JSON](run01/AXIS_PRICES.json)／[CSV](run01/AXIS_PRICES.csv)。

| 区分 | 軸数 | 状態 |
|---|---:|---|
| m0/m3/m4保存再利用 | 6 | 元producer・trial・cohortの意味を保持し、今回候補でC guard受入 |
| m1/q2・X/Y | 2 | 新規直接価格、各11,333 RZ、作用gate/C guard PASS |
| m2/q4・X/Y | 2 | 新規直接価格、各22,359 RZ、作用gate/C guard PASS |
| 合計受入 | 10 | m0〜m4、q1/2/4/8/16、両軸個別 |
| 残るm5〜m12 | 16 | 未取得、価格null、予測/補完なし |

## 範囲・SOURCE・重複照合

利用者が採択した有限案は `486fa84ccf2b63c73f71d3adcedada72dc6ebe99:docs/research/d_pf_width_price_coverage_20261115/NEXT_FINITE_PLAN.md` と同dirのFIRST_FINITE_PLAN / PRICE_COVERAGE。Bの固定Q3結果bec032fcを根拠に幅hを候補に残したとの指示を受け、価格だけを取得した。Bの結果や配分を再計算していない。

元configuration `e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc` は保存C546 descriptorを正本に、既存planner / price_requestsへ接続した。m1 T=.2/q2/actual幅.1/physical7d1659503f9d99381006、m2 T=.4/q4/actual幅.1/physical7fb12f906ab2f8cd8083のX/Yだけを新規取得した。新しい回路pipeline、候補探索、signal/N評価は実装していない。

§10.5は既存幅coverageを起点に、5 carrier / catalog、当branchのm0/m1m2/m6/m9、対象IDの既知private trial path、関連running job、今回の新namespaceを差分確認した。[事前PREFLIGHT](PREFLIGHT.json)、[実行時PREFLIGHT](run01/PREFLIGHT.json)。対象4軸の追加receipt・同等job・未解決候補0。保存6軸の原本locatorとm4 SHAも確認した。未列挙archiveや別サーバーは網羅せず、不在を断定しない。科学retry・追加seed/軸/round代替は0。

- 実行SOURCE：**`be0f81ebd893415de031ee30ee0755f039adb823`**。
- raw / price / implementation proof：**`302d63e9024a0c594d899f407af9331cc7174e86`**。
- bound sidecar / C guard：**`11cb034ef39b1796cf83958a405233d93af14a99`**。
- 全固定locator：[PUBLICATIONS.json](PUBLICATIONS.json)、各価格行の`receipt` / `trial` / `original_receipt`。最終報告・表は最終公開commitに固定する。

## 科学・実装条件

保存linear H4 / 1.00 Å / STO-3G / tol-only DF10、元H/state/snapshotを使用した。H hash c1a6c428…、state9bcc0b0e…、snapshot12219b7d…。P0除外、一体項＋fragment0..6を保存順で保持し、S2 signed順とsymmetric_directional controlを維持した。scalarはc_H＋tail identityをtotal Tに一回。実contextのtotal scalarは2.293146549384713、hex `0x1.2585d37d4573ap+1`、元保存値と完全一致。prep hash fc83f26f…、fragment順[0,1,2,3,4,5,6]も一致した。scalarの置換・分配・tolerance追加はない。

Qiskit1.3.0 / rz,sx,x,cx / opt1 / seed17、compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。価格quantityはscalar位相・X/Y基底変換・最終測定を含むfull controlled Hadamard wrapper全体のcompiled RZ数。native / wrapperは元af284d853の経路、`H46_LONG_TIME_NATIVE_COMPOSE` / `NOT_APPLICABLE_NO_RTE`。旧RTE/Pauliのlowering価格は混在させない。

新cellは元checked_trial / native cost_trialを使い、low-control、high-control leakage、独立occupation作用、Hadamard X/Y、compiled-vector X/Yの7 gateをそのまま通過した。元native<1e-9、compiled<1e-7、位相合わせなし。新規観測の最大保存誤差はnative5.43e-14、compiled2.65e-12。これは保存状態に対する元作用検査であり、全入力状態や全RPE精度を新しく証明したものではない。

## 保存6軸と今回の受入

保存m0 q1/X/Y＝各5,820は既存D m0 receipt、m3 q8＝各44,411は公開PF common原trial、m4 q16＝各88,515は旧long-time原trial＋公開carrierを再利用した。元trialを再simulation・再compileせず、保存metricsを既存record builderの形へ渡した。m0の要求幅表示が旧shortlistでも、T=.1/q1/actual幅.1・prefix・signed順・scalar・wrapper/compilerが同じため、元source付きで今回候補へ結合した。

元SOURCE/cohortと今回のbinding SOURCEを分けた。m0の元D cohort、m4 EXPLORATIONを保持した。m3のnative原本にはcohort fieldがないので`original_cohort=null`のままとし、既存carrier label `SAVED_T0.8_COMMON` は接続labelで、新標本と扱わない。m3の価格dictをmetrics viewへ投影したことを`trial_view`に記録し、原公開trial locatorを保存した。旧保存ファイルは変更していない。

新規run labelは `D_P7_S2_H_DIRECT_V1`。cohort名は決定論runの識別であり、新MC cohortではない。全10軸のsource-bound sidecarをraw公開commitに束縛してC guardを実行し、checked10軸を確認した。`DIRECT_COMPILED_DETERMINISTIC` / `DETERMINISTIC`、`model_qualified=false`を維持する。受入はimplementation/quantityの整合で、model・N・G資格ではない。

## 技術検査・資源

最小接続は新thin caller `scripts/run_pf_p7_h_prices.py`。既存m0 callerのduplicate preflightへ外部の固定rowsを渡せる引数と、明示されたdeterministic run labelだけを追加し、既定の旧scope/labelは保持した。回路生成・compiler・作用gate・fast helper・T3 record builderは変更していない。

人工fixture3件＋別provider processのtoy worker回帰2件、**計5件PASS**。[TESTS](TESTS.txt)。config/physical/m/T/q/実幅/cohortと4軸のscope、保存6軸のproducer保持、既存C guard、fast context・段階計測・異常終了保存を確認した。最初のfixtureでは既存guardの「欠測は例外ではなくcallerへ返す」挙動にtest期待を合わせ、実callerはchecked軸数の一致を要求した。科学gateは緩和していない。全26行のcanonical/physical/axis結合と10受入・16欠測のmetadata検査もPASS（[check_metadata.py](check_metadata.py)、[出力](METADATA_CHECK.txt)）。

2 cellを2 workerで並列化し、route contextと各cellのX/Y evolutionを共有した。元fast fingerprint / signed部品再利用をtrial全体で有効にした。X/Yは別compile、新cacheでcache-hit0。開始時affinity64、観測idle CPU約63.87、利用可能RAM524,442,664,960 byte。対象数と実資源に応じた並列化で、恣意的な恒常CPU/RAM上限・共有設定変更はない。

取得parent wall25.706 s、cell wall m1 13.988 s / m2 21.838 s、context約2.070 s、最大worker RSS421,498,880 byte（401.97 MiB）。binding・報告・公開は別。childと2 workerは終了code0 / signalなし。stage時間はnested/inclusive、compile欄はarchive保存込み、RSSはprocess peakで同時aggregateではない。同条件baselineを再実行しておらず、高速化倍率は未測定。[TIMING](run01/TIMING.json)と協調progress JSONLに保存した。非同期stack読取や他process操作は行わない。

## 主張の範囲・次の入力

この5点は幅hのq1/2/4/8/16に対する直接証拠で、同幅の価格規則を今後検討する入力になる。今回は係数・残差・fit・価格規則を算出していない。任意qの直線・閉包、未使用holdout、長roundの価格・誤差上界としては扱わない。q1の端効果とtotal scalar、幅h固有の角度・境界・終端・compiler数値枝の裏付けが残り、現行16h/7の閉包採択を移送しない。

未取得はm5〜m12、q32/64/128/256/512/1024/2048/4096のX/Y計16軸。すべてnullのまま。435,481,090 RZは幅16h/7の保存達成費用として保持し、幅hの価格へ代入しない。新fit・長round予測・QPY監査・Q3・shot生成・N/G・全額・winnerは0。ロードマップ本体は編集せず、HANDOFF_Bの差分を統合担当へ渡す。結果commit/push・remote照合後**mandatory STOP**。
