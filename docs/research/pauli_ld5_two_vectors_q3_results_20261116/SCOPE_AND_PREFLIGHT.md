# LD5 total view・固定2配分Q3：scope/preflight

2026-11-16 JST。元configuration 4d6e2c9c4511acb0e2d7f4d488e331009504de805ad7826fdeb703907fc919ebの保存信号を用いたtotal view 1件と、利用者が指定した2列各1回・新規Q3最大2件。新配分生成は0件。

## §10.5

開始時git fetch origin。関連all-refのLD5履歴、LD5保存signal/価格、今回の出力directoryを照合した。固定A成果はstage1・stage2・較正価格のみで、同一LD5のQ3保存結果/total viewは見つからない。価格JSONでもQ3_for_LD5=NOT_EVALUATED。own-user process観測で動作中の該当LD5 Q3は見つからない。観測したDのp7/Y4価格processは別候補であり変更・停止しない。LD3の2評価は別configuration/辞書/lawなのでLD5へ流用せず、比較にのみ引用する。新規予定はLD5 FULL_PF__B_PRICE 1件、LD5 VP 1件。旧VP・旧予算・旧STOPは変更しない。

## SOURCE・metadata

- A packet ac46d100e74f3f56b4a3ff4d95c15fdc7fe8ca98 / docs/research/df_head_pauli_tail_ld5_signals_20261116/MEAN_HANDOFF.json。
- config 39a1fc7dfe9b093f0e246be595e2f7938191368b / docs/research/df_head_pauli_tail_ld5_stage1_20261116/config_run01/CONFIGURATION_LD5.json。
- 辞書6255fbc422121fd777cad52763ef5a684abd46a3のLD5_DICTIONARY.json、hash 2947784c…、同directoryのresult.jsonが残余locator。
- 13 raw carrier 1e3985e7eb29894934877a74e2a96c3088dd435a、各mean_source.selector=record。数値SOURCE c67469ad49fd92a3054cb6da9337a33f276494d0。各round自身のSOURCEを維持。
- Aの8c8ce542e9fc25ead709fe6b0eb71b87ba7434ca::b_intake_dictionary_identityと不変P1 jobをそのまま使用。旧LD3 intake/C helperの固定SOURCE dispatchにLD5 assembly/dictionaryを最小追加し、bind_packetのhash・bit順・SOURCE・law・scalar once検査を維持した。
- total ID 42309d00702fa11c43d62eb27a5ff928657dc1a4d9d8e9401547433af5c25403、binding ID 59198e931cb97974dd8a5101f4a2982865ee0b831946170e250d00af551f343e。元configuration/raw/normalization/target/qualificationは変更しない。Aの価格JSON中total_view=nullは元のまま保全し、Bのsource-bound viewのみ作成した。

## 固定2列・価格

順序1：FULL_PF__B_PRICE。X/Y各[48,45,42,39,36,33,29,26,23,19,16,8,2]、exact費用3792207339/7 RZ。
順序2：VP。X/Y各[48,45,42,39,36,33,29,26,23,19,16,13,9]、exact費用6105690627/7 RZ。

価格入力 b8bcdf97b693fa7e99d3d35f44f5ce6d3a691201 / docs/research/a_pauli_ld5_cal_price_20261116/PRICE_26AXIS.json。全26軸はMODEL_APPLICATION_FROM_Q8R4_DECOMPOSITION。各rowのN/productはVP用であり、順序1の積には必ず順序1自身のNを用いる。軸別積から両列の総額を求め、A.comparison[name].T_LD5とexact一致を確認した。cell receiptはround receiptへ付け替えない。E_C/CI/Gはnull、価格の正式資格なしを維持する。

## 数値経路・count-only・停止

LD3の既存generic worker・Q3 validator・source-bound binding・修正済み観測context・高速DPを再利用する。kernel 710aa69b、accelerator 9e294778のbytesは不変。各call内部でoutcome batchをprocess並列化し、2列は指定順に逐次投入する。

LD3のdecoder/target/law/loss/exact tau/optionsをそのまま用いる：Z1+T1 total、保存DF10 solver proxy −2.1663874486347634 Ha、保存ideal finite-K raw、shots/X/Y/round IID、EXACT_AFFINE_OF_BINARY64_RAW、全事象二乗loss。tauは十進0.00159360010199040 Haのexact二乗。A deltaEは加算しない。

順序1のcount-only上限：2401 outcomes / 18210906 verification pairs / 78466853 transitions。
順序2のcount-only上限：2401 / 18263012 / 80154405。
両者とも固定scope 4225 / 55043173 / 141559036、非終端frontier8192、全round rho1に収まる。LD5自身のbinary64閉包・質量・残余loss・MSE/moments gateは各科学call内で確認し、LD3のPASSを流用しない。

上界<=tauなら適合、下界>tauなら不適合、またぐならINCONCLUSIVE、gate不成立はUNKNOWN。順序1の適合/不適合/INCONCLUSIVEでも順序2へ進む。重大UNKNOWN・保存不能・異常時のみ未開始call投入を停止し証拠を保全する。retry、範囲拡張、配分変更なし。終了後mandatory STOP、十分N/global RMSE/G/winnerは認定しない。
