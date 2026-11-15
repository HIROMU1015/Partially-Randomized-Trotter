# LD3 PF予算・固定4列 Q3 preflight

2026-11-16 JST。今回の利用者指示により、LD3自身の26軸モデル参照価格を用いた配分生成・exact条件付き費用・最大4新規Q3のみ実施する。早期終了の意味を保つためscience callは逐次投入する。各call内部は既存8 workerの高速DP。

## §10.5・再利用

開始時に`git fetch origin`実施。関連all-ref履歴のLD3/Pauli budget/reallocation、A LD3 packet/result directory、今回の出力directoryを照合。LD3の同契約保存結果は2de33a4f4のVP点のみ。Aの新価格較正f25a05f8aはrisk評価ではない。LD2やPR-DF/LD7は別configuration/dictionary/lawなのでQ3を流用しない。同一LD3の追加配分Q3や進行中の該当jobは見つからなかった。own-userのQ3/LD3 process表示も照合し、科学jobの二重投入なし。

既存VPはAPI snapshot c7961238679f7dc7dd358baa924ec1e3579a7121を保持し、対応する入力は2de33a4f4。新4列はVPと一致しない。重複vectorをまとめる実装と、VPに一致した場合の既存結果再利用経路を人工/保存metadataで確認した。未取得と判断できないものを新call対象にはしていない。

## 入力・生成・SOURCE固定

- 元/total ID：dd5b881d308f6fc48d3c91d08cec22c5fbbac296f60242e9bfb433c3e0321847 / 6757998a50bec243e1cd14d1b968457c673d383c47cb95c6b52e8b0ed7e1a0eb。
- 信号・canonical binding・decoder/target/law/loss/exact tau・runtime数値gate：2de33a4f4のLD3経路をそのまま再利用。元signal cdbce383f、数値SOURCE 8c8ce542e。rawやnormalization/scalarは再取得・再適用しない。
- 26軸価格：f25a05f8a92aab43cfbe30fc80b974054156aac1、docs/research/a_pauli_ld3_cal_price_20261116/HANDOFF_BC.json。各rowを元/total candidate、binding、q/R/event、元B索引のlaw/辞書/raw SOURCEに照合し、元PRICE_26AXIS rowも読む。全26軸MODEL、qualified_for_candidate=false、E_C/CI/G nullを保持。
- 既存の有理数価格breakpoint生成器：e094b2d60c5c4468711dea244fa2f91fb587ac12のscripts/run_pauli_ld2_reallocation_q3_fixed.py::generateをpure関数としてbytes由来で再利用。A_SHAPE重みVP、B_PRICE重みVP/C_ref、各軸1..VP。
- 予算は今回の指示395,962,678 / 559,457,386 RZ。前者は保存p7 m12=2、後者はa006b8e15のFull PF m12=2の条件付き達成費用。A HANDOFF中の旧PF基準593,555,352 / 838,693,580は今回の予算に使わない。原A資料・旧結果は変更しない。
- caller/INPUTSと全4列・評価順を科学call前にcommit/pushし、remote一致を確認する。配分生成後は結果を見た変更・追加列なし。

## 固定順・count-only

| 順 | 予算・規則 | X/Y各軸整数列 | exact費用 RZ | 最大outcomes / verification pairs / transitions |
|---|---|---|---|---|
| 1 | P7 A_SHAPE | 21,19,18,17,15,14,12,11,10,8,7,5,3 | 30009021696/77 | 484 / 693158 / 14915108 |
| 2 | P7 B_PRICE | 48,45,42,39,36,33,29,26,23,19,15,4,1 | 456463851968/1155 | 2401 / 18180068 / 77696805 |
| 3 | Full PF A_SHAPE | 26,24,23,21,19,18,16,14,12,10,8,7,4 | 10729281172/21 | 729 / 1676822 / 23352994 |
| 4 | Full PF B_PRICE | 48,45,42,39,36,33,29,26,23,19,16,8,2 | 625936421828/1155 | 2401 / 18210906 / 78466853 |

各列のexact予算残額・軸別積はINPUTS。価格全26軸同一SOURCE・すべてモデル参照値。最大scope 4225 outcomes、55043173 verification、141559036 transitions、非終端8192、rho=1全roundは変更しない。count-onlyは新vectorごとに確認した。閉包PASSは保存VPから流用せず各call内で検査する。

## 判定・保存・停止

Z1+T1 total decoder、保存full DF10 solver proxy −2.1663874486347634 Ha、理想finite-K raw・shots/axes/round IID二項、EXACT_AFFINE_OF_BINARY64_RAW、全事象二乗loss、十進epsilon exact二乗を維持。A deltaEを加算しない。上界<=tauのみ適合、下界>tauのみ不適合、またぐ場合INCONCLUSIVE、gate失敗UNKNOWN。

最初の適合で停止。安い全列が不適合なら生成集合内の最安適合点、安い列にINCONCLUSIVEがあれば確認済み最安とし保留を列挙。UNKNOWN/保存不能/子異常など重大gate失敗では新規投入を止める。全列不適合なら保存VP費用173557305044/165を維持。最大4、retry/別配分/範囲変更なし。

既存worker-owned API/partials/CPU/peak RSS保存と親supervisorを再利用。前回の別module instance問題を修正した`L.installed`を実入口で用い、人工実行でもSEGMENTS・子exitと保存returnを確認した。新signal・sampling・MC・compile・fit各0件。正式十分N、物理RMSE、E[C]、G/winnerを認定しない。終了後mandatory STOP。
