# 保存coverageと今回の再利用方針

分類は既存の候補metadata・保存status・fieldの有無による。新しい残差・phase/radius・norm判定、保存値の数値比較・再資格は今回行っていない。全physical ID/出典は [manifest](BASELINE_FINITE_MANIFEST.json) の各roundへ固定した。

## 初回39round

|m|T|Full PF S2/.1|B0I p7 S2/.1|Full-H K2 q=2^m/R448|
|---:|---:|---|---|---|
|0|.1|未取得→新規|未取得→新規|未取得→新規|
|1|.2|未取得→新規|未取得→新規|未取得→新規|
|2|.4|未取得→新規|未取得→新規|未取得→新規|
|3|.8|旧spectral保存、診断不足→新規direct|旧spectral保存、診断不足→新規direct|未取得→新規|
|4|1.6|旧spectral保存、診断不足→新規direct|旧spectral保存、診断不足→新規direct|未取得→新規|
|5|3.2|未取得→新規|未取得→新規|未取得→新規|
|6|6.4|旧spectral保存、元gate PASS→受入再資格1件|未取得→新規|未取得→新規|
|7|12.8|未取得→新規|未取得→新規|未取得→新規|
|8|25.6|未取得→新規|未取得→新規|未取得→新規|
|9|51.2|未取得→新規|未取得→新規|未取得→新規|
|10|102.4|未取得→新規|未取得→新規|未取得→新規|
|11|204.8|未取得→新規|未取得→新規|未取得→新規|
|12|409.6|旧spectral gate失敗→新SOURCEのdirectは未評価|未取得→新規|未取得→新規|

「未取得」は今回照合した固定SOURCEの同一physical candidateに対応する保存meanがないこと。全branchの不存在を主張するものではない。input/referenceはmetadata上で全39roundに供給可能だが、実ファイル・数値gateはruntimeで未確認。

|保存セル|SOURCE/locator|再利用・未資格の区別|
|---|---|---|
|Full PF m3 `83d703b801b8545ce61f` / m4 `dc96042e5f8605a29c8f`|`155529e20a:track_a_h46_long_time/.../signal_records.json.gz#ID`。pf_common由来provenanceも原本に保持|物理一致。保存recordに現在必要なpower norm/unitarity診断が揃わず、旧meanのflag書換えでは埋まらない。新directのproduction meanとして評価し、旧値との新比較は追加しない|
|Full PF m6 `de9bc15cbc089f14b47f`|`b72ea6b30:docs/research/h4_four_method_signal_screen_results_20261113/run02/result.json`、`records[1]`（F2-FINE）|旧spectral backendの元numeric gate PASSと診断fieldが保存されている。承認後に1件だけ受入条件を検査し、PASSなら旧値/旧SOURCE/旧backendを保存再利用。direct-expm再計算/新旧差計算はしない|
|Full PF m12 `4841a53bcaf02927b34a`|同run02のSCREEN_STATUS、F2-FINE/m12|旧spectralの数値gate失敗。新direct経路の科学的PASS/FAILは未評価。旧STOPを保存し、新しい承認scopeで一回だけ評価する計画。旧screenのretryではない|
|B0I m3 `398f55738920acb1d4a5` / m4 `b4156f8b791f26be5ae6`|`155529e20a:signal_records.json.gz#ID`。m3元pf_common `H4_B0I_7_8_2nd`|同物理値はあるが新policy用のpower診断が未取得。新directへラベル変更せず、新規2件に含める|

**無条件再利用0、受入成功時1、新規38、保存受入再資格1、新旧mean比較0。** 保存数値の受入処理も科学実行として許可対象へ明記する。受入失敗時のfresh m6への自動fallbackは計画にない。

## 他の保存成果をそのまま使わない理由

- Full PF S2/Y4/M5 × .1/.4のm3/m4計12、run02のm6計4を元SOURCEのまま残す。今回選んだ同recipeの5セルだけを上で位置付け、それ以外を別formula/幅へ移植しない。
- `b72ea6b30` の直接expm確認は **F2-COARSE/m12、S2/.4、q1024、ID `4f0aca817edccc7fbe17`**。今回S2/.1/q4096とは違う。再実行や新比較をしない。直接expm経路の採用根拠であって新candidateの精度保証ではない。
- 既存B0/p7/S2/.1の13roundはB0で、今回B0Iとはscalar規則が異なる。B0の全13値をB0Iへ位相変換する新解析や、新しいB0 scheduleを件数合わせで追加しない。
- 旧B0I/p7/Yoshida4/.4のm3/m4、未完m6/m12は今回S2/.1と別candidate。
- Full-H旧lightはT.8/q16/R16/K4。旧20セルscreenはT.8/q4/R180、T6.4/q26/R1434、T409.6/q1639/R91751のK2/K4で、今回の全round R448/q=2^mとは別。旧T2.0研究も同時刻ではない。同総event数だけのsignal/価格移送はしない。
- 旧20セルの4保存・6gate失敗・9未開始・1空例外未解決と、残15件という過去の計画を保全。今回その残15件を埋めるrunnerは起動しない。

## compiled価格

Full PFはD `e7dca5f73:COVERAGE_V2.csv/HANDOFF_C.md` の確定receiptを参照する。同物理S2/.1のm3/m4はX/YともEXACT metadata（計4軸）で、consumer V2/source/profile/receipt traceを結んだ後に再利用できる。今回RZ数を再計数・再解析・新record化していない。他11roundは同catalogの保存domain外。

B0I m3は `155529e20a:reused_compiled_prices.json.gz#398f55738920acb1d4a5` に両軸SOURCEがある。Full PFの73 receiptはB0Iを資格化しないため、Dにtruncated domainのreceipt/consumer bindingを依頼する。m4以降を含む他roundは今回の固定catalogで対応価格未確認。

Full-Hの旧light n9価格は別T/q/R/Kで、今回には使わない。Dの現candidate adapterは主にPR用で、3 baselineの取得経路が全て完成したとはしない。必要な価格requestはmanifestの同candidate/axis/control/full wrapper/compilerから後続Dへ渡す。loweringと不確かさ・cohort/law資格が揃うまでCは価格を採用しない。
