# C-S V0保存schema修復後：限定検証完了

2026-11-08 JST／SERVER1 GPUA100-gleap-01。
**修復SOURCEでrun02を完了。D/R両局所specのfinite/native/local capに合格し、7positive keysを取得した。今回の局所literal Tは通常合成より増加し、T削減は見られなかった。原本を公開・照合後mandatory STOP。**

## 条件・修復と実行来歴

保存linear H4／1.00 Å／STO-3G／legacy DF12／8system／全4電子sector ground参照／L_D3、m0,q1,δ=.02,r1,K2,coefficient_atol0、N1 actual Q／selected full。D/R canonical first joint factor0各1、π保持nearest1/256rad/tie-away、residual r_res1/K_res2/root256/proposal160/ηρ10^-12、strict pygridsynth2.0.0/seed0/dps100/10^-6、finite cap10^-12/local cap3e-6を変更していない。

元SOURCE0bd18bbのrun01は保存str parameterの数値符号反転でSTOPした（[旧結果64fcb50](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64fcb50dc09172b28a8d34880789cbbd4f5f717e/docs/research/cs_v0_science_20261108/REPORT.md)）。今回の「意図しない中断は修正・再実行可」という明示許可に従い、既存basis codecで数値へ復元してからinverse thetaを反転する最小修復を行った。XXPlusYY beta、位置、phase、科学値は維持した。新人工5件がPASS、旧18/9/10件の全面再実行なし。

[修復SOURCE e69d1df](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e69d1dfbff0f8ed924109c3c5f30087fc122f595)を科学入力投入前に公開し、16ファイルの本文/blobとremote SHA/treeを照合した。新preflightは完全な新SOURCE hash表を使い、固定13入力/runtimeの検査を省略しない。新output/cacheのrun02で一回だけ起動、process exit0で正常終了した。元run01/STOP/費用はbyte不変。

技術的attemptはrun01＋run02の計2回、完了した限定campaignは1回。今回のrun02内部の追加retryは0。旧run01はnative取得0だったため、技術再実行を含めた実native取得は7keysのままで、上限8を超えない。科学的cap違反の救済、条件探索、別slotへの差替えはない。

## 取得したidentity・適格性

- D：`D/forward/df-one-body-correction/central0/factor0`
- R：`R/order0/rotation/df-fragment-10:Z0/factor0`
- 両joint wordは保存順の `IIIIIIIIZ`。factor1はFIXED partnerのまま。保存selected basis/policyを保持した。
- 入力はN1 744e1a0、N2 5e9bd08、旧engine fdf00ceの13固定参照。同じactual Q/selected full、H/partition/preparation/tail/stateを維持し、N1/N2再生成0。state vectorや旧engine上界を変更後列へ転用していない。

BINDING.json、LOCAL_PLANS.json、REQUESTS.jsonを[原本directory](../../../artifacts/cs_v0_science/2026-11-08/campaign_run02/)に保存。target/coarse/residualはexact表現、各operator/finite meanは2×2 Pauli eigenspaceだけを評価し、full H4 operator/state/Bornは使っていない。

## 誤差と局所T

以下の誤差表示は厳密有理数上界を外側へ丸めた値。価格の≈は表示用で、exact値は原本JSONを正本とする。

| spec | finite error上界 | 合成local error上界 | S0 通常T | S1 coarse+finite residualの局所期待T | S2 Dだけ適用の局所期待T |
|---|---:|---:|---:|---:|---:|
| D | ≤1.305×10^-18 | ≤2.526×10^-7 | 70 | ≈142 | ≈142 |
| R | ≤2.194×10^-20 | ≤2.840×10^-7 | 70 | ≈140 | 70 |

7keys全てstrict operator guard≤10^-6を通過した。両finite≤10^-12、`b_local=b_finite+B_d b_coarse+Σc_n δ_native,n`≤3×10^-6。scalar/event/global W、negative actual adjointを保持した。coarseの共通positive keyを共有できたため最大8案から7keysになり、再合成はない。

係数和B_dの表示はD≈1.0000000055962222、R≈1.0000000007255108。actual dyadic proposal、exact c/q/weight、range、moment、root幅、truncation/finite errorはspec*_finite.json。B_dとrangeがexactに等しいと仮定せず、原本を別々に保存する。finite meanをexact residualと同一視していない。

価格は**選択局所primitiveのliteral native加法モデル**。T/T†、CX、1Q Clifford、W、serial depth・枝別価格とactual residual proposalによる期待countを保存した。full wrapper/partner/basisのcompile、optimized depth、必要shot数N、総費用Gは未取得。今回の2specでは追加T削減が得られなかったという限定結果で、C-S一般のNO-GO・full費用順位を採択しない。

## call数・実行負荷・保全

binding1、local comparison2組、residual mean/error2組・4event、positive acquisition7/返却7/qualified7、local price2組。全14 scientific call-groupはRETURN。state/Born/RTE・測定RNG/量子shot/full build/compile/N/G/他roundは0。合成器内部のseed0手順は採択profile通りである。

runnerの記録値はwall約2.21秒、process CPU約16.40秒、peak self RSS 174,060 KiB。これらはrunner内部の計測点までの値で、起動・末尾I/O等を含む完全なcampaign wall/CPU会計や大規模性能の保証とはしない。実資源観測では空きCPU/RAMがあり、pressureによる中断なし。CPU/RAM/workerの恣意的capや共有設定の変更なし。

runner原本74ファイルをbyte不変で保全し、RUN.json/RUN.logを追加した全76ファイルを[ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json)へ記録した。科学実行の正本はRESULT/STOP/CALL_COUNTSと42journal、各guarded cache/先行raw、spec別mean/error/price。sourceと旧run01も保全した。

元main HEAD/status/3,087既存files、固定13入力、旧SOURCE/result/grant/ledger/marker/STOP、SCV rawとC1凍結、他job/processを維持。最新の結果を索引/status/当日note/manifestへ追記し、過去の記録は書き換えない。詳細はPRESERVATION.json。

## 次の判断と停止

保存schemaの接続不良は修正され、V0の局所資格取得は完了した。今回のcoarse/residual政策で局所T削減の材料は得られなかった。V1/full-wrapper、別coarseやprecision、他slot/分子/round、全量子資源比較は新しい科学判断・有限scope・明示実行許可を要する。今回は開始せず、重要科学レビューも自動実施しない。

結果branch=`codex/cs-v0-repair-results-20261108`、科学SOURCE branch=`codex/cs-v0-repair-20261108`。source SHAは上記e69d1dfに固定し、結果公開SHAは本REPORTの公開commit・最終チャットを正本とする。remote SHA/body/blob/treeを照合後、**mandatory STOP**する。
