# PR-DF LD9：configuration・13信号・共通V0 Q3（2026-11-16 JST）

**LD9の共通V0は適合した。** 既存providerのnormalizerを含む全13数値gateを通過し、元の3.11.0rc1 workerでQ3を一回実行した。

- **event数：変化なし。** 元mapperはLD9も全13 R=1を出力した。LD7のRをコピーしておらず、双方の予定event数q×Rは全roundで同じ。
- **同じshot配分の精度：条件付きモデルMSEはLD9がわずかに低い。** LD9−LD7のMSE差は約−4.036976028462E-12 Ha²。両者とも同じ採択閾値に適合する。
- **価格・総費用：LD9は未評価。** LD7の価格/係数は移送せず、compiled RZ価格・費用・厳密E[C]/G/winnerはnullのまま。

| 分割 | MSE lower (Ha²) | MSE upper (Ha²) | 判定 | 今回の取得 |
|---|---:|---:|---|---|
| LD7 | 4.28414160318294721301353E-7 | 4.28414160318294721365383E-7 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD | 保存V0再利用 |
| LD9 | 4.28410123342266259264443E-7 | 4.28410123342266259328471E-7 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD | 新規1 call |

共通V0はX/Y各 `[64,60,56,52,47,43,38,34,30,25,21,17,12]`。LD7は `0defa286c53a1b9acf513be453907bcad83102b3` の保存V0だけを再利用し、最安A_SHAPEの別配分とは比較していない。MSE差のexact区間は `[−4.03697602846210093932990E-12, −4.03697602846197288263362E-12] Ha²`、[SUMMARY.json](SUMMARY.json)に有理数を保存した。これは保存信号から作るideal finite-RTE raw IIDモデル内の比較であり、物理的な全誤差保証や信頼区間ではない。

## LD9自身のconfigurationとschedule

保存H4 linear1.00Å/STO-3G/tol-only DF10、元H/state/snapshotとfragment順を維持。one-bodyはL_Dに数えない別head、DF0..8をhead、DF9をtailにした**PR-DF**。S2・Δ=.1・K2・共通paper κと元mapperはLD7と同じ。

configuration ID：`ea087a7e3b311be7efae27558347e9eb471f616b8b55f59cbef8e506dda7a833`。

保存contextのprefix9から λ_R=`1.5605698403540042e-07`、tail identity s_R=`5.193351122457893e-08` を取得した。Hamiltonian定数c=`2.29310124732` は保存snapshotのconstant memberとreceiptを別に照合し、c+s_R=`2.2931012992535114` を総時間Tに**一度だけ**適用した。定数・tail identityの別SOURCE、hex、head/tailは[generation/SCALAR.json](generation/SCALAR.json)。

| m | T | q | LD7 R | LD9 R | LD7 qR | LD9 qR | LD9 physical ID |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.1 | 1 | 1 | 1 | 1 | 1 | `85c9c67322750fa684f9` |
| 1 | 0.2 | 2 | 1 | 1 | 2 | 2 | `c6fc0a97391231f7531c` |
| 2 | 0.4 | 4 | 1 | 1 | 4 | 4 | `eb329e71e708963df765` |
| 3 | 0.8 | 8 | 1 | 1 | 8 | 8 | `180d09b892f9407338f0` |
| 4 | 1.6 | 16 | 1 | 1 | 16 | 16 | `e89ab19795af8b2c84e7` |
| 5 | 3.2 | 32 | 1 | 1 | 32 | 32 | `3aecfc31f7c029c48b78` |
| 6 | 6.4 | 64 | 1 | 1 | 64 | 64 | `09fa97187d6eee510d0c` |
| 7 | 12.8 | 128 | 1 | 1 | 128 | 128 | `d31f4204149a7c21b64d` |
| 8 | 25.6 | 256 | 1 | 1 | 256 | 256 | `e12b3ea7c9167e78e514` |
| 9 | 51.2 | 512 | 1 | 1 | 512 | 512 | `f375c4f6be2abe32aa0b` |
| 10 | 102.4 | 1024 | 1 | 1 | 1024 | 1024 | `4b38e5ed2025eccc5f34` |
| 11 | 204.8 | 2048 | 1 | 1 | 2048 | 2048 | `6676a1bf70c04b3a11ae` |
| 12 | 409.6 | 4096 | 1 | 1 | 4096 | 4096 | `33fb90436d34dbb43e41` |

exact T、canonical identity・law bindingは[generation/CONFIGURATION.json](generation/CONFIGURATION.json)。比較表は[generation/LD7_LD9_SCHEDULE.json](generation/LD7_LD9_SCHEDULE.json)と[SCHEDULE.csv](generation/SCHEDULE.csv)。qRは予定RTE event数で、実sampleやcompiled gate数ではない。headとevent内容は変わっているので、同じqRから同じ1 shot価格は導けない。

## 13 roundの信号とnormalizer gate

変更しない `fefd548ecd56f60d72031b3d385f6f5e6ee95d06:src/trottertracks/pr_rpe_signal_provider.py` のdirect-expm head / paired finite-Taylor / matrix-powerを使用。保存Full-H reference pairを13件再利用、新reference action0。新state/H/DF生成・trajectory sampling・回路build・compileは0。

全roundの1 pieceはsigned_time=.1、event_time=.1、τ_RTE=`1.5605698403540043e-08`。保存binary64のnormalizerは **b=`1.0000000000000004`、hex=`0x1.0000000000002p+0`**。元の有限・b>1 gateを通過し、丸め/正規化/閾値/精度設定を変更していない。全13のnorm・unitarity・raw/corrected closure等も元gateを通過した。最初のmacroを計算後、残り12 roundは既存head/macro cacheを再利用した。

b>1の通過は元の保存精度と数値gateでの観測であり、normalizerや信号の厳密数値誤差証明ではない。bの値/hex、B/log_B/attenuation、raw/corrected/exact-tail、referenceとphysical IDの対応は[ROUND_INDEX.json](ROUND_INDEX.json)、全診断は[signals/RESULT.json](signals/RESULT.json)と各m*.json。13複素信号をX/Yで共有し、26独立pathとは数えていない。

最初のgate不成立でm/b/hex/理由と部分結果を保存して停止する入口も、人工入力で確認済み。**実入力では不成立0、技術STOP0、未取得round0。** Q3不適合と読み替えていない。

## 共通V0 Q3の固定条件・検査

科学call前の `54b10d2473df0fdc2c5bfc7991e4071428a88a53:Q3_INPUTS.json` に、LD9自身の13信号・固定vector・実行SOURCE・元評価契約・出力先を固定。既存A intakeが13 raw/identity/law/normalizerを照合し、canonical total viewへ結んだ。

- Python3.11.0rc1、元worker/Accelerator、元kernel710aa69b・accelerator9e294778とAST hash `3d043e716246e57a8b05784b0d5c0f6777aaf43410fed8c4dd8b68df6dd590b8` を維持。
- decoder=`consecutive_dyadic_nearest_branch_principal_energy_total_v1`、保存Full-H solver条件付きproxy target=`−2.1663874486347634 Ha`、全事象squared-energy loss。
- exact採択閾値τ_MSE=`15136964828156312001/5960464477539062500000000 Ha²`。価格holdout用の許容差や他方式の規則を導入していない。
- count-only upper137,923,114、実DP transition94,048,712、元cap141,559,036。新V0呼出内の閉包は `VERIFIED_FINITE_BINARY64_SUPPORT`。

Q3は一回だけ、`MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`、API_RETURN==RESULT、worker exit0。数値不成立・不適合・判定保留/UNKNOWNは今回は0。[CHECKS.json](CHECKS.json)と元[RESULT.json](q3/LD9_V0/RESULT.json)を参照。HEAD/T/sourceの変更で他jobを止めず、期待hashの差替え・guard無効化・retryは0。

## SOURCE固定・§10.5・資源

生成/信号/Q3 caller実行SOURCEは `303ed2ef705d0a2b24cff1c9118058386e247ca1`、再利用helperは1cbc24cfc、元数値providerはfefd548ec。生成入力04090592a→configuration33a308bd7→信号入力d1d2b847f→信号raw2f717a807→bundlef117269c1→Q3入力54b10d247→Q3 raw560ef11e7→C向けjoin778118926の順に固定した。全40桁locatorは[SOURCE_LOCATORS.json](SOURCE_LOCATORS.json)。SOURCEと結果公開commitを別々に保持する。

f89の旧LD9 catalog調査とLD7での照合を再利用し、その後のremote成果、同H/state/snapshot・split/law/条件のreceipt/job、SOURCE・出力競合を差分確認。同等の現行LD9成果は未同定。生成した13 physical IDを旧archiveとも照合し一致0。PR-Pauli LD3、決定論PF/Y4、LD7のC joinはLD9の代替とせず、他担当のworktree/jobは非介入。[DELTA_CHECK.json](DELTA_CHECK.json)、[PHYSICAL_DELTA_CHECK.json](PHYSICAL_DELTA_CHECK.json)、[Q3_DELTA_CHECK.json](Q3_DELTA_CHECK.json)。

[TESTS.txt](TESTS.txt)には、新13行検査（R1に固定せず元R3も許容する人工case、不整合拒否）、新信号入口のb=1停止/保存/残り12未開始、元providerのb=1診断保持・scalar一回のtoyテストを記載。旧保存H4 P0/operator suiteは実行0。作成者自身の照合であり独立レビューではない。

| 段階 | wall(s) | CPU(s) | peak RSS(KiB) |
|---|---:|---|---:|
| configuration/mapper検査 | 3.507 | user4.529 / system5.828 | 42,944 |
| 保存入力restore＋13信号 | 5.066 | user15.978 / system6.923 | 358,376 |
| Q3 worker・保存 | 232.882 | worker221.107 / child202.605 | worker832,040 |

Q3外側entry wall235.405s。各PROCESS、RESOURCE_FINAL、協調stage/round logにexit/signal・資源を保存。元cache/並列acceleratorを流用し、環境全体の設定変更や周期faulthandlerは0。

## 完了範囲とCへの引渡し

今回予定/新規/有効はconfiguration1/1/1、signal13/13/13、Q3最大1/1/1。LD7 V0再利用1、saved reference pair13、compile・path・価格fit・価格予測・配分探索は0。C向け機械可読joinは **`778118926c85158f947928c1e744d565464aab17:docs/research/d_pr_df_ld9_v0_results_20261116/C_HANDOFF.json`**。configuration・original/total ID・binding・配分・Q3結果を結び、価格/総費用はnull。C自身の受入は別で、D2はC受入完了を主張しない。

同じκ・同じV0でR/event数は変わらず、LD9の条件付きモデルMSEはわずかに低かった。head/tail・固有signal/normalizerの個別寄与を分離同定したわけではない。LD9の費用比較には自身の価格証拠がまだ必要で、LD7のモデル係数は移送しない。元H/state条件・ground仮定、実samplerへの移送、厳密E[C]/CI/G/winner、全L_Dの最適性/単調性は未認定。

[HANDOFF.md](HANDOFF.md)に固定locatorとロードマップ差分を残す。LD7のraw・価格fit・独立holdout・最安配分、凍結SOURCE、旧STOPは変更していない。commit/push・remote照合後mandatory STOP。
