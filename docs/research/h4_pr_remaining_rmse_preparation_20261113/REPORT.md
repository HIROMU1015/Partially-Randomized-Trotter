# H4 PR残10roundの証拠補完基盤と全13round RMSE計画

2026-11-13 JST。**m3/m4の保存meanを再利用するS1と、欠測8候補をspectral powerで取得するS2を一つのbatch入口へ接続した。** B3216d27aの受入済みm0/m1/m6 packetを参照し、旧packetの再生成や新しい実数値評価はしていない。今回は技術調査・実装・人工テスト・実行計画のみ。実H4のsignal/normalization/N/RMSE・保存数値再解析・新reference action・decoder統計simulation・C/Gはすべて0。

成果：[固定metadata PLAN](PLAN.json)／[S1/S2/S3実行計画](SCIENCE_PLAN.md)／[B HANDOFF](HANDOFF.md)／[人工・回帰結果](TESTS.txt)／[共通batch](../../../src/trottertracks/h4_pr_rmse_completion.py)／[元spectral powerへの薄い接続](../../../src/trottertracks/pr_rpe_power_mean.py)／[CLI](../../../scripts/run_h4_pr_rmse_completion.py)。A既存branchを継続し、B/C/Dのworktree・process・SOURCEを変更していない。

## 固定条件と現在の資格

Linear H4、隣接1.00Å、STO-3G、保存tol-only1e−8 DF10/completion/元fragment順・scalar、sector(2,2)/dimension36の保存数値state、P=0。S2/L_D1/ΔPF=.1/K2/h=.1/M12、S0 exact有理数κ、εE=.00159360010199040 Ha、εqpe=εtrot=.00079680005099520 Ha、minkowski/finite_B_diagnostic。元H/state/snapshot・prior・decoder・law・compiler/control/full wrapper/profileを保持する。

Canonical IDは **`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`**。m0/m1/m6のnominal各軸64/60/38は既存値の参照のみ。他10roundの新nominal・全13roundの十分N・global RMSE boundはnull。nominalを十分Nへ昇格しない。

| 固定SOURCE | 今回の再利用範囲 |
|---|---|
| A結果 `3e035c0d61386010f59ad7ea792bc7a5cefd18ac`、m1科学 `ac873e7fcc42e937084fa153476cbd87565702b4` | 元result/packet/64,60,38を保全。旧m1専用adapterは変更しない |
| S0結果 `7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75`、科学 `7c3e1e5db0c55dd8249e9030a4c7e9954496fa7e` | 全schedule・exact条件と元scalar gate/allowance式を再利用。実S0再解析0 |
| B `3216d27a69ac8532d2a22e78f5f94491d202d238` | 正本PR configurationと`rpe_diagnostic_intake.validate_m1_packet`。元packetは正式受入済みなので作り直さない |
| m3/m4公開 `155529e20a8fac39e00786c40b57830b71203c4f` | long_timeのraw、endpoint、binding/compiler provenance。m3/R16とm4/R32が完全一致 |
| 元raw科学 `e417ffc9f52b13c18f5493193ce510d858664259` | `h46_long_time_signal.evolve_batch`、`h46_new2_core`のpolynomial/spectrum/spectral_applyを固定Gitから再利用 |
| 保存reference公開 `89c91c692e1d35de94d2e7afbe890aabe6d80b7e`、元科学90dcc3ed2… | S2の8時点すべてで同binding/Tの二経路pairあり。別Rのmeanを流用しない |
| P1 `71ba49bb28b295aa28e51101bc3b1bf07d500767` | reference gate/finite-RTE law SOURCE。`rte.py`、旧action/geometryは不変 |

raw provenanceの記録と元SOURCEのcode hash、元H/state/compiler/profileを静的照合した。必要なGit SOURCEと保存snapshot/receiptの所在は確認できた。私有NPZ配列は今回読み込んでおらず、その数値的資格や実runでの入力hash確認を済ませたとはしていない。実行時には既存loaderの全hash/layout checksを行う。

## m3/m4：保存meanの再利用資格と不足

physical IDはm3 `045843b2429bece6f29b`、m4 `3df0cfd3356c358aeda8`。H/state/snapshot/profile、formula/prefix/K/T/q/R、law実装、phase_alignment=false、元成功recordのorigin、endpointのELIGIBLEが一致した。旧endpointのε=.0005は**過去allowanceのselectorだけ**で、現在のRMSE目標へ採択しない。旧Nを読替えない。

元`evolve_batch`は返却前にexact-tail norm誤差≤1e−9、独立raw/corrected vector closure≤1e−9を検査する。しかしrawには**観測exact_tail_norm_errorがない**。今回のadapterは`observed_error=null`とSOURCE gate limitを別欄に保持する。0や1e−9を「観測値」として代入しない。

**再利用できるのは、SOURCE guardに条件付けた局所empirical診断である。** その証拠契約は今回の提案で、科学的RMSE資格の確定ではない。次のS1承認に、観測scalarの代わりに固定SOURCE guardを使う条件付きrecipeを明示する。観測scalarそのものが必要と判断された場合は、m3/m4同候補の追加mean/対照を別scopeにしてSTOPし、自動で再計算しない。

旧backendはspectral polynomial power/blockで、m1のevent逐次適用とは浮動小数点演算・norm観測箇所が異なる。既存R-aware式のlogical event数・stages・floor・gateを保持し、元endpoint u/marginより小さくしない。SOURCE gateを通ったという理由だけで新backendやrigorous roundoff boundを認定しない。

m3/m4の独立reference pairは不足。既存`h6_df19_rpe_d2.reference_pair`のexpm_multiply/expmをT=.8/1.6で各2経路、計4 actionだけ追加する案とした。保存meanのreferenceを中心として保持し、新2経路との最大差を同じ1e−10 gateで照合する。新pairの元値も保存し、中心を黙って置換しない。

## 欠測8候補と高速化

| m | T | qPF | R_outer | logical event位置数 |
|---:|---:|---:|---:|---:|
| 2 | .4 | 4 | 8 | 32 |
| 5 | 3.2 | 32 | 64 | 2,048 |
| 7 | 12.8 | 128 | 255 | 32,640 |
| 8 | 25.6 | 256 | 510 | 130,560 |
| 9 | 51.2 | 512 | 1,020 | 522,240 |
| 10 | 102.4 | 1,024 | 2,040 | 2,088,960 |
| 11 | 204.8 | 2,048 | 4,080 | 8,355,840 |
| 12 | 409.6 | 4,096 | 8,160 | 33,423,360 |

現在のSOURCE範囲で一致meanがない8候補を固定した。R255を256へ丸める等の候補変更はしない。保存reference8pairをそのまま再利用し、S2の新reference actionは0。qRPE=2^mとqPF=T/ΔPFは区別する。

既存有限K2のpaired lawはTaylor次数0..3のcorrected operatorに対応する。元`evolve_batch`のtail eigenvalue polynomialのR乗を利用し、S2 outerごとのhead-forward → tail → head-reverseの順序を保持する。headとtail全体が非可換なので、全outer unitaryを別の有効Hamiltonianへ置換しない。全eventの確率・順序を変える手法でもない。

高速leafで変更したのはhelper importの固定bindingと最終vector/maximumの露出だけ。元polynomial power、finite distribution、logB、raw/corrected別columnの伝播、exact-tail対照、signed time、total Tでのscalar一回、元norm/closure gateをそのまま使う。rawをcorrected/Bから**終了後に定義しない**。新観測norm scalar、spectral再構成/isometry残差、normalizerのone-ULP log感度を追加記録する。

m12のtail処理は33,423,360回のevent loopから4096回のblock適用へ移る。二headとtailのspectraを1回構成して全候補で共有し、候補ごとは独立vectorを持つ。full snapshot Hのground-state eigensolve、SCF/DF/state生成はしない。36次元sectorのblock spectra計算は将来の科学callに含む。並列数は有限候補数と実affinityを使い、固定CPU/RAM/worker上限を設けない。

**精度の限界**：block境界normは逐次eventの観測履歴ではない。Hermitian polynomialのpowerでは、厳密算術上の中間norm包絡をblock端点から説明できるが、浮動小数点等価性は別である。元R-aware logical event式を維持し、normalizer感度/過去marginを下回らないempirical allowanceにする。spectral残差をrigorous誤差へ自動変換しない。実H4の長時間精度・速度は未測定。次のS2で**m2一候補のみ従来actionとも照合**し、同じallowance下でraw/corrected/exact-tailとlogBが対応するか検査する。これを長時間側のrigorous証明とは呼ばない。

## 実装とB接続

- `plan()`：固定10candidate/全identity/law、保存SOURCE/fieldと8reference locator、入力の所在を列挙。新数値評価0、nominalと十分Nはnull。旧packetはSOURCE参照のみ。
- 共通`diagnose()`：元S0/R-aware関数の算術と閾値を再利用。旧HP PASSへの依存を外し、事前のobserved norm又は固定SOURCE guard検査へ置き換える。この差を新SOURCEとして公開し、過去科学SOURCEを変更しない。旧Hoeffding/analyze全体を呼ばない。
- `execute()`：将来のS1/S2一括科学入口。各mean/ref/局所gateを候補ごとに処理し、新規meanだけをRMSE evaluatorへ一回batch供給。m0/m1/m6のshot算術は再実行しない。個別mean・途中resultを保存し、FAIL/数値未解決時は資格packetを出さずSTOP、retry/fallbackなし。
- `join_records()`：将来の新scope packetだけをB既存`n_from_a`へ渡す。旧m1 packetを再作成する機能ではない。preserved m0/m1/m6の値・SOURCEを保持し、他roundにSOURCE付き新recordを付加する。nominal/十分N/global契約を混ぜない。

Bのcontroller/planner/共通identityを再実装していない。現在B3216d27aのvalidatorはm1 scope専用であり、残10roundの将来packetへ無理に使わない。HANDOFFに新scope・provenance仕様を示し、公開科学結果ができた時点でBが受入条件を更新する。

## 全13round RMSEへ進む条件

S1/S2が完了しても、得られるのは10roundの局所診断と経験的nominal配分である。S3では全raw mean→X/Yの±1分布→固定13round decoder→target energy lossを結ぶ必要がある。[SCIENCE_PLANのS3](SCIENCE_PLAN.md)に統計精度・信頼度・有限試行数の導出条件を記載した。Monte Carlo/binomial simulationは今回0、実行入口も追加していない。

zero座標やbranch tieは元decoderがenergy=nullを返す。これらを削除した成功条件付きRMSEを全試行RMSEと呼ばない。未解決trialのloss/出力、保存数値stateをtruth proxyにする条件、prior/alias、finite-K/PF biasとMinkowski成分誤差、ideal/actual law、検証信頼度・MSE精度はユーザーの科学判断が必要。旧α/βやHP Nを新保証へ流用しない。

## 検証と次の承認範囲

[TESTS](TESTS.txt)：新13件と旧m1 11・S0 7・RMSE17の**計48件**を対象に人工/metadata検証を実施。人工非可換2次元・負時間・巨大event形状・raw独立伝播・元guard/欠測/null・旧policy拒否・群STOP/無fallback・B全scope・旧SOURCE不変を確認した。作者自身の技術確認で、実H4性能や科学的RMSEの検証ではない。

次の推奨は、**ユーザー**がS1＋S2を一つの有限scopeとして承認し、**Codex A**が保存mean2候補の診断＋4 reference action＋新高速mean/対照8候補＋m2直接照合1候補をまとめて実行すること。m3/m4のSOURCE guardを使う条件付きempirical recipeも承認条件に明示する。m0/m1/m6再解析・追加候補・S3 simulation・C/Gは含めない。S3は未確定契約を決めた後の別実行である。

今回commit/push・remote SHA/tree/主要成果物照合後、**mandatory STOP**。新科学計算へ進まない。
