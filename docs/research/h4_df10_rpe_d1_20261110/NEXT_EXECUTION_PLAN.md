# D1後の最小実行案（未承認・未実行）

[完了済みD1](REPORT.md)はT=.8の2pointだけ。h=.1、m0…12の全13roundを設定したが、他roundのN・Gは未取得。本案の採択や閲覧は実行許可ではない。

## 同条件の保存時間と価格coverage

固定carrier `155529e20a8fac39e00786c40b57830b71203c4f` の主研究signal_recordsにはT=.8/1.2/1.6/2.0がある。今回のh=.1のdyadic scheduleに一致するのは.8と1.6。時刻の存在だけを確認し、D1以外のbias/B/phase/Nを今回再計算していない。

| round / T | 保存signal | 対応価格 | 今回の資格 |
|---|---|---|---|
| m0…2 / .1,.2,.4 | この固定corpusにはなし | 同条件の対応表未取得 | 未評価、初期roundを省略不可 |
| m3 / .8 | 指定PF/PR | PF決定論、PR n9/n33直接compiled | D1条件付きNのみ完了 |
| m4 / 1.6 | PF B0/S2/p7/qPF16 `8e50c5560bb8f47f026f`、PR S2/p1/qPF16/R4/K2 `64e4f7e9ec7fba28815e` | PFはexploration_costsに存在。PRは確認した固定価格表に該当IDなし | 保存候補の存在確認のみ、N未評価 |
| m5…12 / 3.2…409.6 | この固定corpusにはなし | 直接価格/適格なDF10長qモデル未取得 | 未評価 |

価格確認の固定表は同carrierのexploration_costs、reused_compiled_prices、extension_parallel_costs、fixed_delta_exploration_costs（artifacts/resource_applicability/track_a_h46_long_time/2026-10-25配下）。これは限定corpus内の欠測判定であり、全私有資産を網羅した不在証明ではない。PRのconfirmation cohortを別candidateの価格へ移植しない。T1.2/2.0をdyadic roundへ丸めて使用しない。

## 推奨する次の最小範囲

1. **ユーザー**：最初にD2a（T=1.6保存PF/PR各1候補のscalar評価のみ）を実行するか判断する。D1のh/β/εE/全round α配分、同H/state、同prefix/R/Kとδ=.1を保持し、上記2 IDに限定する案。新signal/trajectory/build/compile/fit/最適化/全G/H6は含めない。
2. **Codex、D2aの明示許可後**：既存adapterとD1のdisk/normalizationを再利用し、m4のphase/radius/Nと資格を評価する。D1は再実行しない。既存価格は単価照合だけとし、欠測PR価格を0や外挿で埋めない。この段階で分かるのは同じ候補系列の次の保存時刻での成立性で、全RPE費用ではない。
3. **ユーザー**：D2aの結果を踏まえ、D2bの欠測時間signal評価とD3価格評価を具体的に許可するか決める。必要なら候補/誤差資格の方針を先に選ぶ。各小処理ごとの再承認は設けず、一つの固定scopeをまとめて許可する。

D2bの最小初回案は未取得の初期T=.1/.2/.4と最初の長時間T=3.2に限定する。候補系列の初案はS2、PF B0/p7対PR p1/R4/K2、δ=.1、qPF=T/.1だが、長時間の成立は未保証であり固定採択していない。新Hamiltonian/stateを生成せず、保存snapshotを使う既存finite-mean入口を再利用する。正当なreference/誤差評価とlog Bを取得し、trajectory samplingやcompileを自動追加しない。T=6.4…409.6への全展開、候補再最適化やα/β最適化は、この限定結果を見て別scopeで判断する。

全roundへ進むには数値stateをtarget groundへ同定する資格と、reference/numerical allowanceの資格が必要。厳密保証を目指す場合は、単なるsolver residualや経験uだけでは不足する。actual law/実装誤差も未取得であり、未知量を0として成功保証しない。まず条件付きpilotを継続するか、保証を補強するかを明示して選ぶ。

## D3価格・総費用への接続

Nと直接価格Cと予測Ĉを別欄で保持し、まず保存/固定Nに対する価格問題を分離する。最終的な式はG=Σ_m,a N_m,a C_m,a。新compileなしで今すぐ再利用できるのはD1の単価対応と、存在する保存candidateの単価だけ。全Gを新compileなしで完成できるとは現時点で判断できない。

Track Dのwhole-path/frozen holdout/paired covariance/残差管理、B4/B-CのNとCを分けた会計、旧DF12のdirect/model分類は方法・コードを再利用できる。[既存価格調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/caa592a807c533c36e06a94d2fd4e62af4a38734/docs/research/rpe_price_reuse_survey_20261110/REPORT.md)に保存された検証は再実行しない。

一方、Track D number-CDF8/LD2/r32/K0のStrong/Referenceと、本研究DF10/prefix7または1/R4/K2のPF/PRは異なる。DのAffine係数・改善率・共分散を数値移植しない。q64は未完了の旧断面だけで判断せず、上記価格調査が参照する後続完了結果を使う。それでもDF10への移送資格は得られない。旧DF12や新Aの価格もHamiltonian/分割/state/control/compiler/qの意味が合わなければ係数を移植しない。

追加価格が必要なら、最初にm4 PRの同candidate直接価格の既存資産を限定探索する。それで不足する場合のみ、新compileまたは同DF10条件のモデル構築・独立holdoutを具体的scopeとして提示する。新model fit/compileは現在未承認。Nの揃ったroundについてdirect対predictionの既存差を優先して再利用し、必要時には新しい適用検証を別途提案する。現時点では全Gのround寄与を計算していないため、支配roundやPF/PR総優位は未判定である。

H6は共通interface・設定理論の参考まで。H4のprior/h/N/価格をDF19へ数値移植せず、H6の新計算を本案へ含めない。

**mandatory STOP：この案ではD2a/D2b/D3のいずれも開始しない。**
