# H6 DF19 RPE m6：限定2候補の実行準備

2026-11-10 JST。**m6専用入口と人工差分3件を準備した。実H6のT6.4 reference／finite mean／bias／phase／radius／Nは未計算。** 今回指示§6・§8に従い、科学実行はユーザーの別途明示許可後に行う。[PLAN.json](PLAN.json)が固定menu・call範囲、[人工結果](synthetic_tests.txt)と[静的照合](CHECKS.txt)は技術確認の記録。公開照合後mandatory STOP。

## 固定条件・候補

Linear H6、隣接1.00Å、STO-3G、保存tol-only 1e−8／DF19、weighted-Hermitian completion後の保存DF-H、Nα=Nβ=3・sector400、保存数値state、P=0。targetは保存DF-Hの同sector ground energy。signed weights・fragment順・scalar・sector・H/state/snapshot bindingを保持する。

CA=.00159360010199040 Ha、h=.1、βtotal/stat=.4/.2 rad、各方式αtotal=.05、m0..12の13round、αaxis=.05/26、prior [−9.695399206821989,−3.1355322139814668] Haと無shift alias政策は[D1/D2完了結果][D2]の保存設定を転記する。prior/horizon/α/βを再計算・最適化しない。保存数値stateの厳密ground同定、actual implementation/law誤差、rigorousな誤差上界は未資格のまま。

| arm | candidate ID | 構成 |
|---|---|---|
| PF | `201a8e365ab6f4af6ebc` | Full19、m5_best、q_PF4、δ1.6 |
| PR | `ef41b65942d56977d6cd` | prefix7、m5_best、R_outer16/K2、q_PF4、δ1.6 |

両方ともm6/T6.4、q_RPE64。IDは既存`h46_long_time_signal.cid`の規約で作成したmetadataで、適格性を意味しない。既存11 signed pieceとABS_TIME_PER_OUTER配分・各piece最低1を保持し、PR全interrogationのevent数はq_PF R_outer=64。q_RPE64とq_PF4は別変数である。候補差替え・追加探索・m7以降への自動展開は行わない。

## 再利用と最小拡張

基準は公開commit **`95d01b6ab1d820921fcd2521324e41e82389713a`**。その実D2の科学SOURCEは`e0503b769358f239ef63245dda78ef1ee6ffce9b`。旧12科学SOURCE file、旧runner・8 ID guard、旧D1/D2結果とSTOPは変更していない。

追加は[`run_h6_df19_rpe_m6.py`](../../../scripts/run_h6_df19_rpe_m6.py)と人工差分test。既存`h6_df19_rpe_d2`の`make_context/reference_pair/half_step_keys/half_step/evolve/analyze`、旧runnerのinput読取・save・資源観測、m5 schedule／finite Taylor／finite normalizer／D1 geometry・Hoeffdingを再利用する。新しいRPE solver、数値公式、価格器は追加していない。

新入口はm6の2 ID・q_PF4/q_RPE64・T6.4/δ1.6・partition・固定政策だけを受け付ける。旧8候補runnerの`main()`を呼ばず、旧guardの変更やmonkeypatchも行わない。新結果のstageはD2_M6、価格coverageは`NOT_EVALUATED_SEPARATE_PRICE_TASK`として、旧D2の価格表照会結果をm6へ流用しない。

保存snapshot/receiptは従来の固定SOURCE索引から読み、3原入力hash・H/state binding、保存tail context SHAを照合する。prefix7のtail scalar=.0194666612606853、lambda_R=.1649333839054104は保存値の転記だけ。新しいtail抽出や対角化はしない。400次元の保存H作用表現を用い、full Fock行列や新Hamiltonian/DF/stateを生成しない。

## 承認後に行う科学callとSTOP

| 対象 | 固定範囲 |
|---|---|
| full-H reference | T6.4のみ、expm_multiplyとexpmの2経路 |
| PF/PR finite mean | 上記2候補のみ。PRはcorrected/raw別伝播とexact-tail内部対照1件 |
| phase/radius/disk/N | 2候補×X/Y、D1/D2と同じempirical u・margin・geometry・Hoeffding |
| D1/D2 m0..5 | 保存result/COVERAGEの読取と転記のみ。再計算0 |
| prior/horizon/新H・DF・state/eigensolve | 全て0 |
| sampling/build/compile/価格照会・fit・予測/N×C/RPE成功率simulation | 全て0 |

rawをcorrected/Bから事後生成せず、独立にattenuated列を伝播する。scalar/identity一回、負時間piece、normalization・イベント次数を変えない。u floor1e−9、既存stage依存roundoff、raw/corrected closure、reference別経路差、log_B margin、T×保存residualを含むD1/D2の数値政策を再利用する。未知actual誤差はNoneであり、経験的診断をrigorousな上界へ格上げしない。

reference別経路差>1e−10、exact PF norm差>1e−9、vector normalization closure>1e−9、非finite、SOURCE/input/binding不一致は保存結果にSTOPとして残す。正raw半径とβdet+βstat≤.4が成立した場合だけ条件付き十分Nを報告する。条件不成立はN=nullを含む有効な結果として扱い、候補や閾値の事後変更・科学的retryをしない。技術的STOPを正常PASSと表示しない。

半step指数cache・batch列伝播・観測した空きCPU/RAMに基づく並列化は旧D2の方式を使う。PF/PRの候補job数は2という科学scopeであり、CPUやworkerへの固定上限ではない。共有設定・他process・他Codexのworktreeには変更しない。

作業worktree rootで、**明示承認後だけ**実行する入口：

```bash
../../h6-environments/20261011/venv/bin/python -B scripts/run_h6_df19_rpe_m6.py --output-dir docs/research/h6_df19_rpe_m6_results_20261110/run01
```

fresh outputを要求し、既存runの再開や上書きは行わない。実行時のcommitと使用SOURCE/input hash、元D2の保存設定・結果を記録し、完了後にresult/REPORTを公開する。ユーザーのチャットによる明示許可で足り、承認証明書・grant・承認text保管等を追加しない。

## 人工確認・保存coverage・価格担当との接続

追加人工3件は、(1) m6固定scopeと旧8候補guardの保持、(2) q4/R16/K2・負時間pieceを含む非可換2-level伝播の既存spectral経路との一致、(3) m6のq_RPE/q_PF binding・reference診断必須・actual N/C/G未知量の保持を確認した。全3件PASS、約.192秒。旧D2の人工7件は既存記録として再利用し、同じ試験を全面的に再実装・反復していない。人工比較で用いた2-level eigensystemを、実H6の新eigensolveと混同しない。CLIは`--help`だけを確認した。

既存条件付きNは次の保存値をそのまま参照する。以下は数値state仮定・上記DF-H/CA/α政策の下での各軸の値で、actual保証ではない。

| m / T | PF N | PR N | 出典 |
|---|---:|---:|---|
| 0 / .1 | 705 | 705 | D2保存 |
| 1 / .2 | 705 | 708 | D2保存 |
| 2 / .4 | 705 | 719 | D2保存 |
| 3 / .8 | 705 | 765 | D1保存 |
| 4 / 1.6 | 705 | 721 | D1保存 |
| 5 / 3.2 | 705 | 737 | D2保存 |
| 6 / 6.4 | 未計算 | 未計算 | 今回の承認待ち |

価格担当へのhandoffはPLAN.jsonのcandidate ID／T／m／q_RPE／q_PF／δ／formula／prefix／R_outer/Kとscientific binding、保存tail scalar/lambda_R、compiler期待条件。RTE配分は同じABS_TIME_PER_OUTERと既存m5係数から決まり、実行結果にも配分を記録する。H hash `223628b0…`、snapshot `99440a59…`、state `664c6335…`、完全値はPLANに保存した。

compiler期待条件はD1/D2と同じQiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric_directional、測定付きfull Hadamard wrapper、P0、hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。これはm6のcompiled価格や回路build済みという証拠ではない。今回は価格Cを照会・取得・fit・予測せず、別Codexの価格研究を再実装しない。

保存contextの`binding.T=.8`は入力referenceの歴史的時刻であり、今回の発展時間T6.4とは別。PLANの`historical_binding_T_is_input_reference_only`とcandidate.Tを使い、価格へ結合する時間を取り違えない。

m6が条件付きPASSでも最大m0..6の7roundまでであり、m7..12/T12.8..409.6のsignal/N、対応C、全体phase/branch/failure資格と全Gは不足する。次の科学範囲は**本書のm6だけ**。その結果を踏まえたm7以降の最小案は将来の結果REPORTで提示し、今は新しい候補・費用・成功を仮定しない。

既存H6 branchを継続し、他worktreeに書込まない。今回の確認は本担当自身の人工・静的照合で、独立科学レビューではない。公開remote SHA・主要成果物の全文一致を確認後**mandatory STOP**。

次の担当は**ユーザー**：この固定m6/2候補・call範囲の科学実行を明示承認するか判断する。**Codex**：承認後に同じ限定campaignを一括実行し、結果公開・照合後STOP。m7以降、価格評価、詳細科学レビューへ自動進行しない。

[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/95d01b6ab1d820921fcd2521324e41e82389713a/docs/research/h6_df19_rpe_d2_results_20261110/REPORT.md
