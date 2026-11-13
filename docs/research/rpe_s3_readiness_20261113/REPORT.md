# A/C/D最新証拠のV2統合・S3実行準備

2026-11-13 JST。**技術受入・最小adapter・人工/回帰検証まで完了。実H4 S3は未実行。** B基点`170047ef6e7a72aef5f4c6ff9c73f942097a694e`、既存branch `codex/rpe-decoder-statistics-20261113`を継続する。過去のB/A/C/D成果・科学SOURCE・STOPは変更していない。

## 受入結果

| 成果 | 固定SOURCE | 結果 |
|---|---|---|
| A全13round | `169d3316dd008eb98dd9be56fe4ecefc5c1a7d60` | 新10round scopeを独立に受入、旧m0/m1/m6と各SOURCEを保全 |
| A物理input・誤差契約（追加公開） | `bba150834ca30bdb186fbfd58c55a8d51177b855` | 今回の依存項目として公開を確認。保存PHYSICAL_INPUTSを参照し、物理誤差の未資格・未決定を保持 |
| C K0/K1 | `83d219cc68c25540507a5539ebd2da94c29d06b3` | K0 PASSとK1の4構造価格・数値区間を受入。compiled価格へ供給しない |
| D残差interface | `af6f27d7a6467bb355686421acd067544edcbbd2` | module/testをbyte一致で取込み、既存combine/b_price_recordへ接続。実残差なし |
| D Full PF coverage | 既存B170047の受入済みe7dca5f7成果 | EXACT73/CONDITIONAL0/欠測63を保全。旧56/17/63もそのまま |

Canonical configurationは`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`、正本B030f3f548。Linear H4/1Å/STO-3G・保存tol-only DF10/sector(2,2)・元H/state/snapshot、S2/L_D1/K2/ΔPF=.1/h=.1/M12・元exactκ、minkowski/finite_B_diagnostic、prior/decoder/law/compiler/control/full-wrapper/P0を保持した。

## A：全13roundの物理値と経験的nominal

各軸の保存nominalは **[64,60,56,52,47,43,38,34,30,25,21,17,12]**。Bで式を再評価した値ではない。新しい`validate_completion_packet`は、旧S0/m1 validatorの意味を変更せず、S1/S2の新scopeを検査する。

- 新scopeは[2,3,4,5,7,8,9,10,11,12]、保存mean再利用はm3/m4、新meanは他8round。科学SOURCEはc76ffe5dbb2a8dd1a05f1a85038f2d48b959f73e。
- 元m0/m1/m6のpacket行全体・nominal64/60/38・diagnostic_sourceを前packet3e035c0dと照合。新scopeへ再帰属させず、旧validatorに偽装しない。
- 全canonical identity/law/physical candidate/profile/request ID、元numerical SOURCE、physical packetと原resultのselector先を完全照合。Aのmodel recordとraw/corrected/logB/bias/診断をcopyとして関連付ける。
- m3/m4の観測normは欠測のまま、`SOURCE_GATE_LIMIT_NOT_OBSERVED`を保持する。SOURCE gateの閾値や0を観測値として埋めない。u/margin・reference pair・norm/closure等はAの経験的診断を転記し、数値gateを新たに再評価しない。
- 十分N_X/Yは全26軸null、global bound=null、status=UNQUALIFIED。nominal全coverageは十分Nの資格ではない。

[A_ACCEPTANCE](metadata01/A_ACCEPTANCE.json)に各roundのsource/physical IDを保存した。追加公開A bba150834のPHYSICAL_INPUTSは元169d3316からの転記であり、Bが新しい物理計算pipelineを作る代わりに利用した。targetのsolver energy/residual/ground_state_certified=falseも元carrierへ照合し、truth proxyとして新規採択していない。

## C：K0 PASS・K1の4値

元quantityは`FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT`、expectationは`IDEAL_FINITE_K_SOURCE_LAW`。科学SOURCE19b95d8b3c88063d2dbae4e5bb47455b52b00903、preparation hash・全canonical identity/law/q/R/profileを照合した。Bの既存構造envelope `MEAN_STRUCTURAL_RZ_COUNT_NOT_COMPILED_C`へsource付きで接続し、元quantityも残す。

| round / axis | E[S]（保存値の転記） | numerical enclosure（保存値） |
|---|---:|---|
| m0/X・Y | 1743.6682557187337 | [1743.6682557186837, 1743.6682557187817] |
| m6/X・Y | 1764659.275564426 | [1764659.275537014, 1764659.2755917928] |

K0のlocal/terminal parity、law metadata parity、state representation、probability contract、kernel hashを固定参照として保持する。数値区間はCのNUMERICAL_CONTRACTの前提付きであり、Bが再計算・証明した区間ではない。X/Yが同値という保存結果をcompiled Cの同値性へ移さない。

[C_STRUCTURAL_ACCEPTANCE](metadata01/C_STRUCTURAL_ACCEPTANCE.json)は`compiled_price_records=[]`、`mean_compiled_C=null`、`qualified_for_compiled_price=false`、G=null。実sampler距離・compiler residual・未知systematic/covariance/CIのnullを保持。kernelを他11roundへ拡張しない。

## D：残差recordと既存価格系への接続

既存D `compiled_residual.py`の`paired_residuals/cohort_summary/combine/b_price_record`を再利用する。元module/testはD固定commitとbyte一致。実C−Sの値・平均・分散・fitは計算していない。

新`rpe_s3_intake.residual_price`は1round/axisの構造envelopeと1種類の残差だけを受ける。canonical identity/law/physical candidate/profileはDの既存検査を使い、有限cohortの値あり残差には同一path/event digest・軸・compiler・SOURCE・normal stratumの参照を要求する。forced rare・別roleのpool、残差の重ね掛け、Sへraw compiled clusterを足す二重計上を拒否する。Dの同path C/S差分・exact人工加算を使って検証した。

ideal E[S]と実sampler E[C−S]のlawが一致しない場合、Dの診断理由を保持し、B価格recordの値はnullにする。一致して人工のsumが作れても常にUNQUALIFIED_MODELであり、B.assessは資格を拒否する。現在は実残差自体がないので、残差mode・n_pathsを架空に選ばず欠測recordだけを供給する。

[D_RESIDUAL_INTERFACE](metadata01/D_RESIDUAL_INTERFACE.json)：m0/m6×X/Yのcompiled Cは全てnull。保存C1 MODEL false、finite cohortと全law、校正/holdout/rare、law距離/長q/共有不確かさを昇格させていない。元C1/S0aのSとcontrolled loweringの差はDの検証課題のまま、価格定義をBで変更しない。

## S3：入力充足と科学的readinessの分離

既存`rpe_decoder_statistics.mean_from_a/prepare`をそのまま使用し、全13raw/corrected/logB・局所bias・allowance/sourceと全26nominalを接続した。[STATISTICS_INPUTS](metadata01/STATISTICS_INPUTS.json)は保存値copyとmetadataだけで、確率生成・shot simulation・decoder実行はしていない。`input_complete=true`、`physical_qualification_complete=false`、`scientific_execution_ready=false`、`S3_executed=false`。

[S3_READINESS](metadata01/S3_READINESS.json)と[DECISIONS](S3_DECISIONS.md)で次の10項目を区別する。raw meanとnominalが揃ったことを根拠にlossや精度を採択しない。

1. raw complex meanとSOURCE：13round揃う。経験的局所診断のみ。
2. nominal：26軸揃う。十分Nは未取得。
3. count caller：既存exact整数和→binary64 ratioは利用可能。科学検証callerとして未採択。
4. 独立binomial：fresh ideal-law trajectory/Bornをshotごと、X/Y/round独立とする条件付きmodel。実sampler/相関は未資格。
5. zero/tie loss：未決定。試行の除外や便宜energy代入はしない。
6. state/target：保存solver proxyの証拠はあるが厳密ground・新truth採択は未資格。
7. PF/finite-K/state/law系統誤差：signal空間の局所証拠のみ。最終energy L2への変換は未取得。
8. Minkowski：Aの追加公開契約を参照できる。比較推定量/coupling/target/未解決lossと各L2上界は未決定。
9. MSE precision/confidence：未選定。
10. simulation反復数：loss support/precision/confidenceから決める条件だけを保持。H4でのnの新評価はしない。

既存plannerの[EVIDENCE_PLAN](metadata01/EVIDENCE_PLAN.json)には全13未資格Nと欠測価格を渡し、G/winner=null・4方式未完了を確認した。Full PFの[保存coverage](metadata01/FULL_PF_SAVED_COVERAGE.json)は136候補の73/0/63と旧56/17/63を無変更で参照。136候補の正式menu化・実G集計はしない。

## 検証・保全・次の担当

新13＋D元13＋decoder/statistics17＋既存V2 106＝**149技術test PASS、failure/error=0**。結果とlogはTEST_INTAKE/TEST_STATISTICS/TEST_V2_REGRESSIONを参照。人工のC−S arithmeticだけを実行し、実値はcopy/metadata比較だけ。作者自身の確認であり、独立reviewや科学的保証ではない。

新科学mean/N/RMSE再解析、S3 Monte Carlo、shot最適化/十分N認定、global bound新評価、sampling/build/compile、実C−S統計/fit・実G・8次新接続/検証は0。既存controller/statistics/decoder/A物理核/C providerを再実装・変更していない。新scope専用validatorと小adapter、metadata CLIだけを追加した。

次は**ユーザー**がS3のtarget/loss/joint law/caller/precision/有限scopeを判断・明示許可し、**Codex B**が許可範囲だけ統計実行する。**A**は物理誤差契約・必要な追加物理証拠、**C/D**は他round構造/残差・compiled平均の資格を担当する。監査R1〜R6の別改善は今回実行していない。commit/push・remote SHA/tree/主要artifact照合後、**mandatory STOP**。
