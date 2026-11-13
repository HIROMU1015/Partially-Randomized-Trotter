# H4 DF10 PR–RPE：S1＋S2残10round科学実行結果

2026-11-13 JST。**固定SOURCE c76ffe5dbb2a8dd1a05f1a85038f2d48b959f73eのS1＋S2を一回実行し、残10roundの条件付き局所診断・経験的nominal Nを取得した。** 全13roundのnominal Nは各軸 **64,60,56,52,47,43,38,34,30,25,21,17,12**。旧m0/m1/m6の64/60/38は再計算せず保全した。全十分N・global RMSE boundはnull、モデル資格はUNQUALIFIED。

[科学result](run01/result.json)／[B V2 N packet](run01/N_HANDOFF.json)／[13round CSV](ROUND_SUMMARY.csv)／[SOURCE・入力保全](SOURCE_INDEX.json)／[scope/packet検査](CHECKS.json)／[事前preflight](PREFLIGHT.json)／[実行出力](EXECUTION.txt)／[人工テスト](TESTS.txt)／[HANDOFF](HANDOFF.md)。各roundの独立mean/refは`run01/round_XX_mean.json`、消費した固定計画は[run01/PLAN](run01/PLAN.json)。

## 固定条件とSOURCE

Linear H4、隣接1.00Å、STO-3G、保存tol-only1e−8 DF10/completion/fragment順・scalar、sector(2,2)/dimension36の保存数値state。targetは保存DF-Hの同sector ground energyだが、厳密ground同定は未資格。S2/L_D1/K2/ΔPF=.1/h=.1/M12/P0、無energy shift、S0 exact有理数κ、εE=.00159360010199040 Ha、εqpe=εtrot=.00079680005099520 Ha、minkowski/finite_B_diagnosticを維持した。元H/state/snapshot・law・prior・decoder・compiler/control/profileは不変。

Canonical B V2 IDは **`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`**。Bの固定参照は3216d27a69ac8532d2a22e78f5f94491d202d238。既存正本とfull identity/lawを使い、B controller/plannerを変更していない。

| SOURCE | 今回の利用 |
|---|---|
| **c76ffe5dbb2a8dd1a05f1a85038f2d48b959f73e** | 今回の科学実行SOURCE。`h4_pr_rmse_completion.py` / `pr_rpe_power_mean.py` / runnerをbyte無変更で使用 |
| 3e035c0d61386010f59ad7ea792bc7a5cefd18ac | m0/m1/m6の受入済みpacketを保全。元S0結果7da8a35f…・m1科学ac873e7f…をcopy |
| 155529e20a8fac39e00786c40b57830b71203c4f | m3/m4の保存raw/endpoint/binding/compiler provenance |
| e417ffc9f52b13c18f5493193ce510d858664259 | 既存spectral polynomial power、元norm/closure gate、順序・scalarを再利用 |
| 89c91c692e1d35de94d2e7afbe890aabe6d80b7e | 欠測8候補の同H/state/Tの保存reference 8 pair。元科学90dcc3ed2… |
| 7c3e1e5db0c55dd8249e9030a4c7e9954496fa7e / 71ba49bb28b295aa28e51101bc3b1bf07d500767 | 元S0/R-aware scalar式とreference checker。旧Hoeffding N wrapperは呼ばない |

実行前に固定PLANとの完全一致、科学コードと保存snapshot/receipt/tail contextのhashを確認し、実行後にも入力hash一致を確認した。保存H/stateの復元・sector行列構成とhead2/tail1の3 spectraは行ったが、SCF・DF fit・ground-state生成/full-H ground eigensolveは行っていない。Hamiltonian/state bindingの全hashはSOURCE_INDEXと元PLAN参照。

## 一回の承認scopeと実行会計

```bash
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B \
  scripts/run_h4_pr_rmse_completion.py --execute --stage S1_S2 \
  --source c76ffe5dbb2a8dd1a05f1a85038f2d48b959f73e \
  --out docs/research/h4_pr_s1_s2_rmse_results_20261113/run01
```

完了した実行の記録であり、再実行の指示ではない。ユーザーのチャット承認により実行し、新grant・承認証明書を要求していない。SOURCE不一致・数値gate失敗・候補変更・閾値緩和・科学retryは0。

| 処理 | 実施 |
|---|---:|
| S1保存mean/対照の再利用と局所診断 | m3/m4の2候補 |
| S1新full-H reference | T=.8/1.6、各expm_multiply/expm、計4 action |
| S2新spectral-power finite mean / exact-tail対照 | 8候補 / 内部8対照 |
| S2保存reference再利用 | 8時点・8 pair、新reference0 |
| m2逐次actionによる直接照合 | 1候補＋内部1対照 |
| K2 event normalizer | 主8候補＋m2直接経路の計9評価（固定一pieceのコード経路） |
| 新nominal X/Y | 10候補分を一回batch評価 |
| m0/m1/m6再解析 / S3 simulation / trajectory sampling / compile / C / G | **すべて0** |

event normalizer数は固定コード経路からの会計であり、過剰なlibrary内call auditは追加していない。S2のreference recordにある過去の`reference_state_actions=2`は元取得履歴で、今回新規の16 actionではない。

## S1：保存meanと観測norm欠測の保全

m3 physical `045843b2429bece6f29b`（T=.8/q8/R16）、m4 `3df0cfd3356c358aeda8`（T1.6/q16/R32）のmean/対照/logBを保存SOURCEから再利用した。旧endpoint ε=.0005は過去allowanceのselectorで、現在のenergy/RMSE条件へ採択していない。

**observed exact-tail norm errorは両候補ともnull/欠測のまま。** 承認された元SOURCEの返却前norm gate≤1e−9を、条件付きempirical診断の根拠に使った。0や閾値を観測値として補完せず、`norm_evidence.kind=SOURCE_GATE_LIMIT_NOT_OBSERVED`を保持。局所gate成立と厳密保証は別である。

新reference2経路の元値を保存し、保存meanのreference中心を維持して3者を照合した。最大差はm3 **2.424922536e−15**、m4 **3.036921684e−15**で、元1e−10 gate内。raw/correctedの元独立vector closureはそれぞれ2.329413175e−15、2.370316031e−15。SOURCE norm証拠を新たな実測normやglobal RMSE資格へ変換していない。

## S2：高速meanとm2直接照合

元spectral polynomial powerを再利用し、head-forward→tail→head-reverse、K2 paired law、R allocation、signed time、total Tでのscalar一回会計を保持した。corrected/rawは別columnで独立に伝播し、完了後にraw=corrected/Bと定義していない。exact-tail対照を同時取得した。

m12のlogical event位置数33,423,360に対し、実行は4096 outerのspectral block処理を用いた。今回、実H4の全対象をこの経路で完了できた。逐次全長時間runとの速度比を測定したわけではない。

| m2直接照合 | 差 |
|---|---:|
| corrected signal | 2.2957737090157393e−14 |
| raw signal | 2.0654027705266258e−14 |
| exact-tail signal | 2.3024483800080385e−14 |
| 両経路の固定recipe allowance比較値 | 2.8284275688354e−9 |
| log B | 完全一致 |

この限定照合は成立した。全長時間のrigorous同値性の証明ではない。S2の観測exact-tail norm誤差は各候補で取得され、最大はm12の2.4378943308533962e−11、元1e−9 gate内。各roundのvector/scalar closure、有限値、bias分解、normalizer感度、R-aware allowance、Decimal ceilも成立した。

m2直接経路のraw診断にある`allowance.maximum_norm_domain`は共通adapterの既定文字列`ARCHIVED_SPECTRAL_BLOCK_BOUNDARIES`が残っている。**直接経路の実際のmaximumはD2 `evolve.checked()`の逐次half-step/event観測**に由来する。数値やraw artifactは変更せず、このmetadata上の注意をCHECKS/SOURCE_INDEXにも記録した。主高速経路の値はblock境界観測であり、厳密算術の包絡の説明と浮動小数点のempirical検査を区別している。

## 新10roundの局所診断

以下はresultの表示丸め。全signal/reference/bias分解・u/margin・SOURCEはresultと個別mean fileに保存した。B=1への救済や閾値変更はしていない。

| m | log B | corrected絶対bias | referenceに対する位相 [rad] | 条件付きraw半径下界 | βdet [rad] | nominal X/Y |
|---:|---:|---:|---:|---:|---:|---|
| 2 | .04095130345 | 5.361772371e−5 | −5.361772277e−5 | .959824403744 | 5.361913458e−5 | 56/56 |
| 3 | .04097478214 | 1.072220711e−4 | −1.072220660e−4 | .959750416526 | 1.072232238e−4 | 52/52 |
| 4 | .04098065762 | 2.143749050e−4 | −2.143748936e−4 | .959641927022 | 2.143761893e−4 | 47/47 |
| 5 | .04098212685 | 4.286479136e−4 | −4.286479267e−4 | .959434847759 | 4.286493908e−4 | 43/43 |
| 7 | .04114330179 | .001714595120 | −.001714595407 | .958046108633 | .001714599511 | 34/34 |
| 8 | .04114332502 | .003429214703 | −.003429216534 | .956400570435 | .003429235379 | 30/30 |
| 9 | .04114333087 | .006858365790 | −.006858379416 | .953109596109 | .006858476605 | 25/25 |
| 10 | .04114333231 | .013716690930 | −.013716799268 | .946527566240 | .013717336232 | 21/21 |
| 11 | .04114333123 | .027432808587 | −.027433669591 | .933363644749 | .027437174667 | 17/17 |
| 12 | .04114333130 | .054860439752 | −.054867324822 | .907039349647 | .054891246697 | 12/12 |

R-aware式のlogical event会計を保ち、旧u/margin・normalizer one-ULP感度を下回らないempirical allowanceを使った。例えばm12はu=1.9004903921776505e−6、logB margin=9.893361209945881e−7。長時間側でallowanceが増えるのは固定式の結果で、閾値を緩めて通したものではない。certified=false/rigorous_bound=falseを保持する。

## 全13round nominal coverageとB packet

| m | T | qPF | R_outer | nominal N_X=N_Y | 由来 |
|---:|---:|---:|---:|---:|---|
| 0 | .1 | 1 | 2 | **64** | 元S0を保全 |
| 1 | .2 | 2 | 4 | **60** | 元m1を保全 |
| 2 | .4 | 4 | 8 | 56 | 今回S2 |
| 3 | .8 | 8 | 16 | 52 | 今回S1・SOURCE norm条件付き |
| 4 | 1.6 | 16 | 32 | 47 | 今回S1・SOURCE norm条件付き |
| 5 | 3.2 | 32 | 64 | 43 | 今回S2 |
| 6 | 6.4 | 64 | 128 | **38** | 元S0を保全 |
| 7 | 12.8 | 128 | 255 | 34 | 今回S2 |
| 8 | 25.6 | 256 | 510 | 30 | 今回S2 |
| 9 | 51.2 | 512 | 1020 | 25 | 今回S2 |
| 10 | 102.4 | 1024 | 2040 | 21 | 今回S2 |
| 11 | 204.8 | 2048 | 4080 | 17 | 今回S2 |
| 12 | 409.6 | 4096 | 8160 | 12 | 今回S2 |

Nは固定finite_B経験的モデルの値で、十分N/actual minimum Nではない。局所phase/disk/半径をenergy RMSEやεtrot上界へ換算していない。保存E_stateはreference診断だけに使い、prior/branch選択には使っていない。actual implementation/law/ground誤差はnullのまま。

新packetの`diagnostic_scope`はnewly_evaluated_rounds=残10、saved_mean_rounds=[3,4]、new_mean_rounds=8候補、preserved_rounds=[0,1,6]。元m1 packetをprevious_packetで参照し、旧3roundのnominal/model status/diagnostic_SOURCEを保全した。全13identity/law/physical ID・入力・誤差policyは同canonical context。十分N_X/Yは全26軸null、global bound=null、model/global/finite-K/signed資格は未資格。Bの旧m1専用validatorへ偽装せず、新scopeとしてHANDOFFする。新価格recordやGはない。

## 残るS3条件とSTOP

局所証拠とnominal配分のcoverageは13/13になったが、**全RPE energy RMSEの評価は未実施**。全raw meanから有限shot X/Y分布、固定decoder、prior/alias/branch、未解決zero/tie trialのloss/出力、元DF-H target/保存state、finite-K/PFの系統biasとMinkowski成分、ideal/actual sampler、統計精度/信頼度を結ぶ必要がある。旧HP N/α/βや今回の局所βdetを十分Nの根拠にしない。

次の条件案は[既存SCIENCE_PLANのS3](../h4_pr_remaining_rmse_preparation_20261113/SCIENCE_PLAN.md)を保持する。未解決trialを除外したRMSEを全試行RMSEとは呼ばず、必要な検証信頼度・MSE精度・有限反復数と対象lossを決めてから別scopeとして承認する。今回S3/価格/G/Full PF/8次PFへ進まない。

実行前の既存13人工・metadataテストPASS、出力のscope/全identity/null/provenance/input前後hashを照合した。科学code・旧S0/m1/準備・raw/結果/過去STOPは変更しない。作者自身の技術/数値確認で、独立科学レビューではない。

**次の担当**：**Codex B**は新10round scope packetを未資格nominalとして取り込む。**ユーザー**はS3の未解決trial/target・誤差契約・統計精度/信頼度と有限実行を判断・承認する。**Codex A**は別承認後にだけS3へ進む。今回commit/push・remote SHA/tree/主要成果物照合後**mandatory STOP**。
