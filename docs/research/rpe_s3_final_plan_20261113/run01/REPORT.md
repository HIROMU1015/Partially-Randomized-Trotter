# S3 run01：固定H4 PR・全13roundの条件付き統計評価

2026-11-13 JST。採択済み推奨セットで**固定43,819反復を1回だけ完了**した。全trialがenergyを返し、経験的な解決条件付きRMSEは**0.0006847202619833286 Ha**。未解決0件、97.5%二側Clopper–Pearson区間は[0, 9.999788982664716×10^−5]。これは供給mean/solver proxy/ideal IID modelを条件とする観測で、十分N・全RPEの科学的RMSE保証ではない。

[採択条件・実行SOURCE](EXECUTION.json)／[主評価](PRIMARY.json)／[loss感度評価](LOSS_ENVELOPE.json)／[完了状態](RESULT.json)／[全trial圧縮原本](SIMULATION.json.gz)／[原本サイズ・SHA256](ARTIFACTS.json)／[保存照合](VERIFICATION.txt)／[HANDOFF](HANDOFF.md)。科学条件と有限範囲は[公開計画](../SCIENTIFIC_PLAN.md)を採用し、過去の計画・入力・STOPを改稿していない。

## 固定条件・実行SOURCE

Linear H4/1.00 Å/STO-3G、保存tol-only 1e−8 DF10/weighted-Hermitian completion、sector(2,2)、S2/L_D1/ΔPF=.1/K2/h=.1/M12/元exactκ。元H/state/snapshot/fragment順、prior/decoder/law/compiler/profile/full-wrapper/P0、εE=.00159360010199040 Ha、εqpe=εtrot=εE/2、minkowski/finite_B_diagnosticを維持した。

canonical ID=`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。全13roundのX/Y各軸nominalは64,60,56,52,47,43,38,34,30,25,21,17,12。N/M/κや方式の探索は行っていない。

| 役割 | SOURCE |
|---|---|
| 数値API・高速decoder | B `ab12b69120323379b0aae572e7f62692a5819125`、src無変更 |
| 固定caller | `7486b91582c01e0313257c03ca9f58e9c3326847`、`scripts/run_h4_s3_statistics.py` |
| 採択した計画 | `ee1d456dcd1cb47ae1f404b01a62447ede22a8c1` |
| 保存prepared/canonical | B `bf515a976f1987ad6a19c61d14689b27b0894542` |
| A物理契約・全13round | `bba150834ca30bdb186fbfd58c55a8d51177b855`／`169d3316dd008eb98dd9be56fe4ecefc5c1a7d60`。旧S0/m1と新10roundの来歴を保持 |

targetは保存solver energy −2.1663874486347634 Haの**条件付きproxy**。元SOURCEは89c91c692の`records[5].diagnostics`。ground_state_certified=false。energy shiftやprior/branch選択へのproxy使用はない。元preparedの`energy_ha=null`・`truth_proxy_adopted=false`は原記録として保全し、新しいEXECUTIONに採択条件を記録した。

## 実行した統計model

保存raw complex meanを使うideal finite-K/fresh IID shotの周辺binomial model。shot/X/Y/round/反復独立を仮定し、PCG64 seed20261113、binary64確率、exact整数和/Fraction→binary64のcount caller、R3 scalar互換batch hookを使用した。元scalar/vectorized decoder・count変換・loss/CIは変更していない。

corrected meanを確率へ代入せず、Bを再乗除せず、normalization負担を二重計上していない。実RTE trajectory sampling、実sampler資格検証、数値allowanceの新伝播、物理mean/target/reference再計算、compiled価格/Gは0。副評価は**同じtrial群**へ既存summarizeをもう一度適用したもので、新simulationではない。

## 結果とその限界

| 指標 | 保存結果 |
|---|---:|
| 固定反復数／解決数 | 43,819／43,819 |
| zero未解決／branch-tie未解決 | 0／0 |
| prior外の解決energy | 0 |
| 経験的な解決条件付きRMSE | 0.0006847202619833286 Ha |
| 解決分の全trial MSE寄与 | 4.6884183717051806×10^−7 Ha² |
| 未解決率・観測値 | 0 |
| 未解決率97.5% CP区間 | [0, 9.999788982664716×10^−5] |
| 主評価population RMSE CI | null |
| 条件付きRMSE点推定のCI | null（現APIにない） |

DECODEDは正しいbranchの認定ではない。prior外0件もalias不在の証明ではなく、branch_error_probability=null、prior_alias_qualification=falseを維持。未解決0件でも真の未解決確率を0としない。主lossはUNDEFINEDのまま。元APIの`all_observed_trials_RMSE`と零観測時の点区間は標本診断であり、特定の全事象energy estimatorのpopulation RMSEには昇格しない。

副評価は未解決squared loss∈[0,4096] Ha²、energy保守域[−32,32] Haという**採択仮定の感度envelope**。全到達出力supportの数値認証を新たに行ったわけではない。実行時のfinite/support検査は通過したが、観測域内だったことだけでpopulation supportを証明しない。

| 副評価・MC不確かさ | 保存結果 |
|---|---|
| population MSE envelope・97.5% | [0, 28.963512732533932] Ha² |
| 同RMSEスケール | [0, 5.381775983124338] Ha |
| 方法 | 既存Hoeffding・有界loss端点の上下片側 |
| 採択済み精度設計 | η=.01U=40.96 Ha²、粗いpilot |
| 同時confidence | CP familyとMSE端点familyの各97.5%、union boundで名目≥95% |

この広いenvelopeでは目標精度の達成を認定できない。ηはMSE端点へのMC余裕幅で、条件付きRMSEの誤差棒ではない。副評価も特定fallback energyのRMSEや物理global boundを定義していない。CIは供給mean/proxy・IID・support/loss契約を条件とするbinary64の名目統計で、物理系統誤差を含まない。

## 保全・実行検証

callerのsyntax/CLIと入力metadataを実行前に確認し、数値APIは既存60技術testの固定SOURCEを再利用。実S3のscalar比較・別seed・追加反復は行っていない。全43,819trialの全13round位相・branch・距離・energy・SOURCEをSIMULATIONへそのまま保存した。圧縮原本21,862,603 bytesのSHA256は`eec21518e51b866e0e253c0ec2571a95ab0058553ae6ce191b8e939aa1714195`。

全5原本のhash/size、gzip CRC/JSON全読戻し、全trial件数/status/13round、SOURCE/入力無変更、null資格を確認した。再simulationやlossの再計算による別評価は行っていない。runtimeはsimulation約2.130秒、保存約15.204秒、主/副summary約2.463秒。process peak RSSは618.766 MiB。共有環境のCPU/RAM/thread上限を変更していない。

Aの経験的allowance、signal空間のPF/finite-K bias、m3/m4観測norm欠測、state/target/actual law差のunknownを0にしていない。L2/energy上界への変換・Minkowski合成を行わず、全13round十分N/global RMSE bound・適格compiled C/G・科学的RMSE PASSはnull。A/C/Dのbranch/worktree/process/rawには介入していない。

## 次の担当・STOP

**ユーザー**：次に進む場合は、より精密なMSE評価・完全な未解決loss契約・物理誤差資格のうち、次の有限研究範囲を選び、必要な科学実行を別途許可する。**Codex B**は選ばれた範囲だけ計画/実行する。Aの物理L2/target/law、C/Dの価格/残差はそれぞれの既存担当・scopeを維持する。

今回は固定1実験を完了。全結果をcommit/pushしremote SHA/tree/主要成果物を照合後**mandatory STOP**。結果を理由とする反復追加、seed変更、shot最適化、実C/G、4方式最適化へ自動進行しない。
