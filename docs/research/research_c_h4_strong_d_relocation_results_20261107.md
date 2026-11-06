# 研究C：保存CDF8/XDF8のLD2再配置強D・固定2候補比較の結果

2026-11-07 JST。**2候補ともPF/RPE・native correctness・数値合成誤差laneを通過し、登録したideal laneの点値比較は`E2_POINT_BENEFIT_REMAINS`となった。** 再配置CDF8＋Dは108,561,283,488 RZ、再配置XDF8＋Dは101,780,388,260 RZで、保存ENTRY E2＋PR/r16の75,621,395,471 RZより高い。これは固定2候補・固定LD2政策・条件付き資源モデルの点値結果であり、母平均の統計的優位、actual CA、全DF/PFでの最良性を認定しない。

科学runnerは一回、286/286 logical calls COMPLETE、FAIL/inflight/retry/resumeは0。科学計算は終了し、rootと両候補の`STOP.failed=false`、`mandatory_stop=true`を保存した。新fit・分解・候補探索・E2再計算・H2/H6・重要科学レビューは0。

[比較原本](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/science/run01/comparison.json) ／ [公開要約](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/publication/summary.json) ／ [root STOP](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/science/run01/STOP.json) ／ [保存物監査](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/execution/result_audit.json)。

## 固定範囲と実行の来歴

ユーザーが[固定commit e6652ab841aae5ae700286bf1ab249ad4fb85461の契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6652ab841aae5ae700286bf1ab249ad4fb85461/docs/research/research_c_h4_strong_d_relocation_contract_unapproved_20261107.json)の科学条件と一回の独立実行を明示許可した範囲だけを実行した。契約の歴史的`UNAPPROVED`名・準備時metadataは書き換えず、その同じbytesをrunに保存した。承認証明書・grant・承認原文保存は作成していない。

| 項目 | 固定内容 |
|---|---|
| 物理系 | 直鎖H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、spatial4/spin8、sector36/system256/control512 |
| 候補 | `B2_CDF_relative`の保存number-CDF8 J/U、`B3_XDF_spin_relative`の保存signed λ/G元順prefix8 |
| 再配置 | 元順LD2、F2〜F7の線形Zだけを保存一体headへ移す。対応scalar shiftを一回加え、元frame I/ZZとF0/F1を保持 |
| D政策 | 全frame ordered symmetric S2、s1、h=.1、m0..12。再配置をLD8でやり直さない |
| RPE | CA=.00159 Ha、DF reserve1/8=.00019875 Ha、β_total=π/6、β_stat=π/12、候補ごとのfailure .01、axis .01/26 |
| Reference | 各候補の保存own-DF ground oracle。保存H・groundのbinding、norm/residual/lowest energy/aliasを検査。状態準備費用除外 |
| Native | 旧CDF legacy basis、旧XDF exact spin-lift、StrongBuilder relative Gaussian、P0位相保持。Qiskit1.3.0/opt1/seed17/rz,sx,x,cx |
| 会計 | fenced serial、各axisの`2^m×kernel＋head＋readout`をshots加重。RZはnative gate個数、depthはscheduled serial sum |
| 比較先 | [保存E2＋PR/r16](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/artifacts/research_c_h4_handoff/2026-11-06/science/run01/E2/final_r16/summary.json)：C0/ENTRY strength .03/64更新、partial S2/s1/h=.1/K0/元順LD2、selection後final64の条件付き価格proxy |

科学SOURCEは`e6652ab841aae5ae700286bf1ab249ad4fb85461`の[68-file manifest](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/technical/SOURCE.json)、SHA256 `8323d7c76847b24276b500a80545172fd92719b7a82f6a5b298ca601adec4f53`。契約SHA256は`b011325203f1b6e1f5a69f3a32ed058ff9c097f0bb45645135aea93924527bb6`。保存23入力と合わせた91ファイルのhashを実行前後で照合した。

元P-H4 snapshotはSHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`、CDF exportは`c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08`。保存κと元g補正、核scalar一回、signed λ、exact interleaved spin順、元因子・保存sector/groundを既存whitelistと数値gateで照合した。保存DF norm＋guardを再利用し、新DF fit/normは計算していない。PF exact-referenceに必要なsector36のspectral factorizationは契約どおり各候補一回行った。

独立branch/worktreeは`codex/research-c-h4-strong-d-relocation-execution-20261107`。固定CLIの`--execute`を一回起動した。[launcher](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/execution/launch_once.py)は固定HEAD/契約/preflightを確認し、専用process groupだけを監視する。失敗時の停止監視は発火せず、SOURCE・候補・契約を途中変更していない。

## 登録した点値比較と2 lane

| 構成 | Kernel RZ | Planned shots合計 | Ideal lane総RZ | 数値合成lane総RZ |
|---|---:|---:|---:|---:|
| 保存ENTRY E2＋PR/r16 | 平均5,512.3125（保存final64） | 15,118（保存） | **75,621,395,471**（保存proxy） | 全law/実装law証拠不足で未資格 |
| 再配置CDF8＋強D | 11,039（固定回路） | 13,824 | **108,561,283,488** | **108,561,283,488** |
| 再配置XDF8＋強D | 10,774（固定回路） | 13,688 | **101,780,388,260** | **101,780,388,260** |

[CDF result](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/science/run01/CDF/evaluation/result.json) ／ [XDF result](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/science/run01/XDF/evaluation/result.json)。両候補とも全13 roundで`CONDITIONAL_ROUND_PASS`、energyは`CONDITIONAL_CA_PASS`、数値laneは`NUMERICAL_PILOT_CA_PASS`だった。Dの費用は固定回路と登録shots処方の確定的な会計値であり、64回のclassical価格samplingはしていない。

ideal laneはexact-native synthesisを仮定し、Dのlawはdeterministic。数値laneはraw/compiled/referenceのfull512 Frobenius、norm-growth、shell誤差を13 roundへ伝播した。**合成誤差は非零だが、今回の整数shotsと総費用は両laneで一致した。** 誤差がゼロだった、両laneが同じ保証を持つ、PR側も数値laneを取得した、という意味ではない。

| 保存した数値pilot量 | 再配置CDF8＋D | 再配置XDF8＋D |
|---|---:|---:|
| 再利用DF norm＋guard［Ha］ | `7.666215916301914e-5` | `9.533901455091894e-5` |
| compiled/reference max-entry | `7.338309640049294e-13` | `3.1763000819741534e-12` |
| compiled/reference Frobenius | `8.580271713406127e-12` | `5.893911856999753e-11` |
| m12 PF operator error | `.11179983620206387` | `.08503278795769181` |
| m12 synthesis complex-signal pilot | `9.940450746861306e-8` | `6.828237447575093e-7` |
| m12 PF＋synthesis complex error | `.11179993560657134` | `.08503347078143657` |
| m12 deterministic phase bias pilot | `.11203415753268406` | `.08513628078581584` |
| m12 shots/axis（ideal/数値） | `648 / 648` | `611 / 611` |
| Energy error upper-bound pilot［Ha］ | `.0013549794824010532` | `.001373656337788953` |

PF誤差・合成誤差はそれぞれ一回だけraw signalへ接続し、DF［Ha］は最終energy式へ一回だけ加えた。energy上界pilotは実測energy誤差ではない。候補ごとのfailure .01は、E2を含む全比較の同時成功保証ではない。

6指標のD総費用も保存した。主判定は総RZだけであり、今回から別の副指標へ切り替えていない。

| Ideal/数値共通のD総費用 | 再配置CDF8 | 再配置XDF8 |
|---|---:|---:|
| RZ depth | 54,059,393,376 | 51,683,753,316 |
| CX count | 67,581,556,992 | 64,049,629,440 |
| CX depth | 60,304,148,352 | 57,569,091,712 |
| Total depth | 117,235,209,312 | 111,718,513,420 |
| Circuit size | 206,707,998,240 | 192,555,191,756 |

## 再配置前の強Dとの接続

[旧固定強D結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/docs/research/research_c_c1_h4_strong_baseline_ca_resources_execution_20261026.md)は、CDF kernel11,062/総shots13,606/総RZ102,056,444,569、XDF kernel10,885/総shots13,600/総RZ100,267,421,030であった。今回の固定LD2再配置ではkernel RZは減った一方、候補固有PF誤差を用いたRPE shotsが増え、両Dの総RZは旧保存値より高くなった。

Hamiltonian保存からPF分割・shots・総費用の保存を推論できないという前回の注意を、今回の2候補で具体的に確認した。これは全frame Z移送・別順序・別s/PF次数まで探索した結果ではなく、Dにとっての最適再配置は評価していない。旧値はその元の政策の結果として保全し、新しい再配置済み値に置き換えない。

## Correctness・call・資源・保存物

両候補とも再配置前後のscalar込みH、保存candidate/ground/sector、frame順、signed係数、full/sector S2、relative Gaussian方向が既存gateを通過した。全512入力・全512出力に対してraw/compiled/reference、実raw/compiled inverse、dagger、return、control0/1相対位相、shell/X-Yを検査した。component1e-10、signal2e-10、normalization1e-12、rtol0、phase fitなしを保持した。

[CDF full512 segment](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/science/run01/CDF/evaluation/segment.json) ／ [XDF full512 segment](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/science/run01/XDF/evaluation/segment.json) ／ [CDF actual inverses](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/science/run01/CDF/evaluation/adjoint.json) ／ [XDF actual inverses](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/science/run01/XDF/evaluation/adjoint.json)。native/correctness数値処理を結果公開のために再実行していない。

| 実行規模 | 実績 |
|---|---:|
| 候補 | 2 |
| PF round / ideal round accuracy / 数値round synthesis / 数値round accuracy | 各26 |
| Native kernel build / compile | 各2 |
| Actual inverse build | 4 |
| Kernel full512 decode / shell full512 decode | 8 / 12 |
| Reference用sector spectral factorization | 2 |
| Scientific logical calls | 各143、計286/286 COMPLETE |
| FAIL / inflight / retry / resume / 追加候補 | 全て0 |

[起動・資源](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/execution/start.json)と[完了記録](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/execution/completion.json)：**01:36:01〜01:36:26 JST、runner wall25.53秒**、子process合計CPU50.62秒。候補workerのwallはCDF24.23秒/XDF24.08秒、peak RSSは579,896/582,688 KiB。これは各workerのpeakであり、同時合計RSSを測った値ではない。

候補間2 private process、候補内最大4独立decodeを用いた。affinity64 CPU、利用可能memory約520 GBをplannerが観測し、BLASは各候補process内1 thread、shared package/OS/schedulerは変更しなかった。quotaは参照不能で、無制限とは主張しない。過去の人工138 PASSは準備証拠として再利用し、科学結果に合わせた新試験・閾値調整は行っていない。

rawは**79ファイル、41,854,635 bytes**を元のfresh outputに保全した。JSON/JSONL61件は直接公開し、NPZ10＋QPY8の18 binaryは[raw binary bundle](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/publication/raw_binary_bundle.tar.gz)（9,855,974 bytes）として公開する。[全raw file hash一覧](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/publication/raw_files.json)とbundle内のbytesを照合した。bundle SHA256は`543719e279fa5ad2fc782f6ae98ff79ef47de8a40f7542ef3c4d5a8bc7edb2dc`。圧縮・byte監査であり、NPZ/QPY数値の再decodeや新科学再解析ではない。

## 成立する結論・未資格事項・MANDATORY STOP

**共通の固定LD2再配置を保存CDF8/XDF8強Dにも施した後、保存E2＋PR/r16の条件付き総RZ proxyの点値利益は両対照に対して残った。** 両Dが物理・PF/RPE・native・数値laneを通過した後に、登録規則どおり両方のideal総RZがE2保存点より高いと判定した結果である。

E2のfinal64を新標本として数えず、E2自身のPR/D区間を今回のCDF/XDFへ付け替えていない。新Welch区間、比区間、p値、同時優位性、母平均の保証は0。P/r16の旧`UNRESOLVED_DISTRIBUTION_DIAGNOSTIC`も変更していない。

以下は未資格のまま保つ：PRの全law synthesis/implemented-law error/ideal-law価格移送、厳密rounding enclosure、actual chemical accuracy、実機/FT資源優位、状態準備込み費用、全候補・全PFの最適性、PRだけやENTRY目的だけの因果効果、独立新規性・研究Cの正式GO/NO-GO・論文化判断。今回のD level B pilotを、それらの資格へ格上げしない。

旧SOURCE/raw/結果/契約/STOP、旧H2/H6・1.15 Å、grant/ledger/marker、他worktree/process/cacheは不変更。SOURCE68と入力23の前後照合PASS、科学runner再起動0。実行済みrun01は消費済みであり、retry/resumeしない。

結果と来歴をcommit/pushし、remote SHA/tree/公開本文/SOURCE/raw bundleの一致を確認した後に**MANDATORY STOP**。次の担当は**ユーザー／GPT**が今回の限定結果をどう研究上位置付けるか判断すること。Codexの自動的な次作業なし。追加実験・重要科学レビューへ進まない。
