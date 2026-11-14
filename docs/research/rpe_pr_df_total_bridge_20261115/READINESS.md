# 5方式・14物理configurationの精度側readiness

2026-11-15 JST。[READINESS.json](READINESS.json)がSOURCE・全ID・価格cellの正本。以下の「接続」はmetadataを指し、科学MSEのPASSではない。今回、新しいMSE/DP/N/Gは評価していない。

| 方式・物理候補（元ID先頭） | 全13raw | total decoder接続 | total MSE / shot vector | 価格候補との対応 |
|---|---|---|---|---|
| PR-DF LD1 `a4dda8f7` | 保存済み | 今回SOURCE付きview受入 | 未評価。旧partial DP/controlは別結果 | C52軸coverageと元physical/q/R一致。m0/m6有限cohortは正式SOURCE/consumer受入待ち |
| PR-DF LD2 `1fa33556` | 保存済み | 今回SOURCE付きview受入 | 未評価。旧partial DP/controlは別結果 | coverageのphysical/q/R一致。適格平均価格なし |
| PR-Pauli LD1 `d905d219` | 保存済み | 既存受入保全 | caf1417のopt-in診断1件完了、USER_SUPPLIED control | LD1 m0の旧canonicalに対する単一path X/Y sidecar context一致。全law E[C]ではない |
| PR-Pauli LD2 `d7dd4f52` | 保存済み | 既存受入保全 | 同じく1件完了、USER_SUPPLIED control | 確認したcoverageでは価格未同定 |
| Full PF S2/幅h `f2d58f32` | 正式packet未同定 | source-bound canonical view生成のみ、signal受入待ち | 未評価、vector未指定 | 現価格shortlistとは別schedule |
| Full PF S2/16h/7 `11cf8ca6` | packet未同定 | API対応、入力待ち | 未評価、vector未指定 | A followupとC planのm4〜m6 physical ID一致。新compile計画は実行前 |
| Full PF Y4/16h/3 `a799697f` | packet未同定 | 同左 | 同左 | 同じC価格shortlist。局所H_eff・二時点meanを全RPE資格へ昇格しない |
| Full PF M5/16h `70f7b317` | packet未同定 | 同左 | 同左 | M5 m4の保存X/Y受入はある。全round E[C]ではない |
| Truncated B0I p7/S2/幅h `e6754564` | 保存済み | 既存受入保全 | 未評価、vector未指定 | 下記3つの価格scheduleと異なる。全13信号を他scheduleへ付替えない |
| Truncated p7/S2/16h/7 `a6e55777` | packet未同定 | API対応、入力待ち | 未評価、vector未指定 | C planはこの物理候補。S2/幅hの全scheduleとは一致しない |
| Truncated p7/Y4/16h/3 `defc5e88` | packet未同定 | 同左 | 同左 | C planのY4候補。元full-H targetと切断/scalarを保持する必要 |
| Truncated p7/M5/16h `424eecbf` | packet未同定 | 同左 | 同左 | M5 m4の保存X/Yは受入済み。精度側の全13rawとは別の不足 |
| Full-H κ2/K2 `54ac3eb3` | 保存済み | 既存受入保全 | 未評価、vector未指定 | m3/q8/R71の再検証済み単一path sidecar context一致 |
| Full-H R448/K2 `cfa6aee3` | 保存済み | 今回独立configurationのmetadata確認 | 未評価、vector未指定 | κ2の価格・law・normalizerを流用しない。今回の固定参照範囲で対応価格未同定 |

「packet未同定」は、確認済みSOURCE内で全13rawの正式受入packetを特定できなかったという意味。H_eff・二時点mean・旧catalog・個別roundの保存値が存在する場合もあり、「全roundが過去に未実施」とは扱わない。新しいAのcoverageが公開されるまでは所在・受入可否が未確認のroundを保留し、保存記録を回収できた分は再計算候補から除外する。

## SOURCEと資格の区別

- **PR-DF signal**：A714247619に保存された元91349成果。configuration a9abf8de、数値fefd548e、raw carrier16210932。旧partial DPはe7ce2380。新total IDだけを作って旧MSEを移植しない。
- **Pauli signal/risk**：A714247619とB caf1417。保存controlは両軸`[64,60,56,52,47,43,38,34,30,25,21,17,12]`。原論文nominal・十分N・最適Nではない。今回、閾値filterや平方根・比率・rankingを追加していない。
- **Full PF / Truncated**：A714247619のH_eff followup、5fb2c32のT2、C85557306のCELL_PLAN_V2。固有値shiftは全RPE energy MSEではない。保持H・full-H target・scalar一回を区別する。
- **Full-H**：κ2はce6aacfa、R448は旧truncated_fullh packet。one-body込み全H lawであり、DF B3やPauli tailと混ぜない。
- **価格**：C85557306、Pauli00f74f86、Full-H a04470c0の固定記録をmetadata比較しただけ。構造E[S]、有限cohort、単一path、モデル予測、全law E[C]を区別し、適格期待価格を今回認定しない。

## 全方式に共通する未採択事項

Z1＋T1の主推定量採択、targetと全事象loss、model MSE閾値、保存ideal IIDと実sampler lawの対応、数値certificate、必要Nの保証水準はユーザー判断事項である。保存state/solver proxyを厳密groundへ昇格しない。raw直接riskは近似の影響を含み得るのでAのδEを重ねて加算しない。

全行で`sufficient_N/global_RMSE_bound/G_qualified/G/winner=null`。十分Nは0件。原論文型nominalとMSE資格付きNを分離する。全roundを結合したwhole-shot policyを守り、roundごとの独立最適化を前提にしない。

次の有限N案は[前計画](../rpe_required_n_cross_method_preparation_20261114/FINITE_N_PLAN.md)の未採択案を保持する。今回のbridgeによってPR-DFも将来の同じdecoder診断へ技術的に接続できるが、PR-DF用の新vector選択・評価許可は追加されない。価格に基づくcost-aware比較は適格N/Cが揃ってから、別の有限scopeで行う。
