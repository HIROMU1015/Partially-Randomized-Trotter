# A/C/D成果のV2接続・標準4次PF・明示Full-H RTE

2026-11-13 JST。技術統合・保存metadata照合・人工テストのみ。B branch `codex/r-variable-rpe-cost-pipeline-20261111`を再利用。科学的baseline、κ/RMSE契約、実Gや方式の優位性は採択していない。

## 完了した接続

| 対象 | 固定SOURCE・結果 |
|---|---|
| B正本 | `030f3f548a35d98a88e62e92a35f132e0243a293`の保存CONFIGURATION_V2・IDENTITY_MAP・C_REQUESTS |
| A公開packet | `75b5c89bddf8a1ffa4a8ac99b2c06d6e05fa012d`のV2_N_HANDOFF/V2_CONFIGURATIONを正式受入 |
| 元S0 | `7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75`のN_HANDOFFを元の科学証拠として参照。変更・再生成なし |
| C計画 | `d458b5e4f596d628d923263f3b2f1678960f2ffa`のPRICE_REQUESTSを正式B正本へ照合 |
| C provider | `1db4d6e7ebce872c069e337b76632046f60d8b56`のbv2_mean_priceを無変更で継続利用 |
| D物理adapter | `eecaa8030280f8957ad85555bcc89e2daa1a29fa`から既存module/testを取込み、今回の最小拡張を接続 |
| 先行共有仕様 | `575598983059fdf5238da877f62b40250fd9eae9`で[INTERFACE](INTERFACE.md)をcommit/push済み |

対象のcanonical IDは`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。H4 linear 1Å/STO-3G・保存tol-only DF10/元順、PR L_D=1/S2/ΔPF=h=.1/K2/M12・13round・exact共通κ・元H/state/snapshot/prior/decoder・symmetric directional control/測定付きfull wrapper/P=0を変更していない。

`abc_s0_published_intake.accept`は既存`h4_s0_v2_intake.validate_A_v2_packet`を再利用し、全13identity/law、元科学SOURCE、A adapter SOURCE、V1/V2出典リンクと診断値を照合する。A公開configurationはB保存正本と完全一致。A packetをBで作成・再ラベルしない。既存validator内部のcanonical整合検査はそのまま使い、H4 S0の`prepare/canonical`、Aの`connect`、Cの`prepare`や旧13round対照生成workflowは起動していない。

Cの公開計画には**重点m0/m6の2requestと13roundのcoverage metadata**がある。C既存`verify_B_publication`で重点2件の全identity/lawを確認し、他11roundを含む13件のphysical ID/q/R/event/B IDと共通binding/profile/sourceを照合した。B保存正本の26軸requestは全identity・law・SOURCE・測定軸・要求量が対応する。Cが全26軸の価格や独立した全identity packetを取得済みという意味ではない。m0はq1/R2/event2、m6はq64/R128/event8192。26requestは新compile26本の実行指示ではない。

受入結果は[PUBLISHED_INTAKE.json](PUBLISHED_INTAKE.json)。元A/B/C JSONは固定Git参照で読み、7入力artifactの科学的出典を同結果に保持した。以前公開済みのcanonical configuration・identity対照・13round plannerを複製し直さない。既存plannerの複合reason `N_IDENTITY_LAW_OR_INTEGER_SHOTS_MISMATCH`は十分N=nullも含むため記録されるが、今回の正式identity/law照合はPASSしている。

## 技術拡張と再利用

`pf_pr_rpe_pipeline.structure`と既存B `generate/_validate/assess/cost/select`を最小拡張した。新しい評価framework、RMSE数値器、価格modelやnative physical hookは追加していない。

- **標準Yoshida4**：`formula='4th'`、ID=`PF_STANDARD_YOSHIDA4_BINARY64_V1`。既存`h46_m5_plan.weights('4th')`のbinary64 signed列・負時間をそのまま利用し、m5_bestとは別identity。Full/truncated PFと元H/保持Hを区別。PFにはκ/Kを付加しない。PRの標準4次・8次の新接続は対象外。
- **κなし明示Full-H RTE**：kind=`FULL_H_EXPLICIT_LAYOUT_NO_KAPPA_V1`、`kappa=None`専用。入力のM/K/q/Rを保持し、event=q×Rとlayoutを検査。κ付きの従来mapping・論文κ由来scheduleと区別する。未指定roundは`UNRESOLVED_PHYSICAL_LAYOUT`で未評価。selectでも両parameterizationを別groupに保持し、明示layoutを共通κ探索とは呼ばない。
- **Full-H preparation**：D既存validatorを再利用。one-bodyを含む全非scalar DF項、全H λ/SOURCE、全fragment順、scalar一回、law、K、profileをidentityに保持する。B3との別物性と`priced_equivalence_claim=false`を維持。κとの等価性・RMSE成立を意味しない。
- **物理接続**：D既存`requests`の外部対応表を使用し、`validate_requests`を追加。A/C双方が同じcanonical identity・whole schedule・元physical ID/SOURCE/layout/H/state/law/profileを参照する。同event数でもq/Rが違えば別候補。旧physical ID単独ではN/C再利用不可。

Cの新規取込みmodule `h4_s0_price_plan.py`、既存`bv2_mean_price.py`は各固定commitとbyte一致。係数module・既存physical cid SOURCEもDの固定af284d85とbyte一致。D moduleは上記formula/mapping/bridgeだけを拡張し、既存κ付きAPIを保持。過去の8次SOURCE記載は保全し、今回その生成・接続・検証は行っていない。

## テストと実行境界

[TEST_RESULTS.json](TEST_RESULTS.json)／[TEST_LOG.txt](TEST_LOG.txt)：**93件PASS、failure/error=0**。内訳：今回の結合12、D取込み回帰14、既存controller17、4方式coverage14、A/B/C統合12、C provider12、evidence planner12。evidence旧test中、8次を検査する1methodは今回のscope外として実行対象から除外した。旧S0 `prepare`を呼ぶsuiteも起動していない。

検証対象：公開A受入/C対応、nominal/null保持、4次/m5とFull/truncatedの区分、κなし/κ型、異grouping、one-body/B3/scalar、SOURCE/law/compiler不一致、未指定layout、C1 MODEL false/有限cohort、人工exact巨大Nの積和と候補選択、科学runner/compile/fit/実Gの非起動。供給Gの数値テストはMOCK_H/state/λ/shot/priceのみ。これは科学的性能検証や独立reviewではない。

実signal/N/RMSE再解析、normalization、sampling、回路build/compile、価格予測/fit、実G/最適化は0。A/C/Dのworktree/process/cache・既存SOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOPは変更していない。

## 未取得証拠・次の担当

| 範囲 | 現在の不足 | 次の担当・必要scope |
|---|---|---|
| 固定H4 S0 PR | nominal m0=64/64、m6=38/38のみ。他11round未評価。十分Nは全26軸null、global bound=null、モデルUNQUALIFIED | **Codex A**：既存保存mean/referenceの再利用資格を確認し、不足roundの信号・phase/radius・十分N・全schedule RMSE/finite-K/κ証拠を取得する有限計画。新数値評価・再解析は別の明示許可 |
| 同PR価格 | 全26軸の適格平均compiled Cなし。C1 MODEL false、有限cohort、S0a構造SをCへ代用しない。全law/長q/希少event/共有共分散・systematic・CI未資格/null | **Codex C**：今回の正式configuration/26requestを参照し、既存価格の完全一致再利用と不足domainの有限検証を計画。compile/sampling/fit/実価格予測は別scope |
| Full PF/標準4次 | 技術入口のみ。正式formula/Δ/M比較menu未決定。十分N/global RMSE/元H bias・全wrapper価格・公平なbaseline最適性の証拠不足 | **ユーザー＋Codex D/A/C**：menu/物理SOURCEを固定し、Dが物理request、AがN/RMSE、Cが価格を担当。技術利用可能とbaseline採択を区別 |
| Full-H RTE | 全H preparationと明示q/R/K/Mは入力が必要。実N/C/global RMSE、law差、cohortと全law期待、長時間/共分散未資格 | **Codex D**：SOURCE付きlayout/request、**A**：finite-K信号/normalization/十分N/全RMSE、**C**：groupingを保持した全wrapper価格。κ等価性は別の科学判断 |
| 総費用比較 | 4方式比較未完了。G/winner=null | **Codex B**：資格付きA/C証拠公開後の受入。実N×C/G・有限menu最適化はユーザーが別途明示許可した範囲だけ |

公開commit/remote SHA・tree・主要artifactを照合して**mandatory STOP**。今回の技術許可を後続の科学実行へ拡張しない。
