# B：全RPE shot配分の内側評価・費用選択adapter

2026-11-13 JST。**同じphysical V2 configurationに複数の全round shot配分を接続した。人工・回帰48件PASS。新実H4 MC、保存数値再解析、実N/RMSE/G評価・科学的候補選定は0。** loss、評価水準、探索範囲、confidenceを正式採択していない。

[入出力/API](HANDOFF.md)／[初回の有限科学案](SCIENTIFIC_PLAN.md)／[統計・adapter 31件](TEST_STATISTICS.txt)／[既存controller 17件](TEST_CONTROLLER.txt)／[技術確認](TECHNICAL_CHECKS.txt)。前回[再利用設計](../rpe_shot_evaluation_design_20261113/REPORT.md)の未実装だった内側層だけを実装した。

## SOURCEと再利用

| 固定SOURCE | 確認したもの・今回の扱い |
|---|---|
| B `437ba4acdb82708b6fc98e3cd119a2a734f7cf64` | prepare/MC/R3/summaryとcost/selectの再利用、whole-vector coupling、診断と十分Nの境界。既存branchを継続 |
| D `24b0a6eb1d806391a3c6ad454fb0c018148d8545` | HANDOFFのprofile extension、SOURCE付き平均区間、同時box、共有価格の差分相殺。残差model・価格kernel・実測器は作らない |
| A `821c5c1ffd7c95e1690d45d2b6e8ff4ccb15bd1c` | HANDOFFのcanonical configuration/physical ID/全raw・normalization・誤差・law。Aの候補/signal生成を呼ばない |
| C `331a6ffebe551577eb9574aca354e76175697743` | FAIR_COMPARISON_HANDOFFの同target・decoder/loss・P0、Full/Truncated PF・Full-H RTEの区分、旧/新loweringの分離を維持 |

他担当のbranch/worktreeや科学原本は変更していない。参照は固定Git objectの文書・実装で、保存trialや科学値を再集計していない。

## 実装

新module [`rpe_shot_allocation.py`](../../../src/trottertracks/rpe_shot_allocation.py)の3入口を用いる。

1. `prepare`：同じconfiguration・供給meanを各allocationへ既存`S.prepare`で結合。元physical IDを変えず、全round X/Yのexact整数ベクトル、allocation ID/SOURCE、統計契約を保つ。
2. `diagnose`：明示呼出し時だけ既存`simulate_prepared`と`summarize`を実行。batch hookは選択可能。callerがloss/IID/条件付きscope/confidence等を指定し、結果は全trial/statusと診断。N packet・十分N・global bound・Gはnull。
3. `compare`：SOURCE・契約・配分を結び付けた外部N packetを既存`assess`へ渡し、適格時だけ`cost`を呼ぶ。NOMINALは除外。価格のprofile/lowering/law/SOURCEを追加sidecarで照合してからexact費用を計算する。

元decoder、binomial/count変換、RMSE評価器、`pr_rpe_rmse_controller`、費用会計器は無変更。新しいRMSE/N理論・signal/C provider・共通controllerは追加していない。MC summaryを自動で科学的な十分N packetへ変換するproducerは依然として存在せず、所望claimの資格は外部証拠が必要。

## 選択と未資格の扱い

- `point_minimum`は利用できる適格な供給平均価格の下での有限menu点最小。shot総数の最小とは別。未評価配分があればその部分集合だけを表示し、`outer_evaluation=null`として外側へ完全な最適化結果を渡さない。
- 全menuが解決した場合だけ、選んだ1配分を同じconfigurationの代表として既存`select([config], [evaluation])`へ接続する。外側controllerを置換せず、同一IDの重複evaluationを渡さない。4方式比較完了を意味しない。
- 区間はDの外部資格・全selected round/axis同時被覆・domain/SOURCEが一致した部分だけ使用する。共分散不明を0にせず、同時boxで`ΣN C`と`Σ(N_left−N_right) C`をexact計算する。unknown endpointはnull、同一価格の係数0は相殺できる。
- `strict_interval_winner`は供給Nの資格を条件とする価格区間上の厳密優越。区間が重なる場合は点最小があっても優越はnull。有限cohort scenarioでは期待価格winnerを出さない。点meanのないinterval-only recordからmidpointを作らず、GはUNKNOWNのまま。
- 選択後のN/RMSE confidence・Nと価格の同時資格は供給側契約の責任で、adapterはその正しさを証明しない。探索外の最適性、科学claim採択、全RPE保証を新規認定しない。

## 技術検証

新10＋既存statistics16＋S3 batch5＋controller17＝**48件PASS**。全て人工/mock又は保存metadataの受動fixture確認。人工2roundでは12shot/費用1002と18shot/費用216を区別し、後者を点選択する。実H4のGではない。

新testは同physical ID、SOURCE/契約/整数ベクトル混在拒否、scalar/batch試行・summary同値、元decoderの逐次branch依存、未資格RMSE/nominal/欠測Cの算術禁止、2^1200超のNと1/3価格、MODEL false/有限cohort、未知区間・共有価格相殺、部分menu/tieを確認した。初回の新testは高速来歴fieldとexact-rational encodingを直接比較して2 assertionが失敗したため、来歴を別途検査し数値を既存exact_scalarで比較するよう修正した。decoder規則・数値処理の変更は0。

## 残る判断・次の担当・STOP

**ユーザー＋A/B**：target/全事象loss/評価水準・N資格、有限配分menu、confidence/precision/選択後被覆、科学実行範囲を決定。**A/Cのsignal担当**：同configurationの全round raw・normalization・不確かさを供給。**D**：同law/profile/loweringの平均価格と区間・SOURCE・domain資格を供給。**B**：許可された有限範囲で統計評価し、資格が揃った場合だけ実費用比較へ進む。

今回の成果をcommit/pushし、remote SHA/tree/主要成果物を照合後**mandatory STOP**。実H4 MC、保存再解析、実shot最適化、実G/候補選定へ自動進行しない。
