# 承認待ち：Full PF 3公式×5幅のH_eff有限探索

2026-11-14 JST。候補とSOURCEの正本は[PLAN.json](PLAN.json)、全roundは[CSV](ROUND_SCHEDULE.csv)。**本資料作成は実科学実行の承認ではない。** 実H4 step/固有系/cache数値検査/保存値再解析は今回0。

## 固定対象

保存Linear H4/1.00 Å/STO-3G/tol-only DF10/weighted-Hermitian completion/sector(2,2)/dimension36、元H/state/snapshot、Full PF/prefix10、scalar、元full DF-H targetを維持。h=.1/M12/P0、元prior/decoder/εE・εqpe・εtrot/Minkowski契約とcompiler/profileを変更しない。

formulaは`2nd`、標準`4th`、`4th(m5_best)`。要求Δは保存T=.8/q格子に対応する`.05/.1/.2/.4/.8`。binary64係数・exact時間表現・signed順序を既存B/D生成器とC builderのまま使用する。新公式/係数fit/8次/切断/PR条件は含めない。

SOURCE：元点/元固有系69ee0ff、元科学caller3a26b9、準備9af1ac8、C保存loader/step86f37d25、direct-expm kernel dd24dd32、B metadata437ba4ac、D e7dca5f7、base manifest c5466946。新cache対応評価器とmetadataは**本準備成果を含むcommit**で固定する。科学callerを最小接続で用意する場合も実行前にcommitでSOURCEを記録する。追加の承認証明書やgrantは不要。

## 有限call範囲

| 処理 | 上限・内容 |
|---|---|
| 保存context load | 1。既存C loaderのinput hash/H/state/sector gateを保持 |
| 既存S2/.1点 | 69eeの診断をSOURCE付きでそのまま参照。PF step/Schur/旧mean再解析0 |
| 新macro-step | **14**。S2の.05/.2/.4/.8、Y4/M5の各5幅。各cellは一度だけ生成 |
| 新H_eff | **14 Schur**、principal明示分枝、全energy/vector/state weights/分枝/残差 |
| 元full-H固有系 | **新eigh0**。69ee保存reference_comparisonの全energy/basisを使用。元H行列・source/binding・residual/Gramを各診断で確認し、不成立ならSTOP |
| 新直接PF mean | **最大28**。各新cellの最初の使用roundとm12の二点 |
| 全spectral mean | **最大28**。同二点で全state weightsから再構成し、直接PFと照合 |
| 新full-H reference／旧保存signal再解析 | **0／0**。元固有energy/basisの上記再利用は明示scopeに含めるが、元Hからreference信号を追加生成しない |
| normalizer／N／MC／compiled価格／正式G／自動追加点 | 全て0 |

new step-cellはformula×actual幅の14種類。要求幅別の全195round行を別々に評価しない。元Hや同一stepの重複対角化を避け、既存direct-expm half cache・行列累乗を使用する。巨大event列は生成しない。独立cellの並列化は適切に使えるが、CPU/RAM/worker上限は新設しない。

## 二時点の対応

| actual幅 | 最初の確認 | 最後の確認 |
|---:|---|---|
| .05 | m0/T=.1/qPF2 | m12/T409.6/qPF8192 |
| .1 | m0/T=.1/qPF1 | m12/T409.6/qPF4096 |
| .2 | m1/T=.2/qPF1 | m12/T409.6/qPF2048 |
| .4 | m2/T=.4/qPF1 | m12/T409.6/qPF1024 |
| .8 | m3/T=.8/qPF1 | m12/T409.6/qPF512 |

S2/.1の二点は既存結果を参照し、新しく計算しない。粗い要求幅のearly roundは同公式の.1/.2/.4 cellへ対応付ける。例えば要求.8に対し、全roundがU(.8)の整数powerとは扱わない。

## 評価と保存

1. 固定PLANをB/D正本で再現し、全canonical/physical ID、T/q/actual幅、signed recipe、元binding/profile/lawを照合する。
2. C保存入力を一回復元し、69eeの元固有系cacheのSOURCE/hamiltonianを照合する。cacheから旧state weightsを流用せず、同保存stateを使う。
3. 各新cellのSを既存C builderで構成し、元constantをstep時間で一回含むU=e^(−iΔc)Sを評価器へ渡す。直接PFはS^qへtotal-T scalarを一回掛け、spectrum側へscalarを二重適用しない。
4. 元minimumとeffective minimumの差、元最低部分空間との全trace overlap、保存stateの全weightsと縮退groupを保存する。最大overlapの観測と、ground対応の認定を区別する。
5. 指定二時点の直接PFと全spectral再構成を照合する。全13roundの新signal/decoder/shot計算はしない。
6. 元εtrotに対するshiftの局所screen、formula/幅と構造上のpiece数・価格coverageを整理する。C値の再受入・fit・予測・G算術はしない。

数値条件は準備9af1を保持：complex128、state norm1e−12、unitarity/reconstruction/Gram/二時点mean差1e−9、Hermiticity/元固有残差1e−9 Ha、energy cluster1e−8 Ha、phase guard1e−8 rad。cacheは元Hを伴う完全/整列/有限の基底を必須とし、fallback eighを行わない。数値gateは厳密誤差上界ではない。

principal分枝のcut余裕とbranch整数を保存し、元energy/priorに近い値へ無断unwrapしない。Δ‖H‖<πだけでPFの物理分枝を認定しない。対応が曖昧、又はshiftが数値残差から分解できない場合は未確定を保持する。分枝continuation、別window、精度引上げ、追加幅へ自動移行しない。

## STOPと結論の境界

SOURCE/input/recipe/scalar不一致、cache不成立、非有限、数値gate/二時点照合不成立、log分枝が数値的に解決不能なら部分結果を保存してmandatory STOP。閾値緩和・再正規化・unitary投影・候補差替え・自動retryは禁止。

大きいδEが既定εtrotを超えること自体は有効な局所screen結果であり、数値gate失敗とは区別する。比較可能なら残る承認済みcellを評価するが、精度超過を全RPEの厳密INFEASIBLEと呼ばない。小さいδEも十分N/global RMSEを認定しない。

成功時は格子内の有望formula/幅を、分枝・state支持・全round例外と価格の未取得域を添えて提案する。m12のpiece数又は価格だけで最適PFを採択しない。最良が格子端/予算境界なら、必要な隣接点を別計画で示すだけにする。

科学実行後もrpe_bias上界/十分N/global RMSE/Gは資格がなければnullを維持する。結果commit/push・remote照合後mandatory STOP。次の実行はユーザーの本有限範囲への明示承認後だけ。
