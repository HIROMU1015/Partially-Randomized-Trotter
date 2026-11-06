# A1対B1-OL：採択済み752-shot予備検証

2026-11-07 JST、SERVER1 GPUA100-gleap-01。ユーザーは36ee9cの準備版・5c27684dの草案条件を採択し、新規一campaignを明示承認した。旧草案の未採択表示は履歴として保持する。本ファイルを今回のactivationに対応する実行仕様とする。

保存linear H4/1.00 Å/STO-3G/legacy DF12、8 system qubits/全4電子sector、L_D3/m0/q1/δ=.02/r1/K2/coefficient_atol0、X/Y、理想gate・保存DFモデル限定。
N1 744e1a09661ebba6841fe5faf456f88c41c47780、N2 5e9bd08068738cdf1070bdc43bee8194b113ae33、engine fdf00cee3c87d804845497afc89ac6a71f3b3f59を再利用する。actual Q・selected full policy・state/spectral witness・exact Fraction kernelは変更しない。保存厳密engine誤差上界を用いる。再preparation/CDF/solve/spectral/engine認証は0。

## 有限scope

- campaign `A1-N3-PILOT-H4-m0-q1-20261106-v1`、master seed **20261108**、一回のみ。旧397件と新B1339件のbit/係数/word/cacheの流用は0。
- A1 conditional train5層×16×X/Y＝160。3-control固定、既存OLS一回/軸、intercept込みrank4、D_CV≤4。係数を評価前に凍結し、失敗時にrefit/clip/control削除をしない。
- B1-OL conditional probe5層×8×X/Y＝80。probe bitとfull RZのみで草案の配分heuristicを一回/軸適用し、quotaを凍結。
- A1評価unconditional Q、B1-OL評価direct conditional Qで各方式128/軸、計512。全5正層を保持。B1-OLはfloor4、残り108を p×sqrt(((1+min(s²,1))/2)/max(1,cbar)) のlargest remainderで配分。既存128-bit root enclosure中点・固定cell順tieを使う。
- 合計752 logical trial以下、exact engine/Born≤752、compile≤752、selected full build≤1504、path word≤64/trial＋独立測定word1、総word≤48880。bundle decode一回。全supportの列挙はしない。
- PCG64/SeedSequence、32 sorted domains/64子stream。A1 train/prod、B1-OL train(probe)/prod、X/Y/cellを分離。独立uniform53-bitはモデル仮定で、PRNGの数学的独立性を証明したとはしない。
- guard-first：A1 learn X→Y、B1 probe X→Y、A1 eval X→Y、B1 eval X→Y。同group内は空き資源由来の既存poolで独立taskを並列実行。

## 再利用と保存

新規 `a1_n3_pilot_activation.py` と `run_a1_n3_pilot_science.py` は、準備済みPilotDriver/LegacyCollector、既存Campaign.group/worker、facade、sampler、価格provider、qualified manager/observerを接続する。旧科学SOURCEは変更しない。
SOURCE/profile/input bindingを公開・本文/blob/tree照合し、同じHEADのみconsume-once cronへ投入する。cron triggerを除去してからmanager→親を開始し、`/system.slice/cron.service`を検査する。Restart=no。Linger=noのuser sessionへ依存せず、cron daemon/hostの存続までは保証しない。

fresh output `artifacts/o2_a1_new_target/2026-11-07/n3_pilot_752_run01`、fresh cache `cache/`、control `n3_pilot_752_manager_run01`。旧output/cacheを読まない。親・workerの開始/終了/signal/cgroup/pressure、CALL_START/RETURN/EXCEPTIONを保存。trial計画・path/word→bit fsync→価格の順序を保持する。STOP要求はdrainより先に保存し、全worker末尾をreconcileする。
価格はQiskit1.3.0、basis(rz,sx,x,cx)、opt2、seed17、layout/routing/coupling/backendなし、selected full interrogationの状態準備のみ除外。cache hitもfull RZ課金。train/probe/evaluation/古典費用は別記する。
Legacy ACCOUNTINGのconfidence result placeholderは旧serializerの形式であり、pilot結果はPILOT_RESULT/*EVALUATIONを正本にする。精度保証は採択していない。

## 診断・STOP

草案どおり plug-in Vhat_mean と評価実現RZの積scoreを比較。zero/all-equal cellはINCONCLUSIVE、両軸A1 scoreが小さい場合のみFAVOURABLE_PILOT_SIGNAL、両軸不利ならNO_FAVOURABLE_SIGNAL、混在はINCONCLUSIVE。正式ε/α、精度保証、期待費用改善、一般優位性は結論にしない。
入力/SOURCE/stream/word/engine/価格不整合、rank不足、D_CV>4、共有環境への非干渉不成立でSTOP。追加shot/reseed/refit/科学retry/float・GPU置換/B0・B1-O/本検証/他roundは禁止。
完了・中断とも結果とSTOPを保全・公開照合しmandatory STOP。新しい科学実行は別途承認を要する。
