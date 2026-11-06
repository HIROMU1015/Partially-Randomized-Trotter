# A1 N3案B：採択済み一campaignの実行仕様

SERVER1 GPUA100-gleap-01、2026-11-06 JST。ユーザーの「実行してよい」を、直前の案B契約草案の採択・一回実行許可として扱う。追加の承認発行手続きは要求しない。
基準commitは2438170a7d1eada04e8760cedd6fe5f959a2c87aの[N3_NEW_CAMPAIGN_CONTRACT_DRAFT.md](../o2_a1_n3_manager_preparation_20261106/N3_NEW_CAMPAIGN_CONTRACT_DRAFT.md)。旧草案の未採択という歴史的記述を上書きせず、本書で今回の採択を記録する。

## 対象・有限scope

保存linear H4/1.00 Å/STO-3G/legacy DF12、8 system qubits、全4電子sector、L_D3、m0/q1/δ=.02/r1/K2、coefficient_atol0、X/Y、理想gate・独立uniform53-bitモデル。
N1 744e1a09661ebba6841fe5faf456f88c41c47780、N2 5e9bd08068738cdf1070bdc43bee8194b113ae33、engine fdf00cee3c87d804845497afc89ac6a71f3b3f59の保存identity、actual Q、selected full policy、ground、証明を再利用する。
N1 preparation/CDF、N2 solve/spectral、engine認証は再実行しない。

campaign ID=`A1-N3-B-H4-m0-q1-20261106-run01`、master seed=**20261107**。
fresh science output=`artifacts/o2_a1_new_target/2026-11-06/n3_campaign_B_run01`。
manager control=`artifacts/o2_a1_new_target/2026-11-06/n3_campaign_B_manager_run01`。
旧397件・旧係数・旧stream・旧cache・旧STOPは再利用せず、そのまま保存する。

4方式B0/B1-O/B1-OL/A1 × X/Y、一campaign。ε_X=ε_Y=1/8、全8座標の同時α=1/20。
既存g1/g2/g3、A1五層×64 train/軸、rank4、D_CV=2+Σabs(λ)≤4。
B1各正層64 pilot/軸、B0 pilot0、calibration各単位512、production≤8192/method/axis。
A1 cal/prodはunconditional Q、trainのみ直接五層条件付きQ。全正層を維持する。
confidence分配、Bernstein/Hoeffding、J8有限配分、外向き丸め、OLS、N凍結は固定03928の関数と[旧採択契約](../o2_a1_n3_connection_20261106/N3_EXECUTION_CONTRACT_DRAFT.md)をそのまま再利用する。

上限はtrain/pilot1536、cal9216、prod65536、計76288 trial。path≤64word/trial、独立measurement word1。
60 domain、120 SeedSequence子streamを旧sorted domain規則で一回だけ作成する。外側ledgerにcampaign ID/domain/trial/attempt1を記録する。
PCG64の数学的独立性を証明したとは扱わない。乱数保証は独立uniform53-bitモデルに条件づける。

engine誤差は保存された厳密有理数を読んで使用する。表示上界7.8079008e-7への置換、旧1/32の採択はしない。
full selected no-prep価格はQiskit1.3.0、basis=(rz,sx,x,cx)、opt2、seed17、backend/layout/routing/coupling指定なし。状態準備だけを除外する。
実際のtrialのpriceだけを取得し、cache missのみcompile、hitにもfull shot RZを課金する。

## 追加した薄い入口

[a1_n3_B_campaign.py](../../../src/trottertracks/algorithm_codesign/a1_n3_B_campaign.py)は旧Campaign/worker/serializer/word生成/測定facade/価格/統計を再利用し、新output・seed・managerだけをbindする。
旧a1_n3_campaign.run()、旧marker、旧SOURCEは使用・変更しない。certified exact Fraction kernelを維持し、Born/state/真signal/E0/gap/真varianceをCVへ渡さない。
科学bindingをobserver接続context内で一回行い、bound probability methodもcall traceの対象にする。

qualified manager SOURCE=5cb542ece4d208878a1dfc7011007b99297a5a55のruntime/cron/observerを変更せず再利用する。
自分の一時crontab→consume-once dispatch→外側recorder→application manager→親→worker。
consume receiptを耐久保存し、triggerを削除してからapplicationを開始する。application restart=no。
cron.service cgroup確認前に実科学を起動しない。Linger=no、共有daemon/systemd/cgroup設定は変更しない。

/homeの空きが約20GBだったため、管理observerのHEARTBEAT保存のみ15秒間隔にまとめる薄いcontextを追加した。
元のwait pollingは0.25秒、全START/RETURN/EXCEPTION/exit/signal/STOP記録は省略しない。科学計算・予算・並列数には影響しない。
実child/ancestor cgroupとmemory/pressureは定期記録する。共有cron cgroupをjob単独の物理peakとは呼ばない。

親はSTOP_REQUESTEDを保存してからpoolをdrainし、全worker尾部をreconcileする。bitは価格処理より先にfsyncする。
未完了attemptはbit+price/bitだけ/開始のみ/親で計画のみを分け、欠測を0にしない。
process exit0に加え、全8confidence結果・全bit価格の完備性を要求する。科学STOPは非0終了にする。
外側managerもexitと科学STOP.jsonを照合する。重複dispatchはexit72で拒否する。

## SOURCE固定と開始

追加SOURCE・人工test・PROFILE_LOCK.json・PREFLIGHT.jsonをcommitし、remote SHA・本文・blob/treeを照合した後だけ、以下を**一回**投入する。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/AbeHiromu/projects/partially-randomized-trotter-wp11/.venv-server-opt2/bin/python -B \
scripts/tracks/algorithm_codesign/run_a1_n3_B_campaign.py --submit
```

entryがprivate cache環境を固定する。CPU/RAM/workerへの恣意的上限は設けず、開始時の実際の空きCPUからworker数を取得し、worker自身のみnice+10にする。
SOURCE/input/engine、rank/range、十分N、word/stream/probability/価格、出力衝突、非干渉条件の最初の不整合でSTOP。
失敗後の追加shots、再学習、seed変更、float kernel、条件緩和、追加campaign、18round、C1解凍は禁止する。

## 保全・終了判定

成功・中断のいずれでも全データ・実現RZ・古典費用・call数・終了コードを保存し、公開照合後mandatory STOPする。
全比較完了gateを通った場合だけA1対B1-OLの実現費用差を報告する。期待費用改善の保証はUNKNOWN。
旧中断結果・費用を隠さず、新campaign費用と研究全体の累積費用を区別する。
root/cron/host/filesystem全体の障害や全記録processの同時消失への完全な耐久性は保証しない。未保存の終了原因はUNKNOWNで、自動再開しない。
