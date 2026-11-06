# A1 N3案B：修正SOURCEからの再投入と技術的復旧

2026-11-06 JST、SERVER1 GPUA100-gleap-01。ユーザーの今回の指示「修正して再実行してよい　意図しないところで中断された場合は修正して再実行してよい」に従う。
直前の案Bの入口import失敗（ade6a25fa554903bb477e0f32142c154d01fad1d、科学0件）は保全し、修正entryを新しい管理出力先へ接続する。
旧契約の技術障害後に毎回別途承認を求める制限は、今回の明示指示により技術的な修復・再投入について更新された。科学条件の変更や失敗データの選別を許可した意味には扱わない。

## 維持する科学契約

[案B契約](../o2_a1_n3_manager_preparation_20261106/N3_NEW_CAMPAIGN_CONTRACT_DRAFT.md)および[採択済み実行仕様](../o2_a1_n3_B_science_20261106/EXECUTION.md)の科学条件を全て維持する。
保存linear H4/1.00 Å/STO-3G/legacy DF12、8 system qubits/full4-electron sector、L_D3/m0/q1/δ=.02/r1/K2/atol0、X/Y。
N1=744e1a09661ebba6841fe5faf456f88c41c47780、N2=5e9bd08068738cdf1070bdc43bee8194b113ae33、engine=fdf00cee3c87d804845497afc89ac6a71f3b3f59。
actual Q/selected policy/saved guaranteed ground/certified exact Fraction engineを再利用し、N1/N2/engine認証は再計算しない。

B0/B1-O/B1-OL/A1×X/Y、一条件・一campaign。ε=1/8/座標、同時α=1/20、既存g1/g2/g3。
A1五層×64 train/軸、B1各正層64 pilot/軸、B0 pilot0、cal各単位512、prod≤8192/method/axis。
rank4、D_CV≤4、独立train/cal/prod、既存confidence・Bernstein/Hoeffding・J8配分を変更しない。
上限1536 train/pilot +9216 cal +65536 prod =76288 logical trials、path≤64word/trial、measurement word1。
engine誤差は保存厳密有理数を使用。Qiskit1.3.0/(rz,sx,x,cx)/opt2/seed17、full selected no-prep RZを全stageで課金する。
CVへBorn/state/E0/gap/真signal/真varianceを渡さず、全8座標の完了gateとA1対B1-OLの主比較を維持する。
実現費用差を期待費用改善の保証とは扱わない。

## 今回の最小変更

旧scientific SOURCE03928のCampaign/worker/測定/価格/統計、qualified管理SOURCE5cb542e…のruntime/observerは変更しない。
既存B adapterについてDOC/REMOTE_BRANCH/fresh manager outputを今回のSOURCEへbindし、STOP記録に技術的復旧の許可と科学gate失敗の再試行禁止を区別して記録する。
修正entryは自分のroot/srcを登録するため、PYTHONPATH/cwdに依存しない。最小cron環境・cwd=/tmpでの人工CLI testを含む37件を検証する。
実入力前にSOURCE/profile/input identityを固定し、commit・公開本文/blob/tree・remote SHAを照合する。

campaign=`A1-N3-B-H4-m0-q1-20261106-run01`、未消費seed20261107、science output=`artifacts/o2_a1_new_target/2026-11-06/n3_campaign_B_run01`。
今回のfresh manager control=`artifacts/o2_a1_new_target/2026-11-06/n3_campaign_B_manager_run02`。
前回manager_run01/STOP、旧397件/価格/係数、元main/dirty/untracked/C1凍結を保全する。

既存own-user cronのconsume-once/trigger撤去→outer recorder→manager→parent/workerで起動する。
application自動restartはnoを維持し、技術障害を確認した場合のagentによる修復・再投入を今回許可された作業として行う。
自分の一時crontabだけを用い、linger/共有service/cgroup/他ユーザーを操作しない。workerは実際の空きCPUから決め、CPU/RAM/worker数に恣意的上限を設定しない。

## 意図しない中断への対応

import/起動/セッション/実装例外等の技術的中断では、毎回ユーザーへ同じ再実行許可を求めず、保存証拠を確認して原因を修正する。
科学未開始なら未使用seed/科学出力先を維持し、旧失敗記録を残してfresh管理attemptへ投入する。
科学開始後は全durable bit/価格/word/domain/trial identityを先に照合し、取得済bitを再測定・破棄せず、二重課金や成功データ選別を避ける最小の復旧方法を実装・人工検証・SOURCE固定してから再開する。
復旧による実際の追加engine/compile/古典費用とlogical trial数を区別して記録する。保存済みstreamの曖昧さや一意な対応を解消できない場合、未知を0とせず保全して報告する。

rank不足、CV range違反、必要N上限超過等の科学的gateでは、失敗を消す再学習/seed変更/shot追加/精度緩和/control変更をしない。
SOURCE/input/engineの科学的同一性、独立性、非干渉を満たせない場合も無断で条件を緩めない。
最終的な科学完了または科学gate STOPの後、結果・SOURCE・全call数/費用・journal/終了statusを公開照合しmandatory STOPする。追加campaign/18round/C1解除/重要レビューへ進まない。
