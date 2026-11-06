# A1 N3案B：修正後の限定実行、稼働中

**RUNNING_NOT_FINAL。科学計算は開始済みで、持続manager下で継続稼働している。完了・CV改善を報告する段階ではない。**

実行SOURCE：`ee49b53fd6f9a17d715edb1df3aac8cbf5731d7a`。
公開14ファイルの本文/blob、tree `f3450df369fad0f6db578bc0d4ba5640bf6179a2`、remote SHAを照合してからmanager_run02へ一回投入した。
前回のentry import不備を修正し、PYTHONPATHなし/cwd=/tmpのCLI testを含む人工37件PASS。旧科学SOURCE・exact engine・測定/乱数/統計・qualified管理runtimeは変更していない。

## 固定scope・取得状況

保存linear H4/1.00 Å/STO-3G/legacy DF12、8 system qubits/full4-electron sector、L_D3/m0/q1/δ=.02/r1/K2/atol0、X/Y。N1 actual Q/selected full policy、N2保存ground、認証済みexact Fraction engineを使用する。ε=1/8、同時α=1/20、固定g1/g2/g3、科学quota/gateは[EXECUTION.md](EXECUTION.md)どおり。

2026-11-06 **13:47:32 UTC（22:47:32 JST）**の[snapshot](../../../artifacts/o2_a1_new_target/2026-11-06/n3_campaign_B_running_snapshot01/OBSERVATION.json)：

| 方式 | 保存bit | 価格保存 | 実現full RZ（取得済み部分） | 状態 |
|---|---:|---:|---:|---|
| A1 | 125 | 125 | 598897 | X軸trainの途中 |
| B0 | 0 | 0 | 0 | 未着手 |
| B1-O | 0 | 0 | 0 | 未着手 |
| B1-OL | 0 | 0 | 0 | 未着手 |

計画/path固定/開始250 logical trials、exact engine開始250、Born完了125、compile27。残る125 trialは処理中。これらは最終call数でも全費用でもない。
保存bundle decodeは一回。N1 preparation/CDF、N2 solve/spectral、engine認証の再計算は0。旧397件は新campaignへ混入していない。
まだtrain完了・係数凍結・calibration/N凍結・productionを確認する段階ではなく、精度達成・CV改善・A1対B1-OLの差は全て未判定。

## 管理と保全

own-user cron→外側recorder→application manager→親→125 workerの起動を確認した。実jobは`/system.slice/cron.service`に属し、自分の投入triggerは開始前に撤去され、crontabなしに戻っている。
application自動restartはno。親/worker identity、CALL_START/RETURN/EXCEPTION、終了statusと実child/ancestor cgroup資源を記録する。
CPU/RAM/workerに恣意的上限は設けず、観測した空きCPUから125 workerを選び、自分のworkerのみnice+10にした。観測時のOOM/oom_killは0、memory pressureは0。これを将来の非干渉や完走の無条件保証とは扱わない。
bit保存→価格の順序を維持し、既に測定・価格保存から次のtrialへ進んでいる。会話や投入clientの終了を科学jobの終了条件にしていない。

snapshotは稼働processを停止せずに各JSONLの完全行prefixを取得した**非原子的な途中観測**で、元journalを変更していない。
既存reconcileを一時copyへ適用して保存済みbit/価格/call数だけを集計した。Born、gate作用、sampling、variance/Nをsnapshot作成時に追加実行していない。
collectorのRSSをcampaign親のpeakと誤認しないよう、ACCOUNTINGのその欄はUNKNOWNへ戻し、実際の親/proc statusをOBSERVATIONへ別記した。
SOURCE/SETUP/STREAMS/RESOURCE_START、parent/cron identity、公開照合、journal prefix、管理ログの途中copyと各SHA256を保存した。取得時刻以降のlive進捗をこのsnapshotから推測しない。
元main/dirty/untracked、旧397件/価格/係数、前回import失敗/STOP、旧SOURCE、C1凍結は保全している。

## 今後の停止・復旧条件

ユーザーは今回、意図しない技術的中断の修正・再実行も許可した。この許可を継続し、同じ修復について毎回承認を求めない。許可原文を証明する追加手続きは要求しない。
技術障害時はdurable journalを照合し、既存bitを再測定・破棄せず、二重課金・成功データ選別を避ける必要最小限の復旧を行う。SOURCEと実際の追加call/費用は区別して保全する。
rank/range/必要N等の科学gateを回避する再学習・seed選別・shot追加・精度緩和・control変更は行わない。
科学完了または科学gate STOP時に全結果/費用/終了statusを保全・公開照合しmandatory STOPする。**現時点では科学計算をSTOPさせていない。**
