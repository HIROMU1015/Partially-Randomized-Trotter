# H4 次回実行：失敗分を予算から除外する方針

利用者の「失敗した過去分は残さなくてよい」に従い、次回の新規H4 signal/compile mapは実行単体を上限の対象とする。
過去の停止したmapのcarryを次回へ加算しない。従来の「失敗分を返却せず累積する」条件を、この次回実行の予算会計について更新する。
前の累積21GiB・74,806件案は不要となる。旧journal、停止証拠、認可、入力と科学条件は保持し、値を書き換えない。
この指示は予算の扱いを変更するもので、終了済みone-shotを再利用したり、本体を自動retryしたりしない。

旧carry：22 actual /12,956,511,264 bytes /6004.111340102032秒（保守的upper）。
次回へ持ち込むcarry：0 actual /0 bytes /0秒。旧値は履歴だけに保持し、次回の上限判定から除外する。
凍結6入力を再生成しない。旧partial/cache/resultを新しい科学結果へ混合しない。

全mapの新規charge bound：8,736,971,632 bytes、約8.136939GiB。
全74,784 logical wrappersに必要なactual最悪値：74,784（合法cache節約を仮定しない）。
承認済み17GiB・74,805件に収まり、増額不要。静的見積りには全72h observer trace、journal、control、保存temp/final、
record/ledger/worker log/signal、private library cache、block/inode/metadata余裕を含む。
物理容量とquotaは次の起動直前に別途再確認する。

SOURCE `cd162e9305143d81c28908b9732f8eef12cd6b89`（closure46）と前REVIEW `461ddb2c2d613c063b4358ada8ad3de02b798141`を保持する。
今回はsource/testsを変更せず、軽量方針資料だけを固定する。既存validatorは消費済みrun03/CARRY21のままであり、
この資料を渡しただけではcarry0の起動bindingにはならない。次回は新SOURCEのvalidator/seed、未使用run/output/control/lock、
plan/auth/reviewへ利用者の今回の予算指示とcarry0を明示結合する。
固定environment/compiler、CPU12workers/driver/observer、library cache v2、科学条件、各role memory caps、monitor5秒等の既承認条件は保持。
本体は停止中。新しい起動指示は今回の予算変更には含まれず、approved/runtime false・allowed_cpus=[]・未seal・absolute command=null。

予算検証は既存sourceのstatic関数だけをASTで分離し、carry0で評価した。74,784件、8,736,971,632 bytesを確認し、
旧20.203630GiB見積りからcarryを引く計算とも一致。実NPZ、科学array、回路build/compile/transpile、worker、GPUは使わない。
この資料更新のための追加人工test campaignは実施しない。

[方針・静的見積りのbundle](../../artifacts/resource_applicability/track_a_h4_per_attempt_budget/2026-10-09/README.md)。
[前回のSTOPとcache修正](track_a_h4_entrypoint_cache_fix_20261009.md)は履歴として保持する。
