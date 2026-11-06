# 研究C：保存contextの起動修正とfresh実行

2026-11-06 JST。科学条件はH4/1.00 Å/STO-3G、number CDF8/元順LD2、
Nα=Nβ=2/sector36。別条件H4/1.15 Åは開発recipe凍結後だけに生成する。

## 起動修正

前SOURCE `c5c48ba38b88d5c914c51f710113e65f806a900a67d7ade03fd76a29290d9608` は、
01:20:43 JSTの起動から約0.296秒でSTOPした。保存contextを読む前に必要な
`trottertracks._c1_stage1_legacy` の登録が抜けていた実装不備であり、
完了科学stage0、新しいfit/native/移送0。context再利用の登録callは1 attempt/FAIL。
このSOURCE・raw・STOPは変更していない。

別worktreeの `PhysicalWorker.request(initialize)` に、既存
`c1_stage1.legacy_df_module()` の呼出しを追加した。
同じ登録処理は既存のevaluation workerでも使用されている。
数学的な評価器、optimizer、compiler、許容差、科学的な比較・選択政策は変更していない。

実物のSectorCAR/PhysicalSector classを含む人工contextをpickleし、
legacy namespaceをまだ登録していない**別Python process**の本物のRPC経路で読んだ。
そのprocessでCAR constructorを故障注入により禁止しても読込みが成功し、
保存H・basisの全成分が人工原本と一致した。元H/CARを再構築していない。
追加したこのE2Eと既存接続回帰の計12件がPASS。
実分子の保存pickleを技術確認のために先行decodeすることはしていない。

## 今回のSOURCE・契約

- [SOURCE](../../artifacts/research_c_integrated/2026-11-06-picklefix/technical/SOURCE.json)
- [契約](../../artifacts/research_c_integrated/2026-11-06-picklefix/technical/contract.json)
- [passive preflight](../../artifacts/research_c_integrated/2026-11-06-picklefix/technical/preflight.json)

SOURCE SHA256：`8113f23e66c5eca3c09a8ed1c91fc58c01eeaa91d0c07850b685a20c816539eb`。
契約SHA256：`fd111609d72d2a2d4ded7415b3e622c0b3a8951f55591809794ef3e30c867f11`。
worktreeは `research-c-integrated-picklefix-20261106`。
保存17trajectory・初期化・元H/CARを再利用し、新規開発計算はC0/ENTRY/strength=.3の1本。
その後の全登録下流・recipe凍結・移送を含む新規上限118,188 logical unitsを維持する。

今回の会話でこのSOURCE/契約と、数値処理開始前のimport/path/serializationに限った
技術修正・SOURCE記録後のfresh再投入が一括許可された。
その範囲でも過去のSOURCE/STOPを保全し、旧runのretry/resumeはしない。
科学条件、数値engine、許容差、sampling政策の変更や数値correctness failureはこの起動対応範囲に含まない。
許可を証明する別のartifactやユーザー発言の保存は要求していない。

「18本」自体を特別な科学的必要数やCPU/process数の上限とは扱わない。
今回は元の3 start×3 strength×2 armという比較gridを維持する。
独立な下流candidateは実空き資源に応じて36process、affinity計64 CPUに配分した。

## 実行中の記録

01:32:41 JSTにfresh実行を開始した。pid等は
`artifacts/research_c_integrated/2026-11-06-picklefix/technical/launch.json`。
保存contextは約1.79秒で正常に読め、新1本は約299秒で64更新まで完了し、16/64 checkpointを保存した。
optimizerのjac欠落は、許可済みのcaptured-input-gradient経路で評価し、
raw success=false/status3/nit0とraw jac未取得の記録を保持した。

この時点で全開発18trajectory相当の入力が揃い、36candidateの下流評価が進行中。
この記録時点では条件付き総資源の最終比較、移送結果、科学的GOはまだ得ていない。
現在の進捗・結果・STOPは同familyの `execution_progress.jsonl`、各fresh出力、
最終 `INTEGRATED_STOP.json` を確認する。
最初の登録native32の実測時間を取得後、Codexが所要時間の見積りを更新する。
