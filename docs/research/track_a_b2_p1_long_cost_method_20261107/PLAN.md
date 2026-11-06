# B2 P1長round価格方式の人工技術検証

今回のユーザー指示が承認する技術実装・人工build/compileだけを実行する。
専用branch `codex/track-a-b2-p1-long-cost-method-20261107`、基点`2d21c23879aaae063a796fa95c1433fb3d0b5738`。
前回の実H4 m0価格、元SOURCE/bundleと公開snapshot対応、他worktreeは変更しない。

- 既存人工二mode fixtureを拡張したrank4・signed/complex factorsを使用。
  Q1/2/4/8、LD2/3、r4/K2、delta=.02と、Q4/LD2の反復eventを計9ケース。
  eventはcanonicalなorder0/2の固定patternで、分布平均を推定する標本ではない。
- compilerはQiskit1.3.0、rz/sx/x/cx、optimization1、seed7、backend/targetなし。
  hash `a0301472e2fb95ba82d3ce80dace0d6de9637ccc74b1887d8271ca1c3e5c04d8`。
  X/Y測定付きfull no-prep、ancilla最後/LSB system、scalar/identity/sign/orderを維持。
- reference：既存B2.nativeとwhole compile。
  candidates：同じgate列を出すrequest/tail streaming、既存boundary fusion builder、
  既存actual-circuit/compiler/backend cache、分割compile後の接続回路を計測。
  分割depthの単純和を正しい全回路depthとして採用しない。
- 人工CAR oracleでcontrol-0/1と相対位相を位相fitなしで照合。
  whole wrapperのchannel/測定効果とX=Re/Y=Imを別に検証。
  各軸6価格指標のreference/candidate/差を保存し、人工条件に限って分類する。
- build/compile時間、Python tracemalloc peak、process RSS high-water、DAG operation数、
  cache hitを観測。追加の人工Q64はrequest保持だけのメモリ診断で、回路build/compileしない。
- 既存`1b64e46…`の並列providerを原文のまま再利用し、そのschedulerへ人工B2 taskを注入。
  同じ11明示task（9ケース＋重複2件）のserial/parallel結果とsubmission orderを比較。
  worker数は既存memory probeと空きCPU/RAMで選び、独自の低いcapは設けない。
  これは本格MCの標本数/seed/round範囲の採択ではない。
- `np.load`と分子solverを実行process内で禁止。実H4入力、m17展開、追加実H4 compile、
  科学的価格MC/期待価格/信号/branch/shot/強PF/総RPE費用は一切実行しない。
- 不適格なreference、controlled/測定意味の破綻、元SOURCE/科学条件変更の必要、非干渉を
  確保できない状態は理由を保存してSTOP。技術的な人工実装のdebug/retestは許可範囲内。
- SOURCE・人工結果・REPORTをGitHub接続で専用branchへ公開し、remote SHA/tree/内容を照合。
  新しい実H4条件へ進まずmandatory STOP。
