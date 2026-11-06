# B2 P1 m0実H4 streaming限定照合

今回のユーザー指示が、固定m0/Q1 trajectoryの従来native/stream各1回buildと
streaming X/Y各1回compile（最大2 call）を明示承認している。追加確認は求めない。
専用branch `codex/track-a-b2-p1-m0-streaming-h4-20261107`、基点`1bed45be1869f447cfd90cef443877fcaed7af6c`。

1. 既存stream人工testと追加のidentity比較testを通し、SOURCEを公開commitへ固定する。
2. `2d21c238…`の保存価格・trajectory、DF12 snapshot/hash、元H/state/sector/order metadata、
   private file hash、16/16 correctness証拠、既存coreとstream SOURCEを照合。
3. Linear H4/1Å/STO-3G、H_phys target、DF12、generation LD3、partial-S2/r4/K2、delta=.02、
   元H state仮定/prep費用除外、ordinary-control・signed factors・scalar/identity・bit-orderを維持。
4. trajectoryは既存`fixed_events`で復元し、保存eventの全field/符号/確率/順序とidentity
   `c3ea48da7dcd082f948b6df9879b3105fc44145c438342396a9777d0b409fbc5`を比較。
   indices[0,4,4,0]、orders[0,2,2,0]。新sampling・選択なし。
5. Qiskit1.3.0、rz/sx/x/cx、optimization1、seed7、backend/targetなし・all-to-all、
   compiler hash `a0301472e2fb95ba82d3ce80dace0d6de9637ccc74b1887d8271ca1c3e5c04d8`。
   両軸のscopeは測定付きfull no-prep。compiler変更/cache救済なし。
6. 従来nativeとstreamのgate名/順序/qubit/parameter、global phase、再帰的fingerprintと
   保存native identityを照合。X/Y wrapperも両経路と保存identityを比較し、不一致ならcompile前にSTOP。
7. streaming回路のみを既存`_compile_pair`/`transpile_and_measure_cost`でcompile。
   軸ごとに保存済み全6指標・compile前後fingerprint・scope/backendを比較。不一致ならINCONCLUSIVEでSTOP。
8. buildは同一processでreference→stream、各1回、tracemallocを各buildに限定しfingerprintまで含める。
   2回目には1回目の定義cacheと保持回路の影響がある。compileはtracemallocなしで時間、現在RSS、
   process peak RSSを記録する。独立processの増分peakや再現性のあるspeedupとは扱わない。
9. 原本とSOURCEを保全し、m0結果・簡潔REPORTを公開、remote SHA/tree/内容を照合してmandatory STOP。

出力は`artifacts/track_a_b2_p1_m0_streaming_h4/2026-11-07/run01/`。
private state/HDF5/積分/provenance原本は公開しない。旧結果・bundle対応・他worktreeを変更しない。
Q2以降build/compile、追加trajectory/MC、強PF、信号/shot/branch、総費用、詳細科学レビューは禁止。

技術修正：run01はnative build前・compile0回で停止し、その記録を保全する。
`RTEEvent.to_dict()`のtupleと保存JSON配列を直接比較した表現比較の不具合を修正する。
数値・符号・確率・順序・canonical digestの条件は緩和しない。修正SOURCEを固定した後、
同じ承認済みm0 trajectoryで未実施の各native buildと最大2 compileをrun02へ保存する。
入力やcompiler条件の変更、追加trajectory、compile失敗後のやり直しではない。
