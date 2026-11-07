# 実H4 Q2 opt-in fingerprint限定検証

専用branch `codex/track-a-b2-q2-fingerprint-h4-20261107`。
基点`c71bb8227000bd544aa0c4f946c0e6cf28a80bee`のopt-in pipelineを使い、今回承認された固定範囲を実行する。

- Linear H4/1.00Å/STO-3G、保存DF12、H_phys target、generation LD3、partial-S2/r4/K2、delta=.02、m1/Q2/T=.04。
  元H基底state仮定・prep除外、ordinary control、signed/scalar/tail identity、spin/JW/bit orderを維持。
- 保存Q2 referenceは`ce2e6736ca1178a4adafc4af344852d958951ea6`。
  新方式のstream nativeを**1回**buildし、X/Yを**各1回、計2回**whole compile。旧referenceは再build/compileしない。
- Step1[0,4,4,0]、Step2[4,0,0,4]、完全event identity
  `0ecf28e7a5d2d367c6c609a8d8a4da544c2d03b955aef86b0656f0bf107b7ae8`を維持。
  既存event再現/validatorで全fieldを照合し、完全payloadをbuild前に保存。RNG/探索/代替trajectoryなし。
- Qiskit1.3.0、rz/sx/x/cx、optimization1、seed7、backend/targetなし、logical all-to-all、`no_backend`。
  compiler hash `a0301472e2fb95ba82d3ce80dace0d6de9637ccc74b1887d8271ca1c3e5c04d8`、測定込みfull no-prep scope。
- 旧input/law/source確認helper、Q2 event順序helper、段階timer、opt-in native/wrapper/pair compilerと6指標集計を再利用。
  変更された3つの接続SOURCEは基点のbytesに固定し、他emitter/law/compilerは保存Q2と同一SHAを確認する。
- native行digest/recursive fingerprintと両wrapper identityを確認してからcompileする。
  compiler/backend/scope、compiled fingerprint・global phase、6指標、measurement/qubitを保存referenceへ照合する。
- 新pipelineの段階時間/メモリと終了時RSSを保存。
  終了後、同じ完成native回路で旧/新hashを各1回だけ技術計測し、pipeline本体のRSSと分ける。
  この技術計測では新しいnative build/compileや科学条件の変更は行わない。
- 人工test後、SOURCEをcommit・公開・照合してから実行する。
  科学実行はこのnamespaceで1回。SOURCE/input/trajectory/identity/価格の不一致や結果取得不能でSTOPし、自動再実行しない。
- 実H4 m0/m2以降、追加trajectory、SCF/DF/state solve、信号/branch/shot、MC、強PF、総RPE費用、詳細科学レビューなし。
  private原本、既存SOURCE/raw/grant/ledger/marker/STOPと他worktreeを変更しない。
- 結果JSON/REPORTを公開し、remote SHA/tree/公開内容を照合後**mandatory STOP**。
