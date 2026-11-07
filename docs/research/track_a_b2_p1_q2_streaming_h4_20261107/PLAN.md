# B2 P1 Q2実H4 streaming限定照合

ユーザーの今回の指示により、m1/Q2の固定1 trajectory、reference/stream各1回native build、
両方式のX/Y各1回compile（合計最大4回）を実行する。追加承認は求めない。
専用branch `codex/track-a-b2-p1-q2-streaming-h4-20261107`、基点`f6eea2f28be1f15b6ff5e2954b9c8fb48f3c5a0f`。

- H4/1Å/STO-3G、保存DF12、H_phys target、generation LD3、partial-S2/r4/K2、delta=.02、
  T=.04、元H state仮定・prep費用除外、ordinary controlled、signed/scalar/identity/bit-orderを維持。
- 固定eventはStep1[0,4,4,0]、Step2[4,0,0,4]。m0の保存event metadataだけを再利用し、
  8 event全field/符号/確率/basisとcanonical lawを検証する。m0回路はbuild/compileしない。
  完全trajectory identityは実行前に保存metadataから確定した
  `0ecf28e7a5d2d367c6c609a8d8a4da544c2d03b955aef86b0656f0bf107b7ae8`。
  event全体をtrajectory.jsonへnative build前に保存。RNG・価格による選択・自動置換なし。
- Qiskit1.3.0、rz/sx/x/cx、optimization1、seed7、backend/targetなし・all-to-all。
  compiler hash `a0301472e2fb95ba82d3ce80dace0d6de9637ccc74b1887d8271ca1c3e5c04d8`、backend `no_backend`。
  full measured no-prep scope、ancilla最後、各軸measure1回。
- 既存m0 runnerの入力/serialized event比較・native/compiled比較・計測helper、既存emitterとpair compilerを再利用。
  新規はQ2の接続runner、人工test、本PLANのみ。既存科学SOURCEは不変更。
- 人工Q2 CAR oracle・X/Y規約、Step2切替、実compiler4 call経路と失敗時停止を人工入力で検査。
  SOURCEを公開commitで固定し、入力hash・旧core/stream/helper source・元H/state metadataを照合する。
- reference→streamを同一processで各1回build。tracemallocはbuild/fingerprintまで含める。
  順序/cache/reference保持の影響を記録し、単回の時間やRSSを一般化しない。
- native gate列/global phase/fingerprintと全wrapper identityを比較後、reference X/Y→stream X/Yの順にcompile。
  candidateの各軸を今回のreferenceの6指標と前後fingerprintに照合。不適格なら理由を保存してSTOP。
  cache省略・別compiler・boundary fusion・別trajectoryへの救済なし。
- 出力`artifacts/track_a_b2_p1_q2_streaming_h4/2026-11-07/run01/`。
  private state/HDF5/積分/provenance原本、旧SOURCE/result/STOP/bundleと他worktreeを保全する。
- 実H4 m0・m2以降、追加trajectory、sampling/価格MC、強PF、信号/branch/shot、総費用、詳細科学レビューは禁止。
  結果/REPORTを公開し、remote SHA/tree/内容を照合してmandatory STOP。
