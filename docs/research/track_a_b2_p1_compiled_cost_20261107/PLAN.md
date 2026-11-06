# Codex C：B2 P1 full no-prep接続とm0限定compile

ユーザーの今回の指示による実装・人工検証・限定実回路build/compileを実施する。
branchは`codex/track-a-b2-p1-full-no-prep-cost-20261107`、基点はshot成果
`0d5ca357ccfd8fcc850bef800c5bcadfde2a153e`。他worktreeは変更しない。

- SOURCEを人工検証後にcommit固定する。保存DF12は`bf5d3b2405b2bec4f88dd0589b75dd737b13e052`の
  `artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz`、
  SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。
- H_phys/state/sector/orderの正本は`89165cf70958e7e4676d0580e3acb7faf588772c`のinput spec。
  private state/provenance/HDF5は保存hash照合のみで、state/物理Hを再生成しない。
- native/相対位相/測定規約の証拠は`ea2c0915ba2f7e11a6dac58dc01cf1af3c0d0ff5`の16/16 PASS。
  そのSOURCEと今回消費するcoreを照合する。全support/全trajectoryの認証へ拡張しない。
- 実価格compilerは実測済みpilot `0daf01d51c022977a1c0a7d29dde34a1c4c82d7c`のresult.jsonから読む。
  Qiskit1.3.0、rz/sx/x/cx、optimization1、seed7、backend/target/coupling mapなし。
  P1信号のseed17は価格条件へ移さない。compiler wrapperのSOURCEと実runtimeを照合。
- 既存controlled_checksの**outer case1**を一つだけ再利用：保存代表eventのfirst、
  first negative order2、last、first。保存代表リストではindices [0,4,4,0]。
  既存canonical `_make_event`と`B2RPERound.explicit_request`で検証する。
  RNG・価格による選択・別trajectoryへの救済なし。
- Linear H4/1Å/STO-3G、DF12、generation-prefix LD3、partial-S2/r4/K2、delta=.02、
  **m0/Q1のみ**、ordinary controlled、核scalar/tail identity、signed factors、bit-orderを維持。
- `B2RPERound.native`→`build_native_hadamard_wrapper(include_measurement=True)`→
  providerの既存`_compile_pair`→`transpile_and_measure_cost`でX/Yを各1回compile。
  専用出力`artifacts/track_a_b2_p1_compiled_cost/2026-11-07/run01/`に軸別価格、fingerprint、
  SOURCE/input/compiler contextを保存。preflightとtrajectoryをcompile前に保存する。
- instruction allowanceは既存pre-build boundから導出し、CPU/RAM capは設けない。
  対象は2 compiler callであり順次実行し、初回エラーでSTOPする。
- 不一致、native correctness不適格、compiler/backend/scope不適格ならSTOP。
  価格未取得はnull、期待価格/MC/SE/CI/総RPE費用もnull。自動retryやcompiler変更なし。
- 長roundはSOURCEの静的調査とm0で観測した規模の説明に限定。
  m17展開、他mの実H4 compile、強PF compile、価格MC/外挿/信号/shot/総費用計算なし。

結果/簡潔REPORTをcommit/pushし、remote SHA/tree/公開内容を照合してmandatory STOP。
次の価格MCの標本数・seed・round実行範囲は今回採択しない。
