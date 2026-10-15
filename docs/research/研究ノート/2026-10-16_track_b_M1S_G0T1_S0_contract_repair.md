# 2026-10-16：Track B M1-S G0-T1 S0候補の修正・固定準備

Status: `M1S_G0T1_S0_REPAIRED_CANDIDATE_READY_FINAL_ADOPTION_PENDING_MANDATORY_STOP`。

今回ユーザーは、GPTレビューの推奨に基づきG0-T1への縮小と第三案A′を暫定承認した。前回のA/B未選択状態からの変更理由は、T1aの最小信号保証を使い、Checkpoint 1でT1bへの価値を判定するためである。T1b後はCheckpoint 2の重要GPTレビューを必須とし、Gate 0はその結果で別途判断する。これは研究契約の最終採択・証明実施の承認ではない。

[最新GPT原本](../track_b_m1s_g0t1_claude_gpt_independent_scientific_review_20261016.md)を内容変更せず52,629 bytesで保存した。SHA256 `cb24103f22ddfb12afb82c923fee4210fabae4b7dfec2355896a0686fb18f633`、原本保存commit `d1039afae54520f15545177a7c852407830a0927`。GitHub再取得bytesも一致。Claude今回原本は直接受領しておらず、その主張はGPT原本の引用・評価の範囲だけを参照する。

[候補契約](../../tracks/algorithm_codesign/m1s_g0t1_s0_repaired_theory_contract_candidate_20261016.md)では、可換Pauli例にも残るO(D_damp |theta|)局所高次誤差を証明義務へ加えた。Secular比較対象、exact deterministic halfのΨ、alpha=delta_rel/4を含むjoint scaling、状態ノルムとmode/絶対/相対/rho誤差の区別を整理した。一般Hermitian/DF event、一般Jordan/全次元一様理論、Q4/Q5は初期scope外に残した。修正式を証明済みとせず、比較モデル・norm/weight・joint path等は最終確認事項として明記した。

前回事前契約・文献調査・H4/H6データ監査を再実行せず参照し、旧本文・旧manifest entry・結果分類を保持する。Parent allocation D＋E、M1-A、M2、既存source/result/freeze/auth/marker/STOPは不変。専用worktreeで新文書と索引を公開し、親HEAD/index/既存dirty bytesを保護する。

新証明、T1a/T1b理論実施、数値反例探索、Gate 0/E0、行列/simulation、sampling/trajectory/operator guard、回路生成/compile/synthesis、資源forecast、Q4/Q5はすべて未実施。次はユーザー・GPTによるS0候補の正式採択判断。実行承認は別途必要で、mandatory STOPを維持する。
