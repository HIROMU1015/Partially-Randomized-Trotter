# 2026-10-16 Native DF理論ゲートモデル：技術作業記録

利用者の依頼に従い、043f58ac…を起点とする独立worktreeでA/Bカウンタと人工検証を追加した。
[主要資料](../track_a_native_df_theoretical_gate_model_20261016.md)へ、別SOURCE87b6b9f…のnative構成との対応を記録した。
旧研究方針・結果・STOP・grantを書き換えない。

採用した技術上の選択：Aはbasis reuseを無効化、Bは同じidentity・操作列・qubit配置の隣接inverse Gaussianの相殺だけ。
同軸回転mergeは今回は実装しない。RTE identity applicationは元builderどおりreuse barrierを保つ。
whole-trajectory期待値ではprefix/tail/step境界を閉じ、存在しない境界相殺を計上しない。
これは構成モデルの定義であり、compiled費用を保証する上界や最小値とはしない。

finite-RTEでは全偶数次数と全componentを使う。Aは線形期待値、Bはbasis状態DP。
一般identity rotationのphase取消をfloat/symbolicで同一視できないため、1 occurrence最大1 Pの費用幅を残した。
確率をbinary64で評価した誤差は、Fractionによる費用平均のexact性とは別。

27件の標準ライブラリ人工testsがPASS。主担当の技術チェックであり独立科学レビューではない。
実Qiskit・実分子metadataへの接続、history依存basis-plan、一般phase期待値、depth等は未検証。
科学的な比較契約への採用・主張範囲はGPT判断へ戻す。実H6/H8評価・sampling・compile・server接続・原稿改訂は0。
MANDATORY_STOP / DRAFT_NOT_AUTHORIZATION / next_stage_authorized=false。
