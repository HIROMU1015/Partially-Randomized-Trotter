# PR要求step幅の探索接続：今回の境界

現作業は利用者の明示指示により、独立branch/worktreeでの実装、保存metadata照合、人工fixture検証、公開を行う。新H4 signal・path/sampling・compile・価格fit・実Q3・round05本番は実行しない。旧round05設定のscience_approved=falseと旧STOP/raw/SOURCEは保全する。

起点はB2 d8e806265、C/A実step較正02e7cc6a1、D2共通準備0ef9668e9（B2は98b40e439を統合済み）、D2 PriceRules7cd8ea139。後続branch/同等実装を照合し、この起点に複数要求幅の探索から監査済み実step結果の部分登録・再探索までの完了経路は無かった。並行する高速化branch/worktreeへ書き込まない。

旧round05にrequested_widths宣言が無い場合は従来WideDriver経路を保つ。新宣言はsearch_overrides.requested_widths={PR_PAULI:[...],PR_DF:[...]}。h/T/M/target/loss/MSE閾値は固定し、要求幅ごとに既存mapperが決めるq/R/実幅を使う。厳密1/5とbinary64 .2を区別する。

A/C metadata import graphは固定SOURCEから無変更で再利用した（REUSED_SOURCE_FILES.json）。新しい科学関数を作り直さず、既存DF/Pauli driver・audit・価格適用・V2 shot/Q3経路を接続する。係数の他LD/実幅への転用、部分26軸の全面適用、人工結果の本番catalog登録を行わない。
