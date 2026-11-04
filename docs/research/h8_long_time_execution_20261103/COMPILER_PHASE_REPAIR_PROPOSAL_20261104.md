# H8 compiler位相会計の修正案

**提案のみ。補正コードは未作成、補正を使う科学再検証は未実行。**
2026-11-04。元の失敗とHOLDは保存する。稼働中のS2代表2件は変更せず継続する。

## 診断で分かったこと

Linear H8、隣接1.00 Å、STO-3G、保存tol-only1e-8 DF28、元state・fragment順序0..27、T=1.6、ε=.0005、P=0。
Stage A SOURCEは`d1272016592ff06c0567862f770ed796fd61b366`、元native SOURCEは`df93fbb88761b61c2691bf01f761ed20a7987d7f`。

失敗したm5 PR prefix8/q1/R_outer32/K4の探索index0では、保存wrapperと同条件の再compile結果は命令・parameter・global phaseが一致した。
元wrapperに対する状態vector差は約2、内積は約−1。保存からの復元による違いでは説明できなかった。
compilerのHighLevelSynthesis直後では差0、最初のBasisTranslator直後では約2となった。

使用中の固定compiler（Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/topologyなし）を小ゲートで調べたところ、
実際に使ったXX+YYの2494 parameter組で、変換後の作用は元の作用の **i倍** だった。
この余分な位相を解析的な比較対象に含め、他の非basisゲートも検査した計7677小ゲートの最大行列差は3.83e-14で、失敗は0。
これは主担当による実行範囲内の数値診断であり、一般の全compiler入力に対する保証ではない。

対象wrapperのXX+YYは52262個で、`i^52262 = -1`。観測した符号反転と一致する。
[匿名化した診断metadata](../../../artifacts/resource_applicability/track_a_h8_long_time/2026-11-04/compiler_phase_diagnostic.json)に値と来歴hashを記す。元seed、event列、gate parameter一覧、QPY、入力rawはprivateに保全する。

## 許可を求める具体的な変更

ゲート単体の作用から確認した余分な位相を会計し、コンパイル後のfull Hadamard wrapperのglobal phaseへ
`Δφ = −N_XX+YY π/2 (mod 2π)` を加える修正を実装する。
現在失敗している代表では`Δφ=π`。状態vectorの内積を使って位相を合わせる処理は導入しない。

局所operator検査で確認できたgate family・parameterだけへ適用し、確認できない場合はHOLDする。
元の入力、signal、N_cos/N_sin、cohort、n、candidate/index、元seed/event列、compiler設定、RZ価格を維持する。
gate列・命令数・RZ数を変えず、許容誤差1e-8/1e-7も緩めない。
共有のQiskit installation、OS、サービス設定は変更しない。

対象は既存Stage Bで残っている次の3代表のうち、この位相差により再検証が必要なものだけとする。

| candidate | 公式 | prefix | q | index | 現在の状態 |
|---|---|---:|---:|---:|---|
| 7eb0a3d691b94089969b | m5 | 8 | 1 | 0 | 元のcompiled vector検査失敗・HOLD保全 |
| bbd630c81e1f1845b068 | S2 | 7 | 32 | 0 | 元条件の計算を継続中 |
| 3bce8dc9530cddf679db | S2 | 8 | 32 | 0 | 元条件の計算を継続中 |

全て元の探索cohort・R_outer32/K4の同じ標本であり、新しい候補・seed・標本を追加しない。
既に完了した48標本は保全して再利用し、稼働中の2件を補正導入のために停止しない。

許可後、修正コードと人工operatorテストを作成して専用branchへ公開し、補助SOURCEを固定する。
その上で必要な代表だけについて、元と同じ7項目の制御・algebraic event・Hadamard・compiled vector検査を行う。
元の失敗・HOLDと補正後の結果を別の来歴で保存し、結果を混同しない。
全51標本が必要な検査を満たした後にpaired Gを集計する。Stage C、追加標本、top-up、旧runのSTOP解除は含まない。

## 現在の制約と次の担当

自動承認レビューは、この補正条件で失敗代表を再検証するための実行コード追加を、
元の固定条件からの科学的範囲変更で追加の明示許可が未確認として却下した。
そのため、修正実装・人工検証・上記範囲の科学再検証についてユーザーの明示許可を求める。
この文書は許可の証明・記録ではなく、これから行う変更の説明である。

次の担当：ユーザーがこの修正範囲を判断する。Codexは稼働中2件を継続し、許可後に修正・人工検証・補助SOURCE公開・必要な代表再検証を行う。

## 追加修正の実装（11月4日、上記提案後）

上記は提案時点の記録。現在、S2 prefix7代表は元条件でPASSし、完了済みは49/51件。残るm5 prefix8/q1とS2 prefix8/q32のindex0を補助SOURCEで再検証する。
[位相会計](../../../scripts/h8_compiler_phase_bookkeeping.py)は、実際のwrapperにある全非basisゲートのfamily・parameterを小operatorで検査し、検査済みの余分な位相だけをfull compiled wrapperへ加える。
[再検証入口](../../../scripts/repair_h8_compiler_phase.py)は元runtime・元seed/eventを再利用し、新しい保存先に7検査のベクトルと補正来歴を記す。既存のHOLDを上書きしない。
保存された元条件のcheckpointはSOURCE・入力・initial vector・保存vector hashを照合して再利用する。コンパイル後の保存vectorを再利用する場合も、局所operatorから決まったglobal phaseを解析的に掛けるだけで、状態の内積から位相を推定しない。
修正後の価格は元RZ数と一致させる。集計には元native SOURCEと補助SOURCEを併記し、過去の実行記録を残した別の結果文書を出力する。
人工3qubit回路で元の7検査と来歴保存までを通す検査を含め、関連34テストPASS。これは人工検証であり、H8の補正後PASSを意味しない。
補助SOURCEを専用branchへ公開・固定してから、残る2件だけを並列実行する。科学条件・標本追加・Stage Cの範囲は変更しない。

## 最初の補助実行後の技術修正

最初の補助SOURCE `f04ef60c960fd2a68bcf0506f4c505f0bc872b78`では、m5のcosine検査が7.80e-12で通過した。その後のsineのS†未登録と、S2の復旧QPY参照先不足によりHOLDとなった。これらのHOLDを保存し、同じ2代表に限って技術修正を行う。
S†の局所operator検査も実施し、検査を省略しない。元run・既知の復旧先・最初の補助実行のcacheを探索対象とし、元と同じfingerprint・compiler・RZ価格を要求する。元のnative wrapperの構造を検査してevolutionを復元し、既存builderでcosine wrapperを再構築したfingerprintが完全一致する場合だけ再利用する。
保存済みの補正後vectorを使うときは補正回路の命令・qubit/clbit・parameter・global phaseの一致を照合し、既に会計した位相を再度加えない。未補正の元vectorには局所operator由来の位相だけを一回加える。人工3qubitのS†・7検査・全checkpoint再利用・位相二重加算防止など36テストが通過した。科学的条件の追加、許容誤差緩和、標本追加、Stage Cは含まない。
