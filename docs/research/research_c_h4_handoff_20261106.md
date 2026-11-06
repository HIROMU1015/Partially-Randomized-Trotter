# 研究C：H4再配置下流計算の一回の高速化引継ぎ

2026-11-06 JST。ユーザーが今回の実際の引継ぎを明示許可した範囲。基点は高速化版 `6301b630569a9371db26ff511fde819c6fe97060`。旧科学SOURCE `a67036a6baafc00deaa38a1b0b801d313d519ad5`、元16入力、3候補、r8/16/32/64/128、selection32/final64、統計・correctness・会計・最終mandatory STOPは維持する。H2/H6等には進まない。

[契約](research_c_h4_handoff_contract_20261106.json) ／ [SOURCE](../../artifacts/research_c_h4_handoff/2026-11-06/technical/SOURCE.json) ／ [checkpoint](../../artifacts/research_c_h4_handoff/2026-11-06/technical/checkpoint.json) ／ [人工検証](../../artifacts/research_c_h4_handoff/2026-11-06/technical/verification.json)。本報告のこの段階では科学的な比較結果を新たに主張しない。

## 保全と切替

旧process PID2377879へSIGINTを一度送ったが、ThreadPoolの終了処理が残りの直列compileを待った。そこで、検証済み標本がまだ0の処理中E1/r128 batchを保存したまま、同じprocessだけへSIGTERMを送り、実際の終了コード143とprocess消滅を確認した。他process・共有環境は変更していない。

旧wrapperのfinallyはSIGTERMでは実行されず、旧STOP/completionは作られなかった。その欠落状態を維持し、旧STOPや旧PASSを捏造しない。[別の中断観測記録](../../artifacts/research_c_h4_handoff/2026-11-06/execution/termination.json)に予定した引継ぎのための終了であることを記録した。既存のSOURCE・契約・結果・過去のSTOPを上書きしない。これは元実験の一回の予定した実装切替であり、新候補や独立した第二実験ではない。

Pの全5条件とE1のr8/16/32/64の9 batch、合計288の検証済み標本を再利用する。各標本のevents、raw/compiled/実raw逆/実compiled逆QPY、reference、raw/compiled/adjoint/XY検査、費用6指標が揃い、旧SOURCE・候補・r・sampling namespaceと一致することを確認した。旧出力3608ファイルをcheckpointに固定した。未検証のE1/r128の生成物も保全するが、完了価格標本として扱わない。

## 計算の継続方法

- コードのSOURCEは新しい64ファイル。乱数を決めるSOURCEは元の `bb102f16c3bf757853f9e9fc98d811dc230592d71cd6bad57bcce0363fa2046e` を維持する。master、candidate、cohort、split、r、batch、sample indexは不変。
- 完了cohortのresult・calibration・会計はそのまま再利用し、完了標本を再実行しない。原本とは独立したコピーを新出力へ置く。
- 未完了標本だけを同じindex/seedで独立spawn processに投入する。未検証の古い途中ファイルをPASSへ読み替えず、新規の全512 raw/compiled/actual inverses/return/XY検査を行う。新しい観測数を追加したり、データに応じて標本を入れ替えたりしない。
- 保存済み13roundとPF誤差を再利用する。保存されていなかった36×36 PF行列cacheだけを、復元候補ごとに一度再構成する。Dが完了済みの場合は保存ground stateをそのまま復元する。全固有基底のないground-only復元を未完了Dの全operator誤差計算へ流用することは拒否する。
- 同J/U Dの完了結果は再利用し、未完了候補のDだけを既存経路で実行する。全候補のselectionが揃ってから選択を固定し、登録された独立final64のみを実行する。
- 新出力は `artifacts/research_c_h4_handoff/2026-11-06/science/run01`。一回の引継ぎだけを許容し、新runner内のretry/resumeは0。入力/SOURCE/checkpoint/科学条件不一致、correctness不一致、実行失敗では部分結果を保存してSTOPする。

## 人工検証

129 unique tests PASS（新しい引継ぎ7件、従来122件）。完了32件では数値task投入0、5件完了・27件欠落では27件だけ実行し、全32件の費用と旧乱数列が一致することを検証した。元domainの相違、完了標本の必要ファイル欠落、科学条件の変更、未停止process、非予定のSTOP、回収未完了を拒否する。保存ground・reference13復元、全固有基底の必要性、SIGINT時のThreadPool回収、固定32標本・全512検査・既存adapter互換も確認した。人工PASSを実分子での性能・正しさの結論へ格上げしない。主agent自身の検証であり、独立レビュー・サブagent使用はない。

## 科学結果

実行・終了・結果の状態は、公開SOURCEで一度起動した後のstart/completion、science/result、STOPとともに追記する。旧runは中断済みで科学的完了ではない。新runnerのcall countsは継続process内の集計であり、旧processの未保存call countsを架空に補わない。最終結果では再利用数・新計算数と来歴を区別する。科学条件の追加・H2/H6・再試行には自動進行せず、結果公開後にMANDATORY STOPする。
