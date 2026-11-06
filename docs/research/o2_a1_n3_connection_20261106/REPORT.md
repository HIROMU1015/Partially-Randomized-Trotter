# A1 N3準備結果

2026-11-06 JST、SERVER1 GPUA100-gleap-01。**技術接続と科学契約草案を完成。実N3未許可・未実行、公開照合後mandatory STOP。**
対象は保存linear H4/1.00 Å/STO-3G/legacy DF12/8qubit/全4電子sector/L_D3、m0/q1/δ=.02/r1/K2/coefficient_atol0、X/Y。N1 actual Q/selected policy、N2 state・witness、認証済みexact engineをそのまま使う設計。

[技術接続・再利用・人工検証](N3_TECHNICAL_CONNECTION.md)と[科学実行契約草案](N3_EXECUTION_CONTRACT_DRAFT.md)を正本とする。[SOURCE・テストraw・STOP](../../../artifacts/o2_a1_new_target/2026-11-06/n3_connection_technical/)を保存した。

## 完了した技術範囲

Bit-only facade、actual53-bit直接条件付き取得、stage/axis/method/stratumとpath/bitの分離、3-control OLS凍結・rank/range guard、独立cal→外向きBernstein/Hoeffding→全正層有限配分→prod合成、同selected full列の既存compiler/cache価格、全段階RZ・古典費vectorを接続した。
既存kernel・数値library・solverの再実装や変更はない。科学実行CLIは今回付けず、人工専用入口だけを追加した。
新規22＋既存24＝**46人工/mockテストPASS**。2-mode artificial engine fixtureとmock価格の4方式接続検証であり、実科学結果・CV改善・性能/費用改善・外部再現/CIの認定ではない。

## Controlと推奨条件

N1引用のg2≈3.39e-6、g3≈1.89e-9を、1024 unconditional trainで十分学習できるとは置かない。rank不足はminimum-norm値で成功にせずSTOPする。
**Control集合は変更せず**、学習を5層×64 conditional bits/axisへ変える案を推奨する。intercept＋3controlのdesign被覆は得られるが、真のQでの最適係数や改善は保証しない。B1-OLを必須対照とする。
ε_X=ε_Y=1/8、同時α=1/20、cal512/cell/axis、prod≤8192/method/axis、D_CV≤4、直接条件付き取得、主資源compiled full RZ、実現campaign費用を提案する。**全て未採択**。旧草案の1/32も採択していない。

## 未解決事項と次の承認

必要なのは、草案の学習方法・quota・ε/α・word/独立性モデル・range/価格範囲の採択と、一条件4方式X/Yの有限実測・real-state/Born・必要なselected full build/compileへの明示許可。
採択後、薄いdriver/serializerを同APIで組み、実入力前にSOURCE/環境を固定する。現在の基盤だけで実入力へ自動進行しない。
実Fraction engineのruntime/memory、実λ/rank/variance・十分N、価格・費用、CV対B1-OLの優劣はUNKNOWN。期待費用保証は独立価格range/calの別設計・許可が必要で、実現費用差から代用しない。

## 実行境界・保全

実科学callは0：N1 preparation/CDF、N2 solve/spectral、engine誤差再認証、実state作用/Born、実RNG/shot、train/cal/prod、実full build/compile、variance/N/費用の新規科学評価は行っていない。
実inputへのアクセスは保存bytes/identity/sourceの静的照合のみ。計算付きの実bundle復元/ν再評価は行っていない。
元main・既存dirty/untracked、旧SOURCE/結果/STOP・grant/ledger/marker、C1凍結を保全する。[保全記録](PRESERVATION.json)を参照する。
専用branchでcommit/pushし、remote SHA・本文・blob/tree照合後に**mandatory STOP**する。
