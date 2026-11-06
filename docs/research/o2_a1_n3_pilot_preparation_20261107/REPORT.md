# 現行N3の正規停止・保全と752-shot実行準備

2026-11-07 JST、SERVER1 GPUA100-gleap-01。
**正規停止・保全に成功。752-shot driver/serializerと人工検証を準備し、新pilot科学計算は0件のままSTOPする。**

## 現行N3の最終状態

[停止・保全結果](STOP_AND_PRESERVATION.md)：科学SOURCE ee49b53fd6f9a17d715edb1df3aac8cbf5731d7aの親へ、identity/UID/SOURCE/STOPを照合してSIGTERMを一回送った。
STOP要求→worker回収→既存finalizerの順に完了。親exit1は操作STOPのsemanticな終了、125workerは全exit0、finalization_errorsなし。強制kill・group停止・自動再起動は0。

保存H4/1.00 Å/STO-3G/legacy DF12、8system qubits/全4電子sector、L_D3/m0/q1/δ=.02/r1/K2/atol0、X/Y、N1 actual Q/selected fullとN2保証ground・認証済exact engineで、1374 trial開始、**1339 bit、1318価格**。
価格欠測21、bit未保存35をUNKNOWNとして保全した。既知RZは**6,435,609**、総RZはUNKNOWN。全方式精度/CV比較は未完了で、scope移行の操作停止を科学的失敗としない。

[stop artifact](../../../artifacts/o2_a1_n3_stop_and_pilot_preparation/2026-11-07/stop/)にSOURCE/STOP/全会計と完全archiveを保存。science/manager/operatorの1090ファイルについて、archive展開byte hashを全件検証した。cache WAL/SHMも保持し、元ファイルを変更・削除していない。

## 752-shotの実装・検証

[実装説明](PILOT_IMPLEMENTATION.md)：A1学習160、B1-OL probe80、独立評価512。既存3-control/全5正層/actual Q/selected policy/state/exact engineを維持するdriver・serializer・旧pipeline connectorを追加した。
既存の取得/bit先行保存/価格/fit/配分/集計/全attempt reconcileを再利用する。旧code/kernel/managerの原本は変更していない。
人工**34件PASS**。全752 mock、toyでの旧worker接続、provenance/独立性/quota/range、UNKNOWN価格、zero variance保留、fresh cache/SOURCE/一回入口、最小環境CLIと実行拒否を確認した。
`--show-plan`は候補設定の表示だけ。`--run`は実データ読込・RNG・state/Born・compile・出力作成より前に拒否する。今回は新pilotの科学測定・RNG生成・実state作用・Born・実compileは0。

## 未確定事項と次の承認

[採択待ち表](ADOPTION_PENDING.md)に、候補seed20261108、32domain/64子stream、配分proxy/floor、range guard、診断分類、guard-first順、fresh output/cache、最終activation/SOURCE/profileを明示した。
752-shotの準備枠・研究方向と、具体的な科学条件の採択・科学実行許可を区別する。公開したのは準備SOURCEであり、実行SOURCEの確定・実行開始ではない。
次に必要なのは、これら候補条件の採択と、**新しい一条件・A1/B1-OL×X/Y・一回752-shot予備検証の明示的実行承認**。停止許可は今回すでに消化しており、同じ停止を再要求しない。
予備診断は厳密精度/期待費用改善の証明にしない。Step 3の保証付き本検証へ自動移行しない。

## 保全・公開・STOP

元main/dirty/untracked、旧SOURCE/結果/STOP/grant/ledger/marker、C1凍結を維持した。現行新BのSOURCE/入力と旧科学codeはhash照合し、最終raw原本の不変を再確認する。
独立branch/worktree、独立候補output/cacheで準備し、CPU/RAM/workerの恣意上限は新設していない。共有設定・他ユーザーへ操作していない。
専用branchへcommit/pushし、remote SHA・公開本文・blob/tree照合後mandatory STOPする。新752-shotは別途実行承認まで開始しない。
