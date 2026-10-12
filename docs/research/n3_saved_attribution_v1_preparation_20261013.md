# N3保存帰属v1：資料整理・実行認可前の準備

2026-10-13 JST。[完了済みpilot科学レビュー](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/review_intake.md)第12節を、一つのCodex作業として準備した。
**新しい数値的事後評価・PSD再認証の明示認可前で停止。新科学batch・再試行・新入力・精度変更は未認可。**

- [timer/出力義務/未保存field](n3_saved_attribution_v1_timer_and_output_map_20261013.md)
- [研究主張・観測可能性](n3_saved_attribution_v1_claim_and_observability_20261013.md)
- [新数値事後評価の限定契約](n3_saved_attribution_v1_posthoc_contract_20261013.md)
- [保存literal索引](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/saved_literal_index.json)、[18 inaccurateの元pointer](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/inaccurate_event_pointers.json)
- [既存SOS/OCC/他modeの小型証拠接続](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/SOS_OCC_existing_proof_connection.json)
- [module](../../src/trottertracks/representation_exploration/n3_saved_attribution_v1.py)、[gate](../../src/trottertracks/representation_exploration/n3_saved_attribution_execution_v1.py)、[runner](../../scripts/run_n3_saved_attribution_v1.py)、[synthetic tests](../../tests/test_n3_saved_attribution_v1.py)

## 完了したこと

元40source/result filesのSHA registry、216 path rows・432 events・696候補のliteral/pointer抽出、source timerの静的対応を保存した。
18件はSOSのV2_banded_6/external5、全3tier×両層×各3反復。同値SDPの保存eventにinaccurateはない。
raw proposal、修復前matrix/objective、repair r、solver residual/gapは未保存と確認。原因を後から数値で作り足さない。
既存の3証拠は値を変えず約53KBにまとめ、元file SHA/pointer・既存独立監査へ接続した。登録証拠の再認証はしていない。
synthetic23 PASS、包括的な登録raw read拒否guard、実2spawn/AS1GiBとserialの合成算術一致。
元run1/source/input/plan/消費済み認可/STOPは不変。local技術検証でありCI・外部科学再現ではない。

## 次の明示認可

レビュー第12.6節の「保存結果の新たな数値的事後評価、証拠の再認証等を実行する場合は、個別の運用指示に応じた明示的な作業・実行承認を得る」に従う。
固定source/input registry/planに対する**保存データ事後評価一回**の新しい認可が必要。
scopeはphase時間整理、保存q/box/SOS残差のexact算術、既存3証拠の再認証まで。新solver/固有値/証拠探索/元batch再実行は0。
通常source/test/公開のたびに同内容の詳細GPTレビューを要求しない。
その新事後結果を整理・公開した後、同品質/同機能取得の研究価値、task、継続/縮小判断をGPTへ戻す。
Mandatory STOP、N1停止、N2保留、旧primary/gateと中心テーマnullを維持する。既存global manifest description欠落FAILは別に残る。
