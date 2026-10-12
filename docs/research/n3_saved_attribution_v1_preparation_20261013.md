# N3保存帰属v1：資料整理・実行認可前の準備

2026-10-13 JST。[完了済みpilot科学レビュー](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/review_intake.md)第12節を、一つのCodex作業として準備した。
**新しい数値的事後評価・PSD再認証の明示認可前で停止。新科学batch・再試行・新入力・精度変更は未認可。**

- [timer/出力義務/未保存field](n3_saved_attribution_v1_timer_and_output_map_20261013.md)
- [研究主張・観測可能性](n3_saved_attribution_v1_claim_and_observability_20261013.md)
- [新数値事後評価の限定契約](n3_saved_attribution_v1_posthoc_contract_20261013.md)
- [保存literal索引](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/saved_literal_index.json)、[18 inaccurateの元pointer](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/inaccurate_event_pointers.json)
- [既存SOS/OCC/他modeの小型証拠接続](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/SOS_OCC_existing_proof_connection.json)
- [module](../../src/trottertracks/representation_exploration/n3_saved_attribution_v1.py)、[gate](../../src/trottertracks/representation_exploration/n3_saved_attribution_execution_v1.py)、[runner](../../scripts/run_n3_saved_attribution_v1.py)、[synthetic tests](../../tests/test_n3_saved_attribution_v1.py)
- [固定plan](n3_saved_attribution_v1_execution_plan_20261013.json)、[source freeze](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/source_freeze.json)、[未認可template](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/authorization_template_NOT_AUTHORIZED.json)、[STOP](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/MANDATORY_STOP.json)、[preflight](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/preflight.json)

## 最終freeze

source commit：`a7a9f42d53d5efaca8715bee26edd9f1a0bad999`。新source4＋共有stdlib依存5の9 blobをcommitとSHA照合。
source fingerprint：`2ef6f58a0a3ac560b2e1aa2212dc7ee44a53bbb2cf7babf170b4dc11574d2d46`。
保存registry SHA：`d23d165ffebf1bb5bdfa542841cd678becc78dcae9b9e36bcd19733cd869023c`。
plan SHA：`a5f3fd39e47372e277365d9873f234cbd882ae1a84b6ca25a4f051e339f0c61f`。
Python3.12.3を指定し、source/40保存files/準備証拠のidentity preflight PASS。数値評価を含まない。
シェル既定3.10.12を記録した初期draftはpreflightで拒否され、未実行の[history](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/history/python310_plan_NOT_EXECUTABLE.json)へ保存した。
科学条件・source・原結果を変更せず、テスト済みruntimeへplanを訂正した。

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

N3保存帰属v1の[GitHub公開・再取得確認](n3_saved_attribution_v1_publication_20261013.md)を完了。明示posthoc認可待ち。
