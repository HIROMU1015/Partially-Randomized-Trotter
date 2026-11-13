# P2直接paired pilot：実行引継ぎ

[REPORT](REPORT.md)／[PLAN](PLAN.json)／[TESTS](TESTS.txt)。

- runner：`scripts/run_p2_direct_pilot.py`。
- helper：`src/trottertracks/p2_direct_pilot.py`。
- 既存`compiled_price_fast.py`と新ZZ loweringは基点5a67cea7から変更しない。
- 人工テスト：`tests/test_p2_direct_pilot.py`の7項目と既存pipeline9項目、16 PASS。
- 実行はreplay一致gate→freshの1回。出力directoryは新規限定、retry/resumeなし。
- public予定：本directoryの`run01/`。
- private予定：`/home/AbeHiromu/projects/d-p2-direct-pilot-20261113-private/run01/`。
- 価格fingerprintはfull-wrapperのpre-transpile canonical fingerprint。
- 基点P1結果・SOURCE/PLAN/raw/KEEP/grant/ledger/marker/STOPは保全。

結果確定後に、実H4性能・fresh統計・次の有限実行案を追記する。
完了時はcommit/push・remote SHA/tree/主要成果物を照合してmandatory STOP。
