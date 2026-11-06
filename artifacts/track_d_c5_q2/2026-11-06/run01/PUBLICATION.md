# C5成果物の公開範囲

公開JSONの科学check・位相・資源値は、既存checkerが保存した原本と一致する。`local/`に未加工の原本を保持し、公開viewでは実行時間・peak RSS・state-action thread観測欄を除いた。内部CPU/RAM観測・運用log・cache・環境絶対path・非公開raw入力・資格情報は公開しない。元原本は編集していない。

`circuit_payload.tar.gz`はq2のQPY14本・NPZ5本を元bytesのまま保存する。relative member path、owner/group/timeの中立化だけをtar containerへ適用し、QPY/NPZ自体を書き換えていない。`payload_archive.json`にarchiveと全memberのbytes/SHA256を記録する。

`prices.json`は固定q2/00→01/cut1のfinite_fixture_compiled record4件で、law期待値・モデル推定・RPE総費用の資格を持たない。旧C2/C3価格は再compileせず比較JSONに引用した。全結果の公開・remote照合後mandatory STOP。
