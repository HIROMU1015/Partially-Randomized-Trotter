# API入力・出力例

固定技術SOURCE `7d6a8c385fc057de012c7e1e603bc0bc358f313c`。

- `LD3/LD5_A_PREPARATION.json`：公開A7cf4応答から必要fieldをコピー。canonical ID/辞書/scheduleを変更していない。
- `LD3/LD5_SAVED_PRICE_PROJECTION.json`：公開PRICE_26AXISのexact値・軸SOURCEをfieldコピーしたfixture。新しいD2quote/価格再解析ではない。
- `LD3/LD5_REUSE.json`：実CLI出力。各26軸再利用、較正要求なし。
- `LD3/LD5_CAL_INSPECTION.json.gz`：`--inspect-saved-cell`の実CLI出力。元cohort/seed/cellを再現しreuse-only。
- `*.CALIBRATION_REQUEST.json`：実driverの`--request … --metadata`で読める完全request。
- `ARTIFICIAL_*`：未登録LD4の入力検査fixture。P0/SOURCE readerを人工データへ差し替えたtestであり、実H4用に取得・compileできるartifactではない。

圧縮は公開例だけ。通常のCLI応答はJSON。絶対path欄は生成時のA worktreeを示すため、別checkoutではCLIを呼んで現地pathを得る。
人工fixtureの再現は`tests/check_pauli_calibration_request.py`を使う。reader stubなしで実H4 SOURCEとして使用しない。
