# B・B2・A・価格担当向け：厳密1/5のC接続

結論：厳密ΔPF=1/5をbinary64 0.2へ丸めて同一視しない。保存LD3/LD5のmetadataでは、厳密1/5のm0/m1は元Δ=0.1の保存recipeと一致する。m2～m12はAの現行providerのexact実step条件で未対応。SOURCEに基づく接続条件を満たすまでUNKNOWNを保持する。

- 結果：[REPORT.md](REPORT.md)、[ROUND_BINDINGS.json](ROUND_BINDINGS.json)（LD3/LD5 × 厳密1/5/binary64 0.2 × 13round、52行）。元configuration・candidate・saved mean locator・A修正版の判定を保持。新価格recordは0件。
- コード：`src/trottertracks/rpe_optimization_pipeline.py`の`pauli_round_binding(B, config, row, physical_context)`。既存と同じexact検査からround別のdifferencesを返す。通常`plan`の例外はm/q/field付きになる。`same_S2_lowering_operands=true`を接続資格として使用しない。
- 旧Bの接続：`ff9d25d62`のB interfaceはCの`c628e9e84`を固定読取しているため、この公開だけではBのpinは変わらない。Cの新helperを参照する場合も、Bの既存dictionary/SOURCE検査は保つ。Bのbranch/worktreeやCの旧固定SOURCEを変更していない。
- 安全なroundでは既存のC recipe checkが通る。m0/m1を含め、異なるcanonical IDの価格を付け替えるときは元receiptへの明示的な同一性bridgeが別に必要。whole-schedule planはm2で停止し、部分Gへの自動進行はない。
- Aの不足：厳密1/5のm2～m12について、T/qが非binary64-exactのまま評価できるSOURCEとidentity規約が必要。Aの`4e5b8bef6`はこの問題を解決済みにせずUNSUPPORTEDへ修正した。binary64 0.2は別の候補であり代替にはならない。
- 価格担当の不足：各候補自身のq/R/実step/辞書/law/piece/scalar/軸/control/wrapper/compilerに結び付いた証拠。旧Δ=0.1や別qのモデルを転記しない。

ロードマップ差分：厳密stepの価格UNKNOWNをround別に説明可能にした。m0/m1の同一recipeと、m1からの整数化差、m2～m12のprovider未対応を分離。価格・Q3・比較カタログの数値と資格は不変。本体更新は統合担当へ委ねる。

人工5＋既存34＝39 PASS。新規実signal/sampling/compile/fit/Q3=0。公開・remote照合後mandatory STOP。
