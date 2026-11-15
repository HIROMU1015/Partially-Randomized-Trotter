# A共通PR-Pauli入口の通常accept_saved受入

Aの固定共通入口から生成したMEAN_HANDOFFを、通常の`rpe_pauli_signal_intake.accept_saved`で受け入れる。κ・ΔPF・L_Dごとの組立てcommitの列挙は不要。**追加5＋既存26＝31技術test PASS、実信号・compile・実H4 Q3各0件。**

## SOURCEと差分

- 実装・テスト：`bd42dd1e95eb1d4152216d62792ad3421b803f7c`、branch `codex/rpe-total-dp-acceleration-20261114`。
- A共通入口：`93a5ef3b9fed337cc4d26ee684defd304fa0eac0`。AのHANDOFFは`715af778833b8aaf84d26d467cf3f927771a84c9`の`docs/research/pauli_signal_entry_20261116/HANDOFF.md`。
- 前B SOURCE：`1c4a83d1b841bfc23ba87c45f850fb4233f22c30`。関連履歴の確認で同等の通常intake拡張は見つからなかった。Aで済んだbind_packetを再実装せず、その前の組立てSOURCE処理を拡張した。

新しいhandoffの出力commitで、`src/trottertracks/pauli_signal_entry.py`と`scripts/run_pauli_signal_entry.py`が固定A SOURCEとbyte一致することを確認する。packet内のassembly・辞書helper・configuration SOURCE・dictionary SOURCEを照合後、従来のbind_packetへ進む。configuration locatorのselectorにも対応した。Aのlocal-JSONから保存同一identityを再利用する場合のconfiguration_source=nullも、packet内のcanonical配置・元carrier照合を保ったまま受け入れる。数値SOURCEは組立てSOURCEと区別し、各rowのGit carrierから読む。

C接続adapterの辞書SOURCE列挙も除去した。辞書自身の同じdirectoryにresult.jsonがある場合、Aの固定`8c8ce542e` helperでその残余locatorを再現する。従来のglobal-P0記述子方式は保全する。Cのplanner、価格guard、decoder、DP、Q3 kernelの本体は変更していない。

## 検証

| 対象 | 通常accept_saved → C.signal_input / provider binding |
|---|---|
| 保存LD1・LD2（714247619） | 共通入口packet関数で再組立て、従来経路と一致 |
| 保存LD3（cdbce383f） | 同上 |
| 保存LD5（ac46d100e） | 同上 |
| κ×3/2の人工13round | PASS |
| ΔPF=binary64 0.2の人工13round | PASS |
| 新辞書locator・L_D=4の人工13round | PASS、prefix別登録なし |

保存比較ではconfiguration、13 meanのraw/normalization/uncertainty/provider_binding、辞書identity、targetとそのSOURCE、Cが消費するmeansが一致した。組立てSOURCEと全行reuseの記録は新しい組立てを表す。保存値への新しい確率・MSE算術は行っていない。

不正なassembly code/helper/configuration locator/candidate/law/数値SOURCE/carrier payload/dictionary SOURCE/means-only状態を拒否。原packetは不変。新しい人工carrierと辞書はtemporary Git内のみで、元object storeを読取専用のalternateとして利用し、終了時に削除する。実信号のregistry/catalogへ登録していない。

[TESTS.txt](TESTS.txt)が31件のログ。追加のinline配置互換性を含めた5件の再検査もPASS（[INLINE_CONFIGURATION_TESTS.txt](INLINE_CONFIGURATION_TESTS.txt)）。runtimeは既存科学runnerと同じPython 3.11.0rc1、追加テストは`tests/check_pauli_signal_entry_intake.py`。新しいintake検証ではMC/Q3/二項確率/cost/selectの科学入口を禁止している。既存回帰の人工workerは人工signalのみで動作する。

## 残る境界

- **signal受入とC価格bridgeは別。** 厳密ΔPF=1/5の人工例は通常のsignal-input投影まで受入可能だが、Cのexactな実step幅照合ではcanonical有理数とA physical candidateのbinary64が一致せず拒否される。Cのこのguardを緩めず、正常な価格joinとして表示しない。HANDOFFにsignal-only呼出しとUNKNOWNを記載した。
- 高速DPの固定AST照合はPython 3.12.3では一致しなかった。保存環境3.11.0rc1では一致し、最終31件PASS。今回kernelや固定AST値を修正していない。
- UNQUALIFIED、target proxy未採択、ground/sampler未認証を保持。sufficient_N/global_RMSE_bound/G/winnerはnull。新しいQ3や価格の科学的資格は付与しない。

B2が使う固定入口・locatorは[HANDOFF.md](HANDOFF.md)と[INTAKE_LOCATOR.json](INTAKE_LOCATOR.json)。ロードマップ本体は変更せず、PR-Pauliの共通生成packetを通常受入できる技術差分として引き渡す。固定raw・旧結果・旧STOPは保全し、新しい科学実行へ進まない。
