# 案Eの技術的選別と次の境界

2026-11-08。固定設計081959402f54f06048adb038f97911950091ae1cに対する、許可された人工実装・correctness・baseline比較。重要科学レビュー・研究方針の正式採択ではない。

## 判定

**TECHNICALLY_VALID_KNOWN_LCU_APPLICATION**

人工1/2 system qubit・2-labelのPREP/SELECT/逆PREP、flag→successだけのsuffix、失敗0、全attempt decoderを接続。新規14＋関連2テストPASS。平均保存/M2/flag zeroを確認し、PREP誤差で条件が壊れる境界も確認した。

標準positive LCUに同じearly flagを認めると操作列・推定量・分岐費用は同じ。**独自の増分利益0**で、TECHNICALLY_VALID_WITH_ADDITIONAL_STRUCTUREとは分類しない。同じ制御やbaselineを除外して利益を作らない。direct/grouping・理想層別化が人工例で勝つ対照も保存した。

## 次の扱い

案Eは**既知法の動的実装・人工correctnessの成果として一区切りにすることを技術提案**する。人工PASSだけを理由に実H4のpool探索・科学SOURCE・shot契約を発行せず、大きいbenchmarkを自動提案しない。O2全体の終了、他のLCUの一般NO-GO、正式科学的GO/NO-GOは採択しない。

追加検証に意味が生じるのは、同じ強いbaselineでも利用できない具体的アクセス差、または明示的な同target/full-wrapper費用差が別途特定された場合。今回その構造を特定していない。Mの232単独群のE poolへの再分割、D救済、O1/O3・C-Sへの移行を次の自動手順にしない。

## 将来もし科学検証を検討する場合の必要情報

科学契約の発行・採択ではなく、現在UNKNOWNの障害を列挙する。

| 項目 | 実行前に必要な内容 |
|---|---|
| PR入力・target | 同じ保存Hamiltonian/geometry/basis/DF rank/L_D/δ、round/r/K、actual finite Q、selected full policyとphase |
| pool | 正supportを欠かさない事前選択、exact conditional p、共通A/Bとsystem-only W_j、古典生成費 |
| coherentアクセス | 実二重制御、全phase、PREP実効pと逆PREP整合性、gate/readout/bit-grid・省略biasの新一様誤差 |
| 比較 | 同じearly LCU、通常PR、強い層別/direct/grouping。同じε/α |
| 全attempt価格 | 状態消費、PREP/SELECT/flag/feedback、成功suffix、失敗prefix、校正/学習/古典/cache費。未知項を0にしない |
| 有限scope・STOP | input/SOURCE/output、pool/call/attempt数、retry、精度・law・phase・価格・非干渉不整合時STOP |

旧N1/N2/engineの再計算を要求するものではないが、新instrumentの誤差保証は旧engine上界を継承できない。ε/α・seed・shot数・poolを新たに採択していない。

実H4新計算、保存科学値の数値再解析/再集計、実state/Born、RNG/shot、実回路build/compile/native synthesis、旧STOP再開、重要科学レビューには、それぞれ対象・有限scopeを明示した別途許可が必要。今回は人工検証と公開でmandatory STOPし、これらを起動しない。
