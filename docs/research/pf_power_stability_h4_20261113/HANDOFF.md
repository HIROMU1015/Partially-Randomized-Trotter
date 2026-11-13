# A → B：F2-COARSE/m12の二経路診断

2026-11-13 JST。[REPORT](REPORT.md)／[原結果](run01/result.json)／[要約](SUMMARY.json)／[再開案](RESTART_PROPOSAL.md)。同一physical candidate `4f0aca817edccc7fbe17`について最大2 meanを実行済み。新reference・normalizer・他セル・N/RMSE/C/Gは0。

SOURCEはrecipe256caad34、helper dd24dd329、起動入口ade6438f8（全SHAはSCOPE）。旧screen run01/run02をbyteで照合し、上書きしていない。

## Bが保持するべき資格

- 元spectral：state norm誤差4.199174341579237e−11は通過、power unitarity誤差1.0071230339680433e−9は固定1e−9 gate失敗。complex meanは失敗機構を比較する診断値であり、元screenに有資格値として登録しない。
- direct expm：state norm誤差1.1396439347777232e−12、power unitarity5.899874925253737e−12で両gate通過。same formula/order/scalarであり、backend差の局所数値観測として参照できる。
- mean差4.313865617047318e−11、operator power差5.036635696065883e−10。新しい経路一致の科学的許容値は採択していない。
- direct expmに固有vectorの直交性は非該当。元11blockのspectral/Gram診断を共有入力証拠として保持し、directにも新eigensolveを行ったように書き換えない。
- directの保存referenceからのsignal biasも大きい。numerical gate通過と物理PFの精度、energy RMSEは別。十分N/global RMSE/winnerは未資格/null。

`run01/ARCHIVED_SPECTRAL.json`と`DIRECT_EXPM.json`は候補・binding・SOURCEと別のbackendを保持する。`OPERATORS.npz`に各macro-step/powerをrawで保存し、`SPECTRAL.json`と`BLOCK_FACTORS.json`に原因診断を保持した。旧run02の欠測欄へ値を事後挿入しない。元screen packetのallowanceやnominal Nを今回新規生成していない。

## 推奨と未実行の次段階

**A**は、別途承認後の残15セル再開にPF/B0Iの直接expm＋matrix_powerを推奨する。signed piece/negative time/order/scalar/gate1e−9を維持し、lawや物理候補の変更は行わない。**B**は元4＋新direct1点のqualification差を保持し、15未完＝Full PF7・B0I2・Full-H RTE6として計画へ接続する。新direct点のpacket/allowance bridgeや保存値再解析を含めるなら、対象を明示した承認scopeにする。

Full-H RTE6には今回PFのexpmを使わず、有限K polynomial・normalizer・独立raw/correctedの元経路を保持する。この一点の成功でFull-Hのbasis誤差や長時間の数値精度まで解決したとは扱わない。追加gate失敗は別の欠測として保存し、fallback/閾値緩和/自動retryをしない。

**ユーザー**が再開の数値方式と有限scopeを承認した後だけ、**A**がfresh出力・新固定SOURCEで実行する。今回C/Dへの新価格作業はなく、Bのdecoderを重複実装しない。成果公開・照合後mandatory STOP。
