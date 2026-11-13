# A → B：4方式screenの起動STOPと未評価データ

2026-11-13 JST。科学SOURCEは`256caad3414a80de02fb68eca89c13c957ed6354`。承認された20セルbatchを1回起動し、保存context構築の`h46_new2_core` importで`STOP_FAILED_NO_RETRY`。**mean 0/20・normalizer 0/6・新reference0、retryなし。** 詳細は[REPORT](REPORT.md)。

## Bが受け取るデータ

- [run01/result.json](run01/result.json)：固定実装が保存した原STOP。`records=[]`、エラーをそのまま保持。後から成功recordや値を追加しない。
- [run01/PLAN.json](run01/PLAN.json)：科学SOURCEの元PLANと同一。20 physical候補、40保存locator、m3/m6/m12のreference、H/state/snapshot/profile・preparation・lawを保持。
- [STOP_SUMMARY.json](STOP_SUMMARY.json)：20候補は`NOT_EVALUATED_AFTER_LOADER_STOP`。raw/corrected/bias/logB/attenuation、十分N/global RMSEはnull。新しい科学packetとしてPASSへ昇格しない。
- [EXECUTION.json](EXECUTION.json)／[EXECUTION.log](EXECUTION.log)：1回の起動と停止。計画上の20 workersを実際の20セル完了数と混同しない。
- [TECHNICAL_CHECKS.txt](TECHNICAL_CHECKS.txt)：5件の静的SOURCE・metadata・import探索確認。実分子の科学検証ではない。

保存40セルはFull PF12・B0I2・PR13・B013の元SOURCEに参照を残した。保存値の再解析・新比較・N再計算は0。旧PRの科学資格、m3/m4のnorm欠測、十分N/global RMSE未資格も変更していない。

今回、長時間bias・attenuation・数値安定性の新しい判断材料は得られていない。Bは方式のFAIL、候補除外、winner、全13round採択へ変換しない。新canonical IDを発行せず、旧physical IDとSOURCEを保持する。

## 起動条件の差分と今後

直接関数起動で`src/`のみをimport経路に設定したことが原因。既存loaderの`h6_df19_rpe_d2.make_context`先頭では`scripts/h46_new2_core.py`を参照する。`scripts/`も含めれば当該moduleは探索できると静的確認したが、科学計算は再開していない。固定science codeや保存データを修正する必要は、今回特定したimport不備にはない。

再実行をユーザーが明示承認した後の起動差分は次のとおり。**このコードは今回実行していない。**

```python
from pathlib import Path
import sys
root = Path.cwd()
sys.path[:0] = [str(root / "src"), str(root / "scripts")]
from trottertracks.h4_four_method_screen import execute
execute(root,
        root / "docs/research/h4_four_method_signal_screen_results_20261113/run02",
        "256caad3414a80de02fb68eca89c13c957ed6354")
```

使用Pythonは元のNumPy1.26.4環境。run01を保持してfresh出力を使い、新mean最大20・normalizer6・新reference0・保存40再計算0のscopeを変えない。別の依存不備や数値gate失敗があれば、その結果もSTOPとして保存し自動retryしない。

**ユーザー**が再実行可否を決定し、**A**が承認後のみ実行する。**B**はそのscreen結果取得後に全13round候補の判断材料へ接続する。C/Dへの新価格作業はない。今回の停止成果公開・照合後、Aはmandatory STOP。
