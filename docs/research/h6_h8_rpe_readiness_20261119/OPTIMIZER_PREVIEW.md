# H₆ Full PFの通常探索入口への接続

[後続のH6専用経路](../../../scripts/h6_fixed_run/README.md) で、価格取得の高速化、方式別Q3、各環境での入力照合を確認できる。

固定Full19/M5／h=0.1／要求幅1.6／M=12のcanonical configurationは `4147c55d83984e21ea4a26586c561c5a2ca56ab5e99b86a97f154747592f3b59`。既存B/D mapperで発行し、全13ラウンドについて旧physical ID、T、qと一致することを確認した。Q3の基準は保存全DF19 solver値、目標精度は十進値 `0.00159360010199040 Ha`。Q3 workerは未接続。

## 実際に通った入口

同じGit checkoutで、保存H₆入力の場所とhashを指す元private configを読み、独立出力にメタデータを作る。

```bash
PYTHONPATH=src python3 -B scripts/prepare_h6_full_pf_optimizer.py \
  --saved-input-config "$H6_SAVED_INPUT_CONFIG" --output "$H6_PREVIEW/inputs"

PYTHONPATH=src python3 -B scripts/run_rpe_schedule_search.py \
  --problem "$H6_PREVIEW/inputs/PROBLEM.json" \
  --round "$H6_PREVIEW/inputs/ROUND.json" --run-dir "$H6_PREVIEW/run01" --plan-only
```

入力ファイルには保存H₆のsnapshot/receiptへのホスト内参照が含まれる。ローカル検収出力の `ROUND_RESULT.json` で、保存信号13、直接価格証拠8軸、m0〜m2・m7〜m12の不足18軸、`Q3_ROUTE_TEMPLATE_FOR_NEW_CONFIGURATION`、停止理由 `UNRESOLVED_SCALED_SCOPE`、Q3 0件を確認した。部分価格は通常registryが軸ごとに照合し、総額とQ3への投入は未確定のまま。従来の完全価格表は従来どおり扱い、同じphysical ID・軸の価格衝突を拒否する。

H₆部分価格、通常候補生成・選択、旧H₄の保存registry、倍率・Mの対象テスト20件が通過した。拡張回帰で確認した旧互換テスト2メソッドの3失敗（shot selectorの古いAST固定、既存owner探索の2経路）は基点 `97ec98419` でも再現した。旧runの実行履歴は変更していない。

## 短時間と長時間の実行状態

- 短時間のcallerは、固定 `73dcbe236` のH₆ compose、fingerprint、独立X/Y compileを再利用する。保存入力hashと専用runtimeのNumPy 1.26.4・Qiskit 1.3.0、m0〜m2のID、最大6軸を科学call前のpreflightで確認した。対象は `dec32b474f3fc33ddfab`、`7791db0dc2fab94ce734`、`e7200ae9e16ba879a78d`。実行開始はサンドボックス自動審査で停止した。明示的な実行許可の確認が理由で、compile 0件。元runを開き直さず、新しい出力へ取得する設計。
- 幅1.6の保存q=2/4について、固定receiptのfingerprintで元trialと既存キャッシュを照合した。価格指標とfingerprintの記録はあるが、該当するcompiled/wrapper QPYは、それらの場所には見つからなかった。元q=4のcallerも回路を保存する経路を持たない。現状ではH₄の閉包監査をそのまま適用できない。m7〜m12の12軸は未資格。追加取得の判断前に、別の固定locatorに同等回路があるかを差分照合する。
- 全26軸の価格とH₆専用Q3 template・target bindingが揃うまで、M=12・s=1の実Q3と再開の検収は保留する。新規信号・fit・費用計算・Q3は0件。

[保存証拠](COVERAGE.json) · [有限追加計画](NEXT_VALIDATION.md)
