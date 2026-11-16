# HYBRID の先読みと同一構成内の振替

実装と人工検査は adaptive_search 担当サブエージェントによる。主担当が統合する前の作成者検査であり、独立した科学レビューではない。新規の信号・較正・compile・実Q3は0件。

## 採用した最小差分

`src/trottertracks/rpe_priority_search.py` の通常 `run_priority` へ、既定で無効の2設定を追加した。

```json
{
  "lookahead_mode": "HYBRID",
  "adaptive_hybrid": true,
  "allocation_exchange": {"enabled": true, "max_moves": 8, "add_steps": [1, 2]}
}
```

- `adaptive_hybrid`：独立した構成の先頭候補を先に並べる。同時枠が先読みの深さより大きい場合も、独立候補があれば全枠へ提案できる。残った枠にだけ連鎖候補を均等に追加する。完成済みの先読みが8件以上あれば、その採用率を使って連鎖分を縮める。稼働中のjobを未採用失敗として数えず、完了数の観測がないときは採用率を推定しない。完成したが今後採用されるjobもあるので、この比率は科学的予測ではなく投入量のheuristicである。
- 独立候補数・同時枠・残りQ3予算でpreviewを制限する。既定の挙動は維持する。起動済みのjobを止めず、最終的な起動予算は共通executorが予約時に制御する。
- `allocation_exchange`：高価なroundのshotを減らし、別の安価なroundへshotを加える。各roundの既存capを保ち、費用が厳密に下がる配分だけを費用順に生成する。MSEの単調性や適合性は仮定せず、新配分は通常のQ3判定を必要とする。既評価・保存logに答えのない配分を飛ばした後も、次の有限候補窓へ進める。
- 従来の `transfer_count=2` は「近いκの別構成から適合配分をコピーして種にする」機能である。同一構成内の2round間の振替とは異なるため、その実装を重複させていない。
- `selection_timing` に候補生成・分類、preview整列、backend回答待ちの経過秒と呼出し数を返す。CPU時間と同一視しない。

## 人工worker時刻モデル

`artificial/CHECKS.json`、再現は `python3 scripts/check_rpe_adaptive_throughput.py --output <owned-dir>`。既存の人工MSE fixtureを再利用し、実際のV2の候補生成・採用処理を通した。jobは一定10秒の時刻モデルであり、CPU時間は未測定。

| ready構成数 | 方式 | 起動件数 | 同じ採用配分数 | 全起動jobの完了までの模擬秒 | job所要時間の和（模擬秒） |
|---|---|---:|---:|---:|---:|
|1|従来HYBRID|88|32|250|880|
|1|適応HYBRID|59|32|240|590|
|12|従来HYBRID|563|120|910|5,630|
|12|適応HYBRID|128|120|1,080|1,280|

**先読みを絞ると計算量が減る一方で、到達時間が延びる場合もある。既定設定への一律採用や、実H4の高速化率の主張はしない。** 採用配分の順序は両例で一致した。

再開fixtureは完了OUTCOMEを再利用し、同じ配分の二重起動0件を確認した。観測counterを消した人工再開では、1構成例で新しい未採用先読みが8件増えた。完了結果の再利用だけで、先読み予算の再開挙動まで同一になるとは主張しない。通常executorの人工再開では、観測counterがinvocationごとにresetすること、完了結果の再利用と起動済み予算の維持を別に検査した。

振替fixtureは、純粋な減算がすべて不適合になる人工の局所最良費用896から、m12のX/Yを各1shot減らしてm0へ各1shot戻した費用698の適合点へ26番目の回答で到達した。これは振替の技術的有効例であり、現実のMSEの形や実H4でのQ3削減を証明しない。最大候補窓・cap・strict費用減少も検査する。

## 保存結果での比較

`saved_replay/SAVED_REPLAY.json`。round05の固定台帳4,783件からREQUEST identityだけをindexにし、V2が配分を選んだ後にのみexactなOUTCOMEを開く。台帳の保存結果hashとの一致も確認する。初期のC保存適合点、固定したready構成26件、比較予算400件、32枠を共通にした。外側の広域schedule選択と準備時間は、この比較から除く。

- 従来HYBRIDと適応HYBRIDは、同じ23件を選んだ後に同じ未知の配分で止まった。採用最安517,151,301.57 RZ、最後の採用までの時刻モデル751.06秒はいずれも同じ。
- 模擬起動は126件から69件へ減った。うち保存logに答えのない先読みは70件と19件であり、それらの判定は未評価のままにした。最後の起動jobまで待つ時刻モデルは878.18秒と891.57秒である。
- 振替ありは10件の回答の後、m12からm11への未保存振替を選んで止まった。最安541,743,905.57 RZ。予算400件を使い切る前に打ち切られた比較なので、同予算での優劣や実H4の振替の改善は確認できない。
- 400件まで継続した結果を補間しない。未知の配分を不適合と扱わず、logが以前の探索規則に偏る制約を残す。
- 23選択jobの保存CPU時間和は5,602.51秒。変更後の同時実行条件での総CPU時間ではない。未知先読みを含む総CPUはnullとする。

## 起動例

これは技術検証であり、科学実行用の承認を作らない。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_rpe_adaptive_throughput.py \
  --output /absolute/path/to/owned-artificial-checks
PYTHONDONTWRITEBYTECODE=1 python3 scripts/replay_rpe_adaptive_throughput.py \
  --baseline-worktree /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-rpe-split-preparation-20261116 \
  --output /absolute/path/to/owned-saved-replay
```

4方式に共通のV2入口に接続したが、今回保存結果の効果を比較できたのはPR-Pauliだけである。設定を有効にするだけで他方式にも同じ効果があるとは主張しない。
