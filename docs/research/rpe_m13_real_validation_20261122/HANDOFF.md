# H₄の実信号を使う14ラウンドQ3検収

**Full PF S2、B0I p7 S2、PR-DF LD7 S2の3候補で、実信号を使った14ラウンドQ3と保存回収が通過した。** 全候補でQ3は $m=13$ の状態遷移を処理し、MSE上界が同じ厳密な閾値以下だった。結果と出所は[SUMMARY.json](SUMMARY.json)に記録した。全候補で価格と総RZは未評価。

## 実行条件と結果

元実験のH₄直線1.00 Å、STO-3G、tol-only DF10、同じHamiltonian・状態・snapshotを使った。$h=0.1$、$T_{13}=819.2$。Full PF、B0I p7は要求幅 $16h/7$。PR-DFは $L_D=7$、幅 $h$、$K=2$、元の厳密な論文κを引き継いだ。各候補の先頭13信号を保存packetから再利用し、終端信号を各1 complex mean取得した。PR-DFは独立2経路で作った同じH₄・同じ時間の参照対をFull PF終端記録から再利用した。

配分は $s=1/2$ を選び、X/Y各軸とも `[32,30,28,26,24,22,20,18,16,14,12,10,8,6]` とした。通常Q3の計数上限に収まり、対象候補ごとに1配分の検収ができる。14個の位相半径は元の固定値1.0を使用した。固定Z1+T1、DF10 solver proxy、IID exact-affine、全事象平方損失、厳密閾値 `Fraction('0.00159360010199040')**2` を継承した。

| 候補 | 終端物理ID | $q$ | MSE下界 (Ha²) | MSE上界 (Ha²) | Q3 worker実時間 | 判定 |
|---|---|---:|---:|---:|---:|---|
| Full PF S2 | `d14c25536cc1c316c835` | 3584 | 7.48130343825625e−7 | 7.48130343835994e−7 | 95.18秒 | PASS |
| B0I p7 S2 | `103520c45f64903e989c` | 3584 | 7.61649457295157e−7 | 7.61649457305718e−7 | 93.88秒 | PASS |
| PR-DF LD7 S2 | `18495467bbe1b54f4c9c` | 8192 | 2.57254076045489e−7 | 2.57254076054115e−7 | 96.48秒 | PASS |

数値は表示丸め。分子・分母の正本は各 `OUTCOME.json`。全件で `VERIFIED_FINITE_BINARY64_SUPPORT`、`identity.M=13`、両軸14要素が一致した。$m=13$ の遷移数は各270,336。各 `COLLECTED_OUTCOME.json` で採用状態・上下界・全入力identity・元結果SHAが一致した。新規信号3件、保存信号39件、Q3実行3件、回収時の新規Q3は0件、回路build・compile・価格取得は各0件。

## 固定SOURCEと通常経路

- SOURCEは中央版 `7d3d1c2ba6145363f6d61b5eadc8163313f77a46` の通常 `run_rpe_optimizer_q3_job.py`。通常Q3は既存の固定PF/PR-DF workerと、中央版が読む `89fcde14f6ac5f4b97be234f9b0822091e7d9ac7` の14ラウンドrisk。各信号packet、configuration、binding ID、運用上の回収locatorはSUMMARYにある。
- PR-DFの旧intakeは入力行数を13に固定していたため、検収用の [run_pr_df_m13_validation_q3.py](../../../scripts/run_pr_df_m13_validation_q3.py) で、固定原本を照合した上でconfigure済みの行数を検査する形にした。原A-bundle読込み、carrier照合、固定worker、元AST guardを通過した。先頭13行は元のcarrierから同じ物理・数値データを移し、新配置のlocatorを検査し、旧locatorを明示した。PR-DFの最初のdry-checkでselector付き参照を旧A-bundle readerが解決できなかったため、独立のconfiguration blobへ結び直した。最初のdry-checkの時点で科学Q3開始は0件。
- Full PFとB0Iは [FULL_PF_Q3_REQUEST.json](FULL_PF_Q3_REQUEST.json) と [TRUNCATED_PF_Q3_REQUEST.json](TRUNCATED_PF_Q3_REQUEST.json) を `scripts/run_rpe_optimizer_q3_job.py` へ渡した。PR-DFは [PR_DF_Q3_REQUEST_V2.json](PR_DF_Q3_REQUEST_V2.json) を上記検収用wrapperへ渡した。固定Python 3.11.0rc1で各 `--mode dry-check`、`--mode execute`、`--mode collect` を順に実施した。`--output` は方式ごとの専用run directory。公開済みrun directoryは結果保管先として扱う。新しい実行を行う場合は別の実行範囲と出力先を定める。
- §10.5では、公開マップ・固定13ラウンドの旧結果・最新中央branch・server2の関連稼働を照合した。中央の公開済み実終端はYoshida4・物理ID `a660e98c78346c3fa922` だった。今回の3物理IDとの差を確認した。中央ホストの名前解決ができなかったため、中央の状態は固定公開記録で確認した。 [中央の終端検収](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0b9f7bf6a48a86b6638f4958d151c8d36ac99505/docs/research/rpe_m13_central_20261122/HANDOFF.md)

評価と回収はserver2の実adapterで実施した。中央への自動採用では、SUMMARYにある実結果の新configuration・binding・signal locatorを使って期待identityを作る。今回の専用CLIは新しいconfigurationを固定templateと出力で照合した。中央キューの期待identityは旧要求から流用せず、新しいlocatorを登録する段階で構成する。価格2軸は総RZ比較時にそろえる。関連する人工回帰4件がPASS。今回の有限検証は3候補・各1配分で終了。
