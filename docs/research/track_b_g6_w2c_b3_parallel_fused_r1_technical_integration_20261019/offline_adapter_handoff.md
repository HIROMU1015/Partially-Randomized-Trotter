# ローカルCodexへのparallel adapter引継ぎ

基準offline verifierはe4231b30919a4817f37d4b4d3c7a492a90fe4a7e。今回は変更・mergeしていない。
本公開SOURCE/schemaだけで実装できるよう、暗黙の内部仕様をoutput_schema/protocol.jsonへ固定した。

1. source_inventory.jsonとmeasurement_source_history.bundleで元の測定SOURCEと公開bytesを確認する。
2. output_schemaのcontract/runtime/result/pair/f01/receipt/envelopeとprotocol.jsonを読む。
3. measurements/benchmarkの各light/mixed/heavy-P/PF/S/SF-0-raw.tar.gzと対応contractを使う。
4. measurements/faultsのreference/fused各17障害raw、reports.jsonl.gzを使う。
5. fused-parent_crash-raw.tar.gz/observation/contractはsummary未保存・spent/unknownの例として使う。
6. artifact_verification.jsonとverify_artifacts.pyは今回の明示人工fileに対するreadback根拠。正式offline adapterの代替ではない。

Raw tarの内部はartificial-output/だけ。Pair別JSONL、control.jsonl、trial_states.bin、正常時summary.jsonlを含む。
full benchmark reportはreports.jsonl.gz、各run file hash/semantic digestはruns.json。
finalization後budget、host_outer CPU/wall、fake_claim_statusはhost reportへ含む。
公開contractはfake-onlyで、生成に使った一時root/authority fileは永続的な実行環境ではない。
再現時は専用fixtureが新しい/tmp root・fake ledger・fake承認を作る。実authorityへ転用しない。

Adapterで特に追加する検査は、global ID/offsetの一意性、3 barrier、全体予約budget、全stateと物理prefixの区別、
同一trialへの複数failure通知、receipt/summary/digestの結合、parent/worker費用範囲とunknown、未ACK creditの保守課金。
parent crashは最終reportを捏造せず、非terminal reserved slotのunknown解釈とspent維持を確認する。
Runtime rootは人工実測時の絶対位置だが、offline検査は保存されたidentity/pinsを照合するもので、旧server pathで再実行する必要はない。

再現（repository root、出力先は未存在を指定）：

```bash
python3 docs/research/track_b_g6_w2c_b3_parallel_fused_r1_technical_integration_20261019/run_tests.py /tmp/b3-v3-new-tests
python3 scripts/tracks/algorithm_codesign/run_b3_integrated_artificial.py benchmark --output /tmp/b3-v3-new-bench --repeats 5
python3 scripts/tracks/algorithm_codesign/run_b3_integrated_artificial.py faults --output /tmp/b3-v3-new-faults
```

**次の担当：ローカルCodex**が別worktreeでparallel adapterと受入試験を追加し、固定公開SOURCEへ接続する。
**ユーザー**はその技術結果を確認した後、parallel CPUの正式estimand、cooperative STOP/unknown保全、正式SOURCE採択・実行scopeを判断する。
サーバーCodexは本公開照合後にmandatory STOPし、science release/admission/scientific pilotへ自動移行しない。
