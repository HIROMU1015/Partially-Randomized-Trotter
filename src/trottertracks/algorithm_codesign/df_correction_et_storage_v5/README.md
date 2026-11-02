# E/T v5：lossless共有保存

v4出力guard失敗後の技術的storage-only derivative。
[準備・validation報告](../../../../docs/tracks/algorithm_codesign/df_correction_et_storage_v5_preparation_and_handoff_20261012.md)
と[contract](../../../../artifacts/track_b_df_correction_et_storage_preparation/2026-10-12/v5/contract_v5.json)を参照。
登録E/T実行には別の明示one-shot authorizationが必要である。

`native_evaluation.py`、`saved_data.py`、`accounting.py`はv4とbyte同一。
storage、version-specific launch、relative guard readerだけを新namespaceに置く。
旧shared API・結果・sourceは書き換えない。

## GitHubからの取得・復元

root `.json`は小さいpointerであり、scientific record本体ではない。
そのsnapshotと`previous_snapshot` chain、typed treeの全blob、blob indexの全chunkを
result namespace相対pathのまま取得する。すべての新physical fileは512 KiB以下。
chunkはUTF-8であり、index順に結合したcompact blobのbytes/SHAが一致する。
raw blobもJSON。snapshotの`d/l/s/b`タグはdict/list/scalar/blobを表し、
元のscientific dictに同名keyがあっても解釈を混同しない。

固定sourceのreaderは全履歴の閉じた参照graphとcurrent logical JSON identityを確認する。
数値moduleはimportしない。既存legacy root/index/chunkもread-only fallbackで読める。

```python
from pathlib import Path
from trottertracks.algorithm_codesign.df_correction_et_storage_v5.storage import (
    load_json, iter_verified_bytes, artifact_identity,
)

path = Path('取得済みresult namespace/E-r6.json')
record = load_json(path)                 # 完全graph検証後の元のkey/value
identity = artifact_identity(path)      # 元logical bytes/hash + root descriptor identity
with Path('/tmp/restored-E-r6.json').open('xb') as output:
    for part in iter_verified_bytes(path):
        output.write(part)              # 元indent=2 JSON + newlineと完全一致
```

大きいlogical JSONを全量再取得せず、個別blobやsnapshotからPauli／event／IR情報を点検できる。
過去snapshotは同じpath bindingで検証可能だが、`load_json`は最新logical値を返す。
書込み時に全状態を残し、metadata更新の際に大きいblobを再複製しない。
途中blob／snapshot／pending pointerも消さず、破損・missing・path escape・cap超過はtechnical failure。

## 実行制御

`launch.py`：pending拒否、S5のauthorization-only直接子、source manifest、remote/clean、fresh namespace。
`worker.py`：exclusive job開始、固定runtime、native/Tのinventory/native-result binding。
`guard.py`：既存Supervisorとbyte同一、relative loaderをこのcodecへ接続。
`acceptance.py`：元logical identityとphysical root identityをT inventoryに結合。
`storage.py`：POSIX flock、immutable compact blobs、typed snapshots、quota、hash verification。

storageのactive output上限は旧64 MiB−256 KiBと同一。
新resultはv2 namespace、旧v1 one-shotは再実行しない。
保存値I/O検算は`scripts/tracks/algorithm_codesign/verify_df_correction_et_v5_storage_replay.py`、
静的provenance照合は`verify_df_correction_et_v5_preparation.py`。
実験runnerは`run_df_correction_et_resumption_v5.py`だが、承認待ちの準備では呼び出さない。
