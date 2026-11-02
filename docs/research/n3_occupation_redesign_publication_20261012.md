# N3再設計準備：GitHub公開・再取得

専用branch `n3-occupation-redesign-preparation-20261012`。
固定準備commit `c2955ba0ef391eaf9102c711c378ab73b766fc9f` を
`HIROMU1015/Partially-Randomized-Trotter`へpushし、GitHubから別bare repositoryへ再取得した。
新しい科学batchは0、Mandatory STOP・次batch認可falseを保持。

- [数学検証・強化対照・費用](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2955ba0ef391eaf9102c711c378ab73b766fc9f/docs/research/n3_occupation_redesign_mathematical_audit_20261012.md)
- [次の限定比較設計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2955ba0ef391eaf9102c711c378ab73b766fc9f/docs/research/n3_occupation_transfer_design_v1_20261012.md)
- [独立有理証拠検査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2955ba0ef391eaf9102c711c378ab73b766fc9f/artifacts/n3_occupation_redesign_preparation/2026-10-12/independent_witness_audit.json)
- [未評価の新8係数入力](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2955ba0ef391eaf9102c711c378ab73b766fc9f/artifacts/n3_occupation_redesign_preparation/2026-10-12/proposed_transfer_inputs.json)
- [STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2955ba0ef391eaf9102c711c378ab73b766fc9f/artifacts/n3_occupation_redesign_preparation/2026-10-12/MANDATORY_STOP.json)

以前GitHubから取得した旧N3 objectsのcacheを持つ別bare repoを再利用した。
fetch前に新準備commitがそのrepoに存在しないことを確認した。
worktreeのobjects/alternatesは使用していない。完全にfreshなbase再取得とは区別する。
40準備blobs＋固定入力/ledgerの計42blobs、2,032,200 bytesをremote objectsから読み、
local byteおよび固定inventoryのSHA-256/sizeと照合PASS。
remote exportのstdlib checkerで756 checks、7 PSDの元証拠検査が同一結果となった。
基点の保護tree1811 entries、旧manifest90 entries、root1342 files/HEAD/dirty statusを保持。

remote保存検査の最初の結果比較はPython tupleとJSON listの表現差で停止した。
JSON表現へ正規化してから照合し、全exact値が一致した。証拠を変更したり科学batchを再実行したりしていない。

[機械receipt](../../artifacts/n3_occupation_redesign_preparation/2026-10-12/publication_receipt.json)
に全42fileのhashとremote exact結果を保存。
source/一次証拠を変更せず、receiptと本公開記録、
[公開checker](../../scripts/verify_n3_occupation_redesign_publication.py)を後続commitで追加する。

再取得したbare objectsからの保存検査は、上のcheckerに
`--bare-repository`と空の`--export-directory`を渡して実施できる。
checkerはネットワークfetchや科学runnerを呼ばず、保存objectsと有理証拠だけを確認する。
これはimmutable CIや外部環境での科学再現ではない。
既存global manifestの`relevant_commits[7].description`欠落FAILも保持する。
