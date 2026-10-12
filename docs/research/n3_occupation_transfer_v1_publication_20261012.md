# N3 OCC v1 一回比較：公開・GitHub再取得

2026-10-12 JST。`HIROMU1015/Partially-Randomized-Trotter` 専用branch `n3-occupation-transfer-v1-execution-20261012`。

固定8入力の一回比較は **NO_ADDITIONAL_ACCEPTANCE_OBSERVED**。共通強下界で既存OR/NORM/raw OCC各8/8、OCC固有gain0。
one-body＋非負unordered density、一外部mode、active6/7/8/9、ν2/3/3/4、δ=.004/.006/.003/.005。geometry/化学basis/DF rank/L_D/PF windowは適用なし。
詳細は[固定結果報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c/docs/research/n3_occupation_transfer_v1_results_20261012.md)。

- frozen source: `83f51cd8a4848572c66a5535c67d0d0a5f3868cd`（8ファイル保持）
- clean authorization/execution commit: `a4311eece09275e4a27259db3fe12e98974dbadc`
- immutable result payload: `e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c`
- [一次result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c/artifacts/n3_occupation_transfer_v1/2026-10-12/run1/result.json)
- [独立保存audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c/artifacts/n3_occupation_transfer_v1/2026-10-12/run1/independent_saved_audit.json)
- [費用・資源・全case summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c/artifacts/n3_occupation_transfer_v1/2026-10-12/execution_audit/result_summary.json)
- [Mandatory STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c/artifacts/n3_occupation_transfer_v1/2026-10-12/run1/MANDATORY_STOP.json)

GitHubへpush後、別bare repositoryへGitHubからfetchし、42 blobs / 12,898,275 bytesをexportしてlocal byteとSHA/sizeを照合PASS。
過去にGitHubから取得したobjectsのcacheは使用したが、新payloadはfetch前に存在しなかった。
local worktreeからobjectをcopyせず、alternatesも使用していない。完全freshな全履歴取得とは区別する。

remote exportの凍結済みstdlib checkerで **5,008,702 checks PASS**。保存有理PSD/LDL、元行列再構成、入力・source・plan・認可hash、primary seal、raw/継承/fallback採否を再確認した。
optimizer/eigensolver/ground solver/科学batchの再実行0。同一local環境での保存証拠監査であり、外部環境での科学batch再現やimmutable CIではない。
baseから保護された1882 tree entries、旧manifest92 entries、root1345 files/HEAD/既存dirty entriesの保全PASS。
既存global manifestの`relevant_commits[7].description`欠落FAILは残る。今回の保存監査PASSと区別する。

[機械publication receipt](../../artifacts/n3_occupation_transfer_v1/2026-10-12/execution_audit/publication_receipt.json)にpayloadの全取得file hashとexact auditを保存。
[公開監査source](../../scripts/verify_n3_occupation_transfer_v1_execution_publication.py)は保存Git objects/exportを読むだけで、fetchやproducerを呼ばない。
publication commitではreceipt・この文書・監査source、概要・研究ノート・manifestへの公開記録のみを追加する。
source8/plan/input/run1一次結果と固定result報告はpayloadと同じまま保持する。

再確認command（先に当該branchをGitHubからbareへfetchし、receiptのpayload pathsをexportする）:

```bash
python3 scripts/verify_n3_occupation_transfer_v1_execution_publication.py --bare-repository /tmp/prt-n3-direction-generator-published-20261011.git --export-directory /tmp/prt-n3-occupation-transfer-v1-execution-remote-export-20261012
```

一回結果取得・保存・公開・再取得後Mandatory STOP。N1追加探索停止、N2保留、中心テーマnull。
追加scan、再試行、実分子・多外部・一般二体拡張を開始しない。次はこの新科学結果をGPTが評価する段階。
