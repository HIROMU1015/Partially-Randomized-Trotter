# N3保証生成・N1復旧契約の公開確認

準備commitは`23ac5a49a42985ea6ff47d6edbcfe13e36399eae`。
[公開された準備の入口](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23ac5a49a42985ea6ff47d6edbcfe13e36399eae/docs/research/n3_generator_n1_recovery_preparation_20261011.md)、
[N3数学・移送契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23ac5a49a42985ea6ff47d6edbcfe13e36399eae/docs/research/n3_generator_transfer_contract_v1_20261011.md)、
[N1decode復旧契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23ac5a49a42985ea6ff47d6edbcfe13e36399eae/docs/research/n1_decode_recovery_contract_v1_20261011.md)が固定参照。

HIROMU1015/Partially-Randomized-Trotterの
`n3-generator-n1-recovery-preparation-20261011`へcommit・pushした後、
ローカルGit objectやalternatesを共有しない新bare repositoryへ、GitHubからdepth2で直接fetchした。
準備変更23 blobと旧source/結果220 blob、合計243 unique blobのbyte identityを照合した。
取得treeからstdlib-only準備checkerを実行し、metadata確認8175件PASSを再確認した。
旧manifest88 entriesと他fieldの保持、all-context N1 schema/44候補/178 pair、N3未評価literal12件、
実装未着手・未認可・STOPを確認している。数学証拠やdecoderの科学回帰を検証したものではない。

[取得receipt](../../artifacts/n3_generator_n1_recovery/2026-10-11/preparation/publication_receipt.json)に、
固定commit、全243 SHA-256、remote metadata監査とそのhashを保存した。
このreceiptは準備commitの公開確認であり、receipt自身のcommitを自己認証するものではない。
元rootの1339保存file、HEAD、default porcelain statusの保持も準備監査で確認した。

新科学実行0、新科学tests0、compile/対角化/LP/SDP0。旧GPT付録replay0。
N3生成器とN1decoderの修正はまだ実装していない。N2は完了扱い、旧source/failure/結果は保持。
今回の公開は次の科学実行を認可しない。中心テーマnull、Mandatory STOPを維持する。
