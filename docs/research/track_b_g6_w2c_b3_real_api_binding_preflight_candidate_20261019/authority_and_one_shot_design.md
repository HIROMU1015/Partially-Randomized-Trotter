# Authority・one-shot候補

固定SOURCE：[78c571005fe766c61b32dc867e006531099ff608](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/78c571005fe766c61b32dc867e006531099ff608)。直接基点 `d24cccbecdf0928fe977080dd7b28c591df0089e`。ローカル独立受入 `6bffb5b5819918698b522270ff19b255766d3072` は固定参照のみでmerge/rebaseしない。

今回の範囲は静的読取・AST・hash照合と人工技術検証。実科学入口閉鎖、科学的採択なし、mandatory STOP。サーバーCodex主担当一名による自己検証であり、別reviewerの独立受入ではない。

authority.pyはformal release(null)、SOURCE candidate、科学reference/input hash、runtime、contract、scope/claim、call/result/log budgets、seed、retry/resume、worker policy、storage destination、monitoring/STOP digestをbindingする。authority_binding_candidate.jsonは架空の将来fake rootを使う静的例であり認可ではない。

実行試験は新規private /tmp/b3-bundle-artificial-* だけで固定FakeStoreを生成する。BEGIN IMMEDIATE/unique scope+claim、SPENT commit後ACK、UNKNOWN ACKでも返却しない。32競合のwinner1、二重/STOP後/spent再利用、19 binding mutation、人工approval mismatchを拒否。実store/grant/旧scopeを探索・消費・変更しない。fake approvalはbindingの人工tokenで、署名済み実認可ではない。

実authority/bootstrap、trusted store所有・失効・鍵、全worker lineage、承認文書、正式grantの発行/消費・耐久性は後続の別SOURCEと明示認可が必要。保存snapshot/historyは記録された人工状態を照合するもので、未保存atomic履歴・power loss・実authorityの証明ではない。
