# 実G7/F3 API binding・authority・monitoring 閉鎖候補

固定SOURCE：[78c571005fe766c61b32dc867e006531099ff608](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/78c571005fe766c61b32dc867e006531099ff608)。直接基点 `d24cccbecdf0928fe977080dd7b28c591df0089e`。ローカル独立受入 `6bffb5b5819918698b522270ff19b255766d3072` は固定参照のみでmerge/rebaseしない。

今回の範囲は静的読取・AST・hash照合と人工技術検証。実科学入口閉鎖、科学的採択なし、mandatory STOP。サーバーCodex主担当一名による自己検証であり、別reviewerの独立受入ではない。

実APIの静的mapping、別versioned port、人工bits/endpoint、fake authority、preflight、Producerと分離したoffline判定を追加した。実科学へdispatchするSOURCEは作成していない。

新規73 tests PASS、4 portケース（正常2・技術失敗2）、新offline mutation22件拒否。既存回帰は補正版623＋V2 41＋既存独立57＋Bundle107＝828の別versioned ID。過去828/845へ新件数を重複加算しない。

**接続上の未解決点**：固定Global Bundleは組込みfixtureをhardwireし外部endpointを受けない。今回のportは固定pair engineへ人工結合し、変更していないBundleの保存・STOP回帰と証拠を併置した。新portが固定Bundle内部で実行されたとは主張しない。将来の実注入口は別SOURCEの採択対象。

入口：[handoff](handoff.md)、[API調査](scientific_api_static_inventory.md)、[binding設計](real_api_binding_design.md)、[受入結果](artificial_acceptance_results.md)、[未決事項](pending_and_required_decisions.md)。machine inventory、contract、manifest、authority、monitoring、acceptance matrix、SOURCE inventory、verification、publication receiptを同じpacketに保存する。
