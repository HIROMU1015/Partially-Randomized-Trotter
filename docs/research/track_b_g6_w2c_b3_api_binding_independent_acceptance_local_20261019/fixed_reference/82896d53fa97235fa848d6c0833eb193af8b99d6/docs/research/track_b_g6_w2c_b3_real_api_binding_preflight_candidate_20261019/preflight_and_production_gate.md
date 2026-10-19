# Preflight・閉鎖gate

固定SOURCE：[78c571005fe766c61b32dc867e006531099ff608](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/78c571005fe766c61b32dc867e006531099ff608)。直接基点 `d24cccbecdf0928fe977080dd7b28c591df0089e`。ローカル独立受入 `6bffb5b5819918698b522270ff19b255766d3072` は固定参照のみでmerge/rebaseしない。

今回の範囲は静的読取・AST・hash照合と人工技術検証。実科学入口閉鎖、科学的採択なし、mandatory STOP。サーバーCodex主担当一名による自己検証であり、別reviewerの独立受入ではない。

SOURCE62 pins（固定52＋新runtime/harness/test10）、科学入力10 pins、AST inventory digest、runtime executable/library、candidate/schema/closed flags/seed/予算の完全一致を照合する。親と各人工workerでscience MetaPathFinderを設置し、g7_generator/g7_provider/return_aggregation/g6_f3_*のimport要求をload前に拒否。既loadも拒否。科学importを実行してから判定する試験は行わない。

production_entry、real_bits、real_authorityにdispatch処理は存在せず、無条件例外。run_ready/content_adopted/contract_grants_execution=false、science_release/scientific_adoption=null。boolean反転、fake authority/bits転用、grantなし/偽grant要求は解除手段にならない。将来は別正式SOURCE、input/runtime/authority/監視と保存範囲の採択が必要。

新規negative controlsはSOURCE/runtime/signature/入力/contract/schema、missing/wrong result、arm/layer/axis/event、consume/domain、retry/resume、親/spawn import block、F01 generator/audit/angle/requery、fake race/commit-ACK UNKNOWN/spent/approval等。offlineは別process import boundaryも試験した。Pythonコードの悪意ある直接改変に耐えるOS security sandboxを主張するものではない。
