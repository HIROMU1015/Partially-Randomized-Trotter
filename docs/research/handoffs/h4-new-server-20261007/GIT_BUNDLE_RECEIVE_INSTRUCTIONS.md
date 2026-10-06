# 新サーバーへの回答：Git bundleで提供

旧サーバーで作成・独立bare clone/hash/fsck検証済み。origin未公開のまま、転送は未実施。

branch：track-a-h4-lazy-identity-run05-20261006

commit：8f77bebf99c5bd15fa6419c1f58556c3bd2837a9

旧サーバーのbundle：/tmp/h4-run05-git-handoff-20261007-3fvjlyge/h4-run05-handoff-20261007.bundle

SHA-256：1e5d7df8b9207ac1eafb2555b6bd2b735cd270b0fe7df4926ddaad5d1b3fe63f

bytes：36265268

bundleとSHA256SUMS/delivery_manifest_v1.jsonを利用者が新サーバーの自分専用directoryへコピーし、受領pathをCodexへ伝える。コピー先は未指定のためこちらでは転送していない。

新サーバーではhash照合後、既存origin clone内でgit bundle verifyを行い、bundleから指定refをfetchしてcommitを照合する。既存の同名branchを上書きしない。未追跡NPZ/runtime/controlはbundleに含まれず、既存の移行指示どおり別途受領する。本計算は開始しない。
