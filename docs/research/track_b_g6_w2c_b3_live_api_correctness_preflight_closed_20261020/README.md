# Track B W2c-B3：実API correctness実行前準備・閉鎖v1

新versioned SOURCE `src/trottertracks/algorithm_codesign/g6_w2c_b3_live_preflight_v1/`。実科学loaderは存在せず、人工factoryだけを固定spawn/F01/Storage V2へ接続する。
測定SOURCE `4b4496a86f4ab9e0e1266e7323283aff9fe24f26`、SOURCE artifact `8244d1569708e9f786aec0a5c3b010b0d017a624083de59643608e7f66d68896`。
独立受入 `0db83adb370d66a2f7f5254c8037dadda94ecdb8`／Adapter `80b7bf61fe0a735d4031bb719c4b3e788611c3ba`を固定参照し、merge/rebase・旧SOURCE変更は行わない。

[Q1〜Q10](handoff.md)、[静的API](canonical_full_connection_contract.md)、[512候補](correctness_only_scope_candidate.md)、
[人工受入](artificial_technical_acceptance.md)、[未決事項](pending_and_required_decisions.md)、[公開照合](publication_receipt.json)。
科学SOURCE/input/precision、契約、coverage、予算、real authorityは未採択。人工PASSはlive互換性を認定しない。

再現：`python3 docs/research/track_b_g6_w2c_b3_live_api_correctness_preflight_closed_20261020/run_acceptance.py /tmp/NEW_UNIQUE_OUTPUT_DIRECTORY`。
新FAKE_ONLY scopeを作る人工試験であり、実trusted store・旧claimへのアクセスはない。
新SOURCE commitから実施すること。公開側はpublication_transport.jsonのexact-tree対応も参照。

**未採択・実科学未実行。mandatory STOP。次の担当：ユーザー・GPT。**
