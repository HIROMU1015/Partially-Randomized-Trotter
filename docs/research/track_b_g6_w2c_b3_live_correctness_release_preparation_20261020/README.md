# Track B W2c-B3：live接続Release・512契約・authority準備

実行状態はLIVE_RELEASE_CANDIDATE_UNAUTHORIZED。設計準備へのユーザー了承と、正式採択／実行承認を分離する。
新SOURCE `src/trottertracks/algorithm_codesign/g6_w2c_b3_live_release_v1/`、測定SOURCE `PENDING_FINAL_SOURCE_FREEZE`、artifact `5d99b0a319523aee6df095407c231a912345b1bf35b44a7b13608652a9ba39d5`。
[Q1〜Q10](handoff.md)、[実装](live_api_connection_implementation.md)、[契約](correctness_contract_candidate.json)、
[authority](trusted_store_implementation_status.md)、[未決条件](remaining_execution_blockers.md)、[人工受入](artificial_technical_acceptance.md)、[公開照合](publication_receipt.json)。

API形状の人工objectsを新wrapperから既存Exact Adapterへ渡し、新worker/pair/F01→Storage V2→receipt/callerへ接続する。
実module resolverは`.inactive` SOURCEとしてAST/SHA照合のみ。通常dispatchは組込み人工symbolsだけを許し、science入口は引数に関係なく拒否する。
再現：`python3 docs/research/track_b_g6_w2c_b3_live_correctness_release_preparation_20261020/run_acceptance.py /tmp/NEW_UNIQUE_OUTPUT`。新しいTEST_ONLY store/scopeだけを使う。
科学law、実Evidence互換性、実CPU／容量、real authority耐久性は未確立。

**LIVE_RELEASE_CANDIDATE_UNAUTHORIZED。実科学未実行。mandatory STOP。次の担当：ユーザー・GPT。**
