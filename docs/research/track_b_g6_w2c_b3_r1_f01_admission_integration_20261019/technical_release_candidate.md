# 高速化変更前の技術SOURCE候補

状態は**TECHNICAL_INTEGRATION_BASELINE_NOT_FINAL_SCIENCE_RELEASE**。
branch `track-b-g6-w2c-b3-r1-f01-admission-integration-20261019`、固定SOURCE commitはpublication_receiptのverified_content_commitで実値を示す。
基点 `1c15bacab349b8686c09d2981e1c0bc70e9c3ec9`。新SOURCE全bytesのSHA256：

- `src/trottertracks/algorithm_codesign/g6_w2c_b3_driver.py`：`5e6904e782526e96cc6a2dc9e94642a8237b4aa69ce8743aec24ccde5f212694`
- `src/trottertracks/algorithm_codesign/g6_w2c_b3_admission.py`：`b718047a96353f60bfda4ee08694e610be7e53c701185d1ceda539146f1ae750`
- `src/trottertracks/algorithm_codesign/g6_w2c_b3_host.py`：`49e753184b66b22b41287acade1c9ac6a43249321481c6fe53149b43c2bd03b9`
- `src/trottertracks/algorithm_codesign/g6_w2c_b3_readiness.py`：`f6cbb170b99bd67926c66eeb982df67a1dd4506dcc8fc40f5016cf482341b00a`

全tests/fixtureのSHAは[integration_manifest](integration_manifest.json)。driverはF01修正版bytesそのもの。
技術候補[JSON](technical_contract_candidate.json)のfile SHAとexact digestもmanifestへ固定した。
このdigestは**技術候補だけのidentity**であり、将来の実行用正式contract digestではない。後者はnull。

保持する科学bound：G7 P5_general_order、p=(1/5,3/10,1/2)、x5/7、m5、σ+1、4方式×A/B×X/Y、A6 math-only。
seed B3_FULL_N_NEW_20261019_V1、同prefix規則、24 constructor（Full内包分を重複加算しない）/Full prepare3/canonical preflight9、
新memoなし・X/Y保持stateだけ、correctness→baseline→instrumented、初期全文/全trial compact hash/stream/failure prefix/64GiB/256MiBを維持。
256＋15,855,296＋15,855,296＝**総31,710,848**。correctness256を二重加算しない。
旧科学SOURCE、保存N/error/confidence、law/bit順/phase/weight/word、angle/requery、retry0/resumeなし/STOPは不変。

技術候補内のformal v2 templateはrelease commit・scope/claimがnullで、実行できない。
科学実行用最終release採択、最終schedule採択、正式実行digest、実authority/store/claim、real scienceは今回固定・発行していない。
高速化結果が不要でも、今回の技術baselineを無確認で科学実行用へ昇格しない。
