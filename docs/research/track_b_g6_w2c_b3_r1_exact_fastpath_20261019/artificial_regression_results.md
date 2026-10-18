# 人工回帰結果

[machine結果](artificial_test_results.json)に全ID・SOURCE SHA・log SHAを記録した。各suiteは独立processで並列実行。

| Suite | PASS | FAIL/ERROR/SKIP |
|---|---:|---|
| Admission | 69 | 0/0/0 |
| Readiness | 30 | 0/0/0 |
| B3 Driver | 36 | 0/0/0 |
| Cost Preparation | 48 | 0/0/0 |
| F01 Provenance | 26 | 0/0/0 |
| Host Integration | 66 | 0/0/0 |
| 新canonical/driver/guard | 36 | 0/0/0 |
| 新angle recurrence | 28 | 0/0/0 |
| 合計 | **339** | **0/0/0** |

既存275、新64。既存正常8ケースはsubcaseとして保持し件数へ重複加算しない。
Admission/source/contract/host pin、scope/claim競合、reserve後spent、with port外側STOP、保存失敗、retry0/resumeなし、coverage不足でfull-N禁止を維持。
F01の前trial誤帰属、current/last completed、candidate/committed、partial write/flush、failure evidence/cleanupを確認。
新codecに対してsource欠落・改変・reserve後改変を拒否し、新moduleのfine profilingも検査した。

codec/旧275 suitesではG7/F3/root科学moduleのimportを拒否・監視。
angle suiteだけは明示許可した原F3 precision/contract/rootを人工入力で使用する。sampler/F3 session/G7 generatorのimport・実行はない。
全339はLinux6.8.0の本環境で実行でき、fork・/proc/self・local filesystem依存のSKIPは0。
電源断・媒体故障の完全証明、実G7 correctness、全domain科学的同値性、実CPU優位、CI/外部再現ではない。
