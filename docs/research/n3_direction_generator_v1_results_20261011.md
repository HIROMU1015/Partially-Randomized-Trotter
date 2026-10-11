# N3保証生成・限定移送：結果とGPT判断事項

一回の認可batchを完了。終端は **NO_TRANSFER_ADVANTAGE_IN_FROZEN_SCOPE**。技術停止ではない。方向付きの有理数証拠を未使用入力で生成できたが、事前の移送優位基準「norm-onlyとの差が2 base以上」を満たさなかった。追加入力・条件追加・再探索・retryは0。中心テーマnull、次段認可false、mandatory STOP。

実装source `178d69a31dc7f2f1162d76294ba35a94b67d7fee`、実行plan固定 `f0e5e5d254fce0458431dcc59f2ab02562ee4c39`、専用branch `n3-generator-transfer-execution-20261011`。準備source `ba4530678df9d665d2e6005fb08179b0f7a7921f` と [数学・入力契約](n3_generator_transfer_contract_v1_20261011.md) の12入力、比較法、上限を変更していない。N1は別branch、N2と旧N3の4例はこのbatchでは実行0。

対象は実対称number-preserving one-body＋非負density、外部1軌道、active m4/5/7、固定ν1/2/3、full modes5/6/8。Jordan–Wignerの固定粒子数sector、geometry/chemical basis/DF rank/L_Dはなし。scale1/2,1,2、外部位置0,2,7、hdd/scale3/4,1/2,1/4、振幅1/40と1/10、同normのlow/high方向を事前固定。energy goal δ=1/250。時間発展T・PF delta windowは対象外。

## 一次結果

直接生成5/12、共通norm-only4/12、強いscalar4/12、共通五点4/12、旧c五点4/12、元定義の旧scalar0/12。直接生成がnorm-onlyを超えたのはT1_amp_1_10_high_coordinatesの一件、baseはT1のみ。直接生成が共通五点を超えたのはT2_amp_1_40_low_coordinatesの一件だが、この入力はnorm-onlyでも保証された。したがって直接生成の新しい成功点はあるが、登録した移送優位基準を満たさない。

| input | 旧scalar | 共通norm | 強いscalar | 共通五点 | 直接生成 | 旧c五点 | 数値width |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1_amp_1_40_low_coordinates | 未取得 | 保証 | 保証 | 保証 | 保証 | 保証 | 0.0005674814315 |
| T1_amp_1_40_high_coordinates | 未取得 | 保証 | 保証 | 保証 | 保証 | 保証 | 3.971960119e-07 |
| T1_amp_1_10_low_coordinates | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 | 0.009338985077 |
| T1_amp_1_10_high_coordinates | 未取得 | 未取得 | 未取得 | 保証 | 保証 | 保証 | 6.59557878e-06 |
| T2_amp_1_40_low_coordinates | 未取得 | 保証 | 保証 | 未取得 | 保証 | 未取得 | 0.0005065015176 |
| T2_amp_1_40_high_coordinates | 未取得 | 保証 | 保証 | 保証 | 保証 | 保証 | 1.080399215e-05 |
| T2_amp_1_10_low_coordinates | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 | 0.008366289461 |
| T2_amp_1_10_high_coordinates | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 | 0.0001827274452 |
| T3_amp_1_40_low_coordinates | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 | 0.001053313997 |
| T3_amp_1_40_high_coordinates | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 | 7.043636624e-07 |
| T3_amp_1_10_low_coordinates | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 | 0.01685105753 |
| T3_amp_1_10_high_coordinates | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 | 1.147478621e-05 |

T3全4件は共通下界 c≤U のため BOUND_PRECONDITION_FAILED。保証未取得はfull-space fallbackであり、縮約の成功として数えない。真値widthがgoal以下でも係数証拠がない入力は未取得のまま。検索/証拠失敗を数学的・物理的不可能性とは解釈しない。数値truthはselectorに渡していない。

## 保証・監査と費用

全12×6=72 method recordsを保存し、certificate ledger SHA `488c56f2cd8ffdf392dc6fc2912d65f54ed5eca00472f7b5067958229cad7fa7` を閉じてからfull/active診断24 solvesを実行した。独立checkerはsolver/eigenproblem/science moduleを使わず、係数trial、下界、旧scalar定義、全有理数PSD証拠、元κ・μ三条件を復元。49,016 checks PASS。有理中間値最大394 bits。直接μ0のaffine余裕は厳密zero。LDL pivotはPSDの機械証拠であり、最小固有値marginとは呼ばない。

r/旧五点/6外向き候補の下界改善を共通U/cの全primary対照へ与え、生成費用を各入力の共有部分として記録。元旧scalarは別U/c fieldを維持し、旧c五点は副対照。共通五点と旧c五点の受理集合は今回同じだった。数値one-body下界proposal12 calls、直接検索496 eigenvalue calls、直接有理候補64 attempts、ground診断24 calls。数値proposalは保証ではなく、有限generatorの完全性は主張しない。

2 spawn CPU workers、各thread1、GPU0。batch wall 0.194788 s、parent CPU 0.119792 s、children CPU 0.269476 s、parent peak RSS 43163648 bytes、child最大peak RSS 40108032 bytes。実行audit時output 595507 bytes。shared bounds生成と各method固有wallはledger内に別記。逐次版との同条件速度比較はなくspeedup率を主張しない。新synthetic15件PASS。worker AS4GiB、aggregate8GiB、shard32MiB、output512MiB、log16MiB、wall/CPU capなしを保持。

[input identity](../../artifacts/n3_direction_generator_v1/2026-10-11/preexecution/remote_input_identity.json)、[source-bound plan](n3_direction_generator_v1_execution_plan_20261011.json)、[一次ledger](../../artifacts/n3_direction_generator_v1/2026-10-11/run1/certificate_ledger.json)、[result](../../artifacts/n3_direction_generator_v1/2026-10-11/run1/result.json)、[診断](../../artifacts/n3_direction_generator_v1/2026-10-11/run1/diagnostics.json)、[独立監査](../../artifacts/n3_direction_generator_v1/2026-10-11/run1/independent_audit.json)、[実行監査](../../artifacts/n3_direction_generator_v1/2026-10-11/run1/run_audit.json)、[STOP](../../artifacts/n3_direction_generator_v1/2026-10-11/run1/MANDATORY_STOP.json)。

## GPTへ返す判断事項

方向を保つ証拠を自動生成できることと、登録した未使用入力で広い優位性が出ることは別である。今回は後者の基準未達を保存する。T3のc>U前提が成立しない保守性、T2で直接法の新成功がnorm-only既存成功に含まれる点を含め、研究を続ける価値を新結果に基づいて判断してほしい。今回の範囲から一般二体、多外部、実分子、時間発展、量子資源改善、新規定理または中心テーマの採択は結論しない。旧GPTレビューの同結果再レビューは不要だが、この新結果の科学判断は未実施。

local synthetic/saved監査はimmutable CI・外部科学再現ではない。global manifest checkerは旧 `relevant_commits[7].description` 欠落でFAILを維持する。今回entryと旧89 result_sets保持は別のlocal metadata監査で確認する。GitHub固定commit・独立再取得記録はpublication資料に追記する。

公開前metadata監査PASS：旧manifest89 entriesをraw prefix/suffixのまま保持し、個別entry1件のみ追記。旧source/results220 blobsと準備契約/inputを含む計232 blobsのbyte保持を確認。新entry schemaはPASS、global manifestの既存FAILとは区別する。 [metadata監査](../../artifacts/n3_direction_generator_v1/2026-10-11/publication_metadata_audit.json)、[root保持](../../artifacts/n3_direction_generator_v1/2026-10-11/root_preservation.json)。
