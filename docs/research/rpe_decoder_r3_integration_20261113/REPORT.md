# B：R3高速decoderのS3統計pipeline統合

2026-11-13 JST。**既存S3統計APIへ任意batch hookを接続した。既定は元scalar経路。人工入力の全trial・loss・CIは加速SOURCE欄を除いて一致した。実H4のS3、保存科学値の再解析、十分N/global bound認定、価格/Gは実行していない。**

[HANDOFF・切替API](HANDOFF.md)、[人工benchmark](BENCHMARK.json)、[decoderテスト](TEST_DECODER.txt)、[保存metadata回帰](TEST_METADATA.txt)。今回を含む公開commitを実行SOURCEの固定単位とし、過去の科学SOURCEを付け替えない。

## SOURCEと再利用

| 固定SOURCE | 再利用・変更範囲 |
|---|---|
| A `7f1f9053dfbbb81646ac9e483a0354d01cbec535` | `rpe_decoder_fast.py`、12件のテスト、固定Bを読む人工fixture、`B_S3_PATCH.diff`をbyte一致で取込み |
| B `bf515a976f1987ad6a19c61d14689b27b0894542` | 現行branchの基点。対象統計ファイルはAが検証したB170047と同一blob `e4d5aba93a7a91c0061babaccc86f69e81dfa697` |
| 元decoder | `pr_rpe_rmse.decode_supplied_coordinates`、`rpe_four_round_phase_validation`のscalar/vectorizedは無変更 |
| B既存統計 | PCG64/binomial、exact `count_mean`、count metadata、loss/CI、`prepare`をそのまま再利用 |
| S3受入済み証拠 | `rpe_s3_readiness_20261113/metadata01/`とA/C/D packet・資格・SOURCEは無変更 |

研究系列branch `codex/rpe-decoder-statistics-20261113`を再利用。他担当のbranch/worktree/process/cacheに変更なし。旧SOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOPを保持した。サブエージェントは使用していない。

## 結合の内容

`simulate`と`simulate_prepared`へ`batch_decoder=None`を追加し、Aの提案差分を現行Bと照合して適用した。変更はhook分岐・引数転送と説明のみ。全roundのX→Y binomial count生成後に、指定されたhookへ同じarrays、元`count_mean`、元decoder SOURCEを渡す。Noneは元trial loopで、既存出力とも一致する。

高速版は`decode_sampled_counts`を用いる。unique countに対しB自身のexact整数差/Fraction→binary64変換を呼び、math.atan2とAの定数近傍branch kernelを利用する。各trialへ`decoder_acceleration_source`を追加する。SOURCEの元B参照はAのまま保全し、今回公開commitとA固定commitの組で実装を特定する。

元scalarはargminの最小branch ID、旧vectorizedはargsortの完全tie順で、同じ規則ではない。B hookは**scalar互換**を使用する。人工q32/principal0/previous2.650718801466388のscalar13/vector14という差異をそれぞれ保持する。zero座標がbranch tieより優先する規則、未解決energy=None、−phase/h、prior所属診断のみ・alias補正なし・energy shiftなしを維持した。旧vector互換APIもAのまま取り込んだが、S3 hookからは使わない。

## 技術検証

decoder系34件（A既存12、実B結合5、B既存統計16、元decoder1）と保存metadata系26件（S3 13、D残差13）、**計60件PASS**。テストは人工入力、または保存metadataのcopy/資格検査のみである。

- M0〜12、小q全位相grid、wrap/ULP/±0/1e−12近傍、scalar/vectorの完全tieを元参照と照合。
- 実B経路を固定旧Bと比較し、全trialの全項目、位相・branch・energy・statusがSOURCE追加欄以外一致。
- round→X/Y順の全binomial countとPCG64最終状態が独立参照生成と一致。hookは乱数を再生成しない。
- 2^53超・2^62超の人工countに対しexact変換と全trial一致。Nをfloatへ変換しない。
- 未解決loss未定義/有界区間の両方で元summarize結果・RMSE診断・CIが一致。条件付きprepareの全metadataも一致。
- 保存S3の全13 nominalをcopy確認し、欠測・物理未資格gateでsimulation/hookが呼ばれない。全十分N/global bound/Gはnull。

テストPASSは有限shotの科学的RMSE保証や、実H4の成功率を意味しない。同値性を人工検証したdomainはM0〜12。A実装が受け付けるq≤2^52全域の保証へ拡張しない。

## 人工性能比較

結果は[BENCHMARK.json](BENCHMARK.json)。Aの人工benchmark手法を再利用し、実Bの`simulate`を直接測定した。人工13round/M12、各軸shot31/37、raw=(.43cos(.237·2^m), .43sin(.237·2^m))、PCG64 seed441、512 trial、各経路3回。H4保存mean/Nは使用していない。

| 実B経路 | 3回の中央値 秒 | peak RSS MiB | import後baseline RSS MiB |
|---|---:|---:|---:|
| 既定scalar | 16.651018 | 352.031 | 345.941 |
| R3 batch | 0.025542 | 355.355 | 346.691 |

今回の中央値比は**約651.9倍**。各経路の3実測時間・完全な精度の数値はJSONに保存した。

両経路はfresh subprocessで逐次測定し、同じ元decoder/importを事前に読み込んだ。時間はbinomial生成、count変換、atan2、branch復元、全trial dict作成を含み、process起動/import/digestを除く。全出力digestは加速SOURCE欄を除いて一致した。CPU/RAM/thread/worker capは追加していない。

RSSはLinux ru_maxrssでinterpreter/import/input/outputと先行反復のdigest high-waterも含む。kernel専用allocationやメモリ削減の主張ではない。高速化後も全trialの全round診断を保持するため、出力メモリとdict作成は残る。共有サーバーの今回実測であり、Aの約665倍を固定性能値とは扱わない。新たな統計方式・CI・sampler実装や無関係なR2/R4/R5改善は追加していない。

## S3科学条件と次の担当

受入済みH4 DF10/S2/L_D1/ΔPF=h=.1/K2/M12・元exact共通κ/H/state/prior/decoder/law/compiler/P0のmetadataは維持。nominal各軸64,60,56,52,47,43,38,34,30,25,21,17,12を十分Nへ昇格しない。

**ユーザー**：target proxy採択、未解決loss、ideal-law IID仮定、数値不確かさ、MSE精度・信頼水準・反復数と固定実行SOURCEを決め、有限な実S3科学実行範囲を別途承認する。

**Codex B**：その明示承認後に限り、同じ統計APIで選択したdecoderを使う。今回の実H4 S3・新N/C/Gは0。十分N/global RMSE boundは未資格・null。commit/push・remote SHA/tree/主要成果物照合後mandatory STOPし、科学実行へ自動進行しない。
