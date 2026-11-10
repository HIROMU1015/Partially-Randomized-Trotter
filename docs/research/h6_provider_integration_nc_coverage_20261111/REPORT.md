# H6価格provider統合・H4/H6 N/C coverage（2026-11-11）

H6の保存lookup providerをBの固定共通APIへ接続した。人工/mock・保存fixtureの24 test methodがPASS。全13ラウンド×PF/PRについてH4/H6各26行を静的照合し、[COVERAGE.json](static/COVERAGE.json)と[ROUND_COVERAGE.csv](static/ROUND_COVERAGE.csv)へ保存した。新しいsignal/N/価格予測/fit/sampling/build/compile、実N×C/G、科学的数値再解析、詳細科学レビューは実施していない。総費用・順位・joint covarianceはnull、公開照合後mandatory STOP。

## 実装と再利用

- 独立branch `codex/rpe-h6-provider-coverage-20261110`、基点 `cb2b1d6b4309ca7d705f397e25e27d14e8ba125a`。Bの実行中branch/worktree・Python環境・process/cache/checkpoint/private rawへアクセス・変更していない。
- 統合コードSOURCE：`f58428f3021f1ac9d51fad038faa8f78eb560aed`。Cの`0b5c493f2b2728b0c7c6c848d2cbf084db8f7f76`から[H6 provider](../../../src/trottertracks/h6_df19_price_provider.py)と[既存テスト](../../../tests/check_h6_df19_price_provider.py)をbyte一致で再利用。凍結係数を変更していない。
- [共通pipeline](../../../src/trottertracks/pf_pr_rpe_pipeline.py)の変更は`h6_saved_price_provider(read_json, selections=...)` factoryの6行だけ。`c_provider(task, requested_rows)`を既存`run_missing`へ渡す。H4 provider・N側・会計器と既存共通関数は基点のまま。
- [静的coverage helper](../../../src/trottertracks/rpe_nc_coverage.py)は既存`SavedArtifacts`、passiveな保存loader、`plan`、providerの保存lookup/record変換を再利用する。[CLI](../../../scripts/report_rpe_nc_coverage.py)は固定Git参照のJSON/CSV転記だけを行い、数値実行の入口を追加しない。
- 保存path価格はSOURCE索引として保持し、新しい平均を作らない。旧V1/M0/Mq/MqBの保存予測も別scenarioとして保持。別名を独立標本に数えず、探索/確認、direct/model、異なる回路族をpoolしない。

## 固定SOURCEと公開Nの採用

| 対象 | 固定commit・参照 |
|---|---|
| B共通API/H4 provider | `cb2b1d6b4309ca7d705f397e25e27d14e8ba125a`。[基点REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cb2b1d6b4309ca7d705f397e25e27d14e8ba125a/docs/research/h4_price_pipeline_integration_20261110/REPORT.md) |
| C provider公開版 | `ae44283d169b61deace7e1c0d18c4538d4a3fc0c`。[旧HANDOFF](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae44283d169b61deace7e1c0d18c4538d4a3fc0c/docs/research/h6_df19_affine_price_provider_20261110/HANDOFF.md)、コードは上記0b5c493f |
| H6 frozen Affine | `b85837960c80d619a421c46235baf5ef2523d217`の`Q4_FROZEN_MODELS.json`。価格/検証carrier `73dcbe23678bfc50d5654f09e600d4cfbe75e697`の`Q2_DIRECT.json/Q4_DIRECT.json/Q_VALIDATION.json`。再fitなし |
| H6全13ラウンドN | `f4c0bdabdf2afdc5bc4f20573bb10677917f4c1a`。[公開REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f4c0bdabdf2afdc5bc4f20573bb10677917f4c1a/docs/research/h6_df19_rpe_m8_12_results_20261110/REPORT.md)。既存loaderのm0..7保存16 record＋最終結果のm8..12保存10 record |
| H4固定task | `90dcc3ed20f30dfab6b22a68cc4adb9461485a98`の`h4_df10_rpe_batch_preparation_20261110/PLAN.json.task` |
| H4公開全13ラウンドN | `89c91c692e1d35de94d2e7afbe890aabe6d80b7e`の[N_HANDOFF.json.shots](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/89c91c692e1d35de94d2e7afbe890aabe6d80b7e/docs/research/h4_df10_rpe_batch_results_20261110/N_HANDOFF.json)。科学SOURCE90dcc3ed、整数検証SOURCE `0dc17b0329e1a8aa2737c86f259e049985317dc5` |
| H4 C固定成果 | `2a521b156f8ae957d981516ab8a0031fcafae0a5`までの、基点Bが統合済みのdirect/Mq catalogと旧保存scenario |

作業中のremote確認でH4完了結果の公開を確認したため、未公開Nを推測せず、公開済み固定`N_HANDOFF.shots`を受け入れた。Bの作業領域・進行状況ファイルを読んでいない。公開結果に含まれるprivate原本のpath文字列も開いていない。

H4 m3の保存Nと全26 recordのcandidate/科学binding/measurementを照合した。巨大NはBが整数丸め検証したhandoffを正本とし、raw resultのNへ戻さない。今回その検証やN式は再実行していない。Python整数と十進文字列を両方保存し、binary64へ変換しない。H4のPR m10・m11補正も保存値の転記である。H6のNも保存値の転記のみ。

## 全13ラウンドの暫定coverage

共通：Linear H4/H6、1.00 Å、STO-3G、保存DF10/DF19・weighted completion・保存数値stateを使用する条件付きideal lane。targetは各保存DF-Hの同sector基底エネルギー。h=.1、m=0..12、精度0.00159360010199040 Ha、方式ごとαtotal=.05、βtotal/stat=.4/.2、P=0、主資源compiled RZ。保存prior・state/reference・compiler/full Hadamard wrapperを変更していない。Nの条件付きPASSは実装/law誤差のrigorous資格、actual minimum N、全RPE成功保証ではない。

**両systemの全13ラウンド・PF/PRとも保存NはCONDITIONAL_ROUND_PASS、actual Nは全てnull。** 下表は価格の有無と既存検証の役割だけを示す。PF/PRは両方式、PFのみはPR欠測、—は保存価格なし。各価格はX/Yを保持し、N×Cは計算していない。

| m | T | H4 qPF | H4 direct C | H4保存model C・資格 | H6 qPF | H6 direct C | H6保存model C・資格 |
|---:|---:|---:|---|---|---:|---|---|
| 0 | .1 | 1 | — | — | 1 | — | —、Affine対象外 |
| 1 | .2 | 2 | — | — | 1 | — | —、Affine対象外 |
| 2 | .4 | 4 | — | — | 1 | — | —、Affine対象外 |
| 3 | .8 | 8 | PF/PR | PF/PR、既知点・校正、独立検証資格なし | 1 | PF/PR | 旧V1 PF/PR、未資格、Affine対象外 |
| 4 | 1.6 | 16 | PF/PR | PFは既知点、PR Mqは有限独立q検証 | 1 | PF/PR | 旧V1 PF/PR、未資格。Affineはanchorで保存forecastなし |
| 5 | 3.2 | 32 | PFのみ | PF Mqは有限独立q検証、PR未取得 | 2 | PF/PR | Affine校正q2、採用資格false |
| 6 | 6.4 | 64 | — | —、未検証長q | 4 | PF/PR | Affine有限独立q4検証 |
| 7 | 12.8 | 128 | — | —、未検証長q | 8 | — | —、未検証長q |
| 8 | 25.6 | 256 | — | —、未検証長q | 16 | — | —、未検証長q |
| 9 | 51.2 | 512 | — | —、未検証長q | 32 | — | —、未検証長q |
| 10 | 102.4 | 1024 | — | —、未検証長q | 64 | — | —、未検証長q |
| 11 | 204.8 | 2048 | — | —、未検証長q | 128 | — | —、未検証長q |
| 12 | 409.6 | 4096 | — | —、未検証長q | 256 | — | —、未検証長q |

H4はS2/δ=.1・PF B0/p7対PR p1/R4/K2。H6は4次m5_best、PF Full19。H6のPRはm0..3がp5/R32/K2、m4..12がp7/R16/K2。H6 Affineはδ=1.6の後者だけで、qPF1/2を校正、qPF4を独立検証に使用。qPF>4は未検証で、式が存在しても保存値を補完しない。別δ・別prefix/outer R・未観測order2・別compiler/wrapperへ適用しない。q_RPE=2^mとq_PFは別fieldで、特にH6では一致しない。

機械可読分類は重複可能とした：

| category | 意味・該当範囲 |
|---|---|
| SAVED_SCENARIO_INPUTS_AVAILABLE | Nと既存policyで受入れ可能なC scenarioが保存済み。H4 m3/m4両方式・m5 PF、H6 m3..6両方式。実Gは未算定 |
| MODEL_REQUIRES_SCIENTIFIC_QUALIFICATION | 保存modelが未資格、またはmodelのdomainには合うが保存予測がない。数式の計算可能性と採用資格を分離 |
| NEW_PRICE_EVIDENCE_OR_VALIDATION_REQUIRED | 受入れ可能なC scenarioがない。H4 m0..2・m5 PR・m6..12、H6 m0..2・m7..12 |
| N_OR_SAVED_C_VALUE_MISSING | N未取得/不合格またはdirect/model双方が欠測。今回N欠測はなく、上記C欠測が該当 |

`provider_numerical_model_entry_in_domain`はmetadataのdomain判定であり、数値実行・必要descriptorの充足・科学採用の記録ではない。`numerical_model_entry_executed=false`を保持する。旧未資格modelの存在も消さず、direct Cの有無と別欄で示した。

## 不足資格と次に判断できる範囲

- PR q1探索n9/確認n33、H6 q2/q4各n8、H4独立q16 n33はそれぞれ有限cohort。量子shot Nと価格標本数nを区別する。全law期待価格の資格は全PR行で未取得。path別価格・有限平均・Affine model meanを相互変換しない。
- H6 q4の`qualified_for_candidate=true`は固定domainの有限独立検証に限る。全13ラウンド・厳密state・長q systematic・全law平均の資格へ拡張しない。q2 modelは校正点のためfalseを維持する。
- 保存nominal係数共分散・q4予測共分散・paired情報はそのSOURCEとともに転記。完全な係数不確かさ・共有joint covariance・長q systematic・total CIはnull。共有校正groupとX/Y依存を独立とせず、未知共分散に0を補わない。保存誤差統計の再評価や新残差を計算していない。
- H4低q1/2/4、H4 PR q32以降/PF q64以降、H6低時間δ=.1/.2/.4と長q8..256は保存Cが不足。H6早期PRは別回路族を維持した価格資料/モデルが必要。長qの明示数値予測も今回実行していない。
- 固定menuのN側は両systemで公開結果が揃った。次の科学段階は価格の不足範囲・有限cohortを用いるscenarioの可否・期待価格/不確かさの要求を先に決める必要がある。価格が返る行でも総費用採用を自動承認しない。normalization BをCへ再乗算しない。

## 技術検証とSTOP

[結合7件](INTEGRATION_TESTS.txt)はこの独立worktree自身のB API/会計moduleを使用し、mock N＋保存Cで`run_missing`の一括呼出し、null/資格不足時の自動再呼出し抑止、domain逸脱、cohort/校正資格/共有不確かさ、52行coverage、整数転記、公開N差込み時のmissing/FAIL/競合を確認した。予測関数・G入口を例外化して静的処理からの呼出しがないことを確認。[H6既存10件](H6_PROVIDER_REGRESSION.txt)と[H4既存結合7件](H4_INTEGRATION_REGRESSION.txt)もPASS。既存テストの既知q2/q4 golden式照合は保存fixture検査であり、新しい未取得qの科学予測ではない。

これは作者による技術確認で、独立科学レビューや新しい科学性能の証明ではない。[CHECKS.json](CHECKS.json)に固定コード・保存成果の照合を記録。SOURCE/input/raw/result/grant/ledger/marker/KEEP/過去STOPを変更せず、Bの科学計算に干渉していない。

取込み手順と公開N差込み方法は[HANDOFF](HANDOFF.md)を参照。GitHub公開後、remote SHA/root tree・主要成果物を照合し、mandatory STOPする。q8追加検証、未取得q予測、N×C/G実数値集計へ進まない。
