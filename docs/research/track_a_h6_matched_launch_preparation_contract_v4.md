# Track A H6補完：起動準備 v4（科学runner v3維持）

状態：**SEALED_LAUNCH_PREPARATION_ONLY_NOT_AUTHORIZED / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION / mandatory STOP**。
利用者が旧v3起動時STOP後の技術修正・テスト・新実行準備を指示した。今回、科学runnerを起動せず、新approved grantも発行しない。
同じ研究方針のGPT科学レビューを繰り返さない。次の計算は別の明示認可・新grant・remote照合を必要とする。

## 1. 原STOPと今回の修正範囲

[旧起動STOP・原ログ・監査](track_a_h6_matched_supplement_launch_stop_v3.md)、旧公開HEAD `6a3ce771ba343bd5f53b6991e4be1aef506351c7`を保全。
原runnerは親directoryがないまま`out.mkdir(exist_ok=False)`を呼び、FileNotFoundErrorでscience claim以前に終了した。
Phase A、worker、ResourceTraceは未開始で、MemoryError対策の実H6有効性は依然未検証。

科学計算部分の変更は不要。凍結v3 runner/contract/memory/cost port/source224をそのまま使い、起動準備側に二つのmetadata-only scriptを追加する。
[output-parent検査・起動直前preflight](../../scripts/resource_applicability/h6_matched_launch_preflight_v4.py)、
[metadata sealer](../../scripts/resource_applicability/prepare_track_a_h6_matched_launch_v4.py)。どちらにも科学runnerを起動する機能はない。

親がない場合に必要なancestorだけを作成し、最終leafは作成しない。既存leafは空directory・file・dangling symlinkでも拒否。
repo外・`..`・symlink ancestor・file ancestorを拒否。親のwrite/execute権限を確認し、exclusive小型write probeを作成・fsync・削除する。
これはその時点の検査で、directoryの永続的予約ではない。起動直前に再検査し、最終claimは元runnerのexclusive mkdirに任せる。
他の成果directoryをcleanup・rename・上書きしない。

## 2. source・manifest・新しい実行identity

| 項目 | 固定値 |
|---|---|
| Repository / branch | HIROMU1015/Partially-Randomized-Trotter / track-a-ax2b-h4-post-review-20261010 |
| 科学source commit（不変） | `5a903aea2492c2d211c1e15e7737df0d50694c12` |
| 新metadata source commit | `48b489ca3465ef7f4bb202f61c8ca45e9455271e` |
| 新実行identity | `H6_MATCHED_SUPPLEMENT_V3_LAUNCH_V2_20261011` |
| 新manifest canonical digest | `d350e7dcca318eb729e5cfe43fad0d313cb49ec604f79560aeb3e74f28ce2865` |
| 新manifest SHA-256 | `c7e4dcc354515dd9cdd1bf656aa6cea0cf323fefd204585fc801a1e3f5adb617` |
| reuse registry digest（不変） | `c82aa9223c3f67f52085cf821547b92c37b2e110c78e98c2bb256f7c521ead70` |
| 科学plan digest（不変） | `801e1eea5d09b6fbe60fe66102bc21edb1bd50a0d9aa94ca05a2b0659d1f8cf3` |
| 新exclusive output | `artifacts/resource_applicability/track_a_h6_matched_supplement_v3/2026-10-11/launch_v2` |

[sealed manifest](../../artifacts/resource_applicability/track_a_h6_matched_launch_preparation_v4/2026-10-11/sealed_manifest_launch_v4.json)、[identity比較](../../artifacts/resource_applicability/track_a_h6_matched_launch_preparation_v4/2026-10-11/identity_comparison_v4.json)。
旧科学manifestから変更したfieldは`exclusive_output`と追加`launch_preparation_identity`のみ。
新manifestは元v3 schema/kindを維持し、元runnerのvalidate_manifestで受理する。v4は起動準備のversionで、科学algorithmのversion変更ではない。
metadata source2件と新test1件のGit commit/bytesを別closureとしてseal。旧runtime224・validation5のclosureを増減しない。
新outputは同じv3 namespaceの`launch_v2`。旧`launch_v1`・旧grant・原STOP・旧raw・過去のsealは改変しない。
新manifest digest/exclusive outputが違うため、旧c83e5509のgrantは新実行に適用できない。

## 3. 不変の科学条件・取得順序・資源

[元v3実行契約](track_a_h6_matched_supplement_execution_contract_v3.md)の科学条件・比較・sampling/seed・shot・compiler・capsを全て維持する。
linear H6/1.00Å/STO-3G/12 modes/alpha3 beta3 sector400/保存state/T0.8、tol-only1e-8/rank19/cutoff0・signed generation order・採用DF projection。
B0/B2共通prefix5/10/15、B1 full19のS2/S4、92候補、epsilon_sig 0.05/0.01/0.005/0.001。
旧92 signal・84 B3・449探索group・902 wrapperをbytes/task/seed/event/compiler/両軸/terminalで再利用する。
不足143の先頭原index447 `B1_p19_q8_4th`を単独取得。成功時だけ残142を原順で最大2 workerへ。
全592探索を受理してから原B2上位2/精度・union最大8を固定し、独立32 trajectory/X/Y compile。探索8とpoolしない。
新wrapper最大798、統合最大1700、元上限1708/call/instruction guard不変。失敗時救済・retry/resume・候補追加・proxy補完なし。

CPU [0,2,5,6]、cost最大2 process/threads1、coordinator AS soft2/hard8 GiB・cost8/8 GiB・aggregateRSS18 GiB。
BLAS/OMP/Numba/Rayon/Qiskitのthread制御・sector/cache・streaming fingerprint・single-owner wrapperを維持。GPUなし。
phase/total wall capなし、memory/output/log/call/instruction capは元値。科学的精度・compiler・メモリ上限を変更していない。

## 4. 今回の検証と限界

[新synthetic/process test](../../tests/tracks/resource_applicability/test_h6_matched_launch_preflight_v4.py)20 local PASS、skip0。
[JUnit/log/summary](../../artifacts/resource_applicability/track_a_h6_matched_launch_preparation_v4/2026-10-11/synthetic_validation_v4.json)。親欠落/既存/used leafを実際の子processで検査し、
synthetic stand-inの同じ`mkdir(exist_ok=False)`が成功/拒否されることを検証。実H6 runner・分子inputはテストに使用しない。
旧189検査は既存証拠として保持、今回再実行していない。新20検査をH6科学PASSやAS8 GiB feasibilityと呼ばない。

実際の保存入力・source・旧result・manifestを静的照合し、新parent2 directoryだけを準備。
[parent作成・write probe receipt](../../artifacts/resource_applicability/track_a_h6_matched_launch_preparation_v4/2026-10-11/output_parent_preparation_v4.json)、
[親再検査receipt](../../artifacts/resource_applicability/track_a_h6_matched_launch_preparation_v4/2026-10-11/preflight_parent_recheck_v4.json)、[静的準備照合](../../artifacts/resource_applicability/track_a_h6_matched_launch_preparation_v4/2026-10-11/preparation_verification_v4.json)。
科学runner起動0、H6 build/compile/sampling/signal/DF/state生成0、新approved grant0。
葉`launch_v1`と`launch_v2`は両方未作成のまま。親が作成された事実は旧STOPを成功へ書き換える根拠ではない。
旧v2 raw5220/旧source216/入力33、新科学source224が一致。数値uは経験的、母平均winner・ground-state・化学的精度未認定。
[baseline/保全](../../artifacts/resource_applicability/track_a_h6_matched_launch_preparation_v4/2026-10-11/preservation_v4.json)、[GitHub再取得照合](../../artifacts/resource_applicability/track_a_h6_matched_launch_preparation_v4/2026-10-11/remote_preparation_verification_v4.json)。

## 5. 将来の実行認可対象と起動順序（今回は実行しない）

具体的な認可対象は、上記metadata commit・science commit・新sealed manifest digest/SHA・reuse digest・`launch_v2`に結び付けた
**「科学runner v3によるH6補完143探索groupと原規則の独立確認を、新実行identityで一回取得すること」**。
未承認[grant template](../../artifacts/resource_applicability/track_a_h6_matched_launch_preparation_v4/2026-10-11/authorization_request_template_v4.json)のschemaは元`h6_matched_authorization_v3`、approved_by_user=false。
新grantはCPU/exclusive output/source/manifest/reuse/one_shot/retry=false/resume=false/H8=falseへ結び付ける。
新manifest内のlaunch identity・metadata sourceもmanifest digestに含まれる。旧grantを編集・再利用しない。

将来の明示認可後だけ、次の順序で実施する。

1. 新authorization-only grantを発行・commit/pushし、独立remoteからbytes/hashを照合する。
2. 新manifestとgrantのSHAを指定してmetadata preflightを実行し、親存在/書込み・未使用leaf・source/input/environment/reuseを再検査する。
3. 正常なreceiptを確認した同じ維持中のforeground execution sessionで、元v3 runnerを一回だけ呼ぶ。detached sandbox ownerに任せない。
4. 元Phase A/B/Cの条件に従い、成功/STOPともraw/partial/missing/監査を公開・remote再取得してmandatory STOPする。

preflight入口（このcommand自体はscienceを起動しない、receiptは未使用の新pathを指定する）：

```text
python scripts/resource_applicability/h6_matched_launch_preflight_v4.py
  --manifest <sealed_manifest_launch_v4.json>
  --manifest-sha256 c7e4dcc354515dd9cdd1bf656aa6cea0cf323fefd204585fc801a1e3f5adb617
  --authorization <new authorization-only grant>
  --authorization-sha256 <new grant SHA256>
  --receipt <fresh receipt.json>
```

親が欠落している環境では、新leaf未使用と固定identityの照合後に限り`--prepare-parent`で必要なancestorを準備できる。
Gitは空directoryを保存しないため、GitHub取得だけで親が存在すると推定しない。新sealに記載された絶対worktree・同じ環境が実行対象。
元runner入口・source/CLIは変更していない。実行前検査が失敗したらrunnerを呼ばずSTOP。metadata preflightだけのPASSは実行認可ではない。

今回の到達点は準備の公開まで。**H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION / mandatory STOP / next_stage_authorized=false**。
H8、追加探索、科学的GO・PR優位性・論文化判断を認可しない。H6 map/独立確認完成後にGPTへ戻す。
