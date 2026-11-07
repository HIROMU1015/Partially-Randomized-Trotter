# Signed support実H4 metadata-only：一回取得成功・構造的ELIGIBLE

2026-11-07 JST / SERVER1（GPUA100-gleap-01）。**既存Signed Supportを保存H4 LAW/catalogへ正常接続し、全5層exact momentを一回取得した。Σ1>0を確認し、構造的ELIGIBLEでmandatory STOP。**

## 固定条件・SOURCE・入力

保存linear H4/1.00 Å/STO-3G/legacy DF12/8 system qubits/全4電子sector参照/L_D3、m0/q1/δ=.02/r1/K2/coefficient_atol0、N1 actual finite-grid Q・selected full。軸非依存のmetadata評価一組で、X/Y応答は評価していない。

- 科学SOURCE：`71778dae6954ed9a7a6821062a2a6e6be17672c4`。公開SHA・全14変更ファイルの本文/blob/treeと入口/設定/profileのAPI本文を照合後、clean worktreeから一回実行。[照合記録](SOURCE_PUBLICATION_VERIFICATION.json)。
- Kernel：`b23b3c059dd1fdcc86178659311983048dea2257`。契約：`3ac50a98970ad471661cba452423791c7917cae2`。既存kernel14ファイルは無変更。
- N1入力：`744e1a09661ebba6841fe5faf456f88c41c47780` の `artifacts/o2_a1_new_target/2026-11-06/n1/LAW.json` と `PREPARATION.json`全bytes。
- LAW SHA256：`b77980d198cc9edc620daec9bf87df02b66e96575fd85327aed0793f3e86ecef`。catalog SHA256：`ca897db7be52ad30dcfab0512d7530188fc13a9d8ca0c78430809492d01f738b`。
- H hash：`56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5`。preparation：`f02980a5cbec9478aa1f4e6dd010f117d152f89770c40389544cf876dd197299`。
- a=`df-fragment-3:Z0Z1`、b=`df-fragment-6:Z0`、feature=`rotation_coefficient_sign_times_diagonal_support_size_over_2_v1`。partition/tail/policy・元5patternは[固定設定](EXECUTION_SETTINGS.json)に保存した。

## 全5層のexact結果

以下は保存結果の転記。新しい再計算・再fitではない。exact Fraction文字列を正本とする。

| h | ν_h | m₂,h | Σ_h |
|---|---|---|---|
| 0 | `1` | `1` | `0` |
| 1 | `-1878702282177856/8558339509381393` | `8536936706439523/8558339509381393` | `69532480438749347841126879238803/73245175157838542641857534620449` |
| 2 | `-1429842536818257/9007199254740992` | `4492898225899561/4503599627370496` | `78892409423804433296037474490975/81129638414606681695789005144064` |
| 3 | `-1429842536818257/9007199254740992` | `4492898225899561/4503599627370496` | `78892409423804433296037474490975/81129638414606681695789005144064` |
| 4 | `-1429842536818257/9007199254740992` | `4492898225899561/4503599627370496` | `78892409423804433296037474490975/81129638414606681695789005144064` |

全5層のexact質量・patternとsnapshotは [METADATA_RESULT.json](../../../artifacts/scv_signed_support_h4_metadata/2026-11-07/run01/METADATA_RESULT.json) に保存した。h0は定数、h2–4のrotation momentは共通で、保存5正層のmass/label順、feature range、moment整合、call範囲はすべてPASS。

Signed metadata identity：`c5e0c5c26b926eb4a73d1808a1c61222dc45934248c30d8aea4f8fc32c44b7e2`。
Signed control identity：`5e252ff9fa5682850683ced97d007b59fecbf6d7617c561adb4a34d4cc7c7f89`。

Σ1は上表の正のexact Fractionであり、浮動小数のtoleranceなしで非定数と判定した。**構造的適格性だけ**の結果である。finite trainでのrank2、X/Y測定bitとの相関、残差分散低下、学習費回収、RZ改善を示さない。statistical_effectiveness=UNKNOWN、science_ready=false、next_stage_started=falseを維持する。

## 実call・保存・実行負荷

[CALL_COUNTS](../../../artifacts/scv_signed_support_h4_metadata/2026-11-07/run01/CALL_COUNTS.json) と [journal](../../../artifacts/scv_signed_support_h4_metadata/2026-11-07/run01/CALLS.jsonl)：attempt1、LAW/catalog読込各1、signed binder1、SignedControl1、全5層moment一組、h1分類1、すべてRETURN、例外0。retry0。

内部の実観測は、from_record1、scalar/signed Metadata各1、Component324、JSON解析LAW/catalog各2、cells2、pattern_mass10、joint_mean80等で、固定契約の内訳と一致。kernelを改変せずread-only profilingで入口を観測した。静的期待値は別fieldに保存して実観測と混同しない。

禁止call試行0。state solve/eigh、N2 stateロード・engine初期化/Born、sampling/RNG/shot、gate/circuit build・compile・price、OLS/train/rank/paired/calibrationは0。旧224/752 bit・係数・cacheを使わず、engine/O1/O2高速化SOURCEへ変更していない。

入口内の経過時間は保存記録で 0.12671023607254028 秒、process peak RSSは 33792 KiB。CPU/RAM/thread制限の追加、科学worker/manager/GPU起動はない。共有設定・venv・他jobを変更していない。

[SOURCE](../../../artifacts/scv_signed_support_h4_metadata/2026-11-07/run01/SOURCE.json)、[INPUT_IDENTITY](../../../artifacts/scv_signed_support_h4_metadata/2026-11-07/run01/INPUT_IDENTITY.json)、exact結果、CALLS/COUNTS、REPORT、[STOP](../../../artifacts/scv_signed_support_h4_metadata/2026-11-07/run01/STOP.json)、RUN/consoleを新outputへ保全した。保存snapshotと各cellのexact文字列一致を確認し、SignedControlを再実行していない。

## 人工correctnessと最小差分

新入口20件＋既存106件＝126件PASS：[人工log](ARTIFICIAL_TESTS.log)、[結果](TEST_RESULTS.json)。新入口はhash/schema/label/sign/support/5層・Source/env・定数/非定数分類・exact保存・call超過/禁止・partial/STOP・collision/retry拒否を人工bytesで検査した。実H4をsmoke testには使っていない。local人工PASSをCI/独立再現や科学性能の証拠にしない。

既存signed/scalar/law/CV核は無変更。新規のI/O・外側検査・局所call観測・atomic保存/STOPだけを追加した。4索引の追記は新経路の発見用で、元mainへ変更しない。[実行仕様](EXECUTION.md)・[profile](PROFILE_LOCK.json)を参照。

## 次の条件とmandatory STOP

次のfresh学習・paired比較には、独立trainの量と取得法、seed/stream、評価前の係数/配分凍結、全5正層quota、signed range guard、full compiled RZの取得・学習費込み会計、診断基準、call上限・STOPを別途具体化し、ユーザーの明示許可が必要。旧224の予算・seed・guardは流用しない。

Σ1>0はその検討を可能にするmetadata条件だが、追加測定を自動許可しない。今回、新しいshot数・seed・guard、統計的GO/NO-GOを採択していない。重要科学レビューも実施していない。

科学attemptは終了し、STOP.jsonはMANDATORY_STOP・ELIGIBLE・no_retry=true・finalizer_errors=[]。新しい結果commitへ保存し、remote SHA・本文・blob/treeを照合してmandatory STOPを維持する。追加metadata評価・科学retry・fresh train/paired/calへ進まない。旧SOURCE/結果/STOP/grant/ledger/marker、元main dirty/untracked、C1凍結を保全する。

[結果・journal整合](RESULT_INTEGRITY.json)と[保全確認](PRESERVATION.json)はPASS。元main3,087原本とHEAD/status、旧SOURCE2,921原本（自branchの索引2件追記を除く）、旧停止研究1,086・前pilot29・SCV224 raw530原本をbyte照合した。新科学SOURCEのlock対象20ファイルも無変更。旧研究のhistory本文・manifest既存entryを保ち、新結果だけを追記した。
