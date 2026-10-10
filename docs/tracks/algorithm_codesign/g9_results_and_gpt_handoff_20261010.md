# Track B G9：matched-native one-shot failure / GPT handoff

## 判定と停止

**`G9_TECHNICAL_INCONCLUSIVE`、registered native比較0 row。mandatory STOP済み。**

利用者が採用したGPT G8 review §14.2–14.6のG9を一束として準備し、別branchでsourceを固定した。
source固定後のrunner呼出しは一回。新規合成精度のAPI接続でTypeErrorが発生し、比較stageへ到達していない。
これはnative/CTS/return/P5の優位・非優位を示す結果ではない。完了prefixの採点はしない。

- branch：`track-b-g9-p5-matched-native-20261010`
- G8基点：`e4b410746aadcf03c955b5e961c672c04b220de5`（reviewに記載されたG8結果を不変に保持）
- 固定source S / 実行HEAD：`d6ecc6bc82c11158a66d175d78007a121d0fac46`。sourceは実行前にoriginへpushし、remote SHA/clean worktreeを確認。
- source authorization：採用GPT reviewの固定snapshotとcontract。G9は§14.5の委任scopeで実施。
  G8等の旧authorizationを流用・解除したものではなく、別markerを消費した。
- marker SHA256：`80d6f42056f8aadb8ce15dafd3eee21c37ca0857419c28e41015d119339b2dd9`
- raw result SHA256：`93f1bbc25eb931d040276d94eb6a33a373d0bda77494bd0ee6a53954056ec0d9`
- runs=1 / retries=0。marker、contract、source、raw result、旧STOPは不変。

## どこまで完了したか

| stage / evidence | 完了状況 | 科学上の扱い |
|---|---|---|
| P5独立数式・群分解のsource preparation | 31 parents / 63 events / 10 groups、exact Fraction一致を保存 | source準備・数式証拠。native資源結果ではない |
| provider / phase / finite-bit / import focused tests | 23件pass、synthesizer呼出し0 | local semantic evidence。immutable CI・外部再現ではない |
| 結果前inventory | 19 unique keys = 18 reuse + CTS新規1 | 新規角度/精度を結果後追加していない |
| HEAD/source/runtime/protected launch gate | pass、941保護path不変 | provenance |
| 18既存sequenceのidentity検証 | 完了。angle/epsilon/tool/strict phase/hash/countを照合 | 旧primitiveの再利用。G9の新科学結果ではない |
| CTS新規sequence取得 | helper呼出し1回でTypeError、取得0 | 新規sequence/error guardは未取得 |
| runner内のP5/CTS formal audit stage | 未到達 | 準備artifactと区別 |
| shot budget / registered operator matrix / native cost | 0 row、全方式未到達 | primary/secondary比較は未評価 |
| 保存結果監査 | 23 checks pass、source critical69 / old protected941不変 | 保存bytes・provenanceのみ |

## 技術原因と試験の限界

raw reason：`TypeError: cannot create mpf from Fraction(1, 1000000)`。

固定runnerはepsilonを`Fraction(contract['primitive_error'])`へ変換して`numeric.synthesize`へ渡した。
継承したAPIは合成引数を`mp.mpf(epsilon)/4`で構成する。
`mp.mpf`はこのFraction objectを受け取れず、`gridsynth_gates`へ引数を渡す前にTypeErrorとなった。
G8 runnerは同じ精度値をcontractの文字列で渡していた。これは精度の数値やproviderの失敗ではなく、Codex実装の引数型の接続漏れである。

rawの`new_synthesis_calls=1`はhelperに入る直前の予約counterであり、backend成功数ではない。
sourceの評価順序と保存errorからbackend本体callは0と推定できるが、rawには専用backend-entry counterはない。
新規sequenceは保存されていない。新しい合成を呼んで原因を再現・確認する作業は行わなかった。

23 focused testsは数式、V順序、controlled phase、actual adjoint、finite-bit、importを確認したが、
合成関数境界のepsilon型を確認していなかった。import testだけではこの接続を検証できない。
sourceの修正、backend再呼出し、markerの削除、contractの変更は行っていない。

## 固定された科学scope（今回未評価）

known development `p=(1/5,3/10,1/2), x=5/7, m=5`、同じfull first operator moment P5。
新しい3-qubit synthetic providerはG8 reviewのQ0/V1/V2をそのまま採用。
分子、geometry、basis、DF rank/splitは適用外。完全held-out・DF/I0取得優位と呼ばない。

primary direct6方式はordinary / partial-return+tail / closed-P3+tail / local full / closed-P5 full / matched CTS。
最初の5方式のgeneric helperはdiagnostic5 row、計11 row/22 axes。
全方式に同じdirect構成、CZ lowering、adjacent inverse cancellation、strict Rz precisionを固定した。
CTSはPauli収集を使える明示I1 contextで、full first operator meanのfinite特殊化として準備した。
P5 closed fast pathは同研究familyの低次数特殊化で、独立競合法として新規性を採点しない。

epsilon Re/Im=1/200、alpha/axis=49/22000（22軸で0.049）、resource beta/row=1/11000（11行で0.001）、familywise0.05。
provider delta=0はexact Clifford+T model条件。G8の仮想delta=10^-6達成実験ではない。
しかしG9登録operator/phase/error照合stageには未到達であり、今回の成功を示す条件ではない。
primaryはnative T intercept + K*T_prep/readout。CX/1Q/workspace/classical費用を分離する予定だったが全row未取得。
新しいmateriality閾値、cost-aware IS探索、世界最適compiler、一般入力費用保証は設定・主張しない。

## 資源・エラー・禁止処理

- Guard wall=0.265868 s / CPU=0.265716 s / peak RSS=163864 KiB（160.023 MiB）。
  これはguard区間の記録。pre-marker provenance検証と最終serializationを含む総command時間ではない。
- raw result bytes=19462。時間/CPU/RSS/output capを理由とするabortではない。
- per-key acquisitionは完了せず、successful end-key計測値はない。failureはTypeError。
- 新規helper attempts1 / backend本体calls0（source推定） / 新規acquired sequences0 / 再合成0 / retry0。
- registered native rows0 / matrix checks0 / budgets0 / 実量子shots0 / trajectory0。
- LP、fullv4、分子/DF/NPZ、GPU、新p/x/m/provider/grid/backend/seed/precisionは0。
- STOP後はsaved bytes/rational/hash auditと資料作成・公開のみ。generator・matrix・strict guard・synthesisの再評価0。

## 正本と公開資料

- [result_v1.json](../../../artifacts/track_b_g9_p5_native_result/2026-10-10/v1/result_v1.json)
- [one-shot marker](../../../artifacts/track_b_g9_p5_native_result/2026-10-10/v1/one_shot_consumed.json)
- [STOP.json](../../../artifacts/track_b_g9_p5_native_result/2026-10-10/v1/STOP.json)
- [保存値監査](../../../artifacts/track_b_g9_p5_native_result/2026-10-10/v1/saved_output_audit.json)
- [source準備・契約・inventory](../../../artifacts/track_b_g9_p5_native_preparation/2026-10-10/)
- [P5数式・native契約](g9_p5_matched_native_contract_20261010.md)
- [採用GPT G8 review](../../research/track_b_G8_scientific_review_20261010.md)
- [凍結runner](../../../scripts/tracks/algorithm_codesign/g9_p5_matched_native.py)
- [保存値のみの監査script](../../../scripts/tracks/algorithm_codesign/audit_g9_saved_outputs.py)

sourceの完全SHAは実行に先行して固定。結果公開commitはこのSの直接childで、
新しいaudit script/資料と結果、既存索引への追記だけを含む。source code/contract/preparation/markerの変更を含まない。

## GPTへ戻す判断

今回はG9の中心比較に到達していないため、return法の価値・CTSとのwinner・P5 fast pathの資源的優位を判断できない。
G8の肯定/否定証拠とG9 source preparationは保持するが、この失敗から研究上の結論を補わない。

技術的な検討対象は、合成API境界の固定精度値の表現と、その境界を合成せず検証するsource testに限定できる。
それを実装するか、別source/別markerでの新しい実行を認可するかはGPT/利用者へ返す。
現在のG9 markerは消費済みであり、source修正や新stageを自動開始しない。
**mandatory STOP。研究方針・継続・新規性・主methodはGPT側。**
