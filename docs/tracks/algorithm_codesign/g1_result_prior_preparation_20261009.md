# Track B：G1 decision packetの結果前準備

2026-10-09 JST。**`G1_CONTRACT_AND_INPUTS_PREPARED_IMPLEMENTATION_PENDING`**。
構造監査・backend検証は未実施。実行authorizationではない。

利用者の「その方針で進めてください」を、直前に提示した①構造監査範囲の固定・②最大8人工LP契約の
準備に対する指示として扱った。今回の成果は結果前契約・入力・provenance・公開資料である。
[採用方針](../../research/track_b_reassessment_and_gpt_checkpoints_20261009.md)のG1へ戻す一つのpacketを設計した。

基点：`86df033cea392664cf365a80e5f068091dc8f5e8`。
branch：`track-b-g1-result-prior-preparation-20261009`。
既存backend証拠の基点：`7f9062975d9f2b09f12cda6e83e7de1e830beac4`。
旧分類、旧marker、旧source/contract/authorization/結果は変更しない。

## A：独立構造監査の契約

[構造監査contract](../../../artifacts/track_b_g1_result_prior_preparation/2026-10-09/structure_audit_contract_v1.json)
はA01–A10の証明義務を固定する。GPT側の6頂点導出は**独立未検証**のまま。

- 固定7 prototypeから一般解を独立に導出し、非負条件、3次元性、有界性を確認する。
- 6境界平面の3枚組を全20組分類し、特異交点、重複、符号条件を保存して頂点完全性を示す。
- 一般parameterizationと各頂点の有限Taylor degree係数保存を記号的に確認する。
- 旧B2の断面 `r=1-s, 0<=b<=s`、precision復元、zero-mass case、v4変数への対応を確認する。
- canonical samplingと完成構成の選択確率を区別し、数値K3・量子化後のlawへ同値性を拡張しない。

形式変数は`x>0`。実際の`mu=(x²+2)/(x²+6)`は`1/3<mu<1`だが、GPTの頂点主張は
より広い抽象域`0<mu<1`で別に監査する。`x=0`は範囲外。
補助的な有理数代入はoff-domain `x={1/3,2/3,1,2}`、`mu={1/5,1/2,4/5}`に固定し、一般証明の代用にしない。
登録費用表を開かず、登録xでの採点、J1–J3費用評価、registered LP、quantum matrix評価をしない。

判定は`G1_STRUCTURE_PASS_WITH_DECLARED_LIMITS`、`G1_STRUCTURE_COUNTEREXAMPLE`、
`G1_STRUCTURE_TECHNICAL_INCONCLUSIVE`。重大な反例・技術停止ならbackend段階へ進まずG1へ戻す。
構造PASSも資源改善・新規性・元K3全体の同値性の証拠ではない。

## B：最大8人工LPの固定入力

[fixture manifest](../../../artifacts/track_b_g1_result_prior_preparation/2026-10-09/fixture_manifest_v1.json)
に全JSON path、SHA256、元fixture identity、実行順を記録した。

| 順序 | fixture | variables / equality / inequality | 期待する証明 |
|---:|---|---|---|
| 1 | B2_inner | 28 / 13 / 6 | exact primal＋dual、gap 0 |
| 2 | B2_outer | 28 / 13 / 6 | exact primal＋dual、gap 0 |
| 3 | B2_infeasible | 28 / 13 / 7 | exact Farkas |
| 4 | B3_inner | 22 / 4 / 6 | exact primal＋dual、gap 0 |
| 5 | B3_outer | 22 / 4 / 6 | exact primal＋dual、gap 0 |
| 6 | B3_infeasible | 22 / 4 / 7 | exact Farkas |
| 7 | HP100_B2_inner | 28 / 13 / 6 | exact primal＋dual、gap 0 |
| 8 | HP100_B3_infeasible | 22 / 4 / 7 | exact Farkas |

前6件はv2 manifestでNOT_RUNだった人工fixtureのobjectを変更せず保存した。
旧B2/B3-shapedはsynthetic shapeであり、登録RA-D0入力・実問題ではない。
`mixed_scales`と旧`100_digit_infeasible`は今回の範囲へ追加しない。

追加2件はそれぞれB2_innerとB3_infeasibleから、正の可逆な対角変数変換で構成した。
0始まりの列jについて

```
nu_j = (int("1234567890" repeated 10 times) + 2*j + 1)
     / (int("9876543210" repeated 10 times) + 2*j + 1)
old_variable_j = nu_j * new_variable_j
A'_ij = A_ij * nu_j, H'_ij = H_ij * nu_j, c'_j = c_j * nu_j
U'_j = U_j / nu_j; b, f, c0 remain identical
```

全値をcanonical rational stringsで固定し、分母の最大十進桁数は両件とも109。
これは元人工問題の可逆な座標変更で、feasibility/contradiction marginは元座標で保存する。
新たな`10^-99` gapを課すものではない。期待statusの根拠は構成であり、backendの取得結果ではない。
単なるechoより複雑な係数・bounds・証明の扱いを検査するが、scalingで難しさが消える可能性もある。
任意の100桁LP、near-singular問題、production精度への適合をこの2件だけで保証しない。
旧極小gapの取得失敗はそのまま比較資料へ残す。

## 実行順、runtimeと資源上限

[packet contract](../../../artifacts/track_b_g1_result_prior_preparation/2026-10-09/decision_packet_contract_v1.json)
はA→全8入力のecho-only→固定順のsolve＋独立verification→STOPの順を固定する。
echo-onlyは各key一回・計8回まで、optimizationは各key一回・計8回まで。
初回失敗でsuffixを実行しない。別roundtrip LP、extra solve、retry、reconstruction、sign探索は0。

[runtime identity](../../../artifacts/track_b_g1_result_prior_preparation/2026-10-09/runtime_identity_v1.json)
に、既存SoPlex 7.0.0/GMP build、binary、shared libraries、guard、harness source、Fraction verifierを固定した。
既存binary SHA256は`59196dd28bba8b25cc960257f1b255f4f50bc4819aa48a0712e557e125c8b1ae`。
準備時のread-only hash照合は一致。binaryは呼び出していない。
新build、再compile、solver設定・tolerance変更、backend切替は許さない。
binaryが失われた場合も再buildへ自動移行しない。

- packet全体wall：exclusive markerから1,200秒。
- 構造監査：wall 60秒、RSS/address space 256 MiB、一回。
- 各echo/solve/verifier：wall 30秒、同時solver一つ。
- backend RSS/address space：1,536 MiB、新output：64 MiB、sample間隔25 ms。
- CPUは既存guardのwait4/subreaper child scopeを記録・合算する。今回CPU hard capを追加実装したとは言わない。

RSSはguard supervisor＋対象process群のunique PID合算で、共有page、sample間peak、
外側controllerの除外という限界を保持する。outputは新private領域・result artifact・新限定source/文書を
固定scopeで計上し、既存read-only binary・過去証拠・準備入力を除外する。

新しいexclusive marker・key ledgerで実行を管理し、旧runのmarker/STOP/ledgerは変更しない。
source/remote/worktree/runtime/contract/input照合とmarker未存在を確認してから開始する。
source bindingは後の明示実行指示に完全SHAで指定し、commitの自己参照は作らない。

## 失敗分類

旧v2の`ERROR (-15)`をverifier例外で不正証明へ分類した問題を避ける。
旧raw outputやsourceは修正せず、新controllerの仕様として以下を固定する。

| 状態 | 判定 |
|---|---|
| resource/guard failure | G1_RESOURCE_INCONCLUSIVE |
| non-JSON、runtime/readback不一致、payload format failure | G1_TECHNICAL_INCONCLUSIVE |
| ERROR/unknown status、exact payload未取得 | G1_BACKEND_ACQUISITION_INCONCLUSIVE |
| OPTIMAL/INFEASIBLEが期待statusと異なる | G1_FIXTURE_STATUS_INCONCLUSIVE |
| 整形式で取得した証明がexact row/bound/sign条件を破る | G1_BACKEND_INVALID_CERTIFICATE |
| 妥当なupper/lowerだがoptimal gapが非zero | G1_BACKEND_ACQUISITION_INCONCLUSIVE |
| 全8件の予定証明が独立認証を通る | G1_BACKEND_CLOSURE_PASS |

ERROR/unknownはexpected-status mismatchより先に扱う。未取得の証明にverifierをかけて
例外をINVALID_CERTIFICATEへ読み替えない。solver statusだけでfeasibility/infeasibilityを認証しない。
prefixのPASS件数は技術記録として残すが、全体PASS・production採用・科学的positiveへ使わない。

## 未完了と次の境界

今回準備したのはcontractと入力。独立監査script、新しい限定controller、off-domain launch testsは未実装。
**実行可能sourceが固定された状態ではなく、今すぐ実行するためのcommandはまだ提供しない。**
旧v2 `run_backend.py`は旧state・失敗分類・23件の順序へ結び付くため使用禁止。
次の技術準備では別namespaceで限定script/controllerを作り、旧guard/verifierを変更せず利用する。
実fixtureのsolveや本構造監査に先回りせず、off-domain launch分類・marker・no-retryを確認する必要がある。
そのsourceも必要資料としてcommit・pushし、別の明示実行指示を受けて初めてpacketを開始する。
RA-D0 science authorizationの新規作成、v4 production source、登録B2/B3へは進まない。

今回の静的確認はJSON/rational schema、dim/bounds、旧6 object一致、追加2変換の入力恒等式、runtime hashのみ。
数学監査の結果、backend適合、B3>B2、科学的negative、algorithm採択を得たとは扱わない。
結果は全outcomeでSTOPし、構造・backendの双方を同じGPT G1 packetへ戻す。

## 証拠入口

- [preparation静的照合](../../../artifacts/track_b_g1_result_prior_preparation/2026-10-09/preparation_static_checks_v1.json)
- [provenance・公開manifest](../../../artifacts/track_b_g1_result_prior_preparation/2026-10-09/evidence_manifest_v1.json)
- [受領・準備の研究ノート](../../research/研究ノート/2026-10-09_track_b_g1_result_prior_preparation.md)

**今回はLP=0、構造監査=0、backend invocation/build=0、science/synthesis/GPU=0。公開後STOP。**
