# GPTへのTrack A PM-1最終実行前レビュー依頼

2026-10-04。今回はPM-1の結果前authorization draftをレビューします。
科学データを開かず、最終booleanの固定後に一回だけ実行できる条件が整ったかを判定してください。

## 依頼と停止条件

前回の `APPROVE_PM1_RESULT_PRIOR_AUTHORIZATION_DRAFT` を受け、
source・sealed plan・科学条件を変更せず、別commitでauthorization draftを固定しました。
現在の運用statusは `PM1_AUTHORIZATION_DRAFT_FROZEN_AWAITING_FINAL_REVIEW` です。
JSONの `final_review_approved=false` により、現runnerは実行gateを拒否します。

レビューだけでscience runnerを呼ばず、分子NPZ/NPY/pickleのresolve/stat/hash/load、
旧runtime/cache、H4 signal、sampling、circuit build/compile、GPUを実行しないでください。
Track B、全repository tests、既存M1/M2の再実行も対象外です。

`APPROVE_PM1_EXECUTION` の場合も、まず承認記録を別auditへ保存し、
authorizationの `final_review_approved` 一項目だけをtrueへ変更して別commitで固定します。
科学条件に差がないことを照合した後も、利用者の明示launch指示までは開始しません。
この限定的なfinalization手順も今回のレビュー対象です。承認前にtrueにしたとは記録しません。

## 固定identity

- Repository / branch：`HIROMU1015/Partially-Randomized-Trotter` / `pr2-v4-s2-parallelization-20260928`。
- M1/M2 evidence基準：`b6e65c6123475add5e620ec1064f361378bead95`。
- actual PM-1 source：`fd7552edc0334ccf57ecf501a128c85c8d22822a`。
- 前回準備review bundle：`d49555f0f1144aa1de6925ace249dc380ae6f4a6`。
- authorization draft commit：`b6175a0f4fdeb1d2f0cd61c09dce37675624b73a`。
- 本依頼・監査・索引を含む正確なreview bundle commitはhandoffで示します。

| 対象 | SHA-256 |
|---|---|
| sealed plan | `cae692bee2be748ddbf17bace2a5652613537a244fe94238cdf669f5d2ca5624` |
| authorization draft | `3916f1e050bf4d4174f0b2b5d4d1d82cdb12396f9eefe3a34a75e496e716377d` |
| authorization schema | `675453d1a93019db0857743cbbc626cbadd4f15536bfbb280349e89acbf10941` |
| authorization review audit | `1429af3feb6ba08e2595de53f1de8300f822f46fec7d727eda1726763c4cbf71` |
| authorization review manifest | `72326ad83878a4cc28d9c9483170b5520964bd6454ea093eba62f57bc1bf59db` |
| contract module | `08b3acbbd47f10196da19560e36f06ded4fb6f69f6679548aebd68274248a7ad` |
| science module | `62f356010c9a1b81c6828dc87230ff37a79a15541ceefe8f552ee0b0e59adbed` |
| science runner | `c2907f80c96c4833639ca54c319dd9442bf195903243276bc487127bd68cfde0` |
| guarded test runner | `f98619917cbd796354033ae7e8c79efcc08018cbfe68c80ab440a6599ecc90d0` |
| PM-1 test | `cfc30de9557831c03e499654718e7093cd03b9a7633de58ed5d53f68ac2f252a` |

plan fingerprintは `144824b70264dd3d7d1d22898afbb1f9e1c248f105ecac8096f31a2c8f78ec9e`。
134 source hashはplanとauthorizationで一致し、source commitのblobと同一です。
8 candidate fingerprintsと16 unique wrapper keys、環境identityも固定しています。
保存M1-A/M1-B1 JSONと生成済みPM-0/PM-1 preparation manifestを変更していません。

## 読む資料

1. [PM-0証拠帰属・機構解析](pr2_post_m2_evidence_attribution.md)。
2. [固定PM-1契約](pr2_pm1_nearby_discard_contract_v1.md)と
   [結果前authorization draft・固定command](pr2_pm1_discard_execution_authorization_v1.md)。
3. 上記source module/runner/testと
   [sealed plan/schema/source/access/test準備監査](../../artifacts/resource_applicability/pr2_pm1_discard_preparation/2026-10-04/)。
4. [authorization JSON・schema・レビュー監査・manifest](../../artifacts/resource_applicability/pr2_pm1_discard_authorization/2026-10-04/)。
5. [前回準備review依頼](pr2_pm1_preparation_external_review_request_fd7552e.md)と
   [研究概要](研究概要・現状.md)のPM-0/PM-1段階。

旧契約とpreparation auditの未認可・未push等は作成時点の履歴です。
現在の拘束力はこのdraft段階で明示し、過去の監査や科学結果のbytesを公開に合わせて書き換えません。
必要な追加文書を参照する場合は日付・拘束力・sourceとの整合を精査し、旧promptを新規認可と解釈しないでください。

## 科学範囲と終了後の扱い

development H4 linear 1.00 Å、STO-3G、DF rank12、8 system qubits、T=0.8、
同じsnapshot/stateへのB0 discard rank4/5 × q=1/2/4/8、r=K=0だけです。
delta=0.8/0.4/0.2/0.1。追加・除外・置換・再探索はしません。

上限はsignal8、deterministic full wrappers16、development hash/load各1、CPU1、BLAS各1。
random sampling、held-out、quantum shot、GPU、cache reuse、retry/resumeは0。
full-H targetは保存M1-A JSON、biasはdiscard＋PF総biasでpure分解はnullです。
accuracy不適格も記録し、二軸compileを行いますがmatched-workへ入れません。

新8構成＋保存済みdevelopment comparator5件のpoint比較だけです。
既存B2/B3の再compile、新しい10% GO基準、formal CI、厳密winnerを導入しません。
compiler、環境、fixed root/output/commandはauthorization文書のとおりです。

成功は `PM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW`、failureは `IMPLEMENTATION_GATE_FAILED`。
不適格と実装failureを分け、failureのpartial/null ledgerと未解決予約を0計算と読み替えません。
どちらもmandatory STOP、next-stage=false、research_decision=nullです。
その後に研究方針reviewへ戻り、PM-2、追加trajectory、別geometry/分子、strong synthesis、
higher-order PF、energy/RPE接続、Track B統合を自動実行しません。

## 科学データを読まない検査

限定7-file suiteは201 passed、fail/skip0、禁止データアクセス試行0です。
tiny2-qubit synthetic compileを含むlocal実装検査で、H4科学結果、immutable CI、外部再現ではありません。

実際のcommitted draftはfinal booleanがfalseで実行gateが拒否されました。
trueを与えたpositive gateは**in-memory booleanとauthorization blobをmockした仮定検査だけ**です。
現在のproduction authorizationがPASSした、または最終review承認済みだとは主張しません。
source・保存JSON・環境の照合は実値で行い、status/source/plan SHA/fingerprint/caps/permissions/
final boolean/root/outputの9改変とuncommitted finalizationの模擬変更を拒否しました。

science runner invocation、private backend、NPZ/runtime/cacheアクセス、GPU操作は0。
PM-1 output/registryは作成していません。過去NPZ4件stat/hashの違反は元auditへ保持しています。
独立review承認や利用者launchをrunnerが別artifactから認証するとは主張せず、
boolean・commit/blobのmachine gateと運用指示を区別します。

再検査が必要な場合は、分子データguard付きの既存限定suiteだけを実行してください。

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
"/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python" \
scripts/resource_applicability/run_pr2_pm1_preparation_tests.py
```

## 判定する点と回答形式

1. source→sealed plan→別committed draftの順序、134 hashes、8候補/16 wrapper、環境が一致するか。
2. final booleanがfalseの現在は拒否し、独立承認後の一項目commitだけでfinalizationする手順が妥当か。
3. fixed root/output、一回限り、no retry/resume、root-local registryと運用制限が整合するか。
4. eligibility・総bias・shots・exact wrapper・null ledger・saved random point比較が契約どおりか。
5. 成功/failureのどちらも研究方針reviewへ戻り、追加科学計算を認可しないか。

先頭に次のいずれか一つを示し、重大な問題と必要な最小修正だけを列挙してください。

- `APPROVE_PM1_EXECUTION`
- `REVISE_PM1_EXECUTION_BEFORE_DEVELOPMENT`
- `STOP_OR_NARROW_WITHOUT_PM1`

このreviewではデータを開かず、science runを開始しません。
承認後も一項目の別commit照合と利用者の明示launch指示まで停止してください。
