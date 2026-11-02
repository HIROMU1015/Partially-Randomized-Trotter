# GPTへのTrack A PM-1準備bundleレビュー依頼

2026-10-04。レビュー対象は契約・実装・保存済みJSON・準備監査だけです。
PM-1科学実行を承認する依頼ではありません。

## 依頼と停止条件

PR-2のpost-M2 Track Aについて、PM-0で確認した近接discard baselineの欠測を埋める
PM-1準備bundleをレビューしてください。研究方針を広げず、別のresult-prior execution
authorizationを作る段階へ進めるかを判断してください。

分子NPZ/NPY/pickleのresolve/stat/hash/load、旧runtime/cache、H4 signal評価、
circuit build/compile、sampling、GPU操作は禁止です。science runnerを試しに呼ばず、
全repository testsも実行しないでください。Track Bは今回の対象外です。
承認されても科学実行へ進まず、別authorization・最終review・利用者の明示launchまで停止します。

## 固定identity

- Repository：`HIROMU1015/Partially-Randomized-Trotter`。
- Branch：`pr2-v4-s2-parallelization-20260928`。
- M1/M2保存値の基準commit：`b6e65c6123475add5e620ec1064f361378bead95`。
- PM-1 actual source commit：`fd7552edc0334ccf57ecf501a128c85c8d22822a`。
- 準備status：`PM1_DISCARD_PREPARED_EXECUTION_NOT_AUTHORIZED`。
- この依頼・PM-0・PM-1準備artifactを含むreview bundleの正確なcommitはhandoffで示します。
  source commitとbundle commitを混同せず、自己参照hashは文書へ埋め込みません。

| 対象 | SHA-256 |
|---|---|
| PM-0 summary | `182d525116a10cda335f696a89d58204f989b3075fd06a3c303677d9f861a313` |
| PM-0 manifest | `18e7e6a4ecd99f8c09e2fb373354659c8d7728279c496bbe17aca472b3afdcaa` |
| PM-1 sealed plan | `cae692bee2be748ddbf17bace2a5652613537a244fe94238cdf669f5d2ca5624` |
| PM-1 source inventory | `9e72965b5cd0c2e8429de45c2743109fb374882125853cba85fdf4a530dc11f8` |
| PM-1 preparation verification audit | `e1102d239abebf325cb9ef588be1f2958b5402b543d0bc541ea754588faae89c` |
| PM-1 manifest | `28509755c6a753d66d1717c80290920c6bad5ae4c1b6c1869a47f44c159268fa` |

PM-1 plan fingerprintは
`144824b70264dd3d7d1d22898afbb1f9e1c248f105ecac8096f31a2c8f78ec9e`。
134 source hashをsource commitのblobと照合し、16 unique wrapper keysを固定しています。
manifest記載ファイルだけを照合し、artifact内に書かれた科学データpathを辿らないでください。

## 読む資料

1. [現在の研究概要](研究概要・現状.md)のPM-0/PM-1停止位置と
   [PM-0証拠帰属・機構解析](pr2_post_m2_evidence_attribution.md)。
2. [PM-1結果前契約](pr2_pm1_nearby_discard_contract_v1.md)。
3. `src/trottertracks/resource_applicability/pm1_discard_contract.py` と
   `pm1_discard_execution.py`。
4. `scripts/resource_applicability/run_pr2_pm1_discard_contract.py`、
   `run_pr2_pm1_discard.py`、`run_pr2_pm1_preparation_tests.py`。
5. `tests/tracks/resource_applicability/test_pm1_discard.py`。
6. [PM-0の軽量CSV/JSON](../../artifacts/resource_applicability/pr2_post_m2_evidence_attribution/2026-10-04/)と
   [PM-1 plan/schema/source/access/test監査](../../artifacts/resource_applicability/pr2_pm1_discard_preparation/2026-10-04/)。
7. PM-0報告とPM-1 access auditが指す保存済みM1-A/M1-B1 JSON、および対応するsource。
   必要な追加文書は日付・拘束力・証拠との整合を確認し、旧promptを新規認可と解釈しないでください。

PM-0 manifestの`LOCAL_UNCOMMITTED_POSTHOC`、準備audit内の未push等は作成時点の履歴です。
今回のbundleへの収録で数値や当時の監査を改変しません。これはlocal evidenceのcommit固定であり、
immutable CI・外部再現・新しい科学計算を意味しません。

公開前の限定再検査も201 passed、fail/skip0、禁止データアクセス試行0でした。
[公開前照合・再検査audit](../../artifacts/resource_applicability/pr2_pm1_review_bundle/2026-10-04/publication_verification_v1.json)は生成済み監査と分離し、
134 source、16 wrapper keys、既存M1/M2証拠7件、旧中央台帳74 entryの保持を記録しています。

## PM-0から採用する根拠と非claim

PM-0はH4 linear 1.00/1.30 Å、STO-3G、DF rank12、8 system qubits、
T=0.8、ε_complex=0.05の保存値をPOSTHOC再解析しました。
M1は登録L_D=0/3/6/9/12、q=1/2/4/8、delta=0.8/0.4/0.2/0.1、
M2はdevelopmentで固定した5構成のtransferだけです。

同じM1候補domainでは固定q=8でもprimary最小はB2 rank3です。
旧16件selectorのprimary RZ regretは0ですが、6指標actual Paretoの2件中1件を落としました。
共通5構成のP感度ではM1/M2とも大きいPでB1が入ります。
したがってqだけによる方式逆転、selectorのprimary損失、異なる候補集合による差をgeometry効果として主張しません。

B0 rank4/5は未登録であり、rank3不適格・rank6適格から補間できません。
この欠測を最小のbaseline反証として調べるのがPM-1です。
一般的な強いdeterministic baselineへの優位性、全prefix最適性、energy/RPEの最終総costは検証対象外です。

## 将来の科学範囲と上限

H4 linear 1.00 Å、STO-3G、DF rank12、8 system qubits、同じdevelopment snapshot/state、
T=0.8のB0 discard rank4/5 × q=1/2/4/8、r=K=0の8構成だけです。
候補追加・除外・置換・再探索、別geometry、使用済みM2 snapshotの再開封を禁止します。

- signal最大8、full measured Hadamard wrappers最大16、development hash/load各1。
- CPU単一process、BLAS各1、random sampling・quantum shot・GPU・cache reuseは0。
- full-H targetは保存M1-A JSONから取得。新しいground stateやexact truncated-H signalは求めません。
- biasはdiscard＋PF総bias。pure discard/PF分解はnullのままです。
- accuracy不適格も記録し、baseline completeness用にcompileしますがmatched-workはnullです。
- 比較は新8構成＋保存済み5 development comparatorだけ。既存randomを再compileしません。
- primaryは軸別shots×compiled RZの和、6指標matched-workは併記します。
  randomの点推定を使う比較からformal CIや厳密winnerを主張しません。
- 成功は `PM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW`、失敗は `IMPLEMENTATION_GATE_FAILED`。
  どちらもmandatory STOP、next-stage=false、research_decision=null。
- 一回限り、resume/retry・別root/output/serverへのauthorization再利用・未解決予約の救済を禁止します。

compilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17。
既存Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1等をplanどおり維持し、環境変更をしません。
別authorizationはまだ存在しません。root内registryはglobal lockではないことも契約に明記しています。

## 準備検査とアクセス履歴

専用49＋PM-0 18＋helper134、計201 local tests passed、fail/skip0です。
tiny2-qubit synthetic compileはH4科学compileと区別します。
許可する再検査は次のguard付き限定7-file suiteだけです。

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
"/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python" \
scripts/resource_applicability/run_pr2_pm1_preparation_tests.py
```

guardはpytest import前にNPZ/NPY/pickle/runtimeのopen/stat/lstatを拒否しますが、OS sandboxではありません。
前attemptで確認scriptがNPZ4件をstat/hashした違反は、利用者へ報告して停止済みです。
load・scienceは0で、文書-only再開後のアクセス0と監査上分離しています。
過去もaccess0だったとは主張せず、使用済みM2 geometryをfresh held-outとも呼びません。

## レビューで判定する点と回答形式

1. PM-0のclaim限定に照らし、近接rank4/5反証の比較集合と上限が最小で妥当か。
2. signal/cost fingerprint、q依存delta、scalar/identity、wrapper semanticsが既存M1と整合するか。
3. 総bias・accuracy eligibility・shots・null ledgerを混同していないか。
4. actual source→sealed plan→別authorizationの順序と134 source identityに漏れがないか。
5. 一回限り・16 wrapper・no retry・全status後STOPが機械実装と一致するか。
6. 既存evidenceと過去のアクセス違反を隠さず、結果と実装検査を区別できているか。

先頭に次のいずれか一つを示し、重大な問題と必要な最小修正だけを列挙してください。

- `APPROVE_PM1_RESULT_PRIOR_AUTHORIZATION_DRAFT`
- `REVISE_PM1_PREPARATION_BEFORE_AUTHORIZATION`
- `STOP_OR_NARROW_WITHOUT_PM1`

この回答だけでは実行を認可しません。PM-1結果も研究判断も作らず、別authorization前で停止してください。
