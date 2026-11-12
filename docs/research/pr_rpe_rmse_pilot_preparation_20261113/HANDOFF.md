# A → B / ユーザー：二時点PR RMSE pilot準備

A e2a472422f0206a2f09fcc8e28d6938ebceb6d0fとモデル510fbacd01a2d855202de4c045d8ba1972e5ff3d、B f09af82482aad11460c4bb73d3e13d7ada40be6fを基点とした。実M/κ/RMSE条件は未選択、実signal/N/RMSE・保存数値再解析・C/G0。

## 追加API（科学計算を呼ばない）

```python
from trottertracks import pr_rpe_rmse_pilot as P
inventory = P.pilot_inventory(repo_root)
plan = P.prepare_pilot(fixed_B_config, fixed_A_schedule, inventory, root=repo_root)
# 将来の承認範囲で既に取得されたA recordだけを受け取る。
n_packet = P.handoff_supplied_results(plan, fixed_B_config, fixed_A_schedule,
                                      evaluated_A_records, result_source, root=repo_root)
```

`prepare_pilot`は固定BのA adapter専用、PRのみ。H4はm0/m6、H6はm4/m8を選択する。M/κ/予算のデフォルトなし。全scheduleを検証し、対象二時点以外を評価しない。Bのidentity/lawとAのsource_candidate_idを保全する。

旧CLIを拡張し、新runnerは作らない。

```bash
python3 -B scripts/preflight_pr_rpe_rmse.py --two-time-pilot --inventory-only --out /tmp/pilot_inventory.json
python3 -B scripts/preflight_pr_rpe_rmse.py --two-time-pilot --configuration fixed_B.json --schedule fixed_A.json --inventory /tmp/pilot_inventory.json --out /tmp/pilot_plan.json
```

B configの生成は既存B `generate(I.a_spec(fixed_inputs,'PR'))` を使用する。固定inputsがない現段階で実λから新scheduleを生成しない。旧82bc native方式のCLIも従来のflagなしで利用できる。

## 保存値との対応とSOURCE

P1 public71ba49bb…、科学SOURCE a1a79213…の12候補から候補/field/law metadataのみを利用する。H4 m0:R2/4/8、m6:R4/128/256、H6 m4:R11/16/32、m8:R16/128/256。完全なID・H/state/snapshot・λ出所は[P1_MATCH_METADATA.json](P1_MATCH_METADATA.json)。新B IDへ旧Nを移植しない。

κ一致条件はR>Jで `qPF(R−1)<κλ_R²T²≤qPF R`、R=Jで `0<κλ_R²T²≤qPF J`。二時点の条件の共通部分、元piece allocation、λ定義/SOURCEを満たすことが必要。今回は実値の区間評価・κ選択をしていない。MはN_mと全identityを変える。

P1 lawのwrapper宣言と、SOURCE_INDEXが固定する`src/trotterlib/rte.py`実装をjoinする。確認できたのはideal finite meanのSOURCE対応で、actual sampler/law誤差の保証ではない。不明なlawは保留、矛盾するSOURCE・保存FAIL/数値未解決はcall案nullで自動retryしない。

## packetと局所geometry

- 準備時は全nominal N/十分N/boundがnull。evidence_kind=PREFLIGHT_ONLY、global資格=false。
- 将来の`handoff_supplied_results`は対象二時点の既存A RMSE recordだけを受け取り、A.b_handoff→B.n_from_aを使う。対象外round、旧HP N、actual/十分Nや保証の混入を拒否する。未完了roundはnullで、部分結果も保持する。
- 二時点のnominal Nが存在してもBの十分N条件を満たさない。全schedule RMSE boundは引き続きnull。B assess/cost/selectはここから呼ばない。
- `supplied_disk_geometry`は既存c1のρ=A−δ、asin(δ/A)を旧Hoeffding Nから分離。供給されたraw diskとSOURCEだけを使い、未知はnull、局所診断のみ。実際のdisk作成・allowance適用は承認後の科学scope。

B f09はH4 B0/p7とfull-H PFを区別し、PF scopeを未資格として保持する。H6初期prefix5/R32/q1/δ=T・κ_applied=falseも保存した。前回の技術的BLOCKを科学的PASSへ読み替えない。FULL_H_RTEのA evaluatorは未接続であり、B3やPRへrelabelしない。

## 次にユーザーが固定・承認するもの

[PILOT_INPUT_TEMPLATE.json](PILOT_INPUT_TEMPLATE.json)のnull欄を具体化する。1系、M、paper κ、外側H/state/分割/公式/δ/h/prior/decoderの保持、RMSE targetとQPE/発展誤差予算・合成則/normalization rule、固定SOURCE/数値allowanceを指定する。source/hashは科学入力の再現性用であり、承認証明書ではない。

有限science案は二時点PRの保存gate再確認＋局所bias/phase/radius/finite B/nominal N_X/Y。保存一致なら新mean/reference0が可能。不足時だけ最大2 mean/内部対照2、必要pairだけ最大4 reference actionを対象ごとに固定し、別途明示承認を得る。離れた二時点で連続RPE decoderを実行せず、全RMSE認定・sampling・P2/P3・C/G・最適化は含めない。

論文の誤差二乗和と経験的shot配分をfinite-Kの厳密保証とはしない。state/effective evolution・finite-K bias・H6負時間/初期例外・alias/decoder/全round統計の資格は未解決。詳細は[REPORT](REPORT.md)。**公開照合後mandatory STOP。**
