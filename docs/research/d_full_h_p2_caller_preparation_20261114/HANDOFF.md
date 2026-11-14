# Full-H P-2：A/Cへの接続

実装SOURCE：`6069605d50270111d1510478b08e11b8e5a18d86`。正本設計：A `7a0c9468b2499fac9fa1cd279ae2980c9dab048e`。C model/guardは `c628e9e8447c11c984064b922cbce6c52dc72d48` を再利用する。実科学実行・モデル採択は未承認、mandatory STOP。

## metadataの利用

repository rootで次を実行するとmetadataだけを表示する。

```bash
python -B scripts/run_full_h_kappa2_p2.py --design D6
python -B scripts/run_full_h_kappa2_p2.py --design D7
python -B scripts/run_full_h_kappa2_p2.py --design D7_H2
```

`--design D0` は旧18 compile案をrank不足として拒否する。保存した [PROPOSALS.json](PROPOSALS.json) は全cell・physical row・cohort・group seedを含む。[JOB_TEMPLATES.json](JOB_TEMPLATES.json) は未採択の例で、`adopted=false`、`duplicate_preflight_complete=false` のまま実行できない。これらは許可証やgrantではない。

将来、有限scopeの科学実行がユーザーに許可された場合の入口は `--execute-approved-job JOB.json`。一回の呼出しは一つのcellの明示n_pathsとX/Yだけを扱い、後続stageへ進まない。plan採択のboolは処理状態であり、チャットでの科学実行許可を代替しない。今はこの入口を実行しない。

## requestとpaired入力

- `proposal(name)`：Aのcell表/domainを読んだ未採択plan。
- `validate(plan, execution=False)`：A.check_planを最初に実行し、完全なcell集合・q/R/s/T・domain・stage・両軸を確認。
- `physical_row(cell)`：元 `h46_long_time_signal.cid` の規則でFull-H物理row/IDを生成。nonconfiguration cellをB roundへ見せかけない。
- `seed_table(plan, cell_id)`：元 `DF_FULL_H_LIGHT_20261109` seed関数。paired P2はG566 path0とpath1のgroup0を使用する。
- `parents`：上記G566の完全なpath receiptを順序付きで渡す。`row/cohort/index/seeds/events/trial.scalar` を保存したまま使う。親を再samplingしない。

jobにはplan、stage、cell_id、step_seeds、fresh output/private、pairedならparents、HOLDOUTなら凍結モデルの`{commit,path}`を含める。public outputはこの独立worktree内。namespace既存、scalar/physical/SOURCE不一致、gate失敗はSTOPし、自動retryや別seed代替をしない。依存SOURCEの検査・provider `install()`・import-only検査はP-1経路を使う。

## Cが受け取るもの

`records` は供給されたpathごとの直接価格だけから有限cell recordを構成する。今回実価格では呼んでいない。

| cell | 出力 | 用途 |
|---|---|---|
| G566/G4528/G36223/Q8R566/P2R566 | `C_INPUTS.json` のX/Y各 `PRICE_CALIBRATION_RECORD_V1` | 非configuration較正。B round IDは付けない |
| M5_ROUND/M6_ROUND | 同形式のHOLDOUT＋`PRICES.json`、sidecar、implementation evidence | canonical configurationのfixed C guard |

featureは `FULL_H_GROUP_V1=(one,event_count=qR,outer_boundary_count=q-1)`。較正/HOLDOUTのroleとsource/cohort/step_seedsを保持する。X/Yは別のmodel domain。paired cellは親と相関し、n1である。Dはfit、SE、許容誤差、holdout判定を行わない。Cがfitする場合も別途許可と事前固定条件が必要になる。

cell成功後のrunnerは `RESULT.json`、上記records、`SIDECARS_UNBOUND.json`、`IMPLEMENTATION_EVIDENCE.json` を保存してSTOPする。`source.commit` は実行code SHA、`source.result_path` は結果の相対出力先であり、その結果が実行commitに存在するとは主張しない。publish後、Cへ結果のpublication `{commit,path}` を別途渡すこと。

canonical価格はevidence publication commitが確定するまで受入完了ではない。`check_round_records(prepared,cell,prices,sidecars,evidence_commit=公開SHA)` は固定C guardで検査し、公開commitを結んだ `bound_sidecars` を返す。新しいsidecar/C_GUARD_CHECKとして保存する。旧recordや旧evidenceを書き換えない。nonconfiguration cellへround guardを偽適用しない。

`frozen_model_gate` はCが公開したX/Yモデルmetadataのdomain、feature、係数キー、全CALIBRATION cell、design_source、cohort、seed表を確認するだけで、fit/predict/価格判定を行わない。`qualified_for_candidate=false` を要求し、HOLDOUT成功や長時間資格を自動認定しない。

## テストと保全

```bash
python -B -W ignore tests/check_full_h_kappa2_p2_prep.py
python -B -W ignore tests/check_full_h_kappa2_p2_circuits.py
python -B -W ignore tests/check_full_h_kappa2_p1_prep.py
```

P-1既定cohort/index/m、phase gate、new ZZ、parity、compiler/wrapper/cacheは維持。テストの数と制約は [REPORT](REPORT.md)。Pauli B1の変更は別branchであり、このbranchにmergeしていない。決定論PF取得系列と過去P-1 raw/STOPを変更していない。
