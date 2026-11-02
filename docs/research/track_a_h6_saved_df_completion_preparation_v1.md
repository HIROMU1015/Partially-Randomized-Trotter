# Track A H6：保存DF入力完成の実装・検証・実行前固定 v1

2026-10-10 JST。[GPT独立レビュー](track_a_h6_df_hermitization_independent_review_2026-10-10.md)と[準備契約](track_a_h6_weighted_hermitization_preparation_contract_v1.md)に従い、新versionを準備した。
**source・synthetic・保存bytes・実行sealの準備完了。実際のH6入力受理/state生成は未実行・未認可。**
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP。

## 来歴と保全

source commit：`d2d45235724d1b956fb250c11a87985ed9200072`。親診断結果`8a3189e69dd461724fa9e2c01ea08562c1c35f8d`、親source`ff24de4bc410234472a416186b773fc7875ae373`。
[親identity](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_preparation_v1/2026-10-10/input_identity_v1.json)でintegrals SHA `edd0a618f86011757cacae481eff44dc637c11a3f64c55b0cbfb7ffbe637e51d`、raw SHA `2943295c2131ce9896fcad83e3dee78763efce492389ce49a6a66b87b57beebb`、hypothetical・summary・旧source・旧STOPをGit bytes/header/hashで照合した。
H6実配列はdecode・再評価していない。旧失敗runの未保存rawとのbytes同一性は主張しない。
[frozen source](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_preparation_v1/2026-10-10/source_freeze_v1.json)はscience 194件・validation3件。既存source・integrals・raw・結果・STOP/freezesは変更しない。
新旧比較は別policy/version/receipt/target hash。H4 legacy/Track B/H8へ変更しない。

## 新sourceと接続

| 機能 | source |
|---|---|
| 重み付きprojectionと独立係数再構成 | [weighted projection](../../src/trottertracks/resource_applicability/ax2b_h6_weighted_projection_v1.py) |
| stdlib親seal・新grant gate | [contract](../../src/trottertracks/resource_applicability/ax2b_h6_saved_completion_contract_v1.py) |
| 保存raw importer/DF受理/state snapshot port | [port](../../src/trottertracks/resource_applicability/ax2b_h6_saved_completion_port_v1.py) |
| 新policy専用snapshot loader | [loader](../../src/trottertracks/resource_applicability/ax2b_h6_saved_completion_loader_v1.py) |
| phase/wall/log/output監視・process group停止 | [watchdog](../../src/trottertracks/resource_applicability/ax2b_h6_saved_completion_watchdog_v1.py) |
| 保存bytes/schema/来歴監査 | [auditor](../../src/trottertracks/resource_applicability/ax2b_h6_saved_completion_audit_v1.py) |
| default metadata/別認可後入口 | [runner](../../scripts/resource_applicability/run_track_a_h6_saved_completion_v1.py) |
| stdlib保存監査入口 | [audit CLI](../../scripts/resource_applicability/audit_track_a_h6_saved_completion_v1.py) |
| synthetic検証 | [test](../../tests/tracks/resource_applicability/test_ax2b_h6_saved_completion_v1.py) |

旧adapter/loaderを上書き・迂回せず、新versionを使う。bounded_solver、matrix-free sector、HF initial、phase canonicalization、primitive sector certificate、atomic_npzを再利用する。
新loaderは入力完成snapshot用。H6 pilotへの実行接続・grantは別段階で固定し、現pilotを起動しない。

## 固定政策・丸め・独立性

linear H6/1.00 Å/STO-3G、12 modes、Nalpha=Nbeta=3、sector400。全19 fragment、signed real lambdaのexact values/returned order、tol-only1e-8、cutoff0を保持する。
SCF/DF再分解、final_rank、rank fallback、fragment削除なし。旧無重み1e-10は旧政策診断だけに残す。
採用候補 `cI+dGamma(Herm(h+k))+sum(lambda*dGamma(Herm(g))**2)`。correctionを交換せず、h+k一回、plain square/no conjugation。
[固定policy](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_preparation_v1/2026-10-10/fixed_policy_v1.json)：`eta_N=N*d_h+N**2*sum(abs(lambda)*d*(2*s+d))`、N12主・N6保存。追加予算1e-10 Ha。
Frobenius normはbinary64。`Decimal.from_float`で60桁の加算/乗算を行い、1%の工学判定余裕を取り、判定上限9.9e-11 Haとする。
この余裕はinterval norm認証ではない。PASS_ENGINEERING / representation_error_certified=false。将来の実データ結果を見て余裕を緩和しない。

raw/postはfinite/layout、exact cross-spin zero、exact alpha-beta一致、post Hermiticity Frobenius≤1e-12を検査する。
独立係数再構成はone-bodyの明示index和、quarticのouter/reorder、独立pair-index antisymで実装した。元診断のmatmul/einsum/swapaxis経路と保存配列で照合する。
係数照合はFrobenius差≤1e-13+1e-10*max(norms)。保存summaryの同じscalar量は差≤1e-15+1e-8*max(abs(values))。
fragment index/lambda/raw/postのarray hashはexact一致。hypotheticalとのprojection配列一致も確認するが、その存在だけで受理しない。

provider truncation、raw DF残差、projection三角和、累積係数差、PF/RTE、roundoff、measurementを別ledgerへ保存する。
独立再構成は規約整合の工学検査で、厳密total uや元integralsとの物理誤差認証ではない。DF係数作用素上界式評価をprovider truncationと同じ閾値で判定しない。
大きいλで同じ偏差、複数寄与和、signed/zero λ、one-body一回会計、構造違反、summary/配列矛盾・改変をsyntheticで検査した。

## ローカルsynthetic結果と限界

[記録](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_preparation_v1/2026-10-10/synthetic_test_audit_v1.json)と[JUnit XML](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_preparation_v1/2026-10-10/synthetic_tests_v1.xml)：122 passed、新経路41・旧入力生成38・旧診断43。
4-mode synthetic CAR占有基底演算子と正規順序係数の独立照合、read-only importer、policy-bound snapshot、fake solver、shared residual matvec上限、旧grant拒否、default/refused CLI、dummy watchdogを確認した。
fixtureはreal artifact decodeとSCF/DF/eigsh/sampling/circuit処理を禁止する。これはローカルsyntheticであり、CI/外部再現・分子H6のstate/spectrum/性能証拠ではない。
失敗時は保存DF receiptを残し、snapshot不在やparent STOPを成功へ昇格させない。old grant再利用/retry/resumeは禁止。

## 実行対象・予算・停止点

[sealed preparation](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_preparation_v1/2026-10-10/sealed_preparation_v1.json)と[認可対象](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_preparation_v1/2026-10-10/launch_target_NOT_AUTHORIZED_v1.json)を固定。
manifest digest：`5d78f8128ff0fb9392a63058ee404859d134d0032fb64f26c6e46bdf18eb0554`。CPU2/worker1/BLAS1、GPUなし。CPU割当はhost専有保証ではない。[環境/観測](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_preparation_v1/2026-10-10/environment_resource_observation_v1.json)参照。
保存入力確認60秒、DF受理120秒、state/snapshot900秒、合計1080秒（18分上限、予測所要時間ではない）。AS8GiB・output32MiB・log64KiB・snapshot展開16MiB。
受理1回、state solver1回、shared matvec≤10000（rmatvec・residual込み）、progress≤256。SCF/DF/signal/trajectory/compileは0。
solverは既存eigsh k1/SA/tol1e-12/maxiter1000/ncv40・固定HF initial、numba/1thread/chunk1を維持する。数値固有vectorのground-state認証はしない。
exclusive output：`artifacts/resource_applicability/track_a_h6_saved_df_completion/2026-10-10/launch_v1`。現在未作成。新execution identity、新grantとexact authorization SHAを別途固定する。
[認可template](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_preparation_v1/2026-10-10/authorization_template_NOT_GRANTED_v1.json)はapproved_by_user=falseであり、runnerは拒否する。旧480秒診断grantを使わない。

認可後だけ、一回の保存DF受理＋bounded state/snapshotを実行し、accepted_df・独立coefficients・DF/state/snapshot receipt・progress・terminal・監査/source identityを保存/公開/remote照合しmandatory STOPする。
予算超過、構造違反、重大なsummary/係数不一致は救済せずSTOP。科学的条件の変更が必要ならGPTへ戻す。
H6の7 cell/36 wrapper pilot、H6本検証/H8、GO/STOPの代行へ進まない。
現在H6_input_accepted=false、N/Gnull、u未認定・UNDETERMINED。
[別段階inventory](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_preparation_v1/2026-10-10/preparation_inventory_v1.json)を現在の正本とし、過去の科学的manifest/STOPを上書きしない。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP。

## 公開後確認

準備公開commit：`9dea896cf295d88d470da474ffbb2f8ec06e968a`。
[独立remote取得後の確認](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_preparation_v1/2026-10-10/remote_verification_v1.json)でscience194件・validation3件、親raw25件、旧science183件・旧raw20件、repositoryリンクを照合した。
既存4119パス・dirty/untracked・rootのレビュー原文を保全し、新しい実計算0、新grantなし、実行output未作成を確認した。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP。
