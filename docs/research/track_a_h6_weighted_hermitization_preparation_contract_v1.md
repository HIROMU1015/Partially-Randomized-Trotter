# Track A H6：重み付きHermitization・保存DF入力完成の準備契約 v1

2026-10-10 JST。ユーザーが[GPT独立レビュー](track_a_h6_df_hermitization_independent_review_2026-10-10.md)に沿って進める方針を示したため、研究判断を取り込み、次のCodex作業単位を固定する。
**本契約は実装準備の指示と設計であり、H6入力受理・state生成・pilotの実行認可ではない。今回は文書/metadataと保存bytesの静的監査のみ。**
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION。新source、実行caps・環境/CPU・新grantは次の準備で固定する。

## 根拠・来歴

独立レビューはユーザー提供の原文をexact bytesでcopyした。SHA-256：`bbbb6b963f521c103935582624ed4a0c875bd82b0fd791647de17b57ff917203`。
Codexによる新たな独立科学レビュー・再計算ではない。ユーザー側の原文を編集・削除しない。
[来歴/identity監査](../../artifacts/resource_applicability/track_a_h6_weighted_hermitization_review_intake_v1/2026-10-10/identity_audit_v1.json)は診断結果`8a3189e69dd461724fa9e2c01ea08562c1c35f8d`、source`ff24de4bc410234472a416186b773fc7875ae373`、grant/manifest、integrals/raw/hypotheticalのhashを照合した。
source187件・validation2件・raw25件をGit blob/local bytesで確認し、NPZのheader/data SHAをstdlibで検査した。NumPyによる実配列decodeなし。
[既存診断結果](track_a_h6_df_diagnostic_result_v1.md)と旧STOP/source/freezeを保存する。
レビューの上界式・保存スカラー代入値はレビューの導出/評価として扱い、数値certificateへ格上げしない。

## 採用する研究方針

linear H6 / 1.00 Å / STO-3G / 12 spin modes、Nalpha=Nbeta=3・sector400を対象とする。
新診断のactual rank19、19個のlambdaの値/符号/returned order、tol-only1e-8・cutoff0を保持する。
SCF/DF再分解、final_rank追加、rank fallback、fragment削除なし。
旧無重み`1e-10`は旧契約との比較診断として保存し、新受理政策の唯一の条件にはしない。

raw `Hraw=cI+dGamma(h+k)+sum(lambda*dGamma(g)**2)`に対し、採用候補は
`Hacc=cI+dGamma(Herm(h+k))+sum(lambda*dGamma(Herm(g))**2)`。
補正one-bodyを別値へ取り直さず、h+kを二重加算しない。平方へconjugationを追加せず、g†gへ置換しない。
全fragmentと補正one-bodyの前後hash・raw norm・偏差・relative値・fragment別寄与を残す。

レビュー§7の`eta_N=N*d_h+N**2*sum(abs(lambda)*d*(2*s+d))`を、rawから再評価する新gateを準備する。
N=12で全Fockを覆う評価を主判定に用い、N=6も記録する。projection追加予算はレビュー提案の`1e-10 Ha`（DF tolの1%）を採用する。
これは新しい工程政策であり、原論文の定数ではない。丸め方式/判定余裕を結果前に固定し、合格labelはPASS_ENGINEERING、representation_error_certified=falseとする。
数値norm・和の上側評価を保証しない限り、厳密certificate・総u・ground-state認定を主張しない。

構造検査と独立係数再構成を併用する。重み付き量が小さくてもspin/sector/layout/有限性/符号・規約の違反を受理しない。
normal-orderのone-bodyとdouble-antisym二体を別に確認し、補正one-body projectionの変更も別会計で加える。
raw→integralsの残差、provider truncation、projection追加変更は別ledgerにする。採用DFへのPF/finite-RTE・数値roundoff・測定統計と混ぜない。
係数Frobenius normを作用素normと同一視せず、provider truncation値と係数残差上界式を同じ閾値の同じ量として扱わない。

## 次のCodex作業単位（まとめて準備）

1. 新policy/version・saved raw importer・全parent/hash bindingを作る。旧adapterは変更しない。
2. 重み付きgate、post-Hermiticity/sector/spin検査、独立係数再構成、丸め/判定余裕・summary比較規則をsyntheticで検証する。
3. hypothetical診断物を名前変更だけで採用せず、新policy受理receipt/target hashを設計する。必要なら同じprojection bytesかを確認する。
4. 保存integrals/rawからの入力完成portと新policy専用loaderを用意する。bounded sector solver/snapshot writerは再利用する。
5. 新source・source closure・新実行plan/caps・環境/CPU・input/output sealを固定し、認可対象を提示する。

新しい準備sourceの実装とsynthetic追加のたびに同じGPT科学レビューへ戻す必要はない。
重大な意味論/primary target/coefficient policy/比較独立性の変更、予算超過、構造違反、重大なsummary/係数照合不一致が出た場合に戻す。
今回、上記sourceを実装済み/検証済みと扱わない。[machine-readable work package](../../artifacts/resource_applicability/track_a_h6_weighted_hermitization_review_intake_v1/2026-10-10/implementation_work_package_v1.json)へ必要作業と未固定項目を記録した。

## 再利用と必要な新接続（静的確認）

| 現行source | 再利用・不足 |
|---|---|
| [旧tol adapter](../../src/trottertracks/resource_applicability/ax2b_h6_input.py) | DF再呼出し＋無重みgateの凍結経路。保存raw受理用の新versionを作る |
| [旧H6 loader](../../src/trottertracks/resource_applicability/ax2b_molecular_ports_v3.py) | load_h6_snapshotも各変更量≤1e-10を再検査する。新policyを旧loaderへ迂回注入せず、policy-bound loaderを別versionで準備 |
| [旧入力生成port](../../src/trottertracks/resource_applicability/ax2b_h6_input_generation_port_v1.py) | molecule/DF生成を含む入口は流用しない。bounded atomic_npzやphase/state保存の構成は再利用候補 |
| [bounded_solver](../../src/trottertracks/resource_applicability/ax2b_h6_controller.py) | 400-dimension、fixed HF initial、shared before-call matvec、residual会計を再利用 |
| [DF/sector matrix-free](../../src/trotterlib/df_hamiltonian.py) | PhysicalSector / df_linear_operatorを再利用。full Fock dense構築は不要 |
| [primitive sector certificate](../../src/trottertracks/resource_applicability/ax2b_h4_science_v5.py) | exact cross-spin zeros、complete sector、全primitiveのHermiticityを再利用 |

旧module/source/freezeを変更せず、新実装のdependency closureへ再利用sourceを含める。
real signed lambdaを保存し、abs(lambda)は変更予算だけへ使う。complex/nonfinite・hash変更・符号の無断変更を拒否する。
強いlambda/同じ偏差の拒否、複数寄与和、one-body projection一回会計、sector違反、旧policy/grant誤用、raw tamperをsyntheticで確認する。

## 実行認可と停止条件

入力完成は新しい明示認可後にだけ、保存DFからstate/snapshotを作る。新対象・実行caps・CPU/source/parent/output bindingを結果前に固定する。
旧診断の480秒予算・消費済みgrantをstate生成へ流用しない。現在の新caps/CPU/source/output/grantは未固定。
一回の入力完成で、正式DF receipt・state receipt・snapshot・terminal・監査を保存/公開/remote照合してmandatory STOPする。
engineering input受理とground-state/総allowance/精度適格性は区別する。
H6の7 cell/36 wrapper技術pilotはさらに別認可。H6本検証・H8/GPUへ自動進行しない。
RQ-R/補助RQ-P1、旧H4 legacy結果、比較契約・Track Bを変更しない。サイズ比較の前にH4新policy bridge/DF政策差の整理を残す。

現在は研究方針の取り込み完了、実装準備が次。H6_input_accepted=false、N/Gnull、u未認定・UNDETERMINED。
[別intake inventory](../../artifacts/resource_applicability/track_a_h6_weighted_hermitization_review_intake_v1/2026-10-10/review_intake_inventory_v1.json)をこの段階の正本とし、旧科学結果/manifestを上書きしない。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP。

## 公開後の確認

研究方針取り込みcommit：`9714208929bdc3fb0722d7fa3c638b584f777360`。
[remote取得後の検証記録](../../artifacts/resource_applicability/track_a_h6_weighted_hermitization_review_intake_v1/2026-10-10/remote_verification_v1.json)で、レビュー原文のbytes一致、診断science187件・validation2件・raw25件、旧source183件・旧結果20件とrepositoryリンクを確認した。
既存4111パスとdirty/untracked、元のレビュー・root Markdownを保全した。新しい科学計算・source実装・grantなし。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP。
