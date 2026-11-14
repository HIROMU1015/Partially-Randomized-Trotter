# Full-H κ2 P-2 D7 CALIBRATION：実行前STOP

2026-11-15 JST。主agent自身による技術確認。独立レビューではない。

**取得予定18 compile、実行0、PASS 0、FAIL 0、UNKNOWN／未取得18。**
科学実行前の適合性検査で固定callerと今回の指示の不一致を確認したため、G566を含め実行入口を呼ばず停止した。これは科学的な不合格でも、実行済みcellの失敗でもない。

ユーザーの今回の指示はD7採択とCALIBRATION最大18 compileの実行許可として受け取った。追加grantや承認証明は不要。許可不足を停止理由にしていない。一方、指示§6・§9の固定SOURCE維持と、以下の修正の必要性が両立しないため、無断で取得処理を変更していない。

## 1. 停止理由

### 親event列の直接連結が実装されていない

固定SOURCE `6069605d50270111d1510478b08e11b8e5a18d86` の `scripts/run_full_h_kappa2_p2.py:323` は、pairedの場合も `acquire_path(...)` を先に呼び、その後で `check_composition(r, job['parents'])` を行う。`acquire_path:121` は親receiptを引数に取らず、`P1.pilot` → `native.cost_trial` → `native.sample_requests` を通る。seedを親と同じ値に置き換えるため、生成列の一致は確認できるが、親event列を入力として再使用する実装ではない。

既存P-1 toy fixtureとfake compilerを使い、親2 groupを保存した後の `acquire_path` のみをspyした。**sampler呼出し1回、再生成group 2、fake compiler呼出し2回**。親seed一致・順序付きevent比較はPASSだった。H4 sampling／build／statevector／実compileは0。今回の指示§3.2「同じ親pathを再samplingしない」を満たさない。従来のtoy testのPASSは、seed再生の同一性の証拠であり、この禁止を満たす証拠ではない。

### 較正cellのimplementation evidence出力が空

`records:226` は較正cellについて `C_inputs/path_receipts/law_expectation/model_qualified` だけを返す。`sidecars/implementation_evidence/prices` を追加する分岐は `M5_ROUND/M6_ROUND` に限定される（227–238行）。`execute:301` は欠けた値を `PRICES=[]`、`SIDECARS_UNBOUND=[]`、`IMPLEMENTATION_EVIDENCE={}` として保存する。

既存の `fictional_paths` を使った人工recordで、**較正5 cellすべてに同じ出力欠落**を確認した。実際の価格を生成・再解析した結果ではない。指示§7の実装証拠保存・公開後の束縛を、現在のcallerだけでは完了できない。path receiptにはchecksがあるが、今回要求されたSOURCE付きimplementation evidenceの出力を代替するとは認定していない。

C `c628e9e8447c11c984064b922cbce6c52dc72d48` の `price_implementation_guard.check` はcanonical configuration/roundのrecord用であり、D7の非configuration較正record用の入口ではない。固定caller自身も `check_round_records` で非configuration cellを拒否する。round IDを捏造して既存guardを適用していない。較正record用の証拠接続とC側の受入方法の明示が必要である。

## 2. rank・domain・SOURCE

`A.finite_price_design.check_plan` を既存callerから呼び、D7全体のrank **3**、feature `FULL_H_GROUP_V1=(one,event_count=qR,outer_boundary_count=q-1)`、共通domain、CALIBRATION/HOLDOUT分離を確認した。全設計24 compileのうち今回の実行対象は較正18、HOLDOUT 6は対象外。旧rank-2案へ戻していない。

| 項目 | 確認結果 |
|---|---|
| 開始時branch/HEAD | `codex/d-full-h-p2-caller-prep-20261114` / `a43050c786c741d267ddf322232d888780efc03c`。clean、開始時originも一致 |
| caller/P-1/A checker | 指定実装SOURCE `6069605d5` のblobとbyte一致 |
| native/wrapper/provider | 既存 `P1.source_checks()` PASS |
| A設計 | `7a0c9468b2499fac9fa1cd279ae2980c9dab048e` のmoduleをbyte一致で再利用 |
| cohort/seed | 固定 `PROPOSALS.json` のD7 plan・全cell seedと一致。変更なし |
| import-only preflight | 23 module、失敗0、欠落SOURCE_PATHS 0。保存H4 snapshot/receipt・context binding・reference bytes照合PASS |
| profile | Qiskit 1.3.0、rz/sx/x/cx、opt1、seed17、P0、測定付きfull wrapper、new ZZ。実compilerは未起動 |

H4はlinear 1Å/STO-3G、保存tol-only DF10、一体項を含むFull-H、K2。保存binding・scalar・元物理IDと全seedは [PREFLIGHT.json](PREFLIGHT.json) に記載。保存catalogの値は転記のみで再算出していない。

## 3. 5 cell・X/Y価格と親子対応

| cell | q | R | T | path | 予定compile | X RZ | Y RZ | 状態 |
|---|---:|---:|---:|---:|---:|---|---|---|
| G566 | 1 | 566 | 0.1 | 2 | 4 | UNKNOWN | UNKNOWN | 未実行。較正証拠出力欠落、batch preflight STOP |
| G4528 | 1 | 4528 | 0.1 | 2 | 4 | UNKNOWN | UNKNOWN | 同上 |
| G36223 | 1 | 36223 | 0.1 | 2 | 4 | UNKNOWN | UNKNOWN | 同上 |
| Q8R566 | 8 | 566 | 0.8 | 2 | 4 | UNKNOWN | UNKNOWN | 同上。独立seed表を維持 |
| P2R566 | 2 | 566 | 0.2 | paired 1 | 2 | UNKNOWN | UNKNOWN | 親直接連結が未実装。実親receiptも未取得 |

G566 path0/group0 seed `11290319720649780064` → paired group0、path1/group0 seed `9014355981526777692` → paired group1。これは予定対応であり、取得済み親子receiptではない。親の再sampling、親価格の流用、pairedを独立な第3標本とする操作、境界差Δの評価は行っていない。

## 4. 重複照合（§10.5）

Aの既存照合（`7a0c9468b`、A1計画§8）を引き継ぎ、以下の一次metadataを確認した。全projects／agent履歴の探索は行っていない。

| 調査対象 | 今回との関係／判断 |
|---|---|
| P-1 `a04470c0a` のSOURCE・価格・再検証REPORT | H/state/snapshotとnew ZZの配線証拠は参照可能。q8/R71/T0.8・m3・別cohort・単一pathでありD7較正へ転用しない |
| 旧light `9b83b96b` の元 `price_trials.json` | 保存candidate/旧ID/physical tuple/cohortを照合。旧loweringであり新ZZ価格へ変換不可 |
| 旧T2.0 `7a8bc7f1` の元価格trial索引 | 別T/q/R/cohort、旧lowering。新価格へ換算不可 |
| Dのκ接続 `26d4f23e` | `KAPPA_CONNECTION.md` §6の旧価格差、元physical/law/profileを照合 |
| DのV2 catalog `e7dca5f7` | 保存Full-H preparation/binding/scalarと元価格locatorを参照 |
| 現D1・関連A/C/P-1/V2 branch | 現D1に新しいD7実結果なし。関連refと指定固定commitを観測。各worktreeは変更していない |
| 関連実行中job | 同Linuxユーザーの関連processを読取確認した時点で該当jobなし。他ユーザーの実行状況までは確認対象にしていない |
| 新出力namespace | 確認時点でpublic `docs/research/d_full_h_p2_calibration_20261115/` とprivate `/home/AbeHiromu/projects/d-full-h-p2-calibration-20261115-private/` は未使用。publicには本STOP資料のみ作成、cell出力/privateは未作成 |

受入可能な較正価格は0。本STOPによりlive実行preflightのGOは発行していない。再開時は関連jobとfresh cell出力だけを再確認する。資料所在未確認を「全環境で過去に未取得」とは扱わない。

## 5. correctness・資源・C引渡し

実H4 correctness gateは**未実施／UNKNOWN**。既存P-1/P-2 test記録は再利用し、今回追加した診断はtoyのsampler spyと人工record確認だけである。実科学ジョブのwall/RSSは未測定で、大R資源は引き続きUNKNOWN。独自のCPU/RAM/worker上限や共有環境の変更はない。

Cへ渡せる新較正価格record・compiled receipt・implementation evidenceは**0件**。空の証拠を有効と認定せず、価格guard受入PASSも主張しない。[HANDOFF_C.md](HANDOFF_C.md) は停止理由と必要な接続を伝える資料であり、fit実行依頼ではない。

finite-cohort CI/confidence、sampler対ideal law距離、rare event寄与、group境界一般化、m7–m12外挿、全round E[C]、G・rankingは未資格のまま。決定論PFの2%規則は流用していない。

## 6. 再開に必要な具体的変更

1. 既存callerのpaired分岐だけに、保存済み親eventsからrequestを復元する入口を加える。samplerを呼ばず、親SOURCE・physical row・seed・順序・phase・scalarをbuild前に検査する。normal path、native/wrapper/compilerの固定条件は維持する。
2. 非configuration較正cellのX/Y実装証拠を、実path checks・compile record・SOURCEに結び付けて出力する。canonical round guardを偽適用せず、Cが較正recordを受け入れる方法を明示する。
3. 修正SOURCEの固定と限定toy確認後、同じD7 cohort/seed・最大18 compileで再開する。

**今回はこれらのSOURCE修正を実施していない。** 再開には、指示§6・§9の固定SOURCEを上記の限定変更へ更新する明示許可と、STOP後の再開指示が必要。D7の科学条件・seedやHOLDOUT許可を追加する提案ではない。

## 7. ロードマップへの影響（§10.6）

- D7は今回ユーザーが採択し、較正18 compileを許可済み。ただし取得は0であり、旧「未採択」と現在の実行状態を区別する。
- rank不足は採択D7では解消済み。新たな実行上の不足は、保存親eventの直接連結と非configuration実装証拠接続。
- 依存関係は、限定SOURCE変更の許可→技術修正・固定→G566親receipt→pairedを含む全較正→公開・STOP→Cの別指示によるfit/凍結→別指示によるHOLDOUT。
- ロードマップで更新すべき箇所はFull-H P-2の採択／実装不足／取得未開始。**ロードマップ本体は編集していない。**

## 8. 保全・公開・mandatory STOP

既存D1ブランチを継続し、新規科学SOURCE、A/B/C/Dの他worktree、元P-1/raw/保存mean、grant/ledger/marker、旧STOPは変更していない。文書と診断metadataのみを追加した。commit・push・remote SHA/tree照合の結果は最終チャットで報告する。

**科学実行前STOP。HOLDOUT、fit、係数凍結、外挿、N・G、追加Full-H検証へは進まない。**
