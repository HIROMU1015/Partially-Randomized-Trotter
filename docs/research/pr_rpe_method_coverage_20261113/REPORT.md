# PF・PR・Full-H RTEのRMSE評価経路とcoverage

2026-11-13。B基点`f09af82482aad11460c4bb73d3e13d7ada40be6f`から既存branchを継続。**4方式の候補・供給N/Cの資格判定・費用比較を人工データで確認した。新規14＋関連回帰85＝99件PASS。実N/C/G・科学的最適化は未実行。**

## 現在利用できる経路

| 方式 | 技術的な受入・費用経路 | 現在の科学coverage |
|---|---|---|
| Full PF | 既存PF arm＋FULL_H retained-H identity、供給N/C→assess→exact cost/select | H6の保存13round Nは旧HP型。H4にも過去Full PF作用/価格経路はあるが現B0/p7 Nと別。全schedule RMSE N・対応する価格coverageが不足 |
| Truncated PF | PF arm＋TRUNCATED_H/元H/保持H、切捨てを含む全体RMSE証拠を別に要求 | H4 B0/p7の保存13round HP Nは存在。Full PFへ昇格不可。元HへのRMSE切捨て誤差資格、強baseline探索範囲が未確定 |
| PR | 既存A固定grid→B whole-schedule共通κ、C受動lookup、exact会計 | A0 24診断、A-P1 12旧HP型N、固定R全13round N、C1有限cohort等が存在。新RMSE十分N・全round契約、長q/新R/全law価格は未資格又は欠測 |
| Full-H RTE | 既存Bのcandidate/供給record interface・人工cost/select。DF/Pauli過去SOURCEを特定 | A現RMSE schedule/N providerなし。物理event/preparation/law/normalizationとmean compiled Cのbridgeなし。DF B3はこの方式に含めない |

Hamiltonian/stateは各元SOURCEを維持。現DF対象はlinear H4/H6、1Å、STO-3G、tol-only DF10/DF19、保存state/snapshot、H4 S2/δ.1、H6 m5/δ1.6と早期m0–3例外、h.1、P0。Pauli固定時間資産は別表現・別保存state/law/policyであり同一数値条件としない。再利用関数・固定refと適用境界は[SOURCE_MAP](SOURCE_MAP.md)。

## 必要最小限の変更

1. `pr_rpe_rmse_controller`の既存generate/assess/cost/selectを拡張。PF provider armは維持し、Full/truncatedの比較区分とtarget/retained-Hをidentityへ追加。schema V2で旧IDとの衝突を避ける。決定論PFはκ/Kなし。truncated候補の生成と元Hへの誤差資格を分離した。
2. `pr_rpe_rmse_integration.a_spec`でA B0をTRUNCATED_H、FULLをFULL_H、PR決定論側をPREFIXと明示。A科学SOURCEとphysical ID、H6早期override、signed配分、κ scheduleを変えない。新metadataによりB IDは変わるため過去recordをaliasしない。
3. assessに各round/X/Yのcoverageを追加。保存record存在、値存在、資格、出典と理由を区別。全体誤差契約/Nの不備を各roundへ伝え、Cはcohort・種類・law・model/不確かさを保持。Gはnull。OUTSIDE_DOMAIN等のrecordと単一path構造Sを価格として排除する。
4. selectの既定coverageをFull PF/truncated PF/PR/Full-H RTEへ分離。異なる方式の設計自由度に同じ変数個数を要求しない。異なるformulaはsource付き明示menuでだけ比較可能にし、正式探索範囲の採択とは区別。M/time scheduleを出力へ残し、PRの異なるκ roundを混ぜることはできない。
5. 新CLIは既存`SavedArtifacts`で固定Git JSONのmetadataを読み、既存coverageにA-P1の存在情報を追加するだけ。新provider/optimizer/状態管理framework・承認証明機構を作っていない。

DIRECT deterministic／finite cohort／MODEL、全lawと有限cohortの差、C1 MODEL false、巨大整数N、Bと古典価格標本数の非二重乗算を維持する。未知systematic/covariance/CIを0にしない。候補生成・assess・cost・selectを明示的に分離し、metadataから数値providerを自動起動しない。

## 保存metadata coverage

[COVERAGE.json](COVERAGE.json)は既存118物理候補（H4 66/H6 52）と236測定軸。candidate recipe、signed allocation/event数、M/δ/κの意味、H/state/SOURCE/compiler context、N/C存在と新RMSE資格、A0診断を保持する。固定13round N・A-P1 N12件・C1 recordの出所を参照し、**N/Cの科学数値を転記・再計算しない**。A-P1の24軸分を旧HP型として識別した。

このmenuは従来のround別R候補であり、新しい共通κ scheduleではない。全recipeに`current_RMSE_identity_available=false`、全軸に`eligible_for_RMSE_comparison=false`、G/winner=nullを保持した。保存値があっても新RMSE candidateへの再利用を認めていない。

未供給のH4 Full PF、H6 truncated、DF Full-H RTE、Pauli各RMSE経路はroute-level欠測として記録した。架空の物理candidate/λ/event/価格は生成していない。A/C固定成果のSOURCE、raw、結果、grant、ledger、marker、KEEP、既存STOPは変更しない。

再生成（新しい出力pathのみ）：

```bash
python3 -B scripts/inventory_pr_rpe_method_coverage.py --out /tmp/pr_rpe_coverage_NEW.json
```

## 技術検証

[TESTS.json](TESTS.json)と同directoryの7つのlogを参照。Python3.11.0rc1の既存専用venvを使用した。

| test | 件数 | 対象 |
|---|---:|---|
| check_pr_rpe_method_coverage | 14 | 4方式、truncation、B3拒否、κ制約、identity/law/compiler/HP排除、欠測/domain、S0a/局所pilot、formula menu、保存fixture、科学backend非起動 |
| check_pr_rpe_rmse_controller | 17 | 全schedule・exact会計・巨大N・policy・選択 |
| check_pr_rpe_rmse_integration | 12 | A固定grid/H6早期・C受動lookup・人工end-to-end |
| check_pr_rpe_rmse | 17 | A既存RMSE人工/metadata回帰 |
| check_mean_compiled_price | 15 | C既存平均algebra/受動provider人工回帰 |
| check_pf_pr_rpe_pipeline | 12 | 既存N/C/費用pipeline |
| check_rpe_r_selection | 12 | 旧R管理と会計の回帰 |

計99 PASS。既存135件を再実装せず、変更に関連する回帰を選んだ。人工4方式の積和・選択は技術テストであり分子の費用差・RMSE保証ではない。新実Hamiltonian signal/N/RMSE、保存科学数値再解析、C/S計数、sampling、build/compile/transpile、fit、実G/最適化は0。

## 残る科学的判断と次段階

強いPF baselineには、元H Full PFのformula/幅候補を適切に含む有限探索と同じRMSE target/state/prior/decoder/価格unitのN/Cが必要。H4 B0固定対照だけでPR一般優位を結論しない。truncated PFをどこまで含めるか、formula/幅範囲、finite-K RMSE/error合成・κ定義の正式採択はユーザー判断でありBでは確定しない。

A二時点pilotは局所診断、C S0aは単一path S照合。いずれも全round N/平均Cの穴を埋めない。A/C最新準備は参照のみで無条件mergeせず、既存A510/C094 lookupを保全した。

技術で解決できるのは新公開packetのschema/SOURCE/identity照合とadapter追従。保存値のRMSE再解析、新N/平均C、Full-H RTE物理bridgeの科学検証、全law/長q資格・共有共分散・全RPE誤差契約は科学的判断と明示許可を要する。[HANDOFF](HANDOFF.md)に担当別I/Oと、保存再利用を先に照合して不足だけを取得する有限計画案を記載した。

commit/push後にremote SHA/treeと主要成果物を照合し、**mandatory STOP**。次の科学pilot、S0a、実総費用比較へ自動進行しない。
