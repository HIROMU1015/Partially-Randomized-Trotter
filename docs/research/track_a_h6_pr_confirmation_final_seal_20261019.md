# H6新A候補の独立確認：最終技術seal（2026-10-19）

最終n・候補/PF・期待方向、seed、科学SOURCE/入力/compiler/runtime、call予算と終了監査を固定した。**科学実行の承認待ち、mandatory STOP。** sampling・native compile・signal・shot再計算・保存値の新統計解析は今回すべて0。有効grant0、公開push0。

今回の利用者指示を、[前段S_STATS](track_a_h6_pr_confirmation_s_stats_20261019.md)の推奨n48/104を採用して最終技術契約を作る承認として扱った。科学実行の承認には広げていない。Phase 0/S_STATSのSOURCE・結果・STOPはそのまま保持した。

## 固定比較

| ε | PR候補 | 最終n | 直接実測済みPF対照 | 期待方向 |
|---|---|---:|---|---|
| .05 | `B2_p4_q2_2nd_r4_K2` | 48 | `B0I_p7_q1_2nd` | PR不利 |
| .0005 | `B2_p4_q8_2nd_r4_K2` | 104 | `B0I_p11_q4_4th` | PR有利 |

linear H6・隣接1Å・STO-3G・tol-only 1e-8/rank19・prefix4・T=.8・P0。既存二次PR、元の無条件finite-RTE、K2/r4、通常signed bias、allowance倍率1。Qiskit1.3.0、rz/sx/x/cx、optimization1、seed17、既存symmetric-directional full Hadamard wrapperを維持する。

主費用は各trajectoryの保存shot×compiled RZのcosine/sine合計。固定PF費用から新PR paired平均を引いた差の符号が、事前方向−1/+1と一致すれば`REPRODUCED`、反対なら`NOT_REPRODUCED`、厳密な同費用なら`INCONCLUSIVE`。`PR_CHEAPER`は別項目とし、同費用ではnull。結果の方向は技術PASSの条件にしない。候補追加・再選択・旧n8/union n32/条件付き標本とのpoolingを行わない。

nは保存値による計画案の採用であり、実powerや希少tailを含む母平均精度の保証ではない。family α=.05・計画power=.8・SD倍率1/2/4の前段設計を再計算していない。K2のsupportは0/2次で、未観測層を0費用にしない。将来の費用集計は新cohortのみ、1 scopeにつき1回。追加CI/p値や保存探索値の再解析を自動追加しない。

## seedの固定と照合範囲

新namespaceは`H6_NEW_PR_HOLDOUT_V1`。40桁SOURCE、固定計画digest、cell、replicaからmaster整数を導出し、既存のmaster-to-step APIをそのまま使う。152 master＋928 step、計1,080整数は内部重複0、既知除外66,238整数との衝突0。eventは抽選していない。衝突時にredrawしない。

除外は旧H6探索/union n32・新A128を継承し、固定δの予約64 trajectory、条件付き系列、高次PFの旧/追加探索、最小3候補の独立確認13/45/45本（準備時予約と実使用の両方）を追補した。H8新A/独立確認、H4の記録済み旧seed/P確認registryも照合した。旧H6 wrapper台帳から8候補各32 master、計256 master/1,152 master+stepが除外集合に含まれることを再照合した。

対象はcutoff時点の既知固定H6 campaignと参照したH4/H8 registryであり、未知のjobやH4 geometryの全内部派生streamまで網羅した主張ではない。異なるseedは数学的なPRNG独立性の証明ではない。[seed監査](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_final_seal/2026-10-19/seed_audit.json)にcampaign別receipt SHAと範囲を記録した。後続campaignはこの固定registryを除外し、科学起動直前に新たに登録されたcampaignとの衝突も再照合する。追加衝突があれば起動せず、新契約として扱う。

Codex BのH4 tol-only Phase 0は読取専用で確認し、その時点のP seedは未固定。BのHEAD/status・資料更新を凍結条件にしていない。

## SOURCE・入力・再利用

科学SOURCEは`bf0c26dae4477de5fe7264a7c91e077661ef9b38`。SOURCE34 file、入力114行、frozen provider877 file、runtime34,273 leaf、Python binary、compiler/profile、保存law/shot/preparationを固定した。2つの最終private契約を`verify`で照合し、両方`PASS_BYTES_ONLY`。各SHAと契約digestは[固定projection](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_final_seal/2026-10-19/source_input_binding.json)と[実行契約projection](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_final_seal/2026-10-19/execution_contract.json)に保存した。

runtimeはPhase 0の全SHA監査に由来するreceiptを使い、今回34,273 leafのdevice/inode/size/mtime不変を確認してSHAを再利用した。変更leafなら再hashする規則を維持し、今回は再hash0。環境全体の未知fileまで封印した主張ではない。

`pr_cost_native.py`、`pr_cost_batch_science.py`、`h8_pr_confirmation.py`、`h8_pr_confirmation_parallel.py`、`h8_key_compile_parallel.py`、`pr_cost_integration.py`はPhase 0からbyte不変。科学kernelを追加していない。新しい処理はmetadata準備/照合、最終nの変更拒否、終了時のraw receipt/ledger監査であり、既存H6 native contextとH8のpaired会計・control/compile・所有process pool・one-shot/witnessを再利用する。

## 科学call予算

| call | q2/n48 | q8/n104 | 合計 |
|---|---:|---:|---:|
| trajectory | 48 | 104 | 152 |
| occurrence | 96 | 832 | 928 |
| RTE event | 384 | 3,328 | 3,712 |
| native evolution | 48 | 104 | 152 |
| cosine/sine wrapper compile | 96 | 208 | 304 |
| control probe | 4 | 4 | 8 |
| independent event oracle | 1 | 1 | 2 |

SCF・DF生成・state solver・新signal・shot再計算・PF再compile・fit・再選択・追加trajectory・保存値統計解析は0枠。scopeごとに保存contextを1回復元してfork共有する。新cohortのpaired費用集計はscopeごとに1回、合計2回を登録する。

## 並列化・grant・終了監査

2 scopeは順に実行し、各scopeでreplica0の4 probe＋oracleを先行させ、そのPASS後に残りをtrajectory単位で並列化する。cosine/sineは同じtrajectoryから2 CPU groupで並列compileする。probeは実際の5検査に応じて最大5並列。driver CPUはworkerと分離し、各groupをdisjointにする。自分のprocess内のBLAS/transpiler内側threadを1にし、外側poolを使う。

worker数・CPU IDは将来の起動直前に、affinity・idle capacity・cgroup・RAM・同時job割当を観測して選ぶ。過去のworker数を上限にしない。今回の観測はaffinity64、利用可能RAM503,055,384,576 byteであり、固定割当ではない。RAM admissionには旧H6 A_COSTの保守的6GiB/process見積りを使い、L worker CPUに対して`max(2+min(5,L), 1+L+ceil(L/2))` processを見込む。driver・trajectory親・axis/probe子を含めた見積りで、資源上限や実測上界保証ではない。

一時的なPSI/CPU競合ではworkerをkillしない。自分の異常、実OOM/継続不能、入出力・科学的な不一致はSTOPする。他jobへのsignal・共有設定変更・他worktreeへの書込は行わない。quota CLIは未公開なので、将来の科学承認で既存gateが認めるquota政策を明示してから有効grantを作る。[並列/監査計画](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_final_seal/2026-10-19/parallel_and_completion_plan.json)にfresh admissionと終了条件を固定した。

scope別の無効grant2件と共通の科学承認receipt案を保存した。すべて実行承認false、CPU割当空、launch command未発行。有効化には別のユーザー承認を記録した新receiptと新grantが必要。旧draftを上書きせず、SOURCE・契約digest・n・seed registry SHA・call予算・専用output/lockに結合する。

正常終了には固定nの全task/両軸、law/seed/cohort、compiler/fingerprint/control、raw SHA、親/子の非払い戻しledger、driver exit0/wait4、全trajectory/axis/probe子のexit0/回収、終了後SHA照合が必要。終了監査で科学統計を再計算しない。逆方向や同費用も保存し、追加sampling・自動retry・次stageへの移行を行わず再びmandatory STOPする。

## 検証とreadiness

SOURCE版の人工/mockは**66 PASS**（既存H6 31＋既存H8 22＋最終seal 13）。候補/n/SOURCE/入力不一致、後続campaign欠落、seed衝突、ledger欠落、worker未回収の拒否を含む。実CLIの無効grant拒否はrunner/witness×2 scopeの4経路で確認し、科学output/lock/witnessは未作成。既存inputとseed metadata原本のSHA保全もPASS。[監査証拠](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_final_seal/2026-10-19/readiness_audit.json)。作者自身の検証であり、別担当の独立レビューではない。

初回の人工testは/tmpとBLAS既定threadのため既存guardで拒否された。自分専用のhome配下tmp・process内BLAS1で66件を確認し、guardは緩和していない。また最初の技術準備で短縮SOURCE引数を渡したためseed生成前に拒否された。そのSTOPを残し、完全40桁SHAを指定した新しい保存先で最終sealを完了した。科学retry・追加seed setは0。

統合readinessは**最終技術seal完了・科学実行承認待ち**。成果物とコードは専用branch`codex/h6-pr-confirmation-final-seal-20261019`へlocal commitし、公開pushしない。raw seed/event・NPZ・private契約・有効grantを公開bundleへ入れていない。

**次の作業：ユーザー**がSOURCEとn48/104、152 trajectory/304 compile/3,712 eventおよび登録会計の科学実行を別途承認するか判断する。quotaを確認できない場合の既存fail-closed政策を採用するかもその承認に含める。**Codex**は承認までSTOPを維持し、承認後に限ってfresh資源・後続seed照合、新しいone-shot grant、科学実行、全結果保存・終了監査・mandatory STOPを行う。公開pushも別指示とする。
