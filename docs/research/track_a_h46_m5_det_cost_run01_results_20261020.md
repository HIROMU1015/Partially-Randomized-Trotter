# H4/H6 m5：決定論的回路費用run01正常終了・監査報告

2026-10-20。**承認されたH4 38軸・H6 54軸の独立compileは全92軸で完了した。両stageは全pairの検査を通過し、native exit0・全worker回収を確認した。終了監査940項目PASS、既存3137ファイルは不変。保存4軸と合わせ、今回の適格領域にあるm5決定論48候補の両軸単価96行が揃った。ここでmandatory STOPする。**

科学SOURCE：`635ba563766875d688ad6912499b7f90397d0cc5`。起動補助SOURCE：`9e44ec883348bad024d18442bdb97ebb9946908f`。起動記録commit：`7dd69c386816604a2bc3c620e913ab3fd08bb92e`。終了metadata監査器SOURCE：`c1b419cde004d2b07b41f5727cd4722b5e0e0bc4`。branch：`track-a-h46-m5-cost-sampling-readiness-20261020`。

## 1. 実行結果と終了証拠

| 契約 | 新規pair | compile軸 | native経過時間 | 終了記録（JST、約） | native exit・worker回収 |
| --- | ---: | ---: | --- | --- | --- |
| H4_DET | 19/19 | 38/38 | 1時間08分32秒 | 02:53 | 0・3/3回収 |
| H6_DET | 27/27 | 54/54 | 7時間52分22秒 | 09:36 | 0・3/3回収 |

両completionは`UNIT_COST_STAGE_COMPLETE_MANDATORY_STOP`、errorsは空。外部wait4 witnessもexit0/reapedを記録している。retry・次stageへの移行・signal平均再評価・PR sampling・shot再計算・総費用会計・ランキング・fitは0/false。正確な時刻とwall秒は[run_timing.json](../../artifacts/resource_applicability/track_a_h46_m5_det_cost_run01_results/2026-10-20/run_timing.json)を参照。ファイル書込時刻はmetadataとして記録し、科学的な時間発展のTとは区別する。

ユーザー承認原文は「H4の38軸とH6の54軸を、それぞれ独立した契約で実際にコンパイルしてよい」。系ごとに承認receipt・grant・config・output・ledger・消費marker・外部終了witnessを分けた。[起動報告](track_a_h46_m5_det_cost_run01_start_20261020.md)の初期snapshotと旧DRAFTはその時点の履歴として保持し、完了後に書き換えていない。各grantは一回消費済みで、再使用しない。

## 2. 条件と検査の範囲

線形H4/H6直鎖・隣接1.0 Å、STO-3G、保存tol-only1e-8・自動rank10/19、T=.8、P=0。m5係数`PF_M5_BEST_20D64C2_BINARY64_V1`の11 signed S2、q1/2/4/8、保存入力とpreparation、full Hadamard wrapper・symmetric-directional control・scalar onceを維持した。Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、compiler SHA`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`は不変。新SCF・DF・state・reference生成は行っていない。

取得候補は保存5精度(.05/.01/.005/.001/.0005)の同じεで両軸が経験的ELIGIBLEとなったm5決定論候補。bias・allowance・shotの新計算による選び直しではない。H4はB0/B0I p8/9とfull10、H6はB0/B0I p7/11/15とfull19の登録格子が該当する。保存full q2の各pairを除いて新規compileした。費用適格でないm5候補16件も候補表から削除していない。

各pairでは低control恒等性、高control漏れ、独立occupation oracle、cosine/sine Hadamard対応、保存normalization、compiled状態差を確認した。保存gateは各pair8件、計368件。compiled状態差は全PF共通CT01の1e-7未満、その他は1e-9未満で、phase alignmentなし。cosine検査通過後にsine compileへ進んだ順序も照合した。

これらは単一保存入力状態についての技術検証であり、全状態のoperator norm保証・厳密bias certificateではない。終了監査は保存gate/receiptの整合と元閾値の確認に限り、vectorを復元してnormを再計算したり回路を再生したりしていない。

## 3. 単価の保存と再利用

- [新規実測92軸](../../artifacts/resource_applicability/track_a_h46_m5_det_cost_run01_results/2026-10-20/direct_measured_axes_92.csv)：各軸のRZ/CX・depth・circuit size、wrapper fingerprint、preparation、compiler、compiled QPY SHA、receipt/result SHAを保存値から転記した。
- [有効領域の直接実測96軸](../../artifacts/resource_applicability/track_a_h46_m5_det_cost_run01_results/2026-10-20/active_measured_axes_96.csv)：新92軸と保存full q2の4軸を結合した。H4 20候補/40軸、H6 28候補/56軸。新実測と保存再利用・別CT01受入を区分した。model推定はこの表に含めていない。
- [候補88行の費用取得状況](../../artifacts/resource_applicability/track_a_h46_m5_det_cost_run01_results/2026-10-20/candidate_cost_coverage_88.csv)：決定論48件が両軸直接実測、16件は現在の適格領域外として保全、PR24群はn_Eと個別実行承認待ち。

保存12軸は全体を保全しているが、今回の適格領域で使用するのは4軸だけ。残る8保存軸を新領域の費用として補っていない。H6 full q2の元STOPをPASSへ変更せず、既存の独立CT01受入に結合した保存結果として扱った。[準備・再利用監査](track_a_h46_m5_cost_sampling_readiness_20261020.md)の来歴区分を維持した。

単価は各軸を別々に保存した。shotとshot込み総RZはnullで、0費用を意味しない。候補間の費用順位、5精度の最良候補、PR削減率は今回評価していない。H4 legacy DF12・H4 tol-only F/FP・H6強化S2/S4の既存実測/model結果は保全し、旧主profileと新m5 native系列の同一性を無条件に認定していない。PF直接788軸の別研究とは結果・契約・承認を混合していない。

## 4. 終了監査と保全

[終了監査器](../../scripts/audit_h46_m5_det_cost_results.py)はSOURCE/input/compilerのsealを再照合し、result SHA、両軸receipt、保存8 gate、wrapper/compiled QPYのbyte SHA、非返金call台帳、cosine→sine順序、native exit・worker回収を確認した。公開bundleの構造検査37項目もPASS。新人工5件は閾値境界・phase補正・未受理gate・異常metric・compiler混在の拒否を確認しPASSした。作成者自身の検査であり独立レビューではない。科学実行コードは変更していない。

| 項目 | 確認結果 |
| --- | --- |
| metadata監査 | 940項目PASS |
| 保存gate | 368行、元receiptと一致 |
| H4 call台帳 | 1,012ファイル、登録kind/indexと完全一致 |
| H6 call台帳 | 1,430ファイル、同上 |
| private run files | 3,482ファイルのSHA inventory |
| 旧SOURCE/raw/grant/ledger/marker/STOP | 3,137ファイル不変 |
| PR grant | 旧2 DRAFTは無効・未起動のまま |

QPYは各軸の作成時SHAと一致した。vectors.npzは終了時のbyte SHAを記録したが、作成時の別SHAへ照合したものではなく、復元・再生・norm再計算はしていない。`quarantine_*`と`unvalidated_metrics.json`は検査前の保存名を保持している。費用受入の根拠は、その後のPASS gateと`cost_*.json`・`result.json`・completionであり、名前だけで未検証費用を受け入れていない。

[公開bundle](../../artifacts/resource_applicability/track_a_h46_m5_det_cost_run01_results/2026-10-20/)はmetadata、単価転記、SHA、終了witnessのprojectionだけ。有効grant本体・raw入力・QPY/vectorはprivateに残す。private入口は`/home/AbeHiromu/projects/h46-m5-cost-sampling-readiness/20261020/private/authorization_DET_run01/`。これは今回の限定取得の終了記録であり、研究全体の最終evidence freezeではない。他worktreeのHEAD/statusを凍結せず、共有環境・他ユーザーの設定/プロセスを変更していない。

## 5. 未実施と次の担当

PR24群のn_E、fresh seed、探索/確認非混合、rare-eventの扱いは[既存の標本設計案](track_a_h46_m5_cost_sampling_readiness_20261020.md)のまま採択待ち。旧piece-r2 pilot・ordinary n8・別確認13/45/45を新outer32へ混ぜていない。正式shot政策は未採択、経験的allowanceと厳密bias保証は別のまま。m5の主比較採用AD03も保留している。

- **ユーザー**：PRの固定探索n_Eを選ぶか、別承認の同law設計pilotを先行するか判断する。PR実行には対象・n・seed/cohort・rare-event解釈を固定した個別承認が必要。
- **ユーザー**：将来の正式shot政策、強化S2/S4対照とのidentity接続、費用会計/候補選択の実行範囲を判断する。今回のcompile承認を流用しない。
- **Codex**：採択された次stageの実装・固定契約を準備する。新科学計算・会計・探索/確認は別途承認後のみ。

両DET契約の計算・終了監査は完了した。追加compile・自動retry・再fit・shot計算・ランキングへ進まず、mandatory STOPする。
