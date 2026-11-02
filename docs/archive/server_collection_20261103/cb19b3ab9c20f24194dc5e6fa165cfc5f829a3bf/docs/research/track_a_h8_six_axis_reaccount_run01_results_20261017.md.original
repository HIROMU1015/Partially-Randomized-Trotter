# H8主要6軸実測反映：通常・経験的非相殺の限定費用再会計結果

2026-10-18説明追補（A05）：通常biasの簡略projectionは、α・証拠分類・shot状態を[固定SOURCEと元D0/B0I/Phase A accountingへつなぐ説明](pr_alpha_projection_explanation_20261018.md)と併用する。各軸α=.025、EMPIRICAL、n32費用SEと測定αの分離、実測/model区分を明記した。以下の保存数値・shot・順位は変更していない。

固定SOURCEで通常signed biasと経験的非相殺を一つのstageとして一回実行し、3800保存行・40比較を正常完了した。driver/launcher native exit0・全回収・最終gate `PASS_MANDATORY_STOP`。旧32・PR・SE・保存shot・適格性を変えず、指定6単価だけを更新した。再実行・次stageは行わずmandatory STOP。

強化184候補基準のPR点費用削減率（正はPR低費用、負はPR高費用）。

## Allowance倍率1

| ε | 通常bias更新前 | 通常bias更新後 | 非相殺更新前 | 非相殺更新後 |
|---|---:|---:|---:|---:|
| 0.05 | -7.1369% | -8.0230% | 16.6243% | 16.6243% |
| 0.01 | 33.0928% | 33.2373% | 34.6849% | 34.6849% |
| 0.005 | 31.3066% | 31.4549% | 35.1253% | 35.1253% |
| 0.001 | 22.9613% | 22.9613% | 39.2833% | 39.4150% |
| 0.0005 | -51.1403% | -50.8125% | -29.0910% | -29.0910% |

## Allowance倍率10

| ε | 通常bias更新前 | 通常bias更新後 | 非相殺更新前 | 非相殺更新後 |
|---|---:|---:|---:|---:|
| 0.05 | -7.1369% | -8.0230% | 16.6243% | 16.6243% |
| 0.01 | 33.0928% | 33.2373% | 34.6849% | 34.6849% |
| 0.005 | 31.3065% | 31.4548% | 35.1253% | 35.1253% |
| 0.001 | 22.9610% | 22.9610% | 39.2833% | 39.4150% |
| 0.0005 | -51.1418% | -50.8139% | -29.0921% | -29.0921% |

全exact値は[40比較CSV](../../artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/comparison_summary_exact.csv)、[比較JSON](../../artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/comparisons.json)、[3800行CSV](../../artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/accounting_rows_exact.csv)、[通常/非相殺差](../../artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/mode_differences_exact.csv)に保存した。丸め前は倍率1/10で一部異なる。

## 適用範囲と入力来歴

linear H8、隣接1.00Å、STO-3G、保存tol-only DF rank28、16 spin modes、sector4900、T=.8、q1/2/4/8、ε=.05/.01/.005/.001/.0005、allowance倍率1/10、α_axis=.025、P0、state preparationを除くshot-weighted full-wrapper RZ。旧32とPhase A追加152の計184決定論候補、PR確認union6・各n32を別namespace `USUAL_SIGNED_SAVED` / `EMPIRICAL_NONCANCELLATION` で比較した。

- [主要6軸実測](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ce6cd8ead5c0311ec72165c04869e7df4050b75b/docs/research/track_a_h8_key_compile_completed_results_20261017.md)。SOURCE41e466e1…と先頭cosineの旧SOURCE35258110…、全6scalar/native/回収receiptを固定SHAで照合。
- [通常Phase C](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6dc5bf0de01a0aa23cafe0ff022c0cd21fbf1086/docs/research/track_a_h8_phase_c_run01_results_20261017.md)の1900行、[非相殺COST_COMPARISON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23d46e489af054735d9d328aad0342ecdbddf305/docs/research/track_a_h8_noncancellation_cost_run01_results_20261017.md)の別1900行。通常の保存shotは元D0/B0I/Phase A入力と同一SHAのaccountingへjoinした。
- B1結果 `83a3c4a6688e66576f85471bf41a45e29cbfb705`、B2結果 `fef2db7a79886ab682b652da7266b463c98c7d3a` の登録4軸・native・元STOPを維持。新実測→B2登録→B1 anchor→来歴確認済みfull PF→固定B1 modelの優先順を変更しない。
- [入力manifest45件](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aa984d98a7b438dc953a829a24fe3d7edc9d0a6b/references/h8_six_axis_reaccount_inputs_v1.json)、4,162,148 bytesを実行前後にstreaming SHA照合。配列読込・NPZ/QPY・元private rawの一括公開なし。

## 6軸単価とwinner

| 構成（cosine/sine共通） | 変更前RZ/軸 exact | 実測RZ/軸 | 元model検証誤差 |
|---|---:|---:|---:|
| `B0I_p15_q4_4th` | 4670300/7 | 668636 | 0.216902% |
| `B0I_p11_q1_2nd` | 298323/7 | 42268 | 0.827036% |
| `B0_p15_q4_2nd` | 1564196/7 | 223940 | 0.215874% |

元の6軸検証は全軸誤差2%未満PASS。今回compile/model予測/refitは0。費用は保存shotと単価のexact Fraction算術、同値はlexical IDを使用。未測定候補の保存単価を変更しない。

| ε | 通常184 winner | 単価区分 | 非相殺184 winner | 単価区分 |
|---|---|---|---|---|
| 0.05 | `B0I_p11_q1_2nd` | 今回6軸実測 | `B0I_p15_q1_2nd` | 固定model推定 |
| 0.01 | `B0_p15_q4_2nd` | 今回6軸実測 | `B0I_p15_q4_2nd` | 固定model推定 |
| 0.005 | `B0_p15_q4_2nd` | 今回6軸実測 | `B0I_p15_q4_2nd` | 固定model推定 |
| 0.001 | `B0_p15_q8_2nd` | 固定model推定 | `B0I_p15_q4_4th` | 今回6軸実測 |
| 0.0005 | `B0I_p15_q4_4th` | 今回6軸実測 | `B0I_p16_q4_4th` | 固定model推定 |

倍率1/10ともwinnerは同一で、更新前からの変更は**全40比較で0**。通常ε=.05/.0005のPR高費用、.01/.005/.001のPR低費用という正負は変わらない。非相殺でもε=.0005のみPR高費用という正負を保持した。実測化により通常ε=.05は−0.8861pp、.01は+0.1444pp、.005は+0.1483pp、.0005は約+0.3278pp、非相殺ε=.001は+0.1317ppの変化。これは固定した有限標本点費用・比較集合内の結果であり、研究全体の戦略判断を代行しない。

## 不変性・適格性・未測定範囲

[保存出力監査](../../artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/saved_output_audit.json)は元入力とのsaved行完全一致、旧32の640行/20比較、PRの120行（60行/mode）の保存点費用・paired SE・選択、全保存shot・適格性をPASS。非対象1870行/modeは全内容不変。指定3候補60行中、適格34行（通常18/非相殺16）の `axis_costs` / `G_exact` / `cost_status` だけが変更され、不適格26行はnullのまま。

184候補の適格/不適格数は、精度の順に通常156/28、105/79、90/94、60/124、57/127、非相殺140/44、105/79、90/94、60/124、57/127（両倍率同じ）。旧32は通常24/8、13/19、12/20、6/26、6/26、非相殺21/11、13/19、12/20、6/26、6/26。未解決・適格費用欠測は全条件0。PRは各modeで計44適格条件行を保存し、残りは不適格費用null。

[費用来歴件数](../../artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/cost_provenance_counts.json)：決定論184×10行のうち通常はmodel798行・既存実測120行・今回実測18行・不適格904行、非相殺はmodel772行・既存実測116行・今回実測16行・不適格936行。これは条件行数でありcompile件数ではない。通常のwinnerは5精度中4、非相殺は1精度で実測単価へ移ったが、集合全体の比較にはmodel単価が残る。全184の実測最良、未測定格子のモデル誤差保証、PR母平均winner、厳密bias certificateを認定しない。有限時間complex signalの評価でありend-to-end energy/QPE費用ではない。

## 固定・承認・終了監査

- branch：`track-a-h8-six-axis-reaccount-20261017`。
- SOURCE：`aa984d98a7b438dc953a829a24fe3d7edc9d0a6b`。準備REVIEW：`072efb688995cc4b98bad3785c305968c2bb0287`。実行前remote HEAD一致・SOURCE祖先関係・公開blob25件、closure7ファイルを確認した。
- plan canonical SHA256：`bee16900432f5d69369449f134d1ea5b6e02b84f6e66dcb81cdcaea8c392e800`。
- sealed config canonical SHA256：`b07bec1fff96a0cb4843292d122581b5d1a6ecdc30e1b30baa6cd975b89e5e7c`。
- 消費済grant canonical SHA256：`9619363e32a960407d4e2685e77b837ab106f77c392ce4ebd794a74fa6398c62`。
- 入力manifest SHA256：`03e7ce8e1d9149e0d1fb88577aef6913dbed457dabd69aa4222ac7882434c810`。closure SHA256：`95f8808dcc8f6fbd6f27ffab0c605f97beef1cbe429298c991ee53d96b391505`。
- B1 model：`8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0`。compiler：`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。費用来歴profile：`622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`。
- [binding](../../artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/authorization_binding.json)と[one-shot receipt](../../artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/one_shot_receipt.json)。元draft/INVALIDを変更せず、利用者の既存明示承認＋今回の継続指示を新config/grantへ結合。以前の科学attempt0を確認し、新grant1・科学dispatch1・再会計pass1。retry/resume0、次stage非認可。実効grant/config/消費済lock/marker原本はprivateに保持。

`/usr/bin/python3.12 -I -S -P -B`とstdlib profile68ファイルを照合。共有venv・compiler環境のimport/変更0。既存人工30検査のSOURCE/ログhash一致を確認し、人工campaign再実行0。主担当の実行後監査であり、別担当独立reviewは行っていない。subagent0。

[資源receipt](../../artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/resource_observations.json)：fresh gate PASS、許可CPU0–63、load1約109.43、実効空きmemory 468.324GiB、FS空き485.091GiB、空きinode219,269,727。CPU空き占有や専用core予約を主張せず、inherited schedulingで単一軽量算術childを使用。quota utilityが見つからず数値quota UNKNOWN、同FS64MiB実割当probe PASS・削除済み。quota無制限とは扱わない。AS/CPU継承上限はunlimited、独自CPU/RAM/wall cap・affinity変更なし。host PSI avg10=0、観測可能なcgroup OOM/oom_kill増加0。祖先の累積oom_kill1は前後不変の過去記録として保持した。

[native終了](../../artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/native_completion.json)：driver PID2856961 exit0/wait4/reaped、launcher PID2856952 exit0/wait4/reaped、witness CLI exit0。PIDの/proc残存0、cleanup error0。driver wall1.309277秒/CPU1.265617秒、launcher込みstage wall1.560903秒、最大単一process RSS81.754MiB。CPU/RSSは重複するため合算しない。出力6件/1,351,678 bytesを元receiptと照合した。

[最終gate](../../artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/final_gate.json) `PASS_MANDATORY_STOP`。SOURCE7/input45/runtime68/config/grantを事後照合、元draft/INVALID/旧handoff保持。[保全receipt](../../artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/preservation_receipt.json)にあるとおり既存SOURCE・結果・grant・ledger・marker・STOP、他worktreeのHEAD/index/dirty、他job/共有環境への書込・signal・affinity変更0。無関係な作業を凍結していない。新signal/compile/sampling/fit/shot・bias再計算/PR平均SE再計算/モデル予測0。

## 保存・公開と次の担当

[結果bundle](../../artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/README.md)へnative出力6件をbyte同一copyし、軽量な監査・exact CSV/JSON・来歴・資源receiptを追加。SOURCE/tests・元成果・原稿は変更しない。新しい結果commitをSOURCEと区別して専用branchへnon-force pushし、remote SHAと公開blobの一致を実行記録で確認する。認証失敗時は結果commitを保持して正確な手動pushコマンドを提示し、公開済みと扱わない。

次の担当：**ユーザー・GPT**が、今回の固定点費用とmodel残存を踏まえて研究戦略・必要な後続検証の優先順位を判断する。Codexは今回の結果保存・公開（または認証待ち報告）後mandatory STOP。新compile・sampling・bias/shot計算・fit・他分子/精度への拡張や追加科学レビューを自動開始しない。
