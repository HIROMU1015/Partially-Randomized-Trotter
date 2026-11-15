# Full-H κ2 m0：1 shot参照価格規則の提案

**保存資料だけでは、m0-X/Yの数値単価を提案できなかった。推奨は、事前固定したm0 normal pathを1本取得し、そのX/Y各receiptのcompiled RZ数を別々の経験的1 shot参照単価として固定する規則。**

`C_0X = C_RZ(ω0,X)`、`C_0Y = C_RZ(ω0,Y)`。現在の値は両軸null、取得・規則採択は未実施。この規則は同じ物理候補・law・実装のshot数だけを変更する配分に `N_0X C_0X + N_0Y C_0Y` として使える。96 shotは単価へ含めない。厳密な母期待値やCIの規則ではなく、比較用の固定normal-path参照値である。

最小案は**1 normal path、X/Y各1 wrapper、最大2新規compile**。CI・全support証明・m6下界を前提にせず、この低R点の実回路条件を直接反映する目的に限定する。今回は新規取得・集計・予測を実施しない。

## 対象と固定正本

Linear H4 1.00 Å、STO-3G、保存tol-only DF10・H/state/snapshot。Full-H pure RTE（one-bodyを含む全非scalar Hamiltonian、headなし）、finite K2・κ2、P0。

- physical ID：`FULL_H_2a9a68a11d08af234a8c`、m0、T=0.1、q grouping=1、R=9、event位置数9。
- 元configuration：`54ac3eb38de5f90a77e9921100576c89f6c870f68384f4c7d9efe61b2c80c07e`。total decoder view：`265b90dc4a9b81011736664205e23833c420d569e57c923ef45c39eb59489612`。生成時IDとtotal viewを付け替えない。
- scalar：Hamiltonian定数＋Full-H identity抽出、保存total −0.33147781341681615を総Tへ一回。元fragment/basis順、symmetric-directional、full controlled Hadamard X/Y＋measurementを維持。
- new ZZ：`CONTROLLED_ZZ_CX_CRZ_CX_V1`。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17。価格単位は各軸の測定付きfull wrapper全体のcompiled RZ/shot。
- 正本：`61f249e6e18d5afab9ba8797bef5e8f9f14e29de:docs/research/full_h_d7_holdout_admission_plan_20261115/ADMISSION.json#rows[0],rows[1]`。

[PROPOSAL.json](PROPOSAL.json) に各軸のoriginal/total canonical ID、B_join、exact T、law、compiler/scalar、単価null、提案cohort/seedを保存。m1〜m12の価格は触らない。

## §10.5：保存照合と差分

C/D1の関連Git履歴・公開branch、旧IDのFull-H light/T2 trial、D7全cell、P1、Dの既存receipt catalog、同名m0研究と関連Python job名を限定照合した。今回条件に一致する保存価格は**0軸、0 path**、同等の進行中jobは確認範囲で0。未知archive/他ホストの不在は主張しない。新しい価格平均は計算していない。

| 調べた保存証拠 | 元SOURCE・locator | m0価格に使えない理由 |
|---|---|---|
| Cのm0候補・coverage | 61f249e6eのADMISSION `rows[0:2]` | receipt/previous price evidenceはいずれも空。X/Y単価なし |
| D7全7 cell | `7d0c39066d4ed900cfe7f5de6f4c7c5a3b2ec41f:docs/research/d_full_h_p2_calibration_resume_20261115/FROZEN_JOBS.json#plan.cells`、最終e628c0d69/06956c643 | 較正はq1/R566,4528,36223、q8/R566、paired q2/R566。独立holdoutはq32/R283、q64/R566。T=.1の較正でもR9ではない |
| Full-H P1 | `a04470c0ad489eea1798856e648f863f9092cfb1:docs/research/d_full_h_kappa2_p1_results_20261114/run02_revalidation01/PRICES.json` | m3/T=.8/q8/R71の単一path。元roleを維持、m0へ縮小しない |
| 旧Full-H H4 light | `9b83b96b83f62142c9d87858dc24a94db7ac5d61:artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09/H4_costs.json`、`price_trials.json` | H4はT=.8/q16/R16/K4、保存n9。Rが小さくてもK・時間・grouping・lowering/preparation条件が異なる |
| 旧Full-H T2最適化 | `7a8bc7f132e166390c943e6b32c92cc6abe1baf4:artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/price_trials.json.gz` | FH20旧IDもT=2.0。K0/2/4、q8/16等の別候補で、T=.1/q1/R9/K2一致なし |
| 旧receipt catalog | `e7dca5f7301c8633b17a59b6a7161331303aef2d:docs/research/d_full_pf_full_h_v2_final_receipts_20261113/RECEIPTS.json`、`d_full_pf_full_h_rte_v2_paths_20261113/SOURCE_CATALOG.json` | tracesはPFの旧ID。Full-H準備への参照は9b83の旧contextで、新m0の価格receiptではない |
| 同名m0 B2/stream | `0c7ff4c99ab549a69e491e856c1a247d826297c2:docs/research/track_a_b2_p1_compiled_cost_20261107/REPORT.md`、stream f6eea2f28 | 保存DF12、LD3、partial S2/r4/K2、compiler seed7の別研究。m0という名前やX/Y同値だけでは結合しない |

関連branch metadataも照合した。D1 caller最終06956c643、P1 a04470c0、S1 ce6aacfaca、旧light9b83、旧T2 7a8bc7f、C5b2f67ee4。Aの関連公開branch1adb7e65aはPF mean接続で、Full-H m0価格追加ではなかった。Pauli B0やPR-DF m0も別のrandomized representation/splitである。

## preparation hash差の扱い

旧Full-H context `0e3989e739d0e17a5d7af0fe809bd7c10ee765f09c934fe6ebb1f0e8b5c1046d` とD7側 `92325333cfe8c5e6c5f9bf69cdfc0ea6538f8eaa66327a3820d390dc2e8aa100` の差は、5b2f67ee4の記録を再利用した。

元 `prepare_full_h` のhashは、base preparation、two-body tail hash、one-body componentのmetadata・scalarに依存する。cohortやshot数だけのhashではない。basis操作・support・係数等の価格/lawに関係する内容を含む一方、識別metadataだけの差もあり得る。両側の完全payloadを比較できていないため、**価格に関係する差かは未確認**。異なるhashだけで回路が違うとも、単なる名前違いとも断定しない。

旧価格はhash差以前にT/K/q/R・lowering条件が違い、m0単価へ直接使えない。今回新しく旧preparationを生成・数値比較せず、m0取得時はD7側hashと対象H/state/snapshotを照合する。hash不一致を推測で修正しない。

## 推奨規則と他配分への使用

候補規則名：`FULL_H_KAPPA2_M0_NORMAL_PATH_REFERENCE_V1`（未採択）。

1. 事前固定した同条件normal path ω0を一回取得。選別・価格を見た再抽選なし。
2. 元X receiptの整数RZをC_0X、元Y receiptの整数RZをC_0Yとして固定。二軸は同じ1 pathを共有し、独立path数を2としない。値が等しくても軸別receipt・fingerprint・測定SOURCEを保持する。
3. 実測ではあるが、意味は**単一pathの経験的参照**。母law期待価格の保証、CI、代表性の統計的保証は与えない。price kindの候補は `DIRECT_COMPILED_FINITE_SAMPLE / FINITE_COHORT_ONLY`、qualified law E[C]はfalse/nullのまま。
4. 単価は96で乗算せず保存。Nだけを変える比較では同じC_0X/C_0Yを使う。物理time/grouping/R/K/preparation、law、scalar、compiler/lowering/wrapperが変われば規則の適用外。

RTE normalizer Bを価格へ再乗算しない。既存guardの正式な期待価格資格を緩和せず、採択された経験的参照シナリオとして区別する。

採用理由は低Rの実回路とwhole-wrapper最適化を直接反映し、今回の比較用単価という目的に必要な最小取得であること。PFとの価格の大小を選定根拠にしない。n1なので変動の影響は残るが、その弱さを参照tierとして明記できる。保存根拠なしの負値clamp、大R比例縮小、X/Yコピー、モデルrefitを代わりに採用しない。

この規則は費用見積りの参照であり、RPEの全shotでω0を再利用する物理実行規則ではない。元signal/lawを変更しない。また、単価を別Nへ使えることと、その配分がQ3を満たすことは別である。

## 最小取得案・後続との接続

[PROPOSAL.json](PROPOSAL.json) の `proposed_acquisition`：1 fresh normal path、index0、X/Y計2 compile上限、共有seed1個。cohort `D_FULL_H_KAPPA2_M0_REFERENCE_N1_20261115_V1`、seed `15612141344395026302`。元 `9b83…:trial_seed(row,cohort,0,0,0)` を既存D1 metadata readerから呼んで決めたmetadataで、eventは生成していない。

元 `9cbabc617e95a81810924413c1b2b06ced329514:scripts/run_full_h_kappa2_p2.py:acquire_path`、normal sampler、native・new ZZ・wrapper・compiler/cache、index0数値gateとscalar/event/XY/fingerprint検査を再利用する。**閉じたD7計画を編集せず**、m0参照取得だけの別jobとしてSOURCE・cohort/seed・入力・最大2 compile・STOPを実行前に固定する。保存H/stateの復元は再利用し、新SCF/DF/state生成なし。

既存parity gateが要求する未compileのlegacy比較構築も、取得scopeに明記する。追加compiled wrapperはX/Yの2件上限で、legacy回路を追加compileしない。

n1なので各軸の価格を直接転記し、新しい平均処理は不要。両軸の値・実装証拠を受け入れられれば経験的参照単価2件を公開してSTOP。不一致・欠測・gate不成立は該当値をnull/保留、再抽選/代替軸/閾値緩和/自動retryなし。CI・全support閉包・m6下界・価格fitを実行の前提に追加しない。性能は未測定で、旧巨大roundから所要時間を外挿しない。

規則の採択・取得後は、**m0の経験的参照2軸＋m1〜m12のF1保存24軸**という別の比較シナリオとして扱う。元F1の負値・全額null・D7資格を上書きしない。新シナリオの全13 round算出も今回未実施。厳密E[C]/CI/正式G/winnerへ昇格しない。

## 完了範囲・ロードマップ

保存価格採用0軸、新規sampling/build/compile/fit/Q3、価格集計/予測、全額再計算0。今回はmetadata照合・seed選択metadata・規則提案のみ。reader/APIの新実装や科学pipelineの再実装なし。JSONの2軸・1共有path/2 wrapper・単価null・旧成果不変を技術確認した。

ロードマップ差分：m6全support下界探索は利用者指示で終了。現在の課題はm0の比較用参照単価で、保存同条件価格は確認範囲で0。最小1path/2wrapperの経験的単価案を提示、採択/実行待ち。本体は統合担当へ。commit/push・remote一致確認後mandatory STOP。
