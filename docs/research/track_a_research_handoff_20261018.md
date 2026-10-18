# Track A：既存証拠と最新引継ぎ索引（2026-10-18）

2026-10-18 14:18 JSTの既存照合報告を基準とする資料公開追補。原本再照合日時：2026-10-18T14:33:01.490298+09:00。この索引はSOURCEの統合版、科学レビュー、実行許可ではない。新科学計算・数値再解析・再会計・sampling・compile・選抜再実行・新grantはすべて0。

## 入口の使い分け

- [正式共通ノート（10/15）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6b78e5f39917d92d6c60694d3e0b0d06394d7eb5/docs/research/track_a_shared_research_status.md)：正式共有状態の基準。
- [H8 B0+I側10/16追補](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/track_a_shared_research_status.md)：後続実測の追補。上の版と分岐しており、片方で他方を上書きしない。
- [H4〜H8横断証拠索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/621d7d7bed183862db8d9f01b6fdb4406f942309/docs/research/track_a_h4_h8_cross_system_evidence_inventory_20261017.md)：19系列・32固定資料と不足項目の入口。後続s2/RARE/Cは以下を併読する。
- `41ba315b2ad369825577ad671c79b69c77e1f3cc` の共通ノート本文は10/14版の継承であり、最新統合正式ノートとは案内しない。同commitの研究概要は10/18 Cまで追記済み。本索引で固定版の役割を接続し、原SOURCE・旧ノートのmerge/上書きは行わない。

## 完了結果の範囲

基本条件はlinear H8、隣接1 Å、STO-3G、保存tol-only DF rank28、16 spin modes、sector4900、T=.8。各結果のprefix/q/r/K、観測精度、compilerと例外はリンク先の固定契約が正本。P=0のshot重み付きRZ点費用、経験的allowanceと厳密bias保証を区別する。

| 系列 | 結果commit | 実行／監査SOURCE | 公開path |
|---|---|---|---|
| formal_common_20261015 | `6b78e5f39917d92d6c60694d3e0b0d06394d7eb5` | 共通資料、単一計算SOURCEではない | [固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6b78e5f39917d92d6c60694d3e0b0d06394d7eb5/docs/research/track_a_shared_research_status.md) |
| common_followup_20261016 | `5f79257f432bc7e0317f122090f62fec379e2a1b` | 共通資料、単一計算SOURCEではない | [固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/track_a_shared_research_status.md) |
| cross_system_inventory | `621d7d7bed183862db8d9f01b6fdb4406f942309` | `ad9d6a91dbf82f204f1f37298a1785a91847c5eb` | [固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/621d7d7bed183862db8d9f01b6fdb4406f942309/docs/research/track_a_h4_h8_cross_system_evidence_inventory_20261017.md) |
| B2_complete | `fef2db7a79886ab682b652da7266b463c98c7d3a` | `f58da9590597f42ff7260c22d771734096a706bb` | [固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fef2db7a79886ab682b652da7266b463c98c7d3a/docs/research/track_a_h8_b2_completed_results_20261017.md) |
| key_six_axes | `ce6cd8ead5c0311ec72165c04869e7df4050b75b` | `41e466e13f32e9224bb37790cc16f0ff6621c399` | [固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ce6cd8ead5c0311ec72165c04869e7df4050b75b/docs/research/track_a_h8_key_compile_completed_results_20261017.md) |
| six_axis_reaccount | `96e69a01d1fae7a402aac2c4efa570a59f8cc7a0` | `aa984d98a7b438dc953a829a24fe3d7edc9d0a6b` | [固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md) |
| conditional_s2_run02 | `b21e90b8368645a5066c06a2a0a2ceb2850d162e` | `03479a7546a994bf9d02561d1ab74cec67201b4f` | [固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b21e90b8368645a5066c06a2a0a2ceb2850d162e/docs/research/track_a_h8_conditional_s2_run02_results_20261018.md) |
| rare_selection | `cf8e93eef50333bef41f6d75803f6b819d679343` | `3c0c9546701fd953de3b299be561344375c1a78b` | [固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cf8e93eef50333bef41f6d75803f6b819d679343/docs/research/track_a_h8_rare_sensitivity_selection_run01_results_20261018.md) |
| C_EPS_H8 | `41ba315b2ad369825577ad671c79b69c77e1f3cc` | `3156b036fb4a0015e09cce2cb73785d7819dcb75` | [固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41ba315b2ad369825577ad671c79b69c77e1f3cc/docs/research/track_a_h8_epsilon_boundary_run01_results_20261018.md) |
| AF_plan | `41ba315b2ad369825577ad671c79b69c77e1f3cc` | `daf701d8e4fb25a22e5c383202f535f9db7a6e25` | [固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41ba315b2ad369825577ad671c79b69c77e1f3cc/docs/research/track_a_targeted_parallel_validation_plan_20261018.md) |

- **B2完了**：登録S4全4軸PASS。最後のsine SOURCEを表に示すが、保存3軸は軸別SOURCE・別受入契約を保持する。旧STOPを書き換えない。
- **主要6軸実測**：B0+I p15/q4 S4、B0+I p11/q1 S2、B0 p15/q4 S2の両軸。旧完了1軸＋新SOURCEの5軸を来歴別に結合。全格子の実測ではない。
- **6軸反映後再会計**：通常／非相殺の保存3800行・40比較を完了。既存PR、shot、適格性、旧32集合を保持。未実測モデル費用は残る。
- **s2 run02**：p7/r4/K4のq4・q8、それぞれ条件付きs2 n8、計32軸compile。観測した16例は全てN2=1だったが、samplerがsingleへ固定されたという意味ではない。元確認union6/n32に混ぜず、s4/multiの費用を補完しない。
- **RARE_SENSITIVITY / SELECTION_SAVED**：別scope・別grantで各一回完了。RARE40条件（評価32・不適格8）、SELECTIONは元4精度／通常／倍率1の保存値による探索的bootstrap2000。未知s4/multiはnull、母平均保証や選択調整済みcoverageではない。今回これらを再実行しない。
- **C_EPS_H8**：追加ε=.04/.025/.015/.000875/.00075/.000625、通常／非相殺、倍率1。保存anchor1520行を一致確認後2280行・12比較を完了、native exit0・最終gate PASS。既存184決定論＋確認PR6に固定し、新しいA/D結果を混ぜない。連続区間の唯一の逆転閾値の証明ではない。

## 10/18追加計画A〜Fの状態（旧Phase A/B2/Cと別命名）

報告時点は14:18 JST。今回の公開作業はこの実行状態を更新する新計算を行っていない。[計画正本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41ba315b2ad369825577ad671c79b69c77e1f3cc/docs/research/track_a_targeted_parallel_validation_plan_20261018.md)の初期UNAPPROVED/INVALIDは当時の履歴。[後続C完了報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41ba315b2ad369825577ad671c79b69c77e1f3cc/docs/research/track_a_h8_epsilon_boundary_run01_results_20261018.md)に記録された「準備できたら実行」の既存条件付き承認を保持する。承認の存在と、production adapter・SOURCE/input seal・個別grantの完成は別である。

| 単位 | 状態 | 残る実装／gate |
|---|---|---|
| A_SIGNAL_H8 | 科学実行未着手、計画済み | p6/8・r8/q16用scope/preparation、入力binding未完成 |
| A_COST_H8 | 科学実行未着手 | A signalの適格性・別stage境界、新seed/control予算/seal |
| B_RARE_H8 | 汎用人工sampler準備済み、科学実行未着手 | q4/8×s2_multi/s4の4群；native接続/control予算/seal未完成、既存control376/376は流用しない |
| C_EPS_H8 | 完了・正常終了・mandatory STOP | この公開で再実行しない |
| D_H8_S2 | 科学実行未着手 | B0 p15/q8 S2、新scope・入力・予測値・各軸2% gate |
| D_H8_S4 | 科学実行未着手 | B0+I p16/q4 S4、同上 |
| D_H6_S4 | 科学実行未着手 | B0+I p11/q4 S4、H6固有profile/binding |
| E_TIME_SIGNAL_H8 | 科学実行未着手 | T=.4/1.6、δ=.2のreference/signal adapter |
| E_TIME_COST_H8 | 科学実行未着手 | 各T固定5構成、signal gate後にnative費用 |
| F_INPUTS_H6 / F_INPUTS_H8 | 新距離入力未生成 | 各1.40 Å、生成・入力封印後STOP、rank19/28を強制しない |
| F_SIGNAL_H6 / F_SIGNAL_H8 | 科学実行未着手 | 新入力封印後の別承認が必要 |
| F_COST_H6 / F_COST_H8 | 科学実行未着手 | 同じ別承認とsignal gateが必要 |

C以外の新scopeで有効grantを発行した記録は今回の限定確認範囲では確認していない。既存承認を撤回も追加もせず、今回新grantは発行しない。旧H8 Phase Aの追加152候補、B1、B2、通常Phase Cは既に完了した別系列であり、上表の「未着手」と混同しない。

## 今回公開する5群

1. 本引継ぎ索引。既公開結果は固定リンクのみで、新たにコピーしない。
2. [10/16 GPT Level 2原本](track_a_v1_strengthened_baseline_scientific_review_20261016.md)と[10/17 GPT Level 1原本](track_a_h8_strategic_scientific_review_2026-10-17.md)。改行を含むexact copy。作成日・当時の対象結果・後続結果は[別provenance](../../artifacts/resource_applicability/track_a_evidence_handoff_publication/2026-10-18/review_originals_provenance.json)。Codexによる新しい独立科学レビューではない。10/17原本の対象は通常`6dc5bf0de01a0aa23cafe0ff022c0cd21fbf1086`／非相殺`23d46e489af054735d9d328aad0342ecdbddf305`であり、後続主要6軸／反映後再会計とは分ける。原本中の当時の提案・未実施・未承認表記を改変しない。
3. [H8選抜原本照合projection](../../artifacts/resource_applicability/track_a_evidence_handoff_publication/2026-10-18/h8_selection_original_verification.json)。115742 bytesの原本SHAを過去の公開completion hashと照合。現在のbyte観測と、固定SOURCEのwrite-before-confirmation／当時のcoverageを別記。古いmetadataの`original_bytes_verified=false`は書き換えない。原本registrations等はprivate保全。
4. [H6旧B0+I12端点projection](../../artifacts/resource_applicability/track_a_evidence_handoff_publication/2026-10-18/h6_b0i_endpoint_provenance.json)。旧pilot3／新full-family9を区別し、元SOURCE・result・bytes/SHA・公開input binding・保存欄を記録。corrected/exact_truncated/reference/scalar/preparationは存在するが、signed成分はtotalのみ。discard/PF復元・位相回転・closure検証は未実施。
5. [H8原積分・入力政策projection](../../artifacts/resource_applicability/track_a_evidence_handoff_publication/2026-10-18/h8_integral_input_policy_verification.json)。1775 bytes原積分receiptのRHF欄、snapshot/projection親hash、元SOURCEのcharge0 AST欄・file hashと既存環境profileを対応付ける。NPZ配列読込やSCF/DF/state再生成、venvの再監査は行わない。

[公開file manifest](../../artifacts/resource_applicability/track_a_evidence_handoff_publication/2026-10-18/publication_manifest.json)に全対象のbytes/SHA、[限定検証記録](../../artifacts/resource_applicability/track_a_evidence_handoff_publication/2026-10-18/publication_validation.json)に手法と保全範囲を記録。manifest自身の自己hashは循環を避けて含めず、Git commitによって固定する。

## 残る未公開・未保存・未照合

- **O02**：端点12件の存在・同一性と、discard/PF signed成分の復元は別。後者は未実施、科学的解消済みとしない。
- **O05**：H4実event履歴のNOT_SAVED、172件の丸めnormalizationに対する新しい未丸め原記録未回収を保持。新samplingで補完しない。
- **O06**：H4 B0+I親driverのnative wait数値はUNKNOWN。worker exit0／COMPLETEから推測しない。
- **O07**：旧compiler出力bit同値、一部旧profile、prefix19元runtime対応は未照合。
- **O13**：今回接続したH8原入力だけの部分的資料回収。H4/H6全系列のcharge／原軌道政策／環境同一性まで解消したとはしない。
- 原選抜・端点・積分receipt、raw NPZ/QPY、実runtime/checkpoint/ledger/marker、消費済みまたは有効grantの原本、lock、認証情報は今回の公開対象外。安全なprojectionは原本exact bytesではない。
- s4/multi費用、全support上界、PR母平均winner、全184候補の実測最良は未認定のまま。実測/model、有限標本平均、経験的allowance、既存GPTの提案を別の証拠種として扱う。費用の優劣を今回再評価していない。

## 保全と次の担当

起点`41ba315b2ad369825577ad671c79b69c77e1f3cc`から資料専用の独立clone／branchを作成し、worktree登録・source/result mergeを行わない。既存管理rootの958 staged削除、既存研究checkoutのdirty/untracked、SOURCE・grant・ledger・marker・STOP・稼働jobに書込／signalを行わない。原本はread-only、公開先は新規fileの排他的作成のみ。

**Codex**：明示stage・commit・non-force pushを試行し、成功時はremote各fileを再取得して照合、認証不可ならcommitを保全し手動push commandを報告。完了またはpush待ちでmandatory STOP。A/B/D/E/Fの実行や科学判断へ自動移行しない。追加研究を始める次作業は今回の範囲にない。
