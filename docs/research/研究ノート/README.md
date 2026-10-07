> 2026-10-07 Track B RA-D0 v3：**READY_FOR_RA_D0_ONE_SHOT_AUTHORIZATION**（実行承認ではない）。
> [source review](../../tracks/algorithm_codesign/ra_d0_source_review_v3_20261007.md) / [GPT handoff](../../tracks/algorithm_codesign/ra_d0_gpt_handoff_v3_20261007.md) / [manifest](../../../artifacts/track_b_ra_d0_source_review_v3/2026-10-07/evidence_manifest_v3.json)。
> [focused verifier](../../../scripts/tracks/algorithm_codesign/verify_ra_d0_source_review_v3.py) / [v3 tests](../../../tests/tracks/algorithm_codesign/test_ra_d0_source_review_v3.py)。
> exact-certified B2 minimum infeasibilityを正常outcomeに修正。pointをfreezeへ記録しbudget/queryを空にして次nへ進む。
> uncertified failureはtechnical STOP。数値・candidate・grid・call/resource capは維持。登録最適化0、authorizationなし、mandatory STOP。

> 2026-10-07 Track B RA-D0 v2：**READY_FOR_SEPARATE_RA_D0_ONE_SHOT_REVIEW**。
> [source review](../../tracks/algorithm_codesign/ra_d0_source_review_v2_20261007.md) / [GPT handoff](../../tracks/algorithm_codesign/ra_d0_gpt_handoff_20261007.md) / [evidence manifest](../../../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/evidence_manifest_v2.json)。
> [future runner](../../../scripts/tracks/algorithm_codesign/run_ra_d0_one_shot.py) / [focused verifier](../../../scripts/tracks/algorithm_codesign/verify_ra_d0_source_review_v2.py) / [v2 tests](../../../tests/tracks/algorithm_codesign/test_ra_d0_source_review_v2.py)。
> B0_saved/ideal分離、数値B1⊂B2⊂B3、profile-paired budget、batch freeze-before-B3、
> anchor-first、main LP 55,275 / auxiliary込み110,550、resource/launch guardを固定。
> 登録最適化・実budget/minimum/witness取得0、authorizationなし。旧本文・旧STOP・Track Aは保持。mandatory STOP。

## Track B RA-D0準備（2026-10-06）

[dated note](2026-10-06_track_b_ra_d0_preparation.md)：静的table、LP/certificate kernel、30 tests。
数値baselineとquery実行契約をGPT reviewへ返しmandatory STOP。以下の既存本文を全文保持。

## Track B RA-RTE統合数学監査・mandatory STOP（2026-10-06）

R1.5 `af3d014d0a0cfcbbd25bb544f6544652fec92942` 基点、GPT設計案へのDOCS_SYMBOLIC_ONLY_MATHEMATICAL_AUDIT。
[命題別監査](../../tracks/algorithm_codesign/ra_rte_mathematical_audit_v1.md)と[GPT handoff](../../tracks/algorithm_codesign/ra_rte_mathematical_audit_gpt_handoff_20261006.md)：一block／finite table／canonicalのfixed-n LPは仮定付きで成立。
shot-gridの固定total cap保存には反例。log/root・sampler認証、Delta=0、peak workspaceの規約を実行前修正へ返す。
[stdlib人工bookkeeping](../../../scripts/tracks/algorithm_codesign/check_ra_rte_mathematical_bookkeeping.py)の[50 checks](../../../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/bookkeeping_checks_v1.json)を一般証明と分離した。
[manifest](../../../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/evidence_manifest_v1.json)、[dated note](2026-10-06_track_b_ra_rte_mathematical_audit.md)。science/synthesis/solver/資源再採点0、共通API変更0。
既存科学分類・結果・STOPは不変。性能・新規性・algorithm採択・次実装／R2 authorizationは未確定。
**mandatory STOP。次の採択・実装・pilotの必要性／範囲はGPT判断。以下の既存本文を全文保持する。**

## Track B R1.5保存値帰属・mandatory STOP（2026-10-06）

input R1 commit `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b` の保存値だけを用いたPOSTHOC attribution / design input。
[帰属報告](../../tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md)と[GPT handoff](../../tracks/algorithm_codesign/r1p5_gpt_handoff_20261006.md)：新science/synthesis/compile/候補追加0、R1科学分類は不変。
primaryは2-qubit finite P₃、distinct-basis controlled、x={1/8,1/4}、登録native三precision。
Aは登録(G_T,G_CX,G_1Q) frontにx=1/8の1e-4、x=1/4の1e-3/1e-4で残る。
normalizationだけでなくnative費用とbias/shotの関係を整理し、固定合成列への依存も保存した。
[stdlib保存値解析](../../../scripts/tracks/algorithm_codesign/analyze_r1p5_saved_attribution.py)、[全summary](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/attribution_summary_v1.json)、[provenance manifest](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/evidence_manifest_v1.json)、
[日付note](2026-10-06_track_b_r1p5_saved_attribution.md)。共通library変更・独立validation・新algorithm採択はない。
限定診断SUPPORTS_RA_RTE_DESIGNは設計入力のみ。eta探索/R2/DF接続/追加scienceは未認可。
**mandatory STOP。次の数学設計・研究方針判断はGPT側。以下の既存本文を全文保持する。**

## Track B R1一回結果・mandatory STOP（2026-10-06）

固定S `d43d64a821a0249a0dfab12a2472bd3a72fdee74` →直接子authorization-only A
`411f08f768244fe87b600d82308c3851847fe9e4`からrun1/retry0。
[結果照合](../../tracks/algorithm_codesign/r1_one_shot_result_validation_20261006.md)：126 keys / 264 rows / 132 controlled tasks完了、全task適格。
terminal R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW、[保存field専用監査](../../../scripts/tracks/algorithm_codesign/audit_r1_saved_result.py) PASS。
primary distinct-basis controlledではB²改善とnative/shot資源のtrade-offを保存し、自動研究GOはない。
[全264 rows CSV](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/resource_rows_display_v1.csv)、[evidence manifest](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/evidence_manifest_v1.json)、
[GPT判断への入口](../../tracks/algorithm_codesign/r1_post_run_gpt_review_request_20261006.md)。原result/marker/source、既存証拠・共通API・Track A保持。
science終了後mandatory STOP、追加合成/target/grid/分子/DF/trajectory/GPUは行わない。
研究方針・RQ・新規性・着地点・追加検証の必要性/範囲はGPT側。
以下のpending/最新記述は当時の履歴として本文をそのまま保持する。

# 研究ノート

## 2026-10-06 Track B R1 source preparation

[Narrowed R1 source / preregistration](2026-10-06_track_b_r1_source_preparation.md)。
27 focused tests、static126 keys。R1 science未認可、GPT source reviewへmandatory STOP。


## 2026-10-06 Track B R0.5

[Equivalence / novelty closure audit](2026-10-06_track_b_rte_reallocation_r05.md)：
限定method-delta候補、CONDITIONAL-R1をGPTへ返す。R1未認可、mandatory STOP。


## 2026-10-06 Track B R0

[有限mean再配分の独立数学・文献監査](2026-10-06_track_b_rte_reallocation_r0.md)。記号検査一回、科学実行なし、GPT review待ち。


- [2026-10-06 Track B BS-0.5 docs-only設計監査](2026-10-06_track_b_bs05_design_audit.md)：
  candidate独立delta未定義、O/C分離・ordinary形式仕様・多資源Pareto。新science0、GPTへSTOP。

- [2026-10-06 Track B SP-1後GPT review・block合成仕様](2026-10-06_track_b_post_sp1_block_synthesis_design.md)：
  docs-only、三仕様資料・12 target未承認案、実装/science/tests0、mandatory STOP。

- [2026-10-06 Track B SP-1一回結果・GPTへのSTOP](2026-10-06_track_b_sp1_one_shot_result.md)：
  48 rows／96 axes完了、保存値audit PASS、retry0／mandatory STOP、研究判断はGPT側。

- [2026-10-06 Track B SP-1契約案・会計fixture検証](2026-10-06_track_b_sp1_wrapper_preparation.md)：
  science0、17 local focused tests、RUN_READY=false、契約案reviewへSTOP。

このディレクトリには、研究実装を進めた時点の方針、判断、検証結果および
未解決事項を日付順に記録する。後から「なぜこの実装になったか」「その時点で
何が確認済みだったか」を、commitと検証コマンドまで含めて追跡できるようにする。

## 資料としての位置付け

研究ノートは時点ごとの作業記録であり、現行仕様の正本ではない。

- 現在の研究方針と評価条件：`docs/research/`の主資料
- API、回路scopeおよび数学的規約：各実装文書
- 再現可能性と保証status：`VALIDATION_STATUS.md`と
  `artifacts/validation_manifest.json`
- 実際の数値結果：fingerprintと生成条件を持つmachine-readable artifact

過去のノートと現行仕様が異なる場合は現行仕様を優先し、変更理由を新しい日付の
ノートに追記する。過去の記録を現在の理解に合わせて黙って書き換えない。

## 記録規則

1. ファイル名は日本時間の日付に対応する `YYYY-MM-DD.md` とする。
2. 同じ日に複数回更新する場合は、ファイル内に `HH:MM JST` の節を追加する。
3. 実装を記録するときは、基準commit、対象scope、採用方針と採用しなかった範囲を
   明記する。
4. 検証結果は実行コマンド、pass/fail/skip/warning数、既知の環境制約を記録する。
5. 結果には `確認済み`、`部分確認`、`未確認`、`blocked` のいずれかを付ける。
6. 科学的な結論は、対応するartifactとfingerprintがない限り、実装能力の確認と
   区別する。
7. 失敗や方針変更も削除せず、後続ノートから訂正内容を参照する。

新しい記録は[テンプレート](テンプレート.md)を複製して作成する。

## 時系列索引

| 日付 | 主題 | 基準commit | 到達点 | 次の主要課題 |
|---|---|---|---|---|
| [2026-10-03](2026-10-03.md) | PR-2 M1-B1 actual compile map検証 | source `33f436b` + local result | 12,448 wrapperと全checkpoint/cacheを再検査。B2 rank 3、q=1のactual frontierを確認し`CONTINUE_RESOURCE_STUDY` | result-prior held-out transfer reviewを別freeze。H4 1.30 Åは未開封 |
| [2026-09-30](2026-09-30.md) | PR-2 M1-A result-prior authorization | authorization commitで固定 | development-only最大212 signal、compile 0、held-out access 0を結果前固定 | M1-Aを一度実行し、limitedなら停止、clearならartifact freeze後に別M1-B authorization |
| [2026-09-29](2026-09-29.md) | PR-2 V4/S2完了とmatched-accuracy再設計 | `61bbaad` | 旧S2を保持し、M1前研究契約とzero-compute実装契約を固定。208候補・16-cell selector、専用test通過、M1科学計算未承認 | 独立review後、必要ならM1 execution authorizationを別freeze |
| [2026-09-27](2026-09-27.md) | FR-R1b完了と研究完成フェーズ移行 | `ecb7f4c`、`16d4482` + dirty worktree | R5不通過と`MECHANISM_ONLY_NO_PRACTICAL_GO`を維持。C1/C2を中核、C3を条件付き応用とする完成原稿契約を固定 | 新規計算を止め、定理単位の先行研究監査とT1--T4の証明へ進む |
| [2026-09-25](2026-09-25.md) | M06-F・A0・P-B/P-C/P-A停止点 | `3336f03` + dirty worktree | P-C tracking 16/16完了。stretch予測破れと診断不通過によりA/B/C全てcurrent scopeで停止 | P-Dを事前登録するかR3/R6/R8へ問いを再定義 |
| [2026-09-26](2026-09-26.md) | P-D S0契約・S1公平再最適化 | `9a494bd` + dirty worktree | B1b/B2/B4一致、Case C/D不成立。B1a上限依存でCase B＋undetermined、S1停止 | P-Dの研究価値・baseline設計を再検討。S2/H12/長RPEは保留 |
| [2026-09-24](2026-09-24.md) | M06-F all-r coherent opt2初期計算・解析 | `26aa95c` + dirty worktree | 36/36 cell完了。12 group中7通過、5 groupはRZ相対SE 2%基準でfresh-32待ち | 15 taskのfresh 32 trajectory拡張後にcoherent再最適化 |
| [2026-09-23](2026-09-23.md) | M06/L08、N07/P03、WP11限定判断統合 | `efa90e0` + dirty worktree | T4/T7を主軸、T1を範囲変更、T3を保留。次段はall-r coherent opt2再最適化 | opt2未測定$r=1,2,4,8,16$のCPU transpile・再最適化 |
| [2026-09-22](2026-09-22.md) | WP03、Gate S1、WP06-a/b、WP05-a/b/R、WP01-D/C07、G08/M08 | `efa90e0` + dirty worktree | M08の$q=16,32$直接holdoutはRZ最大3.286%で通過し、実測幅によるlocal再集計区間も分離。ただし25%移送区間は重なる | 主張範囲の見直しまたは外部条件での移送検証 |
| [2026-09-21](2026-09-21.md) | 研究方向screeningの実行gateとWP00/WP02/WP01-S/WP04 | `efa90e0` + dirty worktree | $L_D=0$をscreen out。WP04で公平な配分改善後の決定論endpoint差は4.96%へ縮み、5%・25%区間とも重なり未決定。主要因は$\beta$、次いで$\alpha$再配分 | WP03でPF係数選択感度を評価 |
| [2026-09-20](2026-09-20.md) | 4段RPE分枝復元、目標round診断、$\delta$/round別scheduleと中央RTE cost検証 | `2bf3116` + dirty worktree | H4の固定長round設定を棄却し、3個の$\delta$に行列検査を通るscheduleを構成。局所角度・$L=8,16,32$検証後の中央RTE proxyは0.02を全6指標で最小とした | $\delta=0.02$と0.01の制御付きpartial-$S_2$反復・Hadamard 1 shot costを検証 |
| [2026-09-18](2026-09-18.md) | 暫定配分を使った限定4段集計と新配分の短段失敗率 | `2bf3116` + dirty worktree | H4固定条件の$q=1,2,4,8$で1,572 shot、RZ数$3.2673961\times10^7$、8軸$\alpha$和0.05。$q=1,2,4$の厳密二項座標失敗率$2.2246\times10^{-4}$ | $q=8$物理信号、branch復元、必要全round・最終コストを別途検証 |
| [2026-09-01](2026-09-01.md) | RPE短段の信号・shot・cost接続、失敗率、$q=8$代理モデル、配分感度 | `2bf3116`に至る前のdirty worktree | 配分感度から$\beta=(0.02,0.02,0.36)$と重み付き$\alpha$を固定条件の暫定入力に選択 | 限定4段集計（2026-09-18に実施） |
| [2026-08-26](2026-08-26.md) | H5 connected-cluster系サイズ検証と回路cost modelの区切り | `e07a5e6` + dirty worktree | H5、rank 9、$L_D=4$、$K=2$、$L=4,6,8$でpaired K1--K3最大1.665%。独立calibration/holdoutは最大3.776%、予測半幅1.459%で5%/2%基準を通過 | cost providerをRPE shot・誤差/失敗確率配分へ接続。新compiler・$L>8$・不通過条件だけ追加holdout |
| [2026-08-24](2026-08-24.md) | 階層compiled-cost model、$K=2$次数条件付き再検証、connected-cluster運用推定と軽量化 | `e07a5e6` | 固定DF snapshotの$L=4,6,8$ holdoutでK1--K3運用推定は全metric最大2.936%。点誤差5%内だが95%診断5.724%の留保。calibration/prediction/transfer分離と厳密key cacheを実装 | 別$L_D$・short-step・compiler/coupling条件への移送と角度不変性検証 |
| [2026-08-25](2026-08-25.md) | 複数order-2、独立K4、controlled $q=8$の追加・follow-up batch | dirty worktree | paired複数order-2は最大1.679%。$L_D=6$のK1--K4 paired $L=8$は4.008%。controlled $q=8$は0.0529%。全job完走・validator通過 | 系サイズ方向の独立holdoutで運用規則を確認 |
| [2026-08-23](2026-08-23.md) | ランダム回路加法モデルとRTE境界補正の高統計検証 | `e07a5e6` | 1000標本・独立2 seedでcount/sizeのpair-only残差を確認。same/different二分類が別seed pair holdoutを最大0.849%で予測 | count/sizeの$\mu_3$または$L=8$、$L_D,K$、controlled・compiler条件のholdout検証 |
| [2026-08-19](2026-08-19.md) | 論文Eq. (D6)によるPF摂動係数の再検証 | `8418192` | H4全$L_D$とH2--H5の支配位相比較を通過し、H6のD6係数をstate-actionで算出 | GPU経路をH8/H10で確認し、H12の候補$L_D$ごとにD6係数を決定 |
| [2026-08-18](2026-08-18.md) | finite-RTEとPF・摂動・QPE分枝誤差の検証 | `8fdc6b3` | H4全$L_D$の単一位相条件、H2--H5のdense比較、H6のstate-action係数までlocal確認 | GPU経路をH8/H10で確認し、H12の候補$L_D$ごとに$C$を決定 |
- [2026-10-06 Track B SP-1必須修正・source最終review](2026-10-06_track_b_sp1_source_review.md)：
  共通fusion全16 path監査、加法的primitive費用、実adapter/runnerと59 focused tests。
  science sweep0、RUN_READY=false。新source固定後の最終reviewへSTOP。
