# 研究ノート

## 2026-10-10 Track A：H6入力生成準備 v1（未実行）

最新入口は[H6入力生成契約・source/合成検証索引](../track_a_ax2b_h6_input_preparation_v1.md)。
既存tol-only adapter/solver/matrix-free/H6 loaderを再利用し、入力生成専用gate・watchdog・snapshot・stdlib保存監査を追加。
87 local synthetic/metadata tests pass（新38・既存49）、実H6入力/state/signal・sampling/circuit build/compile0。
source67312f3、183 science/3 validation freeze。予算・新outputを固定、CPU null・unsealed・新grantなし。
H6_INPUT_GENERATION_NOT_AUTHORIZED / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・u未認定、mandatory STOP。
入力生成の別指示後にresource/seal/grantを固定し、一回生成・保存監査後STOP。H6 pilotはさらに別認可。
旧H4 source/freezes/結果・既存dirty/untracked・Track Bを保全。以下は段階当時の履歴。

## 2026-10-10 Track A：H4補完実行 v1・STOP

最新入口は[H4補完結果・source/raw/監査索引](../track_a_ax2b_h4_supplement_execution_v1.md)。
別grantでEVENT_CONTROL 4群、S4 2 correctness/4 MPを一回ずつ完了。旧6 correctness/12 MP/STOPを保持し、複数runのunionを記録。
各reference36/primitive537、control100（event単位）、sampling/compile0。保存監査PASS、補完欠測0。
source67aa6bb・seal/grant5bbebb4、CPU1/worker1/BLAS1、予算・精度・stage/probe変更なし。
N/Gnull・UNDETERMINED・u未認定、H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP。
H6入力・pilotは別指示を要する。旧証拠・dirty/untracked・Track Bを保全。以下は段階当時の履歴。

## 2026-10-10 Track A：H4補完準備 v1（未実行）

最新入口は[補完準備・source/予算索引](../track_a_ax2b_h4_supplement_preparation_v1.md)。
[GPT独立レビュー](../track_a_ax2b_h4_limited_stop_independent_review_2026-10-10.md)に従い、旧6 correctness/12 MP/STOPを保持。
cell/dps-local MP cache、S4 2 cellとexplicit 4群の独立単位、atomic進捗を新versionで準備。
122 local synthetic/metadata tests pass。分子load/signal/sampling/circuit build/compileは今回0。
source 67aa6bb、180 science/2 validation freeze。対象・予算・新outputを固定、CPU/grant/sealは未確定。
H4_SUPPLEMENT_NOT_AUTHORIZED / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP。
N/Gnull・UNDETERMINED・u未認定。旧source/freezes/結果・既存dirty/未追跡・Track Bを保全。以下は段階当時の履歴。

## 2026-10-10 Track A：H4限定v3一回実行・GPT引渡し

最新入口は[H4 v3実行結果・レビュー索引](../track_a_ax2b_h4_limited_execution_v2.md)。原status H4_LIMITED_STOP、reason PHASE_WALL_CAP:correctness。
correctness6/8、MP12/16、explicit event0/4。coverage一致・actual全体保存。
source b228f23、seal d2b1511、別認可a699a74を結果前に公開。一回grant消費済み、retry/resumeなし。
原結果/欠測・保存監査を引渡し、科学GO/STOP・u/shot/総費用・H6をCodexは承認しない。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・UNDETERMINED、mandatory STOP。
旧source/freezes/STOP・既存dirty/未追跡・Track Bを保全。以下は各段階当時の履歴。

## 2026-10-10 Track A：coverage serialization準備v3

最新入口は[coverage修正・準備v3](../track_a_ax2b_h4_coverage_preparation_v3.md)。旧v2/source/freeze/STOPを保持し新versionを追加。
tuple/listだけ正規化し、数値・型・順序・coverage変更を拒否。actual/差分をbounded保存。
49 local metadata/mock tests pass、分子load/prepare/signal/sampling/build/compile0。
同じH4 8 cell/capsの新固定・別認可付き一回実行を今回指示の範囲とする。旧grant/output再利用なし。
この段階は準備で分子PASSではない。H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・u未認定。
実行後はmandatory STOP。既存dirty/未追跡・Track B・旧証拠を保全。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4限定一回実行・coverage interface STOP

最新入口は[H4限定実行報告](../track_a_ax2b_h4_limited_execution_v1.md)。固定source/manifestの一回実行はACTUAL_COVERAGE_CHANGEDでSTOP。
correctness0/8、reference/primitive/control/sampling/compile counter0。input/native準備は制御フローから推論。
保存schedule list対runtime tupleの静的interface差を確認。actual bounds全体は未保存。
source/manifest/旧証拠を編集せず、raw STOPと欠測・保存監査を別inventoryへ公開する。
grantは消費済み、retry/resume0。次の修正・新manifest/科学実行は今回未実施。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・u未認定、mandatory STOP。
既存dirty/未追跡・Track Bを保全。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4限定science manifest固定・seal

最新入口は[H4限定seal報告](../track_a_ax2b_h4_limited_seal_v1.md)。保存再監査済みboundsと旧8 cell・capsを固定する。
science source/input/env・CPU3/worker1/BLAS1・専用future outputをmetadataとして結合。
35合成tests pass。新科学計算/array decode/native準備/probe/sampling/build/compile0。
execution_plan_sealed=trueは条件固定だけ。science_authorized=false / launch_allowed=false。
H4_LIMITED_NOT_AUTHORIZED / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP。
旧source/freezes/STOP/結果・dirty差分を保持。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4-P保存read gate v2・再監査

最新入口は[H4-P保存再監査](../track_a_ax2b_h4_native_receipt_reaudit_v2.md)。新gate/runner・36合成testsを追加。
原16MiB aggregate budget内で4MiB超JSONを読める保存専用経路。凍結v1は変更しない。
元source/STOP/receiptのbytesを保持し、実行時sourceと新audit sourceを別に固定する。
新分子計算/native準備/signal/probe/sampling/wrapper build/compile0。H4-P再実行なし。
science manifestのseal/launchなし。`H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED`。
`DRAFT_NOT_AUTHORIZATION`、mandatory STOP。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4-P一回実行・親監査STOP

最新入口は[H4-P実行報告](../track_a_ax2b_h4_native_receipt_execution_v1.md)。保存H4 load1/native準備8とreceipt保存を実施。
親はB3 JSON4,443,419 bytesを4MiB読込gateで拒否しSTOP。総output9,821,513 bytesは16MiB以内。
保存JSONのstdlib補助監査は一致。原STOP・source・結果のbytesを維持する。
signal/probe/sampling/wrapper build/compile0。retry/resume/source修正/science sealなし。
`H4_NATIVE_RECEIPT_STOP` / `H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED`。
`DRAFT_NOT_AUTHORIZATION`、mandatory STOP。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4-P runner・実行前固定 v1

最新入口は[H4-P準備契約](../track_a_ax2b_h4_native_receipt_preparation_v1.md)。専用source・runner・48合成testsを追加した。
CPU3・900秒・AS8GiB・output16MiB・load1/prepare8を未来の計画へ指定する。
今回の実分子load/native準備/signal/sampling/wrapper build/compileは0。
H4-P取得planのsealは認可ではなく、H4 science manifestは未sealのまま。
`H4_NATIVE_RECEIPT_NOT_AUTHORIZED` / `H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED`。
`DRAFT_NOT_AUTHORIZATION`、mandatory STOP。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4実行前契約・metadata固定 v3

最新入口は[H4契約・metadata preflight](../track_a_ax2b_h4_prelaunch_contract_v3.md)。
保存入力/source/旧8 cellを照合し、179 primitive-time組/537 probesを固定した。
専用metadata tests12 passed。新科学計算/array load/sampling/circuit/compile0。
native instruction receipt、CPU実割当、別grantは未固定。manifestは未sealを維持。
`H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。
以下は各stage当時の履歴。旧結果・sourceと既存dirty差分を保全する。

## 2026-10-10 Track A：H4/H6 backend接続準備 v2

最新入口は[接続・実行gate・合成検証の報告](../track_a_ax2b_bound_ports_preparation_v2.md)。
H4独立MP/stage/event port、専用H6 sector/native backendと別grant必須launcherを追加した。
新39＋前回49の88 local synthetic/mock tests pass。分子の正しさ・総u・CI証拠ではない。
source固定のみ。actual input/coverage、CPU/別認可は未seal。新科学計算/sampling/circuit/compile0。
`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。旧証拠・既存dirty差分を保全。
以下は各stage当時の履歴。

## 2026-10-10 Track A：独立レビュー後の準備

[GPT独立レビュー](../track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md)を受け、[準備追補](../track_a_ax2b_post_independent_review_amendment_v1.md)と[H6準備契約 v2](../track_a_ax2b_h6_pilot_preparation_contract_v2.md)を追加。
H4-N/A/E/Mの限定計画、独立small reference・u-aware会計・tol-only adapter、別H6 controller/caps/watchdogを準備した。
専用49 local synthetic tests pass。分子H4/H6検証の新結果・総u認定ではない。
H6 molecular backend/science launcher、H4全stage検証port、input/CPU/別認可は未完了。
旧source/results/freeze/manifestと既存dirty差分を保全。`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。
以下は各stage当時の履歴。

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
| [2026-10-06](2026-10-06.md) | Track A H4追加6距離・最大12 workersの確認と正式契約準備 | server preparation `c2ab34f` + 契約準備文書 | 将来74,784-wrapper cap、数値回路identity修正をhandoffへ記録、新science0 | 契約/schema/zero-compute検査後reviewへSTOP。science port/認可/launchは別指示 |
| [2026-10-05 PM-2結果](2026-10-05.md) | 保存値precision/resource map実行と照合 | source `324435d` / launch `bea4cf0` | 元223候補再現、302点・67,346行、pre/post62 tests、新しい科学計算0 | mandatory STOP、研究方針review。利用者指示でresult commitへ収録 |
| [2026-10-05 source固定](2026-10-05.md) | PM-2保存値解析実装・synthetic検査 | `324435d` | source6 blobs・準備9 files一致、62 synthetic tests。本解析/real reference gate未実行 | 別の明示解析指示待ち。終了後mandatory STOP |
| [2026-10-05追記](2026-10-05.md) | PM-1結果後方針を採用、PM-2精度・測定込み資源契約準備 | `194cc604` + local uncommitted preparation | 保存4 JSONのblob/hash、development218候補・M2元5構成別集合、専用21 local tests。precision解析0、新しい科学計算0 | 契約確認後、保存値解析の実装/source固定。実行は別指示、終了後STOP |
| [2026-10-05](2026-10-05.md) | Track A PM-1最終承認・一項目authorization確定 | `bf9eaee` | final_review_approvedのみtrue、source/plan不変、science-free gate PASS、201 local tests passed | 利用者の明示launch待ち。本計算0、実行後もmandatory STOP |
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
