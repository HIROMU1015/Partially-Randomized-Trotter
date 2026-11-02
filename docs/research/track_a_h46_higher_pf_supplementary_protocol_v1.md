# H4/H6高次PF追加実験プロトコル・記録 v1

2026-10-20。**完了済み実験の条件・固定契約・保存結果を整理した付属文書。新しい科学契約の採択ではなく、本書自体の事前登録も主張しない。** 共通プロトコルv2の正式条件、過去の承認、STOPを変更しない。m5の主比較採用AD03は保留である。

資料基点：`5764aacb0714c6be42cb6dc7c7d8fdf1c8d98bdb`。共通v2の参照版：`3826f566b04ad9c2ead9abf1b80c864f7b34b785`。統合データの入口は[README](../../artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/README.md)、固定commit・ファイルSHA・Git blobの対応は[source catalog](../../artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/source_catalog.json)を正とする。

## 1. 目的と既存比較との関係

S2、Yoshida4、`4th(m5_best)`のPRを同じ保存H4/H6入力とouter32条件で比較できるよう、signal、経験的適格性、m5決定論的対照の実測単価を結合する。現在はPR単価・探索標本数・正式shot政策が未確定であり、費用順位やPRの優位性は本書で評価しない。

既存の強化S2/S4決定論集合は保全する。m5だけを対照にしたり、旧S2/S4 model係数をm5に移したりしない。H4 legacy DF12、ordinary-controlの旧高次PF比較、別研究のPF直接compile788軸は個別の系列である。共通v2のH2/H4/H6/H8既存結果もそのまま保持する。

## 2. 時系列：実験前の固定と実験後の整理

| 記録 | 固定資料 | 本書での扱い |
| --- | --- | --- |
| m5係数版 | 外部PF repositoryの`20d64c2fea1ed2d2188777df4f3d962ee94df486`、[係数manifest](../../references/h46_m5_best_formula_v1.json) | 保存したbinary64版を使用。再最適化していない |
| signal領域・入力・実装 | [88候補manifest](../../references/h46_m5_signal_domain_v1.json)、SOURCE`bec018c64e7a8458dbf13b1c0e9655d628292857`、plan SHA`85bb596c3fce5349d2066d05c5dd1a0cbeebfa49e1f3a35db30db729ac1d8f08` | 修正版run02の実行前固定。run01の初期化STOPは別記録として保持 |
| signal run02結果 | `a960a41da5ebc564b0306e798572257091c8509c`、[結果報告](track_a_h46_m5_signal_run02_results_20261020.md) | 実験後の保存signal・880軸ラベル |
| 費用取得対象・再利用条件 | SOURCE`635ba563766875d688ad6912499b7f90397d0cc5`、[費用取得準備](track_a_h46_m5_cost_sampling_readiness_20261020.md) | signal結果を見た後、費用取得の実行前に対象を固定。signal前の事前選定とは呼ばない |
| H4/H6費用取得 | [起動記録](track_a_h46_m5_det_cost_run01_start_20261020.md)、[終了結果](track_a_h46_m5_det_cost_run01_results_20261020.md) | 系別の独立承認・契約で38/54軸を一回取得。retryなし |
| 本付属文書・統合索引 | 本branchの文書commit | 完了済み証拠の事後整理。過去への遡及採択・最終evidence freezeではない |

manifestに残る過去の`execution_authorized=false`等は、その準備時点の記録である。後続の実行承認・実行完了はそれぞれの起動・終了資料で確認する。過去のDRAFTを有効grantへ書き換えない。

## 3. 入力・状態・共通条件

| 条件 | H4 | H6 |
| --- | --- | --- |
| 分子 | 線形、隣接1.0 Å | 同左 |
| 基底 | STO-3G | STO-3G |
| DF政策 | tol-only `1e-8`、rank自動決定 | 同左 |
| 保存観測rank | 10 | 19 |
| PR prefix | 3 | 5 |
| m5 B0/B0I prefix | 3、8、9 | 5、7、11、15 |
| m5 full prefix | 10 | 19 |
| q | 1、2、4、8 | 同左 |
| T、P | T=0.8、状態準備費用P=0 | 同左 |
| signal精度 | ε=.05/.01/.005/.001/.0005、両軸 | 同左 |
| 主診断 | 通常signed bias、経験的allowance倍率1 | 同左 |

[conditions.json](../../artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/conditions.json)にHamiltonian/state/snapshot/入力receiptのSHA、DF政策、compiler/runtimeの版を記録する。rankは保存観測値であり、rank10/19を生成引数に渡した政策ではない。

保存receiptでは、DF係数の順序はdecomposer生成順、追加cutoffは0、`final_rank_supplied=false`である。H4は保存rawからのweighted Hermitian projection、H6も対応するprojection政策を持つ。表現誤差・基底状態・referenceの厳密certificateを新たに認定しない。

H4のdense eigh・sector36は共通v2の記述を参照したもので、本作業で元generatorの完全な再監査はしていない。H6はSHA確認済みsnapshot receiptのsector400・12 qubit・α/β各3電子、eigsh/SA、固定Hartree–Fock初期値、1回のbinary64正規化、最大振幅の位相固定政策を転記した。過去の生成thread数は歴史的設定で、今後の資源上限ではない。

**SCF・軌道・状態生成規則の全過程の系間完全一致は未認定。** 既存の入力hashと実行契約の結合を示す範囲に留める。DF政策の一致だけからサイズの因果効果・漸近スケーリングを主張しない。H4 DF12の結果・単価をDF10へ転用しない。

## 4. PF公式・係数版・合成順序

固定SOURCEの`src/trotterlib/h46_m5_plan.py:weights`および`h46_m5_native.py`を正とする。本整理ではこれらをimport・実行せず、Git bytesだけを確認した。

| label | S2の合成列 | 版・注意 |
| --- | --- | --- |
| `2nd` | `S2(t)` | 保存実装の対称2次 |
| `4th` | `S2(a t), S2(-u/(2-u) t), S2(a t)`、`u=2**(1/3)`, `a=1/(2-u)` | Yoshidaの3段S2合成。5段Suzuki式とは区別 |
| `4th(m5_best)` | `reversed(tail) + [central] + tail`の11 signed S2 piece | `PF_M5_BEST_20D64C2_BINARY64_V1` |

m5 tailの保存リテラルは、順に
`0.11225428783206336, 0.14181961643121604, 0.1211920365713068, 0.13781643533753404, 0.12547397927357543`。
中央係数は元Pythonと同じbinary64の`1 - 2 * sum(tail)`。この整理で係数や次数残差を新しく計算していない。旧`new_2`の8桁版とは異なる公式IDであり、混合しない。非zero残差だけを理由に棄却せず、固定版の数値次数証拠と適用窓を区別する。厳密代数的次数certificateを得たとは扱わない。

outerごとに、列の順に符号付き時間`T*coefficient/q`のS2 pieceをnative回路へappendする。負時間を絶対値へ置換しない。S2内部のforward/reverse順、独立occupation oracle、scalar/controlを固定SOURCEに合わせる。

PRはK4、`ABS_TIME_PER_OUTER=32`。**32 short stepsは各outerに含まれる全pieceの合計**であり、pieceごと32ではない。各pieceへ最低1 stepを置き、残りを係数絶対値比例のfloor＋最大剰余法で配分する。同率は元index順。実装は固定`h46_new2_core.py:allocation`を参照する。旧`PER_PIECE=2` pilotとはlaw・標本系列が異なる。

prefixはone-body項と生成順の最初p個のDF fragmentを意味する。B0は残部を打切り、B0Iは抽出identityをscalarへ加え、PRはmatrix-only tailを扱う。Hamiltonian定数（B0I/PRではtail identityも含む）のscalar位相は全pieceの後に1回適用する。full prefixは空tail分岐を使う。旧plan moduleのlegacy `SYSTEMS`をtol-only入力として流用せず、実際の入力は固定tol-only bindingを正とする。

## 5. signal・normalization・経験的適格性

保存full-H referenceと同じ保存状態を使用する。PRのexact-tail、有限RTE平均のcorrected、独立に減衰伝播したrawを保持する。K4有限多項式の具体的な次数規約は固定SOURCEの`polynomial(..., cutoff=4)`を正とし、別のTaylor規約に置換しない。

- `total_signed_bias = corrected - reference`
- `outer_PF_or_discard_bias = exact_tail - reference`
- `finite_RTE_bias = corrected - exact_tail`
- `log_B`はpieceごとの`r*log(b)`の和。rawと`exp(-log_B)*corrected`のvector closureを保存する。

決定論的候補にはPRのtail samplingがなく、保存log_Bは0。以上は過去SOURCEにおける定義の説明であり、今回この差やclosureを計算し直していない。

保存経験式`M5_SIGNAL_EMPIRICAL_OBSERVATIONS_V1_PROPOSED`は、closure・最大中間norm・stage数に基づくroundoff観測からuを置く。固定式は`u=max(1e-9, 10*closure+roundoff)`、log margin floorは`1e-12`。軸別headroomは`ε/√2−|bias|−u`であり、正でなければINELIGIBLE、正でも`u/headroom>.01`ならUNDETERMINED、それ以外がELIGIBLEという保存政策である。

reference誤差は厳密に0と認定しておらず、`reference_absolute_error`・`strict_bias_certificate`はnull。適格性は保存reference精度を仮定した経験的診断である。compiled状態差の閾値をshot allowanceへ使わない。

88候補それぞれ5精度・real/imagの880行を、保存値のまま参照する。費用取得対象は**同一εで両軸ELIGIBLE**なm5決定論48候補。異なるεの片軸同士を組み合わせない。残る16候補は削除せず`INACTIVE_DOMAIN_PRESERVED`で保持する。

## 6. 回路単価とnative受入

full Hadamard wrapperのcosine/sine、`symmetric_directional` control、scalar onceを維持する。native policyは`M5_TOL_ONLY_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1`。Python3.12.3、NumPy1.26.4、Qiskit1.3.0、basis=`rz/sx/x/cx`、opt1、transpiler seed17、backend/target/coupling/layout指定なし。compiler settings SHAは`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。

主単価はcompiled RZ count。CX count、CX depth、RZ depth、total depth、circuit sizeも保存する。P=0であり、状態準備の生成実験が行われたことと状態準備費用の計上を混同しない。将来の主会計は両軸の単価×解析shotの和を予定するが、本データは**単価のみ**で、shot・総費用・順位はnullである。

compiled対uncompiledの保存状態vector差はCT01の**厳密に1e-7未満**、その他のnative低control/高control・漏れ・独立oracle・Hadamard・保存log_B検査は**厳密に1e-9未満**を要求した。位相合わせ・補正をしない。これは保存状態での受入基準であり、全状態operator normや厳密biasの保証ではない。

各pairはcosineの受入後にsine compileへ進み、同一evolution/preparationを両軸に使う。失敗時のquarantineを受理費用へ昇格しない。新92軸は46pair×8の保存gate368行へ結合する。保存再利用4軸の受入証拠は旧pilot/CT01系列にあり、今回の368行に再検査として足さない。

## 7. 実測の再利用・来歴

| 対象 | 新規実測 | 保存実測再利用 | 計 |
| --- | ---: | ---: | ---: |
| H4 m5対照 | 38軸 | full10/q2の2軸 | 40軸・20候補 |
| H6 m5対照 | 54軸 | full19/q2の2軸 | 56軸・28候補 |
| 合計 | 92軸 | 4軸 | 96軸・48候補 |

12保存軸のうち今回の両軸適格領域に属する4軸だけをactive表へ結合した。残り8保存軸を消去したり、未取得0軸と表示したりしない。保存RZ以外のmetricは、SHA確認した4原receiptからそのまま転記した。H4再利用receiptにcompiled QPY SHAが明示されていない箇所はnullを維持する。

H6 full/q2 cosineの元SOURCEは`f743a88225f7cb2c8d2c91d7e19813a5780f803b`、独立受入SOURCEは`885bbd7199c9ada26704b7db75b99767734e92e7`。旧1e-9基準のSTOPは保持されている。CT01受入と旧PASSを混同しない。H6 sineもCT01の別実行結果を使用する。H4は元pilot SOURCE`c7d53f04c3be131078ee91e8bef3c78ca434ed3e`。

再利用の条件は入力ham/state/snapshot、preparation、PF版・順序、control/scalar、native政策、compiler/runtime、receipt/result hashの整合である。profile `4e32bb9b…`は`profile_is_reference_only=true`であり、旧主比較runtime全体との一致証明ではない。

新92軸の科学SOURCEは`635ba563…`、終了metadata監査SOURCEは`c1b419cde004d2b07b41f5727cd4722b5e0e0bc4`。SOURCE、原結果SHA、軸receipt SHA、wrapper fingerprint、取得/受入の区分は統合表に保持する。実行時間は[run provenance](../../artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/run_provenance.json)に保存値を転記した。native exit0・worker回収・mandatory STOPは元終了監査を参照し、今回再compile・vector再生をしていない。

## 8. PR探索・独立確認：未実施の設計事項

PRはH4/H6各12群、合計24群の費用が未取得で、n_E・fresh seed・n_Cはnull。対象24群を保存適格性で保持したことは、samplingやshot政策を承認したことを意味しない。

新しいouter32 lawと同じ条件の分散は未取得であり、piece-r2 pilot、旧ordinary-control n8、別の高次PF独立確認13/45/45をn_Eへ流用しない。固定探索nを有限標本目的で採択する案と、別契約の同law設計pilot後にn_Eを固定する案があり、数値は未採択である。

trajectory内のcosine/sineはpaired、5精度では同じtrajectoryを使う。候補間のpaired構造を仮定しない。cohortとseedのidentityを固定し、既知seed一覧との衝突検査をfresh gateに含める。探索後に候補を固定して新seed・新cohortによる独立確認を行い、探索・pilot・旧確認標本とpoolしない。n48を一律に要求せず、候補別に事前設計する。

K4の無条件lawを保持し、s4を強制・棄却しない。s0/s2_single/s2_multi/s4、piece seed・order列・event digestを記録する。条件付きrare pilotを無条件標本へそのままpoolしない。未観測層の費用・分散・母平均tail上界はnull。標本平均・paired SE・順位再現は母平均順位保証とは区別する。

## 9. 現在の完了範囲・欠測・次の承認

| 項目 | 状態・証拠範囲 |
| --- | --- |
| signal | 88候補（新82＋保存6）、5精度・両軸880行、正常終了 |
| 決定論m5単価 | 48候補・96直接実測軸（新92＋保存4）、model推定ではない |
| 非active対照 | 16候補を保存。今回の両軸適格εがない |
| PR24群単価・n_E・seed | 未取得・未固定 |
| shot政策・会計・q選択・SE | 未採択/未実施。null、0補完なし |
| 強化S2/S4対照との接続 | 保全。新nativeへの完全identity接続は別途必要 |
| m5主比較採用AD03 | 保留 |
| 厳密bias・母平均保証 | 未確立 |

H4DF10 F/FPは入力・compiler一致の証拠があってもnative/preparation/profile全結合は未完了。H6強化PFは実測とmodel推定を分けた既存結果を保持し、全て直接実測と表示しない。PF788研究は今回NOT_VERIFIEDで、費用・grantを輸入しない。

次にユーザーが判断するのは、PRの探索目的・n_E設計経路と標本数、別sampling/compileの実行範囲である。会計前にはε軸分配、α、shot整数式、normalization・bias・u・log margin・guardを元SOURCEと照合し、正式shot政策と数値実行を別途採択する必要がある。AD03の採択も別判断である。

将来の実行でもSOURCE/input/compiler不一致、数値FAIL、seed衝突、台帳不整合、worker異常・実OOMで当該契約をSTOPし、retry・範囲追加・自動次Phaseを行わない。通常の他worktree更新や一時的PSIだけで無関係な正常workerを停止しない。独立契約の保全規則と共有環境保護を維持する。

本作業は既存値の転記・構造結合・人工テストだけで終了する。新しい科学計算、正式shot政策、AD03、既存v2条件への自動適用はなく、**mandatory STOP**する。
