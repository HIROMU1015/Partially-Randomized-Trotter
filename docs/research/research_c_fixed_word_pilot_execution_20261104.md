# 研究C：固定4-word native/reference有限pilot・実行結果

2026-11-04 JST。**固定86-call契約を無変更でfresh run01へ一度実行し、86/86 COMPLETE、correctness FAIL・retry・resume0。4/4 wordの全登録検査がPASSした。**

## 固定SOURCE・契約・対象

- 科学実行SOURCE：`ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`。実装SOURCE：`6e8475cc871916473d48fd38fb4b4a0297e84d15`。
- SOURCE manifest SHA256：`22284b98e3d2fa37fc134e734f402d96b3f3989f5c0e3fba5a2098fb03d7847d`。
- 契約SHA256：`634b0c3480e124acd540852f0490ddd77300efcf9e4500bbdb86cfd04614b614`。
- 元h/g/scalar packet SHA256：`1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。
- [固定契約](research_c_fixed_word_pilot_contract_unapproved_20261104.json)／[技術準備・85人工回帰PASS](research_c_fixed_word_pilot_preparation_20261104.md)。今回のユーザー明示実行許可を適用し、歴史的unapproved/metadataを変更せず、新grant・承認証明・承認原文記録を作成していない。
- [SOURCE](../../artifacts/research_c_fixed_word_pilot/2026-11-04/run01/SOURCE.json)／[契約コピー](../../artifacts/research_c_fixed_word_pilot/2026-11-04/run01/contract.json)／[result](../../artifacts/research_c_fixed_word_pilot/2026-11-04/run01/result.json)／[STOP](../../artifacts/research_c_fixed_word_pilot/2026-11-04/run01/STOP.json)。

Linear H4・隣接1.00 Å・STO-3G、Nα=Nβ=2、number CDF8、sector36/system256/controlled512、元frame順LD2、partial ordered symmetric S2/s1/r32/K0、h=δ=.1。PR endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。元g補正κ・scalar一回・signed atomとidentity位相を保持した。Own-DF ground oracle・状態準備除外の研究条件を維持するが、今回のX/Y検査は保存数値vectorによる。

P0はQiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/initial-zero false、既存MPFR160 declared phase・Gaussian/relative fusion・basis_key共有規則。検証対象は1 h-stepのcomplete controlled kernel。親研究のm0..12/CA=.00159 Ha/DF reserve=CA/8は変更していないが、今回round profile・RPE・CA・shots・総費用を評価していない。

## 実行前照合・既存実装の再利用

[実行preflight](../../artifacts/research_c_fixed_word_pilot/2026-11-04/execution_preflight.json)：固定HEAD、SOURCE143 file、保存入力16件、10依存package、fresh outputを照合しGO。準備時の85人工・回帰PASSから実装は変更せず、科学SOURCE/契約を修正・再起動していない。

既存 `c_fixed_word_pilot`→`c_pr_integrated_performance.native(fixed_words=...)` を使用。Full256 CAR/sector比較、immutable head・signed atom cache、StrongBuilder、P0、SharedQiskitCalls、full512 Operator、actual inverse、X/Y shell、metrics/QPY/Calls/STOPを再利用した。新fit・optimizer・RNG・DF norm・ground solve・B2再計算は0。

保存J/Uとのexact一致、元SOURCE・保存DF予算PASS、216 atomのcoefficient/frame/support一致、K0/r32を確認した。保存DF sector norm4.843671019084795e-9 Ha・engineering guard3.2165809758004982e-9 Haは同一endpointの既存結果を引用し、新しい誤差保証へ昇格していない。保存groundのenergy=-2.1663874492829245 Ha、残差8.554429181071078e-16、normalization差1.1102230246251565e-16、alias margin1.6179938779914944 radを記録した。保存vectorと真のgroundの距離は未認定。

Full256/sectorの9因子と和、signed tail law＋identity、候補非依存head/X/Yのraw/compiled shell3組もPASS。個々のGaussian gateをsectorへ投影してfull controlled回路の正しさに代用していない。

## 固定wordとfull512 correctness

辞書順atom ID・frame/support長という事前固定の構造規則だけで選んだ4本。各32event。過去の価格/残差を見て選定せず、sampling・cohort推定・事後のword追加は行っていない。

| word | 事前固定した列 | 判定 |
|---|---|---|
| same_atom | F2:0を32回 | PASS |
| same_basis_Z_ZZ | F2:0 / F2:0,1を16回交互 | PASS |
| alternating_frames | F2:0 / F3:0,1を16回交互 | PASS |
| six_frame_cycle | F2..F7の:0,1を巡回、先頭32event | PASS |

Frame名とbasis_keyを同一視せず、合法なexact basis共有は既存規則に従う。独立CAR/event referenceは元head順・tail identity・scalarを保持する。Control=0 identity/control=1の参照に対し全512入力列・全出力成分を比較し、global phase fitは行わない。

以下は保存binary64観測値。成分allowance1e-10、unitarity成分1e-10、complex X/Y2e-10、rtol0を維持した。厳密な上界・rounding enclosureではない。

| word | raw/ref max entry | compiled/ref max entry | compiled/raw max entry | compiled actual-inverse/ref max entry | compiled inverse復帰 max entry | X/Y最大complex error |
|---|---:|---:|---:|---:|---:|---:|
| same_atom | 5.08111373837e-15 | 2.29401590897e-13 | 2.28203352437e-13 | 2.31912229847e-13 | 1.03695068235e-13 | 9.61071384308e-14 |
| same_basis_Z_ZZ | 5.88424749141e-15 | 2.2630407603e-13 | 2.25109357536e-13 | 2.2545167092e-13 | 1.02477081812e-13 | 9.49152748747e-14 |
| alternating_frames | 7.55834757457e-15 | 9.63394662629e-13 | 9.62171773762e-13 | 9.64711679534e-13 | 1.14908203725e-13 | 9.80297736662e-14 |
| six_frame_cycle | 5.89566337222e-15 | 5.82339282208e-13 | 5.81119037942e-13 | 5.82231167422e-13 | 9.97792251945e-14 | 8.50079698009e-14 |

Rawとcompiledのactual inverseはそれぞれ実際の回路操作で生成し、reference†との比較・元回路との復帰を検査した。行列のdaggerをactual回路の代用にしていない。Raw/compiledのunitarity、head/X/Y shellを経由したcomplex signalとnormもすべてPASS。X/Yは各固定unitary trajectoryの信号であり、finite-RTE平均信号ではない。Attenuationの外挿や再乗算も行わない。

Compiled/ref Frobenius観測値はword順に1.819123026163536e-12、1.8142800551081315e-12、1.2428037837020514e-11、4.549028095137155e-12。参照構築・Operator再構成・norm計算の丸めを包囲した値ではなく、ここから厳密operator boundを認定しない。詳細は各 `samples/wordNN_{raw,compiled,adjoint,XY}.json` に保存。

## 固定P0 kernel費用

この4本それぞれのcomplete controlled kernelを測定した。X/Y shell・state preparationの総費用、13roundの反復・shots加重費用、母平均価格は算定していない。

| word | RZ count | RZ depth | CX count | CX depth | total depth | size |
|---|---:|---:|---:|---:|---:|---:|
| same_atom | 3409 | 1663 | 2068 | 1812 | 3610 | 6491 |
| same_basis_Z_ZZ | 3568 | 1683 | 2148 | 1850 | 3676 | 6844 |
| alternating_frames | 6470 | 2945 | 3074 | 2530 | 6542 | 12806 |
| six_frame_cycle | 8018 | 3141 | 3608 | 2608 | 6813 | 15982 |

Basis共有/切替やsupport/orderの違いを含む固定4word間の費用差を観測した。各要因の独立した因果効果、IID-grid lawの期待価格・support、他DF候補や別分子の性能差を示す結果ではない。旧native32のsample平均と混合したり、有限最大値を全law上界にしたりしない。

## Calls・資源・保存物

| 登録区分 | COMPLETE |
|---|---:|
| input/evidence/候補/保存state/lawの束縛 | 11 |
| event catalog・4word束縛 | 2 |
| 共有CAR/native/basis/shell setup | 35 |
| 各word9種×4 | 36 |
| 有限summary・全体summary | 2 |
| 合計 | **86** |

[Call journal](../../artifacts/research_c_fixed_word_pilot/2026-11-04/run01/science_calls.jsonl)は86 attempt/86 COMPLETE、全kindが固定上限内、FAIL0。Actual inverse pair1単位の中にraw/compiled両方の生成・decode・検査を含む。Retry/resume0、未使用旧call枠の転用0。

Wall 58.949028637秒、process CPU 142.128206439秒、peak RSS 926728 KiB。CPU affinity32、利用可能RAM約52GiB、4つの独立wordに4 worker、各BLAS1。Qiskitの共有状態に関わるbuild/compile/adjoint作成は既存mutexを保持した。CPU/RAM/worker/wallの人工cap追加0、共有venv/OS/global cache/他jobの変更0。実測対照がないspeedupは主張しない。

新runは60ファイル・10,528,232 bytes。QPY16件（各wordのraw/compiled/両actual inverse）、system256 CAR reference4 NPZ、検査/metrics/契約/SOURCE/Calls/STOP/resourcesを保存。候補非依存shell6 QPYは原本再利用・構造再確認のみ。[実行・保全照合とraw inventory](../../artifacts/research_c_fixed_word_pilot/2026-11-04/execution_verification.json)で、SOURCE143 file・input16件・契約コピー・event順・calls・QPY数と旧artifact treeの不変更を確認した。公開後はremote SHA/tree・報告/主要結果/原本blobを照合する。

## 得られた資格と残件

- 得られたもの：固定候補・固定4word・1h-stepについて、complete full512 native/raw/P0/actual inverse/X/Y評価器の登録数値correctness PASSと費用観測。
- 得られていないもの：全合法wordのWhole-P0一様synthesis、per-word rigorous reference uncertainty、categorical unitary trajectory前提、実PCG64のIID保証、13roundのraw bias/合成誤差、RPE/actual CA、母平均価格/総資源優位。対応するunknown/null/falseをresultに保持する。
- B2のArb上界はsector36の保存数学的平均referenceの二項。今回のfull512・per-word binary64 reference誤差へ移送せず、B2/Stage Aを再実行していない。Literal saved Uに基づくatom mapとexact unitary native trajectoryの厳密同一性も今回証明していない。
- 各固定回路の全operator数値検査であり、有限個のstate-actionだけを全operator保証と呼ぶものではない。一方、全operatorを比較したことも浮動小数点丸めの厳密包囲を意味しない。

**Stage C ready=false、whole-law bound=null、actual CA=nullを維持した。科学実行はここでmandatory STOP。**

## 次の研究判断

次担当はユーザー／研究C。今回の有限correctness結果を踏まえ、fresh IID-grid校正のestimand（kernel期待operator差、tail、round raw bias）と資格水準・confidence/failure政策を選ぶ。Codexはその選択後、同じ数学的target・有効なper-sample誤差評価・必要なSOURCE/calls/fresh STOPを固定する。今回の許可で追加word、fresh sampling、N探索、RPE/Stage C、重要科学レビューへ自動進行しない。
