# 研究C：新selected固定・r16/32/64再調整の限定科学実行

2026-11-03 JST。固定科学SOURCE `1cdeaa8bd55eb07a1d5b8cf2080259e929d405d1` と[1,265-call契約](research_c_rte_r_retuning_contract_unapproved_20261103.json)を無変更で、一つのfresh run01へ実行した。**全登録範囲を完了。1265/1265 attempts、COMPLETE 1265・FAIL 0、retry/resume0。** 歴史的なunapproved filename/metadataは変更せず、ユーザーのチャット上の実行許可に従った。承認証明・承認文保存・新grantは作成しない。

**登録集合内ではr32が条件付き総RZの最小点推定で、保存r128比0.513432544710となった。** 一方、同一J/Uの強D比は1.01155731406で、主RZの点推定では強Dを下回らない。r32の総RZ記述的SEは約1.19209e9で、Dとの差と同程度である。これを母平均の順位や統計的有意差の確定に使わない。全新39roundは潜在適格、96/96回路のraw/compiled・actual adjoint・X/Y検査はPASS。実CA達成・保証付き資源優位は未資格のまま。

[全runner結果](../../artifacts/research_c_rte_r_retuning/2026-11-03/run01/result.json)／[報告用JSON投影](../../artifacts/research_c_rte_r_retuning/2026-11-03/execution_summary.json)／[新39＋保存13round CSV](../../artifacts/research_c_rte_r_retuning/2026-11-03/rounds_new39_saved13.csv)／[journal](../../artifacts/research_c_rte_r_retuning/2026-11-03/run01/science_calls.jsonl)／[STOP](../../artifacts/research_c_rte_r_retuning/2026-11-03/run01/STOP.json)。以下は計算済みJSONの値の投影・引用で、追加の配列decode・operator/norm/shots/費用再評価や統計検定を行わない。

## 1. 固定条件・再利用・正しさ

保存直鎖H4・隣接1.00 Å・STO-3G、通常number CDF8、一般signed対称J・実直交U、Nα=Nβ=2/sector36、system8/full256・controlled512。統合実験の未収束`update005`を固定し、元g補正κ・scalar一回・spin/frame順0..7を維持した。PRは元順LD2・partial ordered symmetric S2・s1・δ=h=.1・K0・m0..12。CA=.00159 Ha、DF reserve=CA/8=.00019875 Ha、βtotalπ/6・βstatπ/12、failure=.01・axis=.01/26、own-DF ground-state oracle・状態準備費用除外。

Candidate SHA256は`938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`、入力snapshotは`1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。契約SHA256 `657aa5439bf66d6a9b19e7e726bd558ad82e4c19827f5386a84221c006409fd6`、SOURCE manifest SHA256 `65c8aadbbbc2dd0179ef2465ce02c9fa546a0e942f3caa4aaf3920dd44eaeef7`、SOURCE67/input27をpre/postflightで照合した。[事前確認](../../artifacts/research_c_rte_r_retuning/2026-11-03/execution_preflight.json)／[事後確認とraw hash](../../artifacts/research_c_rte_r_retuning/2026-11-03/execution_postflight.json)。

固定SOURCEの人工・回帰検証91件PASS（新18＋既存73）の[準備証拠](../../artifacts/research_c_rte_r_retuning/2026-11-03/technical/verification.json)を再利用した。科学実行前のpassive preflightもSOURCE67/input27/依存10packageの一致を確認した。実行中・実行後に技術SOURCEを編集せず、同一テストを不要に再実行していない。

保存DF norm5.23337351802e-9 Ha＋元engineering guard3.22739270291e-9 Haの予算PASSを来歴付きで再利用した。新fit/optimizer/DF norm/PF-error norm/exact-time evolution/ground eigensolveは0。Selectedと独立物理exportのJ/U・2組のsector G/H・basisを照合し、一般K/CAR9因子と保存groundのnorm/residual/aliasを確認した。PF kernelと13round PF誤差はrに依存しないため、同一selectedの保存値を使った。旧J/UのM3/M4データを新candidateへ移していない。

新規rは登録順16→32→64。r128は同じselectedの統合実行`980b514c81eb415fe053f558d7d3c0b8cfb5b9b4`由来の保存13round/native32/費用を添付し、新規sampling/build/compile/QPY decodeを行わなかった。強Dは同一selectedの固定科学SOURCE`5f27539ae0ab88181d9cfcd45badb54daf855c28`、全frame S2/s1の保存結果である。

## 2. RTE・attenuation・signal/phase room

全新39roundの原本は各`r16/r32/r64/m00..m12.json`。PF、補正RTE、A、raw complex signalと、RPEのraw-signal誤差・半径下界・位相予算を分離した。次表はm12の引用である。

| r | 全13round判定 | 保存PF誤差 | 補正RTE誤差 | A | Ideal半径下界ρ | Ideal βdet | Ideal N/axis |
|---|---|---:|---:|---:|---:|---:|---:|
| 16 | POTENTIALLY_FEASIBLE | 0.0945468045945 | 0.100683310535 | 0.661354691433 | 0.532238338883 | 0.196492074692 | 1804 |
| 32 | POTENTIALLY_FEASIBLE | 0.0945468045945 | 0.0491346141047 | 0.813236372497 | 0.696389416759 | 0.144180436852 | 1054 |
| 64 | POTENTIALLY_FEASIBLE | 0.0945468045945 | 0.0242727318556 | 0.901796082078 | 0.794645089633 | 0.119100911755 | 810 |

K0のper-step normalization、configのλ/h/r/K、signed atom/identityとdistributionを元allowanceで検査した。数学的な`A=(1+(λh/r)^2)^(-r·2^m/2)`とbinary64の評価順を区別し、既存libraryの`exp(-r·2^m·log B0)`を変更していない。Aはraw complex signalの絶対値そのものではない。PF誤差は同じ保存値で、r再調整によるPF改善と説明しない。

## 3. Complete native32校正

P0はQiskit1.3.0/opt1/seed17・rz/sx/x/cx・approximation degree1・qubits_initially_zero=false・MPFR160 declared phase。Head内部relative Gaussian・tail exact basis_key共有を維持した。r別の固定32 PCG64 trajectoryを新SOURCE/cohort/axis/sampleへ結び付け、非paired・非adaptive・poolなし。各coherent occurrenceの独立trajectory法則を維持し、校正sampleを同じ回路のcoherent反復として割り引いていない。

Full256 CAR/atom reference、Gaussian9/head相対4、共通ancilla head/X/Y shellは同一armで一度準備し共有した。各trajectoryのfull512 raw/compiled reference、unitarity、actual raw/compiled adjoint、complex X/Yを元component allowance1e-10・signal2e-10・rtol0・phase fitなしで検査した。Gaussian gate単体のsector射影で代用していない。QPY保存は新384件。各sampleの診断・seed/event・metrics・参照は各rのsamples/circuits/referencesに保存した。

| r | Mean RZ | Mean RZ depth | Mean CX | Mean CX depth | Mean total depth | Mean size |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 5536.1875 | 2104.9375 | 2716.6875 | 2048.5 | 4593.5625 | 10780.8125 |
| 32 | 7636.5 | 2612.1875 | 3401.9375 | 2348.875 | 5713.6875 | 15006.6875 |
| 64 | 11411.5625 | 3544.15625 | 4644.1875 | 2914.40625 | 7783.125 | 22627.3125 |
| 128（保存） | 19825.625 | 5632.1875 | 7399.3125 | 4152.59375 | 12393.1875 | 39603.9375 |

Finite32のmean/SEは記述的価格で、母平均・全law support・厳密合成上界・選択後CIではない。標本min/maxや全96回路の数値PASSから、全sampling lawを資格付けていない。未実装のcross-head/tail融合や全長追加compileの削減は仮定しない。

## 4. Shell込み条件付き総RZ

主指標の会計は`C_total=W*μ_kernel+S_shell`、`W=Σ_m,b N_m,b 2^m`（s1）。rはkernel内部event数、32は古典校正数で、Wに再乗算しない。各interrogationのancilla head/readout shellは一度ずつ含める。共通価格校正のSEはW×SE_kernelで伝播し、13独立校正として縮小しない。

| r | Ideal shots X+Y | W | Mean kernel RZ | 条件付き総RZ | 総RZ descriptive SE | r128比 | 同J/U D ideal比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | 17452 | 21238192 | 5536.1875 | 117578674155 | 1538851860.28 | 0.565501898448 | 1.11414359565 |
| 32 | 15202 | 13979232 | 7636.5 | 106752458375 | 1192089996.44 | 0.51343254471 | 1.01155731406 |
| 64 | 14398 | 11519264 | 11411.5625 | 131452851483 | 1299823829.36 | 0.632230611582 | 1.24561153341 |
| 128（保存） | 14048 | 10487392 | 19825.625 | 207919150188 | 2249141484.14 | 1 | 1.97018542062 |

同一new-J/U強Dはshots13728・kernel RZ11026・総RZ105532782860である。D numericalとD idealは保存結果で同じ整数shots/費用だが、位相誤差の資格自体は異なる。主比較は共通ideal/idealの反実条件を明示し、numerical D対PR idealの比較だけで保証付き同精度を主張しない。

| r | Assumed energy状態 | Assumed shots | Assumed総RZ |
|---|---|---:|---:|
| 16 | CONDITIONAL_CA_PASS | 17528 | 118141306084 |
| 32 | CONDITIONAL_CA_PASS | 15268 | 107260835684 |
| 64 | CONDITIONAL_CA_PASS | 14454 | 131966965393 |

Evidence laneは全law synthesis/implemented-law誤差が未知のためshots/費用nullを維持する。Idealはexact synthesis・ideal finite-RTE IID、assumedは各raw-signal誤差≤.001Aという未検証仮定であり、未知の実誤差を0または既知boundへ変更しない。Energyの`CONDITIONAL_CA_PASS`は宣言モデルにおける十分条件で、測定したエネルギー誤差・actual CA達成ではない。

副指標も保存した。Depthはfenced serial scheduled sumで、全長再compileの最小depthやhardware実時間ではない。

| r/方式 | 総RZ | 総RZ depth | 総CX | 総CX depth | 総total depth | 総size |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 117578674155 | 44705127855 | 57697530729 | 43506436312 | 97559075777 | 228965079229 |
| 32 | 106752458375 | 36516428297 | 47556473562 | 32835468564 | 79873061951 | 209782064927 |
| 64 | 131452851483 | 40826121894 | 53497621878 | 33571814997 | 89655965207 | 260650079885 |
| 128（保存） | 207919150188 | 59067007298 | 77599490718 | 43549878473 | 129972306754 | 415342108618 |
| D（保存） | 105532782860 | 53292834864 | 66998834000 | 60069240312 | 115793216908 | 200221319010 |

## 5. Calls・資源・保全・結論の範囲

実績は1265 attempts、COMPLETE 1265・FAIL 0、上限1265、retry/resume0。SCI SOURCE/契約/数値allowance/候補集合を変更せず、新grant・旧run再開・未使用枠の流用・追加rを行っていない。詳細内訳はSTOPとexecution_summaryを参照する。内側library操作を追加fit/量子shotsと混同しない。

実行前はCPU affinity32・約54GBのMemAvailable・pressure0を確認した。固定resource_planで独立13round/32nativeを並列化し、BLASを協調、Qiskit build/compile/adjoint作成排他を維持した。実績wall=1531.30160716秒、CPU=24431.4102443秒、peak RSS=2742360 KiB。Speedup比較はしていない。CPU/RAM/worker/wallの人工上限、共有venv/cache/設定や他ユーザーjobへの変更はない。

この固定H4・未収束selected・s1/K0・登録r集合・oracle準備除外・serial会計の結果を記録する。旧PRのr再調整は今回未実施であり、新selectedのr再調整をDF目的だけの純粋な因果効果と呼ばない。全law合成・implemented-law距離・価格母平均/support・厳密CA・保証付き総資源優位は未資格/nullのまま。固定cohortの点比較から、PR一般の不可能性・全指標支配・独立新規性・論文化や研究終了を採択しない。

次の担当は**ユーザー／研究C**。登録rの条件付き資源・強Dとの差・記述的SEと未資格事項を確認し、後続の科学的な問いを選ぶ。Codex-Aの今回scopeは公開照合で完了し、旧J/U再調整・追加r・U更新・全law保証・重要科学レビューへ自動進行しない。共有索引への反映が必要ならA/B統合担当へ引き渡す。

Branch `codex/research-c-rte-r-retuning-execution-20261103`。新run・SOURCE/input・契約・call/STOP/QPYと報告を公開対象とする。Remote SHA/tree・公開本文・raw証拠・旧成果不変の照合を完了条件とし、公開commitが本報告と結果の識別子になる。**Mandatory STOP。**
