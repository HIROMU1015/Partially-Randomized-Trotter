# C1 Stage3：決定論S2・固定shotsの精度資格付け準備

2026-10-25 JST。起点公開commit `5d7faf7121cc281b7e2f38c2e04c224e594ae53d`。
Branch `codex/research-c-c1-stage3-deterministic-accuracy-20261025`。

**4候補の保存結果から13 roundの合成誤差、固定shots、共通CA条件、保存総費用を接続する経路を実装した。新規人工12・独立process回帰140件＋29 subtests PASS。実H4のdecode・新計算・保存科学値再解析・実回路build／compileは0。科学実行は未承認、公開照合後mandatory STOP。**

[未承認110-call契約](research_c_c1_stage3_deterministic_accuracy_contract_unapproved_20261025.json)／[SOURCE・人工検証・preflight](../../artifacts/research_c_c1_stage3_deterministic_accuracy_technical/2026-10-25/)／[前段の費用結果](research_c_c1_stage3_cost_compat_execution_20261025.md)。

## 対象・固定条件

保存P-H4（直鎖1.00 Å・STO-3G、Nα=Nβ=2）、CDF rank8 number・元g補正κ／scalar／元frame順、XDF rank10。決定論ordered symmetric S2の各arm s=1／4、計4候補、s=1を主対象、s=4を対照とする。PRの探索・資格付けは対象外。CA=0.00159 Ha・DF reserve CA/8、δ=0.1、m=0..12、既存β／α配分を変更しない。13 round×X/YのIID理想Born測定、own-DF sector ground oracle、状態準備費用除外。deterministic protocolにはstochastic RTE lawがないのでlaw誤差0は構造上の値で、PRの未知誤差を0にする操作ではない。

P0 Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation1／qubits_initially_zero=falseと既存basis共有・融合・角度結合・serial fence費用を保持する。回路を作り直したりcompileしたりしない。前段のCDF/XDF約27%費用差を同CA優位として事前認定しない。保存DF誤差CDF約7.6660e-5 Ha／XDF約3.3752e-15 Haは異なるままで、今回数値を読み直したりnormを再計算したりしていない。

## 再利用と最小限の追加

新module `src/trottertracks/c1_stage3_deterministic_accuracy.py`、entry `scripts/run_research_c_c1_stage3_deterministic_accuracy.py`、人工test `tests/management/test_research_c_c1_stage3_deterministic_accuracy.py`を追加した。旧SOURCEは変更していない。

| 保存資産／既存関数 | 使用方法 |
|---|---|
| Stage1のCDF full256 CAR base＋8 frame | 元κ／constant／frameの独立referenceを再利用 |
| `stage1.car_bilinears`／`native.xdf_car_generator` | 保存元Gとone-bodyからXDFの未保存local reference用generatorのみ構築 |
| Stage2のcompiled対CAR full Frobenius report | s=1両armのkernel合成誤差の**数値pilot**入力としてそのまま再利用。QPY／reference再評価なし |
| cost-onlyのD(s4) raw／compiled差・unitarity report | compiler誤差を再利用。未保存のraw対CAR local reference差だけ2件を追加する将来の設計 |
| cost-only head／X／Yの保存QPY | 個別残差が保存されていないため、この3 segmentだけraw・compiledをdecodeして数値評価 |
| `reference.reference_kernels`／`component_check` | 2 local S2 referenceと位相を保つ照合。新しいGaussian合成器・fitter・数値ライブラリなし |
| 保存feasibility／shots／native費用 | 4候補のPF誤差・DF予算・alias／ground確認・固定N・費用をjoin。PF全round・ground／alias・DF normは再計算しない |
| `AccuracyTarget`／`assess_energy`／`aggregate_totals`／`compare_totals` | 既存のhorizon、DF一回計上、全13round・failure予算、固定N費用集計・比較を再利用 |

保存回路の候補は決定論だけであり、有限集合をすべて扱う。PRのsample最大値や未観測supportを使用しない。parent proposalに残るPR関連metadataは来歴用で、このentryからdispatchしない。

## 13 roundの誤差接続

各kernelのunitary距離の数値代用値をηとする。s=1は保存Stage2のcompiled対CAR Frobenius残差を用いる。s=4は新たに数値評価するraw対CAR残差と保存raw対P0残差のFrobenius和を用いる。元のmax-entry1e-10／unitarity1e-10／rtol0／位相合わせなしはcorrectness gateのままで、1e-10をoperator norm上界として流用しない。

`n=s*2^m` とし、headとaxis readoutの距離η_head、η_axisも含めて

```
d_axis(m) = min(2, n*η_kernel + η_head + η_axis)
e_syn_pilot(m) = 2*hypot(d_X(m), d_Y(m))
```

を用いる。unitaryのtelescoping `||C^n-U^n||₂≤n||C-U||₂`、`||A||₂≤||A||F`、期待値差≤2倍のunitary距離から得る。X/Yは別readoutなので両軸を含む複素信号差へ変換する。これは局所max-entryを全roundへそのまま転用するものではない。cap2はunitary距離の普遍的上界で、巨大shotsのclipではない。

決定論ではattenuation=1、RTE補正誤差=0を保存条件として確認し、同じraw複素信号単位で `e_m = e_PF_saved(m)+e_syn_pilot(m)` とする。DF誤差（Ha）はここへ加えない。最後に既存energy評価で `DF_saved + β_total/(δ*2^12)` を評価する。own-DF ground oracleと保存sector normに対するWeyl関係を使い、DFを一回だけ計上する。

**代入する残差はbinary64数値であり、CAR／expm／Operator再構成／norm計算の丸め誤差を厳密に囲んでいない。** そのため上記の解析的不等式と、代入値がrigorous enclosureであることを区別する。`rigorous_signal_error_bound=None`、`CA_achieved=None`を保持する。数値残差を0と観測した場合も未知のreference丸め誤差が0という認定にはならない。新しい数値allowanceの科学的採択や厳密なcertificate化は行っていない。

## 保存shotsのまま判定する

新しい最小Nを解いたり、実行結果に応じてNを増やしたりしない。保存の同一X/Y Nと元のα_axisに対して既存Hoeffding式を逆向きに用い、

```
epsilon_N = sqrt(2*log(2/alpha_axis)/N_saved)
rho_pilot = 1-e_m
sqrt(2)*epsilon_N <= rho_pilot*sin(beta_stat)
asin(e_m)+beta_stat <= beta_total
```

を固定条件として確認する。これは統計仮定を含む数値的十分条件で、保存楽観Nを無条件の十分shotsに昇格させる処理ではない。巨大整数はPython intのまま扱い、binary64で正の量がunderflowする場合はUNRESOLVEDにする。候補ごとに全13round・DF reserve・energy horizon・αの総和を確認する。失敗確率は旧契約の候補単位の配分を保ち、4候補同時・選択後の追加保証へ流用しない。

| 状態 | 意味／費用への接続 |
|---|---|
| `NUMERICAL_PILOT_CA_PASS` | 数値代用値と固定shotsで登録十分条件を満たす。保存費用を数値pilot条件付きで接続 |
| `INFEASIBLE_REGISTERED_SUFFICIENT_CONDITION` | 固定shots／phase／DF予算等の十分条件を満たさない。物理的不可能・研究全体NO-GOではなく、CA-qualified費用は出さない |
| `UNRESOLVED` | 必要値や数値解像度が不足。未知を0で補わず、CA-qualified費用は出さない |

rigorous資格は全候補で別field `UNRESOLVED_NUMERICAL_REFERENCE_ENCLOSURES`を保持する。同じsのCDF／XDF両方が数値pilot PASSした場合にだけ費用比較へ進み、s=1とs=4の2登録比較以外へ拡張しない。総費用の再集計は既存関数で保存N・保存round費用の同じ和を照合するだけで、PF/RTE・shots式・native費用の再計算はしない。最終protocol順位付け・同精度の無条件優位・実hardwareの誤差保証は対象外。

## 人工検証と環境

新規12 tests PASS。成功／不適格／未解決の4候補end-to-end、110-call完全計上、位相不一致FAIL、controlled scalar／one-body、独立CAR／保存QPY経路、13roundのnとX/Y係数、固定N不適格／underflow、費用接続gate、入力join、fresh output／失敗STOP、passive import／byte-only preflight、task内cache隔離を検証した。実backendの人工2-mode CAR＋人工NPZ/QPYでも110-call経路を実行し、PF round／ground再計算・native buildをmockで遮断した。

既存回帰は独立processでStage3 100、Stage2 native 27、一般K13＋29 subtests、計140 tests PASS。新規と合計152 tests＋29 subtests。最初の既存群一括実行は126 PASS／14 FAIL／teardown1 errorで、Qiskit実importと、clean `sys.modules`・別mock emitterを前提とする旧テストのprocess状態が衝突した。旧SOURCEやassertionを弱めず独立processで検証した。[短い失敗要約](../../artifacts/research_c_c1_stage3_deterministic_accuracy_technical/2026-10-25/combined_regression_summary.txt)を残し、巨大なsys.modules dumpを含む原本はtask内`.cache/combined-regression/`に保全した。一括ハーネスの互換性は未対応で、独立processという検証条件を明記する。

既存のtask用venvをread-onlyで使用し、Python3.11.1／NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0／Nature0.7.2／Algorithms0.3.1と既存依存lockを維持。SOURCE27・保存ファイル49・元inputのbyte preflight PASS。NPZ/QPY decodeや保存科学JSON解析を行うのは将来の`--execute`内部だけで、既定CLIはbyte preflightのみ。実H4の科学call・decode・再解析0、実回路build／compile0。

## 未承認の限定実行契約

科学callは最大**110 logical units**。複合call内部の仕事はSOURCEで固定し、元input・保存結果・局所reference・QPYの必要範囲を限定した。

| call | 最大 |
|---|---:|
| `saved_bundle_load` | 1 |
| `input_load` | 1 |
| `saved_operator_load` | 9 |
| `car_context` | 1 |
| `xdf_full_generator` | 11 |
| `local_kernel_reference` | 2 |
| `qpy_load` | 8 |
| `operator_decode` | 8 |
| `kernel_assessment` | 4 |
| `shell_assessment` | 3 |
| `round_accuracy_assessment` | 52 |
| `energy_assessment` | 4 |
| `cost_attachment` | 4 |
| `cost_comparison` | 2 |

QPY／Operator各8はs4 raw2＋head/X/Y raw/compiled6。local reference2だけを新計算し、s1のreference／Operator・s4 compiled Operator・旧PF/RTE round／DF norm／ground／aliasは再計算しない。新fit／optimizer／retry／resume／新build／compile／shot-count solvingは0。

実行順序はbyte preflight→保存join／alias来歴→必要CAR準備→共通shell gate→4候補kernel／13round→energy／固定費用→s1/s4の比較→証拠・mandatory STOP。数値correctness FAIL、schema／SOURCE／契約不一致では依存する新規callを停止し、開始済みcallだけ回収してpartial evidenceを保存する。候補の十分条件不適格／未解決は候補状態として記録し、他の登録候補の評価だけ続ける。

重い部分は保存データ再利用で削り、必要なXDF10 frameと4候補は空きCPU／RAM／quotaに応じた既存resource planで並列化する。BLASはprocess内でworkerと調整し、共有環境・他ユーザーjobへ変更を加えない。人工wall／CPU／RAM capなし、未測定speedupの主張なし。

契約SHA256 `59c920cbd08a6b7ad7abda0216ef7d0399d78ea5edbc410a58290aa327b3a957`、SOURCE manifest SHA256 `0aee1202e8bf7150cb0dd2a15805a5fdbaa2b0e2689f2581c310ad4afa284512`。元input／保存fitは旧identityを保持し、新SOURCEを別に固定した。承認はチャット上の許可で足り、証明書・承認記録・新grantは作成しない。旧結果・SOURCE・grant・ledger・marker・STOPは保全。

## 次の限定実行で得られるものと未資格事項

別途明示許可後、Codexは4候補×13roundの合成誤差数値代用値、固定Nの統計margin、PF／DFを含むCA数値pilot判定、該当する保存総RZ等とs1/s4のCDF/XDF比較を取得できる。固定shotsのまま不適格となる可能性も保ち、N変更や候補拡張で通さない。

厳密CA保証には保存DF／PF／CAR／expm／Operatorの誤差enclosureが不足しており、この限定実行だけでは取得しない。実測数値と解析的不等式を混同しない。次の担当はユーザー／研究Cで、この数値pilot範囲と110-call未承認契約を確認して実行を許可するか判断する。重要な科学レビューも別の明示許可が必要で、自動開始しない。

今回のSOURCE・人工結果・契約をcommit・push・remote照合後、**mandatory STOP**。実H4の評価へ自動進行しない。
