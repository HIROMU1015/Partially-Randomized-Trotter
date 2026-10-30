# CDF7 native32：保存119-call結果からfull512校正への接続

2026-10-30 JST。保存直鎖H4・1.00 Å・STO-3G、通常number CDF7、Nα=Nβ=2／sector36、元順LD2／tail F2..6、partial ordered S2、δ=h=.1／s1／r128／K0を対象とする。Own-DF ground-state oracleを仮定し、状態準備費用を除外する。

ユーザーの「進んでよい、私の承認はいらないので進められるところまで」に基づき、**native32の技術準備・条件固定・一回の限定科学実行・結果公開を一括して進める**。再承認手続きや承認証明書・新grantは設けない。以下は実行前のSOURCE／条件の固定であり、実行結果は別の報告へ保存する。

## 再利用・最小差分

[前段119-call結果](research_c_cdf7_PR_execution_20261030.md)は13/13 round潜在適格、8因子／ground-alias PASS、ideal仮定shots21,768、RZ kernel損益分岐は通常CDF8γ0に17426.372、M4に6506.293で、nativeの別提案対象となった。ただし実CA・全law誤差・価格・総資源優位は未資格だった。これらの値を再計算してnativeへ進むのではなく、保存済み結果を入力にする。

新しい[c_cdf7_native32.py](../../src/trottertracks/c_cdf7_native32.py)と[専用CLI](../../scripts/run_research_c_cdf7_native32.py)はP4／第三γ0のnative orchestrationから必要な接続を派生した。旧moduleを変更・再起動しない。旧9因子／rank8固定の`sector_match`は不適合のため、同じ比較式で8因子を検査する薄い専用接続を作った。CAR、sampler、Gaussian、compiler、Operator decode、adjoint、X/Y、metrics、calibrationを再実装していない。

- 保存J/U、元h/g/scalar、8 sector因子・H・mapping、保存own-groundを読む。DF norm、PF／RTE、eigensolve、RPE／shotsは再計算しない。
- B1の一般K／spin対応κ／lawを再接続し、保存component ID・符号・確率・mass・identityを検査する。
- 既存raw CARでfull256の8因子を作り、保存sectorとの対応と和を確認。独立full-CAR law gateを通す。
- 既存StrongBuilder／precision／P0で8 Gaussianと4 head-relative transitionを再利用。Tailの共有は隣接する厳密basis-key一致に限定し、head-tail跨ぎの融合はしない。
- 32本についてraw/compiledのfull512 controlled operator、actual adjoint、complex X/Y、norm/unitarity、metricsを既存gateで検査する。有限probe検証で代用しない。

## 固定SOURCE・入力・契約

| 項目 | 固定値 |
|---|---|
| Branch／起点 | `codex/research-c-cdf7-native32-20261030`／`ad6f6913638c8cb43c9442543663408fc1d03c76` |
| Scope | `C_H4_CDF7_NATIVE32_CONDITIONAL_COST_V1` |
| [科学契約](research_c_cdf7_native32_contract_20261030.json) SHA256 | `8922f3ef93f5a226d0ea6527e147c4eedeb4c3fe112549d0715acb872bb6a8ee` |
| [SOURCE107 manifest](../../artifacts/research_c_cdf7_native32/2026-10-30/technical/SOURCE.json) SHA256 | `e90e04f4ebc9a957cd6089eb224fd70e01a34e851e83d61fe7a1c58ada5d7a13` |
| 保存fit SOURCE／export SHA256 | `c13d6f82a22d497b3334eaaa40d8430a5010f70f`／`4e4939d1380386ebb98c3b39a13954cbfc3148e35fb8e2cfd3cf536bd0b57ff4` |
| 保存PR SOURCE／契約 SHA256 | `3e75b16901f0f4a9966d2bd5b9ff6bde8e75f472`／`55032e96e745095611b52aebe300056f0ce12231742dce2f6007ee5d0e427c81` |
| 元P-H4 input SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| 固定input／旧保護 | 33ファイル／2292件（既存mapを参照して重複した巨大mapは作らない） |
| Fresh output | `artifacts/research_c_cdf7_native32/2026-10-30/run01` |

前段SOURCE102をそのまま保全し、新module／CLI／test、既存shared-lock回帰test、環境lockを加えた107件を固定する。旧SOURCE・結果・grant・ledger・marker・STOPと非実行draftは変更しない。今回、旧119-call runnerを再実行しない。

## Callの登録：本体323＋独立entry9＝332

旧323-call案はnative本体の内訳で、独立entryの入力読込・再接続を含めていなかった。今回の契約はこれらを隠さず別計上し、**全体上限332 logical units**とした。32本・128 event・P0等の科学条件は変えていない。

| 段階 | 内訳 | Units |
|---|---|---:|
| 保存入力接続 | evidence、input、fit、saved reference、ground、構造、model、law、metadata gate各1 | 9 |
| Full CAR | context1、raw8、full-sector1、independent reference1、full-law1 | 12 |
| Gaussian | basis8＋head-relative4 | 12 |
| Shell | 保存raw/compiled6読込＋3検証 | 9 |
| 32 trajectory | sample／build／reference／raw gate／compile／compiled gate／adjoint pair／X/Y／metrics各32 | 288 |
| 集計 | calibration1＋conditional native result1 | 2 |
| **合計** | **新fit0、optimizer0、PF/RTE/shots再計算0** | **332** |

保存QPYはraw／compiled／それぞれのactual adjointの計128件。32 compileのみを新実行し、6 shellは固定QPYを再利用する。各coherent occurrenceはfresh independent trajectoryを使うという費用モデルを維持し、一つのsampleを全roundへ反復使用した量子実行と解釈しない。

Samplingは既存PCG64有限bit実装。Master2026103007、cohort `CDF7_s1_r128_native32_v1`、axis `CDF7_NATIVE_KERNEL`、SOURCE digest／JU hashをkeyへ含め、sample0..31を事前固定した。旧cohortと非paired、poolなし、価格を見た選別・停止・追加sampleなし。Ideal IID lawとの距離は未資格のままにする。

## 数値条件・費用の意味・STOP

P0はQiskit1.3.0、opt1、seed17、basis `rz,sx,x,cx`、approximation degree1、qubits_initially_zero=false。Gaussian `nature_jw_exact_zero_lapack_atan2_v1`、phase `qiskit_P0_mpfr160_declared_phase_v1`を比較先と同じ既存実装で使う。

B2のallowanceを無変更で使用：operator max-entry／unitarity絶対1e-10、complex signal2e-10、normalization1e-12、mass相対1e-12／絶対1e-14、rtol0、global-phase fittingなし。Full512数値pilotは厳密rounding enclosureや全law合成誤差保証ではない。

32本のsample mean／variance／descriptive SE／観測min-maxを保存する。観測極値をsupport boundにしない。保存済みの2仮定laneのW/Sへ`C_metric=W*mean_kernel_metric+S_metric`を接続する。新cohortのSEは`W*SE(mean)`とし、13 roundの独立した再校正として縮めない。比較先は固定された保存finite32点費用で、joint confidence／母平均の厳密優位を認定しない。

Evidence laneの合成／law誤差とactual CAは未資格のまま。条件付き点費用が上下しても、実CA達成、状態準備込み総資源優位、新DF一般化や因果効果へ昇格させない。

最初のSOURCE/input／表現／law／phase／compile等のcorrectness FAILで依存処理を停止し、開始済みtaskを回収してpartial evidenceを保存する。Retry/resume0、fallback・新cohort・追加fit・条件の事後変更なし。正常終了／FAILとも証拠を公開しmandatory STOPする。

## 人工検証・実行効率

人工testは保存H4を使わない。**新9件＋既存17件＝26件の独立testがPASS**。保存119-call相当→332-call mock end-to-end／QPY128、rank7実人工128-eventのfull512 raw/compiled・adjoint・X/Y、phase改変拒否、旧rank8 gateの保全、SOURCE/sector/law不一致STOP、fresh再利用拒否、seed再現性／32分離、共有SEを確認した。[新test](../../tests/management/test_research_c_cdf7_native32.py)／[最終log](../../artifacts/research_c_cdf7_native32/2026-10-30/technical/new_tests_final.log)：9 PASS／182.02秒。[回帰log](../../artifacts/research_c_cdf7_native32/2026-10-30/technical/regression_tests.log)：25 PASS／11.60秒（新test8件を含むため、独立した旧回帰は17件、重い新full512 testは新suiteで確認）。Passive preflightもPASS／science_calls0で、実分子の性能や厳密保証を示す人工結果ではない。

初回の人工suiteではfixtureが既存の人工JSONを排他的なsaveで上書きしようとして失敗した。人工fixtureの更新方法だけを直し、既存科学SOURCE／rawを変更せず最終suiteを再確認した。[開発時log](../../artifacts/research_c_cdf7_native32/2026-10-30/technical/development_initial_tests.log)。人工検証の再実行を科学retryとして扱わない。

Task専用venv/cache、read-only依存再利用。既存resource plannerで32独立taskのCPU/RAMを選び、BLASを調整する。CAR／reference／Operator decode／matrix gate／X/Yは並列化し、Qiskit shared equivalence libraryに対するbuild／P0／adjoint buildの既存mutexを維持する。共有環境・他jobを変更せず、人工CPU/RAM/worker/wall上限は設けない。同規模の過去wallは約920／1229秒であり、今回の実測時間やspeedupを予告するものではない。

この技術SOURCE・契約をcommit/pushしremote SHA／SOURCE／inputを照合した後、今回の一括許可の範囲でfresh run01を一度だけ実行する。結果・call・資源・QPY・STOPを公開し、別の分子／rank／探索／保証実験へは自動拡張しない。
