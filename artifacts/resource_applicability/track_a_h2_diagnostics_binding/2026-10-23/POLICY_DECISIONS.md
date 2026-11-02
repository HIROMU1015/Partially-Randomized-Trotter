# 共通native・allowance・referenceの採択判断資料

これは採択案であり、正式採択・科学PASS・費用順位の判断ではない。対象はlinear H2/隣接1 Å/STO-3G/tol-only DF1e-8/rank3、T=.8、ε_sig=.0005、α_axis=.025、P=0、system4/ancilla4、指定4候補に限る。Hamiltonian/state/snapshotは[契約](DIAGNOSTICS_contract.json)のbindingを固定する。

## A1 共通native

推奨は `H2_COMMON_ADJACENT_QR_SIGNED_PIECES_V1_PROPOSED`。取得SOURCE `4ec26921c6872ed6e8667f167074e1e405cf1a3f` の `h46_new2_native.prepare/gaussian_operations` と `h46_m5_native.compose` を使い、provider `c31ef424175c02379eafebe8fc6c7548c1d9aabe` を別役割で結合する。[全SOURCE索引](source_index.json)にGit blob/SHA256を保存した。

| 共通にする実装条件 | 方法固有で保持する条件 |
|---|---|
| adjacent complex QR、rotationを丸めずgate/hex angle/wiresを記録 | Full prefix3、PR prefix1。prefix間でprep hashの一致を要求しない |
| signed piece順序、方向、operator orderを保存 | S2/q4とYoshida4/q2。Y4の負pieceを絶対値化しない |
| symmetric-directional controlled evolution、cos/sin full wrapper | Fullは全DF項をdeterministicに作用、PRはK4有限RTEの期待作用 |
| scalarを最後に一回 | Fullは元constant、PRは元constant＋抽出tail identity。二重適用しない |
| Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routingなし | PRのABS_TIME_PER_OUTER=32と固定q/r/K、unconditioned cohortを保持 |
| 同じH2 phase guard。phase alignment/state fit/postcompile補正なし | PR rawはattenuated、correctedはBを戻した信号。FullのBは1 |

Gaussian QR内部にはunitarity/reconstruction strict 1e-12検査があり、既存SOURCEのまま使用する。prepare receiptはrotation byte hashとoperation streamを新しく記録するが、**実preparationは今回生成していない**。旧Fullのdense/native費用や旧prep hashを新系列へ自動流用しない。保存PRのexplicit QRラベル・prep hashだけでも新系列との完全identityにはならない。

DIAGNOSTICSではoccupation上のprimary/independent path差、closure/log_B差のstrict `<1e-9` を検査する。native low/high/control/wrapper/compiled correctnessを実行する分岐では、native/occupation/wrapper `<1e-9`、compiled saved-state vector差 `<1e-7` と既存guard receiptを要求する。**DIAGNOSTICSはcompile/simulation予算0のため、後者の実証を完了しない**。全state operator証明でもない。

採択可能条件は、この共通実装条件・方法固有差・保存state限定のcorrectness scopeをユーザーが承認すること。4候補で診断が通るか、QRが実runtimeで正常に準備できるかは別途承認後の検証事項。

## A2 共通の経験的適応allowance

推奨は `H2_COMMON_ADAPTIVE_A_RULE_V1_PROPOSED`。同じ入力意味と同じ演算式を4候補に適用する。旧Fullのadaptive allowanceと、旧PRの保存u=1e-9/margin=1e-12は政策・instrumentationが異なる。保存uを新系列へ移植しない。

新提案は既存 `h6_matched_accounting_v1.empirical_allowance` と同じ式・演算順で、roundoff_multiplier=64、safety_factor=10、floor=1e-12。以下は**式の記述だけ**で、保存値での新u計算ではない。

```
rounding = 64 * binary64_epsilon * (1 + W) * max(1, M)
agreement = e_reference + max(path_signal_errors, path_vector_errors) + e_closure
u_real = u_imag = 10 * max(1e-12, rounding, agreement)
log_B_margin = 10 * max(1e-12, rounding, e_log_B)
```

| 入力 | 意味・単位・計数 |
|---|---|
| e_reference | 旧FPのcombined empirical diagnostic。corrected signal単位。厳密boundではない |
| path signal差 | 同じ列のprimary対independent複素信号差。raw列は各pathのexp(log_B)でcorrectedへ変換 |
| path vector差 | 同じbasisのstate vector norm差。raw列をcorrectedへ変換。phase fitなし |
| e_closure | PRの `abs(exp(log_B)*raw_signal-corrected_signal)`。旧raw-unit closureで代用しない |
| e_log_B | 2 pathの自然対数normalization差。信号差とは異なる単位 |
| W | primary＋independentのwork。deterministicはvectorごとのpropagator action、tailは実行した行列vector action。FLOP数・compile数ではない |
| M | 両pathの全required stageの最大norm。Taylor term/sum、spectral basis内、rawのcorrected換算を含む。endpointだけでは代用しない |
| log_B_margin | 後続shot式へ渡す自然対数normalizationの経験的余裕。uそのものをlog_Bに足さない |

新workは2列を別vectorとして数え、primaryのtail spectral基底出入とindependentのdegree0..5 Taylor処理を区別する。旧Fullのoracle/action counts・stage_count、旧PRのend norm/proxyと数値を同一視しない。保存PRには個別path-vector差・work countが欠測だったため、新instrumentationが必要となった。

欠測・NaN/Inf・負値・bool代用・単位/coverage不一致・他reference・他native/inputのreceiptはSTOP。実装内部のmaxのdefaultを欠測補完に使わず、adapterの事前検査でpath/trace/workを必須にする。Fullのtail work0やB=1によるclosure0は数学的な構造上の値であり、欠測referenceを0と置くこととは異なる。

このuは経験的数値診断であり、厳密な誤差上界・厳密な信頼区間・全入力誤差保証ではない。2 pathは同じ保存入力、signed schedule、setupで構成した行列を共有する。特にindependent行列はoccupation spectraから再構成されるため、共通の入力/表現誤差を独立に検出する証明ではない。

後続の会計では通常signed biasの軸別絶対値を使い、有限で正のheadroom、解決可能なshot整数、`u/headroom <= .01` を必要条件とする。このguardを満たしてもuのrigorous certificationやcohort母平均保証にはならない。DIAGNOSTICSではu/headroom、N、G、順位を評価しない。

採択可能条件は、上記instrumentationの単位・経験的scope・eligibility guardの明示採択。係数64/10/floorを計算結果に合わせて調整しない。未解決の不確実性は、診断の共有誤差、経験的proxyの妥当性、厳密reference誤差の未取得。

## A3 旧referenceの条件付き再利用

推奨は `REUSE_OLD_FP_SCALAR_AND_COMBINED_EMPIRICAL_DIAGNOSTIC`。cc7404edの保存excerpt全byteを照合し、旧FPの複素scalarと保存errorを結合する。旧referenceは同じ保存tol-only H2 Hamiltonian、state、T=.8、P=0で、reference_rotated=false。scalarはexact mathematical full-H解の証明ではなくbinary64による保存評価。

保存 `error=2.0973428995247362e-16` はexpm/eighのstate差を含むcombined empirical diagnosticとして扱う。信号差とoccupation discrepancyも保存されているが、厳密なreference誤差boundは欠測。このerrorを0に置かず、新runtimeで再計算したreference精度とも呼ばない。[入力保全](preservation_observation.json)はFP resultを当時のSHAと照合する。

条件付き使用は「同じH/state/snapshot/T/scalar conventionへの保存浮動小数referenceを基準とする新しい経験的診断」に限定する。4回のreference_rebindは各新corrected signalから旧scalarを引く処理で、新full-H reference生成ではない。

新referenceが必要となる条件は、binding不一致、保存referenceを受け入れない科学判断、厳密uncertainty要求、または診断結果から旧referenceの疑義が生じた場合。この場合はDIAGNOSTICSを拡張せず、solver/precision、行列・basis・scalar、生成件数、独立検証件数、厳密/経験的誤差意味を明示した別SOURCE・別契約・新grant・新run_rootを準備する。件数未確定のlive reserveは作らない。

今回のfresh runtimeは既知のPython3.11候補を新しく封印したもの。保存PRのPython3.12 runtimeと同一とは認定しない。このruntime選択と条件付きreference解釈も、ユーザーの次段階承認事項に含む。
