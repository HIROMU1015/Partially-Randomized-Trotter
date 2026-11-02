# H6高次PF：独立確認run01結果（2026-10-19報告）

**目的Aの固定2比較で、探索時の点順位が独立標本でも再現した。** 新103 trajectory・206 compiled軸・control536件が完了し、2026-10-18 23:28:31 JSTにnative exit0で終了した。所要約42分12秒、56 worker全てexit0・回収済み。追加sampling・compile・再選択を行わず、mandatory STOP。

SOURCEは `e3ac6fd8494a612d7a1aa1061fa79b3fc50d278c`。承認範囲は最小3候補・13/45/45本・固定2比較・目的Aのみ。正式採択後のplan SHA256は `9fc105e9590b15dbc86a743e037056942d2eb83242d62c79afd366e941e290b8`。前段の[準備報告](track_a_h6_higher_pf_confirmation_native_readiness_20261018.md)、draft/無効grant、探索n8の結果は保持した。

## 1. 条件と確認範囲

H6直鎖・隣接1 Å・STO-3G・保存DF rank19・prefix5・保存状態、T=.8、K4、outer当たり32 RTE short steps。signed時間の配分・generation order・scalar phase・ordinary control・explicit XXPlusYY/number Phase loweringを維持した。

精度ε=.001、各軸ε/√2、α_axis=.025、経験的allowance1e-9、log_B margin1e-12、状態準備P=0。保存bias/normalization/shotを再利用し、新signal・shot再計算・SCF/DF/state生成は0。これは固定条件のshot込みcompiled RZ費用であり、全RPEや状態準備込みの総資源評価ではない。

Python3.12.3、Qiskit1.3.0、NumPy1.26.4、SciPy1.14.1、OpenFermion1.6.1。compilerはrz/sx/x/cx・optimization_level1・seed17・backend/couplingなし。profile hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。provider SOURCE `c31ef424175c02379eafebe8fc6c7548c1d9aabe`。

H4/new_2や別のA_COST系列を含めず、旧探索n8を今回の平均・分散へpoolしない。目的B（希少eventを含む母平均費用順位の保証）は保留である。

## 2. 独立確認の費用

費用は各trajectoryについて `N_real*RZ_cosine + N_imag*RZ_sine` を整数で計算してから平均した。SEは標本平均のMCSE。以下の軸別平均はcosine/sineで同値だが、会計では各軸の保存shotを独立に重み付けしている。

| 候補 | 新n | cosine/sine平均RZ単価／軸 | real shot | imag shot | shot込みRZ平均 | 標本SE |
|---|---:|---:|---:|---:|---:|---:|
| S2/q8 | 13 | 279,417.076923 | 22,907,931 | 27,494,282 | 1.40832390269143e13 | 9.71945599188138e10 |
| Yoshida4/q2 | 45 | 163,040.733333 | 35,724,491 | 47,416,584 | 1.35553818381217e13 | 4.18174546970716e10 |
| Yoshida4/q4 | 45 | 324,299.155556 | 21,115,055 | 21,415,717 | 1.37926934447259e13 | 3.36637259984281e10 |

今回の点平均でYoshida4/q2はS2/q8より**3.7481%低費用**（分母S2/q8）。Yoshida4/q4はq2より**1.7507%高費用**（分母q2）。これは今回の独立標本の結果であり、母平均・全q格子の最適性を認定するものではない。

### 保存signalとの対応

| 候補 | 保存signed bias real | 保存signed bias imag | log_B | 保存normalization-only shot倍率 |
|---|---:|---:|---:|---:|
| S2/q8 | 8.72723745941606e-5 | 1.413269150842833e-4 | .0021078034526976377 | 1.004224505075521 |
| Yoshida4/q2 | −1.2385728605590618e-4 | −2.008482277895407e-4 | .16344702584056256 | 1.3866545556783623 |
| Yoshida4/q4 | −7.930071789896687e-6 | −1.2855382794385584e-5 | .08181069434428587 | 1.177768314261549 |

上表は保存値の転記。今回新たにsignalやnormalizationを評価していない。PF/finite-RTE成分、scalar/λ、preparation hash、保存shot receiptは `signal_and_shot_binding.json` に結合した。経験的allowanceによる適格性を厳密bias保証と呼ばない。

## 3. 事前固定した2比較

正の差は探索時と同じ点順位を表す。主判定は整数費用から得た有理数平均差の符号であり、両比較とも `REPRODUCED`。新たなq選択はしていない。

| 左 − 右 | 確認平均差 | 差の標本SE | family2 Bonferroni正規近似区間 | 点判定 |
|---|---:|---:|---|---|
| S2/q8 − Yoshida4/q2 | 5.27857188792641e11 | 1.05808704722974e11 | [2.90697269422221e11, 7.65017108163062e11] | REPRODUCED |
| Yoshida4/q4 − Yoshida4/q2 | 2.373116066042e11 | 5.36837588609337e10 | [1.16984683065217e11, 3.57638530143183e11] | REPRODUCED |

補助区間は、候補間独立を仮定し `sqrt(s_left²/n_left+s_right²/n_right)`、α_MC=.05、family size2、両側critical2.2414027276049464で算出する固定仕様。候補間CRN共分散を仮定しない。両区間とも0を含まないが、正規近似・plug-in分散に基づく診断であり、厳密なcoverageや希少tail込みの母平均順位を保証しない。名目power90%も設計時の近似値で、実際の検出力の証明ではない。

## 4. 希少eventと解釈の上限

| 候補 | s0 | s2_single | s2_multi | s4 |
|---|---:|---:|---:|---:|
| S2/q8 | 13 | 0 | 0 | 0 |
| Yoshida4/q2 | 40 | 5 | 0 | 0 |
| Yoshida4/q4 | 44 | 1 | 0 | 0 |

全候補でs4・複数2次eventは未観測。未知層の費用・母分散上界はnullで、0や恣意的な値を補完していない。今回の目的Aを完了しても目的Bは未確立である。他系列の条件付きpilotを今回の無条件標本へ混合していない。

## 5. 技術的完了証拠

- 103 raw結果・103 confirmation envelope・206軸receipt・206非払い戻し予約・control536行を照合した。
- full scalar/λ・preparation/生成順・normalization、各候補初回のlow/high/leakage・occupation oracle・Hadamard両軸・compiled statevectorを検査。最大誤差2.8365450926553323e-10で許容差1e-9未満。
- SOURCE・入力・provider/runtime・seedのclosure306 fileが完了後もSHA一致。SOURCE専用checkoutはcleanで不変。
- fresh seed914個、既知除外3,329個との衝突0。cosine/sineは同じevent/seedでpair。異なるseedをPRNG独立性の数学的証明とは扱わない。
- one-shot grantを1件発行・1件消費し、返金・再利用・retry・追加compile・n追加・refitは0。元draft/無効grantや既存の結果・grant・ledger・STOPは変更していない。
- 起動時の空きCPU約60.94 core相当・affinity64からpool上限60を決め、依存gateに従って実際に56 workerを使用。最大pending56、全56 worker exit0/回収。内側transpiler/BLAS poolを抑え、trajectory単位で並列化・preparation cacheを再利用した。
- launcher pool wall2519.791秒、native監督wall2531.624秒（約42分12秒）。wait4 user90168.820秒・system258.153秒。wait4最大RSS1,683,714,048 byteは単一processの最大で、pool合計ではない。
- 科学SOURCEは準備時の人工回帰92件PASSの版を変更せず使用。今回の終了監査も作者自身の確認であり、独立レビューではない。報告作成時の最初の読取監査は汎用Pythonにpackage metadataがなく拒否されたため、固定Pythonで監査を行った。科学runの停止・再計算はしていない。

## 6. 公開成果物とSTOP

`artifacts/resource_applicability/track_a_h6_higher_pf_confirmation_run01/2026-10-19/` に、保存result、正式approved plan、206軸CSV、103 trajectory receipt台帳、536 control台帳、保存signal/shot binding、native完了・seed/入力監査、scope/保全、summary/inventoryを収録した。

内部パス・生event・seed値一覧・有効grant・private config/ledger・生ログは公開しない。実行中の封印済みSOURCEや過去のSTOPを書き換えず、結果公開後も新Phaseへ移行しない。

- **ユーザー**：目的Aの結果の採用・掲載範囲を判断する。希少event込みの母平均保証を追加で求めるかは別の研究判断とする。
- **Codex**：mandatory STOPを維持する。目的B、H4/new_2、別A_COSTへの追加実行は今回の完了から自動起動しない。
