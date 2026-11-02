# 通常CDF7：保存fitからPR reference・RPE・損益分岐への技術接続

2026-10-30 JST。**rank7専用の119-call経路を実装し、新規11＋既存回帰28＝39件の人工テストPASS、実入力をdecodeしないpassive preflight PASS。今回の科学callは0。** 保存直鎖H4・1.00 Å・STO-3G、Nα=Nβ=2／sector36、通常number CDF7の後続評価を準備した。実H4の新規PF/RTE・ground・shots・価格評価は開始していない。

ユーザーの「その方針でよいので続きを行ってよい」に基づき、前回提案したrank7 adapter・人工回帰・SOURCE固定を完了した。科学実行は準備結果に対する別途の明示許可後とし、承認証明書・新grant等は設けていない。

## 固定するSOURCEと入力

| 項目 | 値 |
|---|---|
| Branch | `codex/research-c-cdf7-pr-preparation-20261030` |
| 起点commit | `794875af24cb954985ac5e43fc799421f5d8736a` |
| 新module | [c_cdf7_pr.py](../../src/trottertracks/c_cdf7_pr.py) |
| 専用CLI | [run_research_c_cdf7_PR.py](../../scripts/run_research_c_cdf7_PR.py) |
| Scope | `C_H4_CDF7_PR_REFERENCE_RPE_PRICE_V1` |
| [新しい未承認・実装済み契約](research_c_cdf7_PR_contract_ready_unapproved_20261030.json) SHA256 | `55032e96e745095611b52aebe300056f0ce12231742dce2f6007ee5d0e427c81` |
| [評価SOURCE102 manifest](../../artifacts/research_c_cdf7_PR/2026-10-30/technical/SOURCE.json) SHA256 | `e79781e8d63acb60f993aeb1d13ee8d9132d38d24f8905f4c8ac3ca086c85d16` |
| 保存fitの科学SOURCE | `c13d6f82a22d497b3334eaaa40d8430a5010f70f` |
| [保存final.npz](../../artifacts/research_c_cdf7_onefit/2026-10-30/run01/fit/final.npz) SHA256 | `4e4939d1380386ebb98c3b39a13954cbfc3148e35fb8e2cfd3cf536bd0b57ff4` |
| 元P-H4 input SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| 入力h/g/scalar snapshot SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| 将来のfresh output | `artifacts/research_c_cdf7_PR/2026-10-30/run01`（今回未作成） |

保存fitの来歴と、新評価SOURCEのidentityを分けて固定した。新manifestはlocal import closure（再利用module中の未使用の分岐も含む）＋CLI・人工テスト・依存lockの102件。含まれている旧fit/native関数の実行を許可する意味ではない。保存input/export・fit/sector/optimizer・比較先・shell・physical-groundの20ファイルはbytes/hashで固定し、旧J/Uを今回decodeしていない。

[旧119-call設計案](research_c_cdf7_PR_contract_unapproved_20261030.json)は`executable=false/implementation_ready=false`のまま保全した。新しいready版の`executable=true`は技術的にentryが完成したという意味で、科学実行許可を表さない。Shared-basisの非実行契約、旧1-fit SOURCE・raw・STOPも変更していない。

## 最小接続と再利用

新規の科学式、fitter、Gaussian合成器は作っていない。旧rank8 runnerとvalidatorを変更せず、新moduleから適合する下位関数を呼ぶ。

| 部分 | 再利用とrank7での接続 |
|---|---|
| Input・一般K | CDF7の構造validator、B1の可変rank `prepare_arm`。float64 J/U `(7,4,4)`、一般対称J／実直交U、元g補正κ・核scalar一回・frame元順0..6 |
| 表現・ground | 既存SectorCAR、一般Kの独立factor経路、eigenpair／alias。base＋7 frameの**8因子**をsector36で検証する |
| PR law | 元順LD2、tail F2..6、δ=h=1/10、s1/r128/K0。符号・identity・massを既存gateで確認する。Source hashはCDF7固有で、旧候補のlawやPFを移送しない |
| Component記録 | 修正済み`with_components`を再利用。後続consumerが必要とするprobability/sign/basis/fragment情報も出力する |
| PF/RTE | B1の部分ordered symmetric S2／corrected finite-RTE kernelと13 roundの独立sector reference、既存十分条件classifier |
| RPE | tail-P4の`stage_accuracy`をそのまま使用。CA／位相／failure配分、attenuationと統計誤差の対応、unknown処理を維持する |
| 損益分岐 | M4の`C_metric=W*μ_metric+S_metric`と既存threshold。保存head/X/Y shellを実行ごとに一回加え、新CDF7 kernel価格μは未知のままにする |

B1での内部role名`baseline`はデータ構築用であり、CDF8の結果をCDF7へ転用する指定ではない。7 frame・source/input hash・候補のown ground/law/referenceで識別する。旧M2のrank8 validatorや9因子ループへの緩和はしていない。

## 次の限定科学実行の条件とcall

対象は上記の保存H4／CDF7一候補のみ。CA=.00159 Ha、DF reserve=CA/8、δ=.1、m=0..12、own-DF ground-state oracle／状態準備費用除外、ideal finite-RTE IID参照モデルを維持する。保存1-fitで報告されたDF誤差＋guard `3.022641747295844e-7 Ha`を再利用し、物理DF誤差を再計算しない。この引用は[既存結果](research_c_cdf7_onefit_execution_20261030.md)に基づき、今回の新測定ではない。

| 段階 | 内訳 | 最大units |
|---|---|---:|
| Reference | evidence/input/export/構造、model、sector、8因子、和、own-ground/alias、law/tail、target/kernel、13 exact＋13 round＋13判定、summary | 62 |
| RPE | 資格3、13 round×3 lane=39、energy3、shots summary3 | 48 |
| Price | affine3、2仮定×2比較4、将来nativeの提案可否1 | 8 |
| 最終summary | 結果とmandatory STOP | 1 |
| **合計** | **新fit0、optimizer0、sampling0、native build/compile0** | **119** |

登録PF/RTEの十分条件が不十分なら、全13 roundの限定的な不適格理由を保存し、RPE/priceへ進まず**62＋summary1=63 units**で終了する。これは物理的不可能性ではない。SOURCE/input、表現、ground/alias等のcorrectness FAILは依存処理を停止する。並列開始済みtaskを安全に回収し、partial evidenceを残す。Retry/resumeは0、別rank／s／r／K／分子への拡張はない。

比較先は保存通常CDF8 γ0とPR-aware CDF8 γ.1 M4。両方とも同じH4・sector36・LD2・s1/r128/K0・13 round・CA/failure条件であることを来歴・契約で検査する。保存されたfinite32 cohortの条件付き費用summaryを読み、新CDF7の平均価格をその値で埋めない。

数値条件は旧`ALLOWANCES`をそのまま固定した。例えばcomponent絶対1e-10、normalization1e-12、mass相対1e-12／絶対1e-14、signal絶対2e-10、ground residual1e-10、alias窓[-10,10] Ha、rtol0、global-phase fittingなし。厳密enclosureはnull。結果を見た閾値変更はしない。

## 資格・未知値・nativeへの境界

3 laneを区別する。

- `evidence`：全law合成誤差とimplemented-law誤差は未資格。Shots／W／価格の資格が得られなければnullを維持し、未知誤差を0に置き換えない。
- `ideal_exact_native`：native誤差0・ideal lawを**仮定**した条件付き評価。実測やCA達成の証明ではない。
- `assumed_raw_signal_budget`：既存の合成・law各`.001*A_m`仮定を維持する。取得済み上界とは呼ばない。

PF/RTEのbinary64 operator評価、saved DF engineering guard、統計Hoeffding条件、仮定した合成／law条件を分けて報告する。点推定や仮定をstrict enclosureへ昇格させない。実際のCDF7 kernel価格・cost support・実CA達成・総資源優位はnullのまま。

将来native提案の機械的目安は、all13 potentially feasible／ground-alias PASS／ideal仮定のenergy適格／正の整数shotsとW／**二つの比較先の両方に対し有限で厳密に正のideal-lane RZ損益分岐**とした。一方だけなら登録条件では提案対象にしない。この条件は未実行の段階で固定したもので、今後結果に合わせて変更しない。`ELIGIBLE_FOR_SEPARATE_NATIVE_PROPOSAL`になっても、本entryはnativeを呼ばない。

将来323-call native案は今回の119-callから除外した。新rank7 cohort／seed、専用SOURCE・実装検証・契約・別の明示許可が必要であり、今回これらを採択していない。Frame削減から価格改善を予告したり、未観測のbasis共有・相対Gaussian融合を仮定したりしない。

## 人工・回帰結果と実行効率

| 検証 | 結果・範囲 |
|---|---|
| [新人工テスト](../../tests/management/test_research_c_cdf7_PR.py)／[実行ログ](../../artifacts/research_c_cdf7_PR/2026-10-30/technical/new_tests.log) | **11 PASS／4.09秒**。Invented 7-frame一般J/U（det=-1も含む）と人工h/g/scalarで既存CAR/PF/RTE/RPEを使い119-unit end-to-end |
| 既存回帰／[結果](../../artifacts/research_c_cdf7_PR/2026-10-30/technical/regression_results.json) | **28 PASS／2.22秒**。Stage C精度・unknown・law/support・統計、旧rank8 reference/RPE、legacy gate、threshold符号 |
| 分岐・STOP | 119-unit正常、63-unit十分条件不適格、correctness FAILでground以降停止、全lane未資格でnull／no-nativeを確認 |
| Provenance・preflight | SOURCE/input/fit rank/予算/comparator条件の不一致、fresh再利用拒否。Opaque NPZによるbytes-only preflight、passive importでnumpy/scipy/qiskit/ffsim/jax未loadも確認 |
| Component consumer | 実B2 law consumerへ人工sector36 atomとcomponent記録を渡して整合、probability改変を拒否。Full-Fock/native検証の代用とはしていない |
| 禁止分岐 | native build／sample関数をraiseで遮断したままend-to-end PASS。QPY／native outputなし |
| 固定inputのpassive確認 | **PASS、science_calls0／maximum_calls119**。実NPZはhashのみ、fresh run01は未作成 |

これらはローカルの人工・回帰テストであり、CI／外部独立再現や実H4の科学性能の証拠ではない。人工テスト内の119-unit dispatchは科学実行実績に数えない。

Task専用`.venv`とcacheを用い、既存依存をread-onlyで再利用した。[Environment lock](../../artifacts/research_c_cdf7_PR/2026-10-30/technical/environment.lock.json)はPython3.11.1／NumPy1.26.4／SciPy1.14.1等を記録する。新install、共有venv・設定・他job変更はない。将来の13 sector36 roundは既存resource plannerで空きCPU/RAMから並列度を選び、BLAS oversubscriptionを調整する。CPU/RAM/worker/wallの人工上限や無計測のspeedup主張はない。

[Technical verification](../../artifacts/research_c_cdf7_PR/2026-10-30/technical/verification.json)にSOURCE102／保存input20、既存保護2214件、テストとfresh不在を記録する。原本の保護mapを再利用し、承認証明システムは追加しない。

## 完了・次の担当

本119-call経路について、人工検証で判明した未修正の技術的blockerはない。実H4で登録十分条件を満たすか、仮定下のshotsや損益分岐が有利かは未評価であり、科学実行後に初めて判断できる。合成／lawの厳密保証と独自native価格も未資格のまま。

Commit/push・remote SHA／SOURCE／契約／成果物照合後、**mandatory STOP**する。次の担当は**ユーザー／研究C**が本ready契約と公開SOURCEを確認し、最大119-callの保存CDF7科学実行を明示許可するか判断すること。許可された場合のみ**Codex**がfresh run01で一度実行・証拠公開する。今回の作業から実科学計算、native、追加fit、詳細科学レビューへ自動進行しない。
