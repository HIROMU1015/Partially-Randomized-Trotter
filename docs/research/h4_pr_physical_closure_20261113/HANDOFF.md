# H4物理閉状態診断：B/C引渡し

結果正本は[REPORT](REPORT.md)、[run02 local metadata](run02/RESULT.json)、[REDUCTION](REDUCTION.json)、[STATE_PROOF](STATE_PROOF.md)。branch `codex/r-variable-compiled-price-design-20261111`を継続。BやAの作業領域は変更していない。

## 結果をBが扱う単位

- B正式接続`75b5c89bddf8a1ffa4a8ac99b2c06d6e05fa012d`のV2 configuration `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`と、前回C requestのm0/m6 physical recipe・identity・law_bindingが完全一致した。
- H4 DF10/S2/L_D1/δ.1/K2、physical IDs `53e00d8b07d1547d19a7`（q1/R2）と`2a8c6718220eec59d419`（q64/R128）、元exactκ/M12/P0に限定する。
- 今回の出力は**価格状態の診断metadata**。`FULL_WRAPPER_MEAN_COMPILED_PRICE` recordとして取り込まない。既存C/B price provider・pipelineの修正はない。
- 価格・cohort mean・平均S/Cはnull。C1 MODEL false、law距離/rare/長q-R systematic/共分散/total CIは未資格のまま。Bは全13roundの価格欠測を維持する。

## SOURCE・初回失敗・実行範囲

| 段階 | SOURCE | 結果 |
|---|---|---|
| 初回 | `c5e2e5046f674277778b224329301c04a52eec8f` | descriptor APIへNone seedを渡して失敗。run01/STOPとEXECUTION.txtを保全 |
| 同scope内修正 | `eeccdf18ee839ccddb824afbc5c8d0aa8b52b7b6` | 整数seed0はmetadataのみでRNG未使用。run02成功、324 component/9 basisと2lawのlocal効果を取得 |
| 同local結果からの縮約解析 | `2b3b0ee7ffea92de25a2c0c549d64f258c34a611` | 追加preparation/law生成なし。十分条件と有限wire state enclosureを確認 |

対象は同じpreparation hashの1個、R2/R128の2law。失敗復旧を含めdescriptor復元処理は2回起動したことを明記する。C1保存snapshot/receiptと元S0a Contextを再利用し、新しいSCF/DF/H/state生成はしていない。旧S0aのpathやSを今回再評価したものではない。

実sampling/full-path S/平均S/回路build/compile/fit/平均C/N/G、最大8wrapper pilotは0。新grant/承認証明は作らず、今回のユーザー指示の科学scope内で実施した。元一回実行契約・旧grant・ledger・marker・STOPは変更しない。

## 今回の状態縮約と限界

元SのRZ数に対し、共有active basis・order2-seen flag・wireごとの正確なpending Zを状態へ残す。元basis operations/inverse/qubit順、binary64順加算/literal zero、tail終了の遅延basis、固定head/outerとscalar/XY terminalは元順のmapにする。

実9 basisの正逆wordはいずれもsystem全wireをresetし、合法product/rotationも各wireでreset又はuntouchedだけだった。tail phaseは合法非identity K2では非負で、order2未出現なら0、それ以外は正。reverse headのcontrol resetより前に別control RZがないため、その正値の大きさをS observableでは保持しなくてよい。phase flagはbasis flushにも影響するので全wire marginalへ共有する。

tail application境界の保守的state enclosureはsystem8wireで100/100/100/100/100/100/120/80。これに実boundary/control mapが必要で、完成kernelサイズや実測性能ではない。wireの期待reward和は相関の有無にかかわらず成り立つが、covariance/CIの独立合成を許可する根拠にはならない。

**有限S observable状態の十分条件はSOURCEから確認したが、物理kernelの実装・全transition/boundary parity・実用性能はUNKNOWN。** 同じ縮約をfull reportやcompiled priceへ移さない。`run02.state_equivalence=UNKNOWN`は変更せず、後続REDUCTIONの構成的論証と分ける。

## 最小実装とテスト

新module `h4_pr_physical_closure.py`、runner `run_h4_pr_physical_closure.py`、test `check_h4_pr_physical_closure.py`だけ。元Accounting/MeanKernel/Streaming/providerは変更していない。新tracerは元native loweringを再利用し、物理full-pathのSやfinishを禁止する。`projected_step`＋既存MeanKernelの接続は相関した人工2wire例でのみ確認した。

最終10技術test PASS（[log](TESTS_REDUCTION.txt)）。初回7/seed修正8件の記録も保持。実行runnerを再起動して結果を再現する許可は今回の完了後STOPからは得られない。通常の人工testは実物理処理を起動しない。

## 次の一括科学scope案

**ユーザー**は、同じ1 preparation・2lawで次のK0→成功時K1を一括許可するか判断する。**Codex C**がその実行SOURCE/精度/STOPを確定して、許可された範囲だけを実行する。

1. K0：今回保存descriptorを再利用して、物理wire-marginal transition/rewardとentry/close/head/outer/control/XY suffixを構成する。元Accountingのlocal/boundary効果、literal zero、全合法component/order、確率massと演算精度を照合。小状態集合を丸めで閉じたり、missingを0へ置かない。
2. K1：K0を通過した場合だけ、m0/m6×X/Yの実E[S]を4値評価する。full event列やsampling/compile/fit/N/Gは不要。ideal finite lawの平均Sであること、実sampler/丸めとの距離・誤差資格を保持する。

K0不一致/未閉包/未定義精度ならUNKNOWNでSTOPし、近似や直接compileへfallbackしない。今回はK0/K1未実行。E[C]には同条件のE[C−S]がなお必要で、前回の最大8 wrapper案はそのための別development pilotとして保留する。使用する場合はseed/cohort・SOURCE・価格law・必要な精度/記述目的・STOPを別途明示し、n2を全law精度保証にしない。

直接価格案の既存正本は[前回SCIENCE_PLAN](../h4_s0_mean_price_plan_20261113/SCIENCE_PLAN.json)の`D_DIRECT_FINITE_PILOT`。候補cohort名は`H4_S0_M0_R2_DIRECT_DEVELOPMENT`/`H4_S0_M6_R128_DIRECT_DEVELOPMENT`、各index0/1、計4path/XY8 request。元C1 `1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7`のsampler/native wrapper/cache/compilerを新leafから再利用し、元run/grantを再起動しない。元trial_seed規則と新しい重複しないnamespaceから実seed列を実行前に固定する案で、今回seed生成/samplingは行わない。実sampler有限cohortの価格参照とpaired C−Sを目的とし、ideal全law平均に必要な標本数・価格精度は未確定。binding/法則/phase/compiler不一致やcheck失敗なら部分結果を保存してSTOP、pool/追加path/fit/自動retryなし。

**Codex B**は固定V2にこの診断の出典を結び付けるだけで、価格値は欠測に保つ。成果公開・remote照合後mandatory STOP。
