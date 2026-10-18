# V1〜V6：将来の検証の受入条件

**テスト設計の文書だけであり、新しいtest SOURCE・fixture・値・回路は作成していない。既存testも実行していない。** 下表の数値tolerance、精度、confidence、試行数、分子、backendは未採択である。既存test内のtoleranceはその当時のscopeに限り、新契約へコピーしない。

S/T番号の固定commit・test名・行位置は[source inventory](source_inventory.json)。実装状態は[interface matrix](interface_contract_matrix.md)。新しい科学的保証が必要なら本計画で補わずユーザーへ戻す。

| 検証 | 目的／既存命題・SOURCE | Fixture／前提 | 受入条件・negative/boundary | 新しい実行・承認 |
|---|---|---|---|---|
| **V1 入力・正規化** | E1/E2、M1/M10。S04/S05/S06/S09〜S12 | 既存synthetic DFBlock/DFHamiltonian testの定義を再利用可能。保存分子NPZは別scope | label→component/basis/fragmentが一対一、sign/identity/constant/threshold/D/R不変。p>0、正確なΣp=1、Λ_Rとtarget identity・誤差証拠一致。零成分除外、Λ_R=0／h=0 bypass、L=1/m=1、bool/noninteger、NaN、異basis同support、x>1を区別。型変換で元H誤差0と宣言しない | syntheticの新実装・人工testは別途承認後。保存分子のdecode/係数評価は科学実行の別承認後 |
| **V2 Event意味論** | M1/M3〜M7/M9/M12。S01/S02/S06/S08 | 小さいformal wordの独立raw oracle。現在のG6/G7 testsを基礎にできる | wordのoperator順とreverse時間順、child単一回転、phase_i_power、σ±、π/c̃/W、metadataを全て保持。raw非reducedを別parentへremapしない。ordinary `_make_event`への変換でangle/lawを作り直さない。現在のpartial-S2 guardが拒むことを検出し、guardを弱めず別bindingの仕様を検証 | 実装と人工testの明示承認後。今回はformal oracleも生成しない |
| **V3 Finite-bit** | M9/M10/M11。S01/S02、既存G6 finite-bit監査 | 正確な有理入力と、固定policyが事前選んだH/K_root/η/ρ | full support、dyadic分母、実bit-map、πW=c̃、root enclosure、bias/range/momentの**既存**上界を検証。support喪失・広いinterval・低bit・非正残余幅はfail closed。例外をzeroに変えず、結果を見たprecision rescueなし。元MとM̃を別検証 | selector実装や新しい精度policyは未承認。人工計算も別途許可が必要。新しい精度定理は作らない |
| **V4 Controlled位相・回路** | M5/M12。S03/S07/S10/S19、T04/T07/T10 | まず既存小operator/registryの定義。実DF/nativeは別段階。provider routeの選択が前提 | control=0のIとcontrol=1の目的eventを**位相を消さず**比較。parent phase、negative component、identity rotation、constant shift、σ±、noncommuting word順序、actual inverse、workspace cleanup。direct DFとhelperのworkspace/costを混ぜない。位相破壊・逆順・二重sign・σ落ち・basis hash改変が拒否される。strict errorの証拠scopeを明示 | synthetic回路build/行列検証でも今回禁止。将来その範囲の明示承認後。native合成・実分子はさらに科学実行の承認後 |
| **V5 全trial estimator** | M7〜M10。S02/S14/S15/S17/S18 | 明示された人工zero/accepted/failure列とbit convention。fresh-IID条件は別 | 全trial分母、rejected=0、accepted=WY、axis分離、実W保持、N/accepted/quantum-callsの区別。accepted-only・再normalization・失敗をzero化・成功までretry・counter改変をnegative例にする。未知Z/B_RAやtrue signalをbudgetへ使わない。range/moment/bias/familyが同じtargetに対応。空のaccepted集合と未完了runも正しく扱う | 人工testの実装/実行も別承認後。backend shotsを行う場合はその科学実行承認が必要。人工固定列PASSをIIDの実証にしない |
| **V6 End-to-End** | E1〜E7、M1〜M12。S01〜S18と既存ordinary経路 | V1〜V5が接続され、target、signed-time、precision/coverage、provider、state、confidence、failure policyが固定済み | 入力identity→label→π/W→strict controlled→bit→全trial記録→budgetが追跡可能。各誤差項目と費用vectorを別保存。ordinaryにも同じinput/state/precision/compiler/cache/合法なbasis共有を与える。測定結果・prefix・source差替えでcertificateが不整合になることを検出。multi-stepは採択済みのpath/weight/order/測定位置の保証がある場合だけ | 現在は未承認。最小synthetic接続から始める場合も別指示が必要。分子・native・実測shotを含む段階には固定scopeと独立した科学実行承認が必要 |

## 既存検証の再利用範囲

G6の16件、G7の18件、G8の17件、G9の原23件とv2 API19件は、それぞれ固定された過去のlocal記録である。G9 v2の19件で原23件を再実行したことにはしない。現在のDF test SOURCEには、Z/ZZ・sign・identity・basis hash・controlled block-diagonal・ordinary partial-S2・Hadamard bit/signの検査がある。これらの存在だけを今回のPASSや全G6接続のPASSへ格上げしない。

`test_h2_df_tail_reconstructs_selected_conjugated_fragments`は分子生成を呼ぶtestであり、既存数値fixtureを読むだけのtestではない。将来の「人工回帰」へ混ぜて一括pytestしない。実行可能なsubset、依存関係、科学callを明示してから承認を受ける。

## 受入状態と失敗

将来の結果は、要求する証拠が揃った範囲のPASS、具体的な不変条件違反のFAIL、入力/証拠/仮定不足のNOT_VERIFIED又はUNDETERMINEDを区別する。source hashが一致するだけでcorrectnessを認定しない。backend/precision/resource failureでconfidenceを持つ完了結果が作れない場合、失敗・部分記録を保全してSTOPする。再実行、marker/claim発行、cap変更は本計画が承認しない。

技術検証の人工PASS、分子correctness、native certificate、科学的価値、研究GOは別の判断である。CPU/RAM/worker/wall-timeに新しい人工capは設けていない。既存の固定cap/STOPは変更しない。
