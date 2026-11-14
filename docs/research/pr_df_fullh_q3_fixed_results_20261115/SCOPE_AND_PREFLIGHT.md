# 固定3候補Q3：実行前の条件・重複・保存検証

2026-11-15 JST。計画0629a8cf359b3aa99400f79de8f61a5ef50d8162のFINITE_SCIENCE_PLAN/CANDIDATE_INPUT_INDEXを正本とする。今回のユーザー指示は、PR-DF LD1/LD2とFull-H κ2の不足分に対する各1vector・各最大1回、計最大3新Q3と閾値判定の明示許可。新しい承認証明書等は作らない。

## 計画後の差分・重複照合

既存B branchのHEADと公開refは0629a8cf3、cleanを確認。計画後の関連公開ref/履歴では、同一3候補・vector・decoder・target/law/lossの新しい受入可能Q3結果は未同定。旧DF partial DP、Pauli control/3/4、PF6結果は計画記載の条件差を保持して再利用境界を引き継いだ。信号39round・旧bridgeは再利用し、信号再取得は0。

確認した関連Python processにはFull-H P2価格callerがあったが、同一Q3ではなかった。D1の価格完了は待たず、Dのprocess/worktree/inputに干渉しない。確認範囲外の全記録の不存在を証明したものではない。新output namespaceは着手時に未存在。同等取得の有無が判断不能な候補は実行せず保留する。今回はmetadata受入3件成功・保留0としてINPUTSへ固定した。

## 入力・モデル・有限scope

固定索引の全ID/physical/context/SOURCEを消費し、PR-DFは既存K.accept_savedと版付きfixed-grid total bridge、Full-Hはκ2の原B_SIGNAL_INPUTとbaseline SOURCE付きbridgeを使う。R448・Pauli辞書・generic-gridへの変換をしない。normalization/scalarは再適用しない。

両軸vectorは`[64,60,56,52,47,43,38,34,30,25,21,17,12]`、allocation IDも計画どおり。別の明示evaluation_targetはsolver proxy −2.1663874486347634 Ha、元packet/資格は変更しない。Z1＋T1、全事象平方loss、ideal finite-K raw/IID、EXACT_AFFINE_OF_BINARY64_RAW、τは十進εE0.00159360010199040のexact二乗。旧Minkowskiを別参照に保持し、δEの加算なし。

ρ1、round outcomes4225、verification pairs55043173、非終端frontier8192、transition pairs141559036／候補は変更しない。数値certificate・ground・実sampler移送・十分N/global/Gは未認定。

## count-only上限の切り分け

最初の準備で、全countを未集約で数える一般helperの上限143,995,786組が固定上限を超えたため、**科学call0のままINPUTS固定を保留**した。科学範囲を拡張せず、既存の同vector・同total decoderの保存supportを使った、より鋭い計数上限を確認した。

caf1417の`pauli_total_q3_accelerated_results_20261114/run01/PARTIAL_COARSE_LD1.json`はm0の全4225 Cartesian countを含む。同runの`PARTIAL_TOTAL_DP_LD1.json:rounds/0`はgrouped outcomes=retained states=2593、residual=0で、初回全supportが失われていない。

初回decoder状態は同じN/count→binary64→phase/zero規則で決まり、確率重み、辞書、Hamiltonianの違いは新しいcountを追加しない。新候補のbinomial count支持はこの全Cartesian支持の部分集合なので、初回状態数は2593以下。それ以後は非終端frontier8192以下である。したがって`C0 + 2593*C1 + 8192*sum(C2..C12)`（C_m=(N_m+1)²）というcount-only上限は**137,923,114組**で、141,559,036以内。m0についてはgrouped数より大きいC0を用いて保守的に数えた。round上限4225と隣接上限55043173も収容できる。

SOURCE/保存計数・式はINPUTS.finite_scope_preflightに保持。保存PauliのMSEを移送したのではなく、同一decoderの全count→初期状態の計数だけを再利用した。旧partial DPの幾何上限をtotalへ無条件適用していない。新しい確率・phase・DP計算はこの確認で呼んでいない。候補固有の閉包・質量・allowance・残余・上下界は各科学call内のgateで確認する。

## worker保存と人工fixture

既存PF workerの数値gate・T.evaluate・高速backendをそのまま再利用し、新callerは3候補の適切なmetadata routeを接続する。T.evaluateの返却直後、**worker内でAPI_RETURN.jsonを保存**し、その後RESULT/RESOURCE_FINAL/PERSISTEDを保存してから親へ小さいreceiptを返す。CPU/peak RSSはworker自身で測り、原API return・partialsの保存を親monitorに依存させない。

親monitorは既存のFuture→label修正を再利用する。例外時は共有failure flagで未開始callを止め、監視エラーを保存し、開始済みworkerの結果を回収する。監視不能を数値PASSへ混ぜない。保存不能はUNKNOWN/STOPとし、無断retryはない。

人工fixtureでは、実際の子processが人工T.evaluateへ入ったところで待機し、親monitorに例外を発生させてから解放した。親の運用エラーを保ったまま、子が原API return・3種partials・CPU/peak RSS・RESULT/PERSISTEDを保存することを確認した。未開始callの拒否・R448 ID混在拒否・固定metadata routeも確認。INPUTS未作成時のmetadata test不足は科学開始前に解消した。最終5 test PASS。実H4科学評価をtestとして呼んでいない。

## 資源・一回性・STOP

観測時64 CPU、available約478 GiB、load約1.7で、価格jobと同時でも空き資源を確認した。既存benchmarkでplateauを確認した8worker/batch8を各候補で再利用し、3構成を独立並列にする。恒常的なCPU/RAM制限ではなく共有設定を変更しない。

caller・INPUTS・本範囲をcommitしてから最大3科学call。原API return・資源・数値判定と運用statusを分離して保存する。完了又は重大異常でmandatory STOP。新配分・追跡拡大・別backend・他方式risk・価格/G・十分N認定へ進まない。旧PF6欠測も埋め直さない。
