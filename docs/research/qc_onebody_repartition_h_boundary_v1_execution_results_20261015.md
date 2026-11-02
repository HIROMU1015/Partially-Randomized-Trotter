# 固定H4再配分PF・h境界v1：収集完了／独立監査停止（2026-10-15 JST）

固定one-shot収集は完了したが、独立数値監査が `NUMERICAL_ALLOWANCE` で停止した。retry0。**本段階の状態は `COLLECTION_COMPLETE_INDEPENDENT_AUDIT_FAILED_MANDATORY_STOP`、新しい性能値は `UNVALIDATED_PRODUCER_ONLY`。数値監査PASSや研究の成功・失敗・継続・終了判断とはしない。**

## 実行認可と固定scope

ユーザー「次の作業に進んで大丈夫」を、直前の完了報告で次段階と指定した新h境界契約への一回実行承認として扱う旨を実行前に伝えた。IDE metadataは使わない。別認可をcommit/push・GitHub再取得後に実行した。[今回の認可](../../artifacts/qc_onebody_repartition_h_boundary_execution_v1/2026-10-15/authorization.json)、[実行前GitHub照合](../../artifacts/qc_onebody_repartition_h_boundary_execution_v1/2026-10-15/prelaunch_remote_receipt.json)。旧36-cell認可を流用していない。

準備正本 `361cf233fb2e4bb00b340311d27d936d7b44fbb6`、source `5e3341c12bd84f30f19072cefa590f4499c36c38`、contract SHA256 `f1417c0eac36ed2dae935fc3b5aff74f3bfa3adc77b10e74006161250de2fe05`、source manifest SHA256 `35cb38263997eb26f636959799691660c64c2512f903673521f0e6b614d10b7b`。[固定契約](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/contract.json)と42file freezeは不変。未発行target、旧source/results/認可/STOPもbyte保持した。

linear H4 / 隣接1.00Å / STO-3G / 保存DF rank12、generation順全12density、8spin / 4spatial、Nα=Nβ=2、sector36dim。PR L_D/delta変更なし。H=constant I+dΓ(one_body)+Σλ_l dΓ(g_l)²、normal-order追加補正なし。入力pack SHA `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`、Hamiltonian hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。v1 selector seal、response/known shift、state、orbital frames、leading、truthを凍結して継承した。E0は性能biasだけに用い、候補/aliasの救済には使わない。

共通比較h=0.002745119306432635..1Ha^-1、h/2・追跡anchor補助hは0.0013725596532163176..1。全12methodへ同じ比較点を与え、2roundの登録refinementだけを実施。全強対照（original/known raw・CS2、全S4、全Richardson）、target10^-4/10^-5Ha、P1/P2、予算分配(.5,.25,.125,.125)、uniform10%gateを維持。新h/分子/係数/selector/precision・compileの救済なし。

## 収集できた一次データ

追跡h762点、比較h381点、primitive cells4572（旧anchor12を含む）、method records4572、cost rows18288。新PF seed/Schur各4560、saved raw科学array load1。新H/spatial eig/RDM/leading/selector/optimizer/molecule/ground preparation/compile/T0。

collectorは75.801秒、producer最大RSS 479604736bytes、checker最大RSS 264949760bytes、attempt出力240718839bytes、console合計614bytes。観測したメモリ/output/logは契約512MiB/512MiB/2MiB内。worker/thread1、wall/CPU capなし。producer exit0、checker exit2。旧27/35/37testsはこの実行で再実行していない。

[一次JSON](../../artifacts/qc_onebody_repartition_h_boundary_v1/2026-10-15/attempt_1/raw.json)、[cell arrays directory](../../artifacts/qc_onebody_repartition_h_boundary_v1/2026-10-15/attempt_1/cells)、[result identity](../../artifacts/qc_onebody_repartition_h_boundary_v1/2026-10-15/attempt_1/result_identity.json)、[producer stop](../../artifacts/qc_onebody_repartition_h_boundary_v1/2026-10-15/attempt_1/stop.json)、[audit stop](../../artifacts/qc_onebody_repartition_h_boundary_v1/2026-10-15/attempt_1/independent_audit.json)、[summary](../../artifacts/qc_onebody_repartition_h_boundary_execution_v1/2026-10-15/saved_results_summary.json)。

producerはdiscrete-grid・boundary双方 `NO_UNIFORM_REGISTERED_COMPONENT_GAIN` を保存した。以下は保存されたproducer値の転記で、独立監査未完了である。肯定/否定の科学成果として確定しない。

| target / Ha | price | response | response h | control | control h | 未検証ratio |
| --- | --- | --- | ---: | --- | ---: | ---: |
| 1e-04 | P1 | CS2:response_selected | 0.601881039 | CS2:known_linear_extraction | 0.585234372 | 0.970902866 |
| 1e-04 | P2 | CS2:response_selected | 0.601881039 | CS2:known_linear_extraction | 0.585234372 | 0.971585144 |
| 1e-05 | P1 | CS2:response_selected | 0.338427654 | CS2:known_linear_extraction | 0.330209304 | 0.974272805 |
| 1e-05 | P2 | CS2:response_selected | 0.338427654 | CS2:known_linear_extraction | 0.330209304 | 0.974957451 |

価格はstatic controlled Givens/linear/pairのP1=(4,1,2)/P2=(8,1,4)仮説、coherent queryはceil(π/(h εstat))のHeisenberg proxy。h選択は性能oracle、情報取得・state preparation・T/compile・full QPE/RPE totalは未評価。producerの局所bracket「resolved」も本監査停止後は未検証の診断値として保持する。一般不能性、global optimum、新規性、正式採択/終了を主張しない。

## 独立監査の停止箇所と不足情報

一回の独立checkerは `INDEPENDENT_AUDIT_STOP_NO_RETRY`、reason `NUMERICAL_ALLOWANCE`。固定checker64行の `abs(saved_allowance−independent_allowance)<1e-10` が不成立だった。[失敗記録](../../artifacts/qc_onebody_repartition_h_boundary_execution_v1/2026-10-15/failure_record.json)。

source読取りで、producerはspectral/cached Uと保存eigenpairs、checkerは独立Padé Uと保存eigenpairsからresidual/defect/orthogonality、円上cluster幅を使ってallowanceを構成することを確認した。この違いは不一致の候補であり、原因を確定したものではない。失敗handlerはcell key、双方のallowance、partial counterを記録しないため、それらはUNDETERMINED。追加のexpm/phase計算を行って不足値を補っていない。

独立PF/Padéの実partial回数は保存されていない。checker sourceはprimitive inventory上限と各expm上限を確認するが、完全な数値・費用・refinement監査へ到達したとは認定しない。new eig/RDM/compile0と観測RSS/logはsource/実行記録の範囲で示す。byte/SHA/保護監査の成功を数値監査PASSと混同しない。

## 保全と次の承認境界

[identity/resource/protection記録](../../artifacts/qc_onebody_repartition_h_boundary_execution_v1/2026-10-15/execution_identity_caps_audit.json)：515旧files、42frozen files、156旧manifest/non-result fields、global description欠落FAIL、root138pathsを保持する。N1STOP/N2HOLD/N3CLOSED/THRIFT現template CLOSED、Track A/B/C1–C4/第二候補は変更・再開しない。

attempt1、audit attempt1は両方消費済み。sourceや許容差を変更して合格へ合わせず、producer/auditを再実行していない。GitHub公開・再取得後Mandatory STOP。

[技術停止引継ぎ](qc_onebody_repartition_h_boundary_v1_audit_stop_handoff_20261015.md)を用意した。次の候補は、失敗値を記録できる新checker versionの限定準備と、保存データへの別数値監査/診断scopeの固定である。ここでは新version実装や再実行を開始しない。新しい数値診断には新source/capsの別明示承認が必要。重要GPT科学解釈・継続/終了判断も別開始承認を必要とし、今回代行していない。
