# N5-B2：m17全consumer経路への因子分離の条件

2026-11-08 JST。[REPORT](REPORT.md)。固定base `f250bdb99a15804adfe2cfebc0ce16d5d8be6c1e`。新しい保証の正式採択・実回路実行はしていない。

## 1. 二つの接続を分ける

[N5-B1命題](../pr_reference_n5b1_scalar_20261108/JOINT_SCALAR_FACTORIZATION.md)は、**全controlを構成しcompiler Aを通した後の同一joint native回路**を開始点とする。許容native入力の各RZへ同じlogical wordを選び、同じnamed gate列・wire・順序をemitするなら、logical body Lとemitted body Eは

`L_omega = exp(i*phi_L(omega)) A_omega`,

`E_omega = exp(i*phi_E(omega)) A_omega`

と分離される。各local scalarをjointへembedしても全体scalarであり、CX、actual whole inverse、inplace composeを通じてこの型を保つ。

この**native→logical/emittedの代数**には、Dのevent非依存性やframe価格の共通性は必要ない。一方、元H/SOURCEの信号へ接続し、m17の保存literal・role・旧diskを使うには、**SOURCE→nativeのcompiler条件Cとword供給・terminal sinkの資格**が別に必要である。候補Gを理由にCを省略せず、逆にCの未確定部分をGの代数自体の反例とも扱わない。

## 2. SOURCEから測定までの追跡

| 処理 | 固定SOURCEと確認内容 | m17へ残る条件 |
|---|---|---|
| Canonical event／RTE | [DF RTE builder](../../../src/trotterlib/df_rte_qiskit.py)の_buildはbasis共有を同じapplication列で決定し、controlはcentralへ付ける | 固定component／basis／finite-bit lawを保つ。新samplingなし |
| SOURCE phase | Controlledならaccumulated_phaseをancilla PhaseGateへ、uncontrolledならcircuit.global_phaseへ置く | ここでのphaseは後のrelative補正に使われる。Gで消去しない |
| B′構成 | [rewrite_rte／build_saved](../../../src/trottertracks/pr_reference_bprime.py)はcentral crzzだけを既存4-CX構成へ替え、tail phaseを保持。[symmetric_fused](../../../src/trottertracks/pr_reference_interference_pilot.py)はq依存scalar Pを置く | SOURCE inverse／既存D融合／相対位相を保つ |
| Reference構成 | [assemble](../../../src/trottertracks/pr_reference_transfer_qualification.py)はm17のq>=2 splitでF/Gを作る。reference_coreはCSWAP、F.inverse、G、vacuum P、CSWAP | Vacuum補正の生成までGをさかのぼらせない |
| X/Y wrapper | wrapperはH、core、YならSdg、H、最後のmeasure | このwrapperは**compile前**に含まれる。同じreadout・clbitを維持 |
| Joint native lowering | [compile_reference](../../../src/trottertracks/track_c_compiler_a.py)の固定passを全jointへ適用。後置controlはprofileでも未資格 | 下記C、実binary64 literal、同じbasis/同値則を維持 |
| Prefix reset | 既存prepare_oneではReferenceだけにcompile後のprefix reset群を追加し、その後nativeをcompose | m17でも同じ配置を使うこと。新buildで観測したとは主張しない |
| Word選択 | [provide](../../../src/trottertracks/pr_reference_dyadic_dictionary.py)は実thetaを読み、domain判定後にC0優先、missは固定dyadic構成 | 必要全入力へ適格wordが返ること。現m17では成立していない |
| Negative／emission | [lower_native](../../../src/trottertracks/track_c2_b_verify.py)はpositive wordを逆順emit、負側はcomplete actual inverse、Wだけphase欄 | Wordのexact Wとstrict guardを維持。Negative sequenceを先に渡して二重inverseしない |
| Segment／outer | [consume_native](../../../src/trottertracks/pr_reference_common_word_consumer.py)は同wireのinplace join、phase欄とnamed命令を分離 | Numeric有限phase・既定命令意味・許容alphabet、同じconsumerを使う |
| Reset／measure／clbit | Consumerはfenceを同順序・wire／clbitへappend | N5-B1局所診断とN4-B保存projectionの範囲を超えた本番経路は未資格 |
| Terminal sink | Gは、出力をそのまま同じ最終測定へ渡すことを要求する | 新control／compiler／relative-phase転送／別serialization経路を後置しない |

M17はq/rと反復数が変わるが、上記SOURCEで内部にreset／measure／barrierを追加する分岐はない。同じC下でprefix reset＋unitary body＋末尾readout、nonempty segment数S<=1という[N4-Cの構造証明](../pr_reference_n4c_accounting_20261108/REPORT.md)を条件付きで再利用できる。旧shell値やRPEへの寄与は再評価していない。

## 3. 許容入力クラスと端点

Consumerが要求するのは、bound numeric parameter、有限numeric global_phase、native rz/sx/x/cxとreset/measure/barrier、条件付き命令なしである。Bodyは通常のQuantumCircuit、登録wire／clbitとnamed ideal gate意味論を保つ。Float phaseがword選択やgate引数へ流れないことをSOURCEで追った。

このクラス内では、任意の有限word列・qubit数についてSOURCEのループ不変条件を適用できる。巨大m17 wholeをbuildしないこと自体が代数の穴ではない。ただし、**成功するword供給と実際のterminal sinkの証拠がない入力を、このクラスの完成実行と扱わない**。

固定commitの`src`と`scripts`を対象としたconsume_native／module名の参照確認では、library定義、N4-Bの局所／projection検証呼出し、N5-B1のAST再利用、静的auditでの参照が確認できた。m17本番のconsumer→terminal実行経路はこの対象から確認できない。これは外部／別branchの不存在証明ではない。統括兼Dの別adapterは未取得で、意味論を推測しない。

## 4. Compiler条件C

[CT-5BのC](../pr_reference_ct5b_20261107/STRUCTURAL_CERTIFICATES.md)を維持する。

- Qiskit1.3.0、compiler A、rz/sx/x/cx、Target／routingなし、保存のSessionEquivalenceLibrary。
- SOURCEが保存post-unroll gate族・標準同値則・parameter bindingへ閉じる。
- 実signed binary64 literal、actual SOURCE inverse、joint relative phaseを保持する。
- DのSOURCE構造はevent非依存、frame pair multisetはpaired armで共通。Native keyへの移送はC条件付き。
- 初段取消しが対象D/frameのRZ族を変えず、最終取消しはCX/Xだけ。新しいpassなし。

M17 r32中央の値、q依存scalar、vacuum／eventの値はm7とは異なる。保存templateのsymbolic parameterへその実値を束縛するという条件を保ち、新しいangle cutoff／numeric rewriteが起きるとは仮定しない。実SOURCE inverseの保存templateを使い、forward native列のinverseと同一視しない。

## 5. 最終X/Y信号

同じprefix reset map R、入力rho、axis別の同じobservable Mについて、条件下では

`Tr[M E_omega R(rho) E_omega†] = Tr[M L_omega R(rho) L_omega†]`。

Trajectoryごと・axisごとの恒等式を同じcanonical lawで平均する。Chi_omegaはtrajectory依存でよく、normalization／range／samplingを変えない。Uncontrolledな平均operator一致や平均channel一致をcontrolled coherent signalの根拠に代用しない。

Logical RZ近似、SOURCE scalar/event/vacuumのrelative angle、compiler disk、finite-RTE／finite-bit／normalization側の旧誤差は残す。Strict operatorでは一般にE!=Lであり、保存certificateを0へ変えない。

**結論：SOURCE条件付きの全経路仕様は明示できるが、m17の完成policy・全入力・terminal資格は未完。** N5-B1の160件は引用のみ。新しい経路のruntime検査や保証採択はせずmandatory STOP。
