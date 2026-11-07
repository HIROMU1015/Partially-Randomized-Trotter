# N5-B1：joint scalar因子分離の条件付き命題

2026-11-08 JST。[REPORT](REPORT.md)。これは固定SOURCEへの記号的な技術分析であり、新precision・emitter・誤差規約の正式採択ではない。有限数値診断を一般証明の代用にしない。

## 比較の型と開始点

開始点は、**control構成とcompiler Aを終えた、同じpost-compiler joint native回路**である。SOURCEのF/Gやuncontrolled Vへ先に合成を入れる問題とは区別する。

NativeのRZ位置iに渡す保存positive wordをs_i、そのW出現数をK_iとする。Written productとして、exact Wを `exp(i*pi/4) I` と解釈する。W以外のnamed gateからなる同じ行列積をA_iとすれば、positive logical wordは

`L_i = exp(i*K_i*pi/4) A_i`。

Negativeは**word全体のactual adjoint**なので、logical wordは `L_i† = exp(-i*K_i*pi/4) A_i†`。正のwordをもう一度合成したり、各chunkを元の順序のままinverseしたりしない。±0のempty wordでは既存SOURCEの `theta < 0` 判定をそのまま保つ。

型は次の三つに分ける。

| 対象 | 意味 |
|---|---|
| `RZ(theta_i)` | Actual binary64 target。SOURCE scalar/event/vacuumのrelative角を含む |
| `L_i` | 保存wordのexact Wを含むlogical unitary。Targetへのstrict近似差は残る |
| `E_i` | 同じnamed gate列を既存emitterで具象化し、float global_phaseを持つunitary |

## 固定SOURCEで確認した前提

1. Wordのwritten productを逆順に時系列emitし、Wだけをnumeric global_phase欄へ加える。他文字は同じH/S/T/Tdag/Xのnamed gate。
2. 負側はzero-phase localへpositive wordをemitした後、local全体を一度inverseする。Named inverseと順序反転、phase欄の否定は分離している。
3. Wire embedding、local→segment、segment→outerのcomposeは同じwireへのinplace経路。SXは位相込みの恒等式SX=HSH、X/CXは同じnamed gateを使う。
4. Float phaseはglobal_phase欄だけに保存される。RZ angle、word選択、gate引数、wire、control、reset、readout、clbitへ転送されない。
5. Consumer後にcontrol、再compile、QPY／別instruction化等の新しい実行経路を加えない。Numeric phaseが有限で、固定runtimeのnamed ideal gate意味論を用いる。

具体的な対応は[SOURCE_PATH_VERIFICATION](SOURCE_PATH_VERIFICATION.md)。任意の別compiler／consumerがこの前提を満たすという主張ではない。

## Unitary bodyでの因子分離

Nativeの保存共通phaseをphi_0とし、各signed W指数をk_i（positive K_i、negative -K_i）とする。W以外を同じ順序・wireに置いたjoint named productをAとする。各局所scalarのembeddingは

`(exp(i*a) A_i) tensor I = exp(i*a) (A_i tensor I)`

で、CXを含むすべての後続gateとscalarは可換である。したがってlogical bodyは

`L = exp(i*phi_0) exp(i*sum_i k_i*pi/4) A`。

一方、emitterとconsumerはnamed product Aを変えず、位相加算・inverse・normalizationをglobal_phase欄だけへ行う。実際に保存されたouter phaseをPhi_Eとすると、ideal named gate意味論で

`E = exp(i*Phi_E) A`

ゆえに

`E = exp(i*chi) L`,

`chi = Phi_E - phi_0 - sum_i k_i*pi/4`。

これはchiを小さいと仮定せず、W loopの8状態閉包や真piとの差の数値上界も必要としない。Prefixのtau→0再設定、joinの丸め、shell phaseも、前提4が成立する限りこの共通scalarへ入る。**Strict operator距離が0になるという命題ではない。**

Finite W residue閉包はruntimeのphase経路照合には再利用したが、上記の一般因子分離は「全prefixが8状態へ閉じる」という偽の前提には依存しない。W=72のb2も、exact logical側では72、float経路の照合ではresidue0として別々に扱う。

## Reset・readoutとX/Y

固定RPE形はprefix reset、unitary body、末尾readoutである。同じprefix reset map Rと入力rhoに対し、任意の同じjoint observable Mについて

`Tr[M E R(rho) E†] = Tr[M L R(rho) L†]`。

Readout gateもbodyへ含められるのでX/Y wrapperのどちらでも成立する。一般の複数unitary segmentと同じCPTP fenceの合成でも、各segmentのscalarはその密度行列作用で消える。ただし、その場合にwholeを単一unitaryと呼ばない。今回の新fixtureはprefix resetと末尾measurementだけを使い、途中fenceの実行は既存N4-B結果を引用した。

各trajectory omega、各axisでこの測定恒等式が成立すれば、chi_omegaがtrajectory依存でも、**同じcanonical lawで平均したX/Y測定信号**は変わらない。Uncontrolledな平均operator `E[exp(i*chi_omega) V_omega]` の一致を主張していない。平均channel一致からcontrolled coherent signalを推定した議論でもない。

## この命題で消せないもの

- 保存logical wordとRZ targetのstrict誤差。W込みguard、dyadic量子化、basic guard和はそのまま残る。
- SOURCE scalar／event／vacuum Pの角度差。これらのgateはrelative eigenphaseを持ち、今回のfloat global_phase欄の差とは異なる。
- F/Gのphaseを後でreference干渉のPへ転送する処理。命題の開始点より前にあり、field名だけで無視しない。
- SOURCE→native／finite-RTE／finite-bit／vacuum算術など既存raw diskの項。今回減額・再会計していない。
- 合成後のcontrol、別backend／compiler、symbolic phase、numeric matrix評価の丸めを物理的なnamed gate誤差とみなす別モデル。

判定は **JOINT_SCALAR_FACTORIZATION_SOURCE_SUPPORTED**。対象consumerの前提を満たす入力クラスについてのSOURCE上の支持であり、m17全supportの入力・policy・域を資格した結果ではない。次の判断は[保証評価](SIGNAL_INVARIANCE_ASSESSMENT.md)へ分離し、mandatory STOP。
