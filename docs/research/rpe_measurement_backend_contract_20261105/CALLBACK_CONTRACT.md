# Trajectory・測定callbackと整数和/phaseの契約

2026-11-05 JST。対象条件・版は[対応表](INTERFACE_SOURCE_MAP.md)に固定。
既存`measurement_adapter.py`のAPIを変更せず、外部providerに必要な意味論を確定する。
具体backendの正式採択、型annotationによる証明、新しいsampler実装は行わない。
`prepare_df_partial_s2`のcoefficient_atol既定値は0.0なので、固定contextの準備では1e-12を明示する必要がある。

## 1. trajectory_factory(request) → T

入力は既存MeasurementRequest。Partial L_D3ではrequest.settingsにm/q/r/k/δ/N等、
axis∈{cosine,sine}、shot∈range(N_axis)、key=(3,m,axis,shot)を持つ。
keyはrequest identityでありseedや独立性の証明ではない。外部experiment namespaceを必要とする場合、
provider側で区別し、既存4-tuple自体を無断変更しない。

API上Tはopaque object。既存request表現を使う接続候補では**DFPartialS2RepeatedRequest**を返す。
別backendのlazy handleを使う場合も、以下の同じtrajectory意味論と照合可能なdescriptorが必要で、
その具体型はbackend接続時の技術的bindingとして固定する。現在採択済みのproduction戻り型はない。
factoryは測定結果や保存μ₀を返さず、次の不変条件を満たすTを生成する責務を持つ。

| 項目 | 必須契約 |
|---|---|
| 固定context | 同じH/state/export/preparation/tail/law/basis、identity抽出/cutoff1e-12、training fingerprintとsupport policyをclosure等へ明示束縛。RoundSettingsだけから推測しない |
| 時間・構造 | `step_time=δ`、`repetition_count=q`、各stepの`rte_steps=r`、`finite_taylor_order=K=2`。総evolution時間qδ、event数q*r。δ/qをstep時間にしない |
| law/τ | 同じ保存signal τ64とorder weights。`make_rte_config`はtail.lambda_r*(δ/r)を使う。元opt2のmodel τや別閾値を代用しない。既存`signal_config`のhex照合を再利用できるが、local probe用q1/2 request関数を長qへ流用しない |
| event | n∈{0,2}。抽選tupleは(rotation,product₁,…,productₙ)、回路のapplication順は(product₁,…,productₙ,rotation)。n+1個のcomponent applicationなのでq*rはgate/application数ではない |
| sign/phase | Pは係数signを含むinvolution。rotation=exp(−iθP)、θ=atan(τ/(n+1))、event phase=(-1)^(n/2)。符号・非可換product順・event phaseを保存する |
| controlled指定 | logical diag(I,V_T)、controlled=True、system0..7・ancilla8。scalar/identity/event phaseはcontrol branchの相対phaseとして保存する |
| F/Rと反復 | 各stepは時間順F→r events→R、q回反復。matrix actionはR E_r…E_1 F。Rは同符号halfの逆sweepであってF†ではない |
| 構成policy | 今回の平均保証へ接続するのは`boundary_optimized`／`fuse_equal_boundary_block_and_aggregate_step_phases`。maker既定の`raw_concatenation`と区別し、任意切替やglobal phase省略をしない。boundary phase差は既存Δ_meanに計上済みで再加算しない |
| 鮮度 | fresh_per_shotで各callに新しいdraw。axis内でtrajectoryを固定してN回測定するreuse_within_axisは今回のIID平均保証の対象外 |

### run_le_1の適用単位

`apply_run_threshold_policy`は**q個の各occurrenceごと**に`make_run_threshold_basis_plan`を呼ぶ。
occurrence内ではevent境界を越えてapplication_sequenceをflattenし、source basis ID＋hashが同じ
maximal runを判定する。run長≤1だけsupport-restricted basis、長>1は登録full basis。
identity applicationはrunを切る。今回の保存218成分はidentity抽出後の条件に固定する。
保存training fingerprintは`9dcdc4a28866dc223afb0ea212df645a3ed4a28e5c6dad6ddfc0ec1fcb28ba9e`。

step境界はresetする。例えば同じAが2つの隣接eventに現れれば同step内はrun長2だが、
別stepの末尾A・先頭Aをglobalに連結して長2へ変更しない。
deterministic boundaryの最適化は、このRTE policyの評価単位を変更しない。
既存`make_df_partial_s2_repeated_request`のdefaultはbasis plan=Noneなので、呼ぶだけではrun_le_1が適用されない。

### Seed・freshness・cache

既存repeated samplerはq1でmaster seedをそのまま使い、q>1ではPython `random.Random(master_seed)`の
randrange(0,2^63)でstep seedsを作り、各occurrence内をNumPy PCG64で抽選する。
`_seed_for_shot`のSHA-derived seedやfresh-IID診断のstream_seedは別の実装規則である。
request key→seedのbindingを明示し、既存規則を混ぜない。
seed衝突検査は技術検査であり、非衝突でも数学的独立性を証明しない。

cache可能なのは固定のimmutable preparation、law、basis descriptor、同じ指定Tの合成結果など。
fresh drawを省略するtrajectory cache、outcome cache、同じTで複数shotの実験への置換はしない。
抽選結果が偶然同じことは合法で、trajectory内容やobject IDの重複を機械的に禁止しない。
同じTの回路を再利用する場合も、drawを完了した後の同一descriptorとの照合とfresh measurement/resetが必要。

## 2. uniform53 lawと実sampler

`rte.iter_sample_rte_events`はPCG64 Generator.choiceを用い、先にorderを選び、
そのn+1個のcomponentをwith-replacementで選ぶ。component確率はmath.fsumで再正規化する。
[NumPy1.26.4 SOURCE](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_generator.pyx#L841)
のp付きreplace経路はfloat64 cumsum→最終値で正規化→random→searchsorted(side='right')。
固定SOURCEを実際に取得して確認した。

[PCG64のdouble入口](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_pcg64.pyx)と
[uint64_to_double](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_common.pxd)は
64-bit値の上位53bitを2^53で割る。従って出力gridとの対応を説明できるが、
固定seedからの値列が独立uniformであることはこの変換から従わない。

既存`mean_adapter.categorical`は同じCDFと、独立U=j/2^53 (j=0..2^53−1)を仮定した
bin質量(ceil(2^53 c_i)−ceil(2^53 c_(i−1)))/2^53を定義する。
CDF境界と一致するUは次binへ進む。保存lawはこの離散lawであり、連続uniformや表示確率そのものではない。

**同じcomponent順/確率binary64、fsum、order確率、NumPy choice経路/CDF/境界規則で、
各drawを独立uniform53と仮定するときに限り条件付き対応。**
固定PCG64＋二段seed列の数学的IID性は未保証で、今回研究しない。
保存H4 lawのCDF・massは再計算せず、既存ファイルのbyte identityだけを確認した。
新しいrandom samplerは作らない。

## 3. measure_one(request,T) → 組込みint ±1

入力requestとTのq/r/K/δ/axis/law/state/policyが一致していなければproviderは停止する。
測定結果は**`type(Y) is int and Y in (-1,1)`**。bool、NumPy scalar、float±1はこの保証契約から除外する。
生bitのschemaを固定し、bit0→+1、bit1→−1を検証後にPython intへ変換する。
値の無条件int化、非整数bitの丸め、counts/bitstringの先頭文字からの暗黙抽出はしない。
wrapperのmeasurement_clbit_index/registerとbackend結果のwire順を明示的に結び付ける。

条件付き期待値の責務は次の通り：

    E[Y_cos | T] = Re <ψ|V_T|ψ>,
    E[Y_sin | T] = Im <ψ|V_T|ψ>.

wrapperはancilla H→controlled V→（sineのみSdg）→H→Z。
systemは保存normalized ray、ancillaは|0〉の積状態で各shotを開始し、毎回reset/再初期化する。
同じTに複数の量子shotを割り当ててfresh trajectory shot数Nを置換しない。
各軸内のIID測定と独立なtrajectory drawを仮定する。union boundはaxis/round間独立を必要としない。

system qubits0..7、ancilla8はQiskit little-endianの最高bit。
full statevector indexは(system integer j)+(ancilla bit<<8)。保存sectorの70成分は
保存basis_indices/ordered_sectorで256次元へ埋め、OpenFermion sector順とのpermutationを省かない。
hash一致だけではbackend浮動小数点正規化やstate actionの誤差上界にはならない。
初期状態は保存normalized rayを前提とし、実状態準備誤差・費用の研究は今回行わない。

constant、抽出identity、event位相は既存step/repeated builderと同じく一度だけ保持する。
controlled gateに昇格した際にglobal phaseを捨てず、無条件global phaseで相対phaseを代用しない。
attenuationをB_K^(rq)で戻したcorrected signalを±1測定平均へ入れない。
仮想Born samplerのclip/abort allowanceは、そのままideal測定の精度保証ではない。

既存collectはこのdtype契約を強制しない。新規テスト中のstrict_int_fixtureは差を示すfixtureだけで、
production wrapperや既存interfaceの修正ではない。

## 4. 整数和からscalar decoder

組込みint ±1ならN個の和SはPython任意精度整数としてexact、−N≤S≤N、S≡N(mod2)。
固定Nはmetadataとして261..510であり、旧証明の1≤N≤2^52内。
旧Float64追補のbinary64 RN/ties-even・通常の演算順序・有限性等の条件下で、
`S/N`は最後のbinary64除算による座標丸めだけを持ち、絶対誤差≤u=2^−53。
このN上限は数学的な証明域であり、CPU/RAM/shot数を新たに制限するものではない。
`2.0*k/N−1.0`のcount-grid経路は別で、旧上界3u。N3の人工例でもbit値が異なる。

既存collectは2個のfloatを作り、complex(cosine,sine)が成分を保持する。
有限標本complex値の半径は1を超え得るのでdiskへclampしない。
MeasuredMeansへの外部直接入力は有限・各座標rangeだけを検査し、整数標本格子やdtypeを証明しない。

`decode_measurements`は順序m=0からの連続q=2^m、同じld/δ/settingsと結果数を検査するが、
長さ18を必須にする汎用関数ではない。全18の完了確認は外部callerに必要。
zero complexならZERO_SIGNALでArg(0)を呼ばない。

[CPython3.11.0rc1 SOURCE](https://github.com/python/cpython/blob/v3.11.0rc1/Modules/cmathmodule.c#L1010)
では`cmath.phase`→c_atan2→通常の有限・虚部非zeroではlibm atan2。
虚部±0等は専用分岐でsigned zero/±πを返す。
**`np.angle`の人工有限格子結果をcmath.phaseの一様保証として採用しない。**
同じlibmへの既存binding観測も、新しい一様精度証明ではない。

次のscalar `reconstruct_rpe_phase`は最初のwrap、q個の全candidate生成、
NumPy complex128 exp→angle→absの距離、argminを使う。
最初のphase抽出がcmathであることと、内部距離でnp.angleを使うことを区別する。
isclose(abs_tol=1e-12)のtie/alias診断は証明ではなく、元branch選択を変えずenergyを保留するだけ。
負実軸±0、±π近傍、tie、zero、callback例外ではstatus/STOPを保全する。

最終energyは`-final/δ`、alias windowはphase[-π,π)、energy(−π/δ,π/δ]。
alias priorの正当性、最終lift、diagnosticがG上で誤停止しないこと、float energy除算も実装保証に必要。
数学的Gは**exact標本平均**の事象であり、丸め後座標/Arg/decoderへの移送は別の誤差証明を要する。
旧a/c/dとFloat64 STOPを維持し、uだけで全decoderを認定しない。
