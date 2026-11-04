# 次期最小科学検証案：float state-actionの上界と測定完了域

2026-11-04 JST。**未実行の計画案。新しい科学実行SOURCEは未完成・未freezeで、現在は実行可能状態ではない。**
今回のcommitは静的preflight SOURCEであり、実H4の数値評価を起動するSOURCEではない。
別途、必要最小限のadapterと人工検査を完成させた後、固定scopeに対するユーザーの科学実行許可が必要。

## 候補比較と第一候補

| 候補 | 解消する不足 / 既存成果で代用できる範囲 | 新計算・入力・精度・負荷 | 判定力と残る制約 |
|---|---|---|---|
| **1：float64 state-action平均差と完了域** | raw norm、P/F/R/trig/runtime丸め、guard完了を調べる。ideal包含やrun02では代用不可 | 保存export/local matrix・F/R64・state・lawと新roundoff adapter。70次元行列の残差包含、少数scalarのq/r伝播。96 bitから。直接trajectory不要 | 現在nullのModel II bridgeを前進させる。guard保証・上界の鋭さが不足すれば未判定。PRNG/実機/他roundは残る |
| 2：確率とBernoulli有限grid | 受理されたtに対する差は解析的3·2^-53で整理済み。人工exact rationalで検査 | 新H4計算不要。binary64/RNEと独立一様measurement wordが前提 | この項だけ確定してもstate-actionとguardがnullなので局所認定は進まない。候補1へ組み込む |
| 3：PCG64とIID | stream設計・有限衝突検査と確率仮定を既存SOURCEから区別できる | 今回の静的整理で足りる。経験的PRNGテスト増加は推奨しない | 数学的IID保証は得られない。Model IIの条件付き保証とModel IIIの妥当性を別に報告 |

**第一候補は1**。数学的IID wordモデルを維持し、既存数値作用との差をboundする。
科学的な問いは「この固定m17 float contextで、既存検査を全supportで通り、
保存N/α/εに必要な局所mean誤差上界を置けるか」。
全support完了の証明が得られなければ、数値誤差majorantが得られた点と測定平均未定義の点を分けて終了する。

## 固定条件・既存入力

linear H4、1.00 Å、STO-3G、保存legacy DF12、B state、4電子sector70、
L_D3、単一δ=.02、m17/q131072/r32/K2、extract_identity_phase、cutoff1e-12。
N411/axis、α=.016957089147912167、ε=.15244325851620388、保存μ0/PF/RTE/stat予算とβを保持。
μ0は保存binary64、α/εは既存包含と同じdecimal文字列のexact rationalとする。
既存sの保存包含を再利用し、今回のpreflightでは比較を再計算していない。

| 入力 | 固定commit / path | 用途・保全 |
|---|---|---|
| latest4原本 | `34840ce9bf81d985b32764d9bc9d77645ed52e59` / `artifacts/rpe_m17_rigorous_mean_enclosure/2026-11-03/run01/` | RESULT.json、source_replay.json、categorical_laws.json、enclosure_operators.zip。hashはVALIDATION.txt。新replay/CDFなしで読む |
| raw ψ | `e22d7718c47358ff650dbc04e8e72f6ab1c8bf41` / `artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/physical_state.npz` | SHA256 `793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4`。state/sectorをlatest exportと照合、書換えなし |
| numeric F/R64 | `cd749066904d3f29b28b03c19ab69370372b157f` / `artifacts/rpe_m17_mean_validation/2026-11-03/run01/operators.npz` | SHA256 `e1a65238e9d9b61a6d0aa542ca7f509827b0a761fc751615f1f2018a0e7020df`。forward/reverseをexact dyadicとして比較。保存A/tail平均を実装誤差の代用にしない |
| 保存μ0/N/α/β/ε | 同元m17 RESULTとV2 `78c49d0b5f1d71e63b5959e78813237923bbc648`のround17行 | literal identity確認。再推定・最適化・他roundなし |
| SOURCE | science `18fffce48e33136b91eb6ac5a6f9a5f77a08d36f`、A `d0b1da1b04c266a1f7abf8e43966ca6e289a8695`、carrier `4591b8d75a2a6a55c74a51225f0c61648023c8ee` | ideal gate/ball、数値basis/P組立て、作用/測定順を再利用。旧科学runnerの--runは使わない |

### 新m17 contextの明示

run02のcompleted numeric payloadは保存されていない。preparation/tail hash一致で同一性を補わない。
次期案では、**上表のF/R64、Bのraw ψ、latest exportのlocal binary64 matrixから旧compositionで構成するP_i64**を
新しいm17仮想数値contextとして固定する。旧run02の履歴的再現を保証する実験にはしない。
constant/identity係数を丸め済み角度から逆算せず、保存F/R64を使ってscalar差もまとめて包含する。

新SOURCEをfreezeする前に、行列順/sector permutation/成分ID/signと、保存m17 order vector/θ64の対応を
人工fixtureと静的SOURCEで固定する。実データによるidentity・norm・差の検証は科学実行scopeへ入れる。
そのcontextを将来の実測定runnerに使う場合にも、この入力定義を引き継ぐ必要がある。
旧Aのr1 `prepare_context`/`execute`をm17として起動しない。

## 再利用と必要最小限の追加実装

| 既存実装 | 適合する再利用 / 差分 |
|---|---|
| export_adapter.py | exportのschema、exact binary64表現、sector/ID/operation対応。`replay()`は呼ばない |
| mean_adapter.py | ideal basisとsigned diagonalの構成。保存lawを読み、CDF再生成・新meanのq乗をしない |
| ball_adapter.py | exact rational→ball、norm/global radius、外向きendpoint・区間比較。新しいruntime丸めmajorantを小さく追加 |
| run_fresh_iid_a_v2.py | numeric basis composition/P組立てをraw local matrix入力に限定適用。旧全体preparationは再利用しない |
| legacy AST作用関数・conditional_probability | 演算順と範囲検査の規範。人工テストへ直接再利用し、実H4 trajectoryは呼ばない |
| 今回のbridge_math.py | exact grid式・growth伝播・null/strict比較の人工仕様。Fraction巨大冪を実H4 production backendとして使用しない |

必要な新機能は、固定numeric contextの構成、idealとの差の外向きnorm上端、
runtime matvec/vector演算/vdotの一様丸め上界、平均majorantと全support guard majorantの区別、結果保存だけ。
承認証明システム・grant・ledger・一般的runner基盤は新設しない。

**先に解決する技術条件**：実際に使うcomplex128 kernelについて、丸めのモデル・加算深さ/演算数・FMA、
subnormal/overflowの扱いから、状態一様な係数を正当化する。
全依存libraryのauditは求めず、結論を支える@/vdot/scale/subtract/trigとguardに絞る。
根拠を確定できなければ、仮のγ係数で保証を宣言せず、実H4数値評価の前に不足を報告する。
別の算術kernelに変更する場合は新contextとして明示し、旧NumPy実装との同一性を主張しない。

## 提案する新科学callの範囲

以下は**将来の別許可対象**であり、今回実行していない。
新SOURCEが完成した時点でこの範囲を具体化し、公開commitを示して実行許可を受ける。

1. 固定inputsの数値読込と照合を1回。latestの成分順・law/θ/sectorと新contextを対応付ける。
2. 保存local matrixからnumeric basis11個を共有構成し、P_i64を218個構成する処理を1回。
   F/R64は保存行列を直接使用する。math.cos/sinは保存θに対する有限個の出力を1回取得し保存する。
   SCF、DF因子化、H生成、state solve、Gaussian gate再生成、controlled回路生成は0。
3. 1つのbound pass内で、ideal component/basisとP_i64、保存ideal F/R包含とF/R64、
   trig出力、保存norm²、runtime丸めを外向きにboundする。
   Componentごとのresidual/必要normはこの1 passの内部対象とし、途中で候補を増やさない。
4. [BRIDGE_EVIDENCE_AND_GAPS.md](BRIDGE_EVIDENCE_AND_GAPS.md)の式で平均誤差majorantと、
   全supportのsignal/guard majorantを**別に**伝播する。q/rはscalar binary powerで扱い、q×r state-actionを実行しない。
5. 全support完了・law/context identity・算術条件が揃えば、保存logical intervalからU_Lを新規算出し、
   action/raw boundと解析3uを加えたModel IIのU_virtualを保存s/εへstrict比較する。
   完了不明ならU_virtual/U_execはnullのまま。可能な部分上界のみ保存する。
6. 正常、幅不足、未保証kernel、domain不明、不整合、中断の全てで原本を保存してmandatory STOP。

既存Acb/Arbとpython-flint0.9.0/FLINT3.6.0、binary64/complex128 NumPy1.26.4を基準にする。
bound passは96 bitで1回。**包含の丸め幅だけ**で判定できない場合に限り160 bit、必要なら224 bitへ進む案とする。
max3 pass。numeric P/basis/trig生成はprecisionごとにやり直さない。
guard上界やmajorant自体の保守性、欠落するkernel根拠、入力identityが原因なら高精度反復せず終了。
これは新しい契約案の上限であり、旧one-shotの再開/許可流用ではない。

新しいCDF構成、event全列挙、直接m17 trajectory、sampling/RNG replay/仮想shot、B-1b/B-2、
他round、compile/noise、成功率・費用再計算は全て0とする。
将来出力は専用fresh directoryへ、SOURCE・入力・数値context・係数の前提・区間上端・部分結果を保存する。
今回そのdirectoryや実行markerを作っていない。

## 判定規則と研究判断

| 新しい局所結果の種類 | 必要な根拠 / 解釈 |
|---|---|
| `CONDITIONAL_VIRTUAL_IID_SUFFICIENT_CONDITION_CONFIRMED` | law/context identity、明示算術条件、全supportの検査完了が揃い、両軸upper(U_virtual+s)<lower(ε)。Model IIだけの局所条件 |
| `ARITHMETIC_BOUND_OBTAINED_BUT_COMPLETION_UNRESOLVED` | 数値作用の上界は得たがguard通過を全supportで示せない。ν_virtualの無条件meanを認定せずU_exec=null |
| `NOT_CERTIFIED_BY_THIS_BOUND` | 有効な上界がstrict条件を満たすことを示せない。RPE失敗や真のmean誤差超過の証明ではない |
| `UNDETERMINED` | 入力対応、kernel条件、radius等が不足。未評価項を0にしない |
| `INTERRUPTED_OR_INVALID_MANDATORY_STOP` | 入力不整合・非有限・exception・中断。partial結果を保存し、retry/resumeを自動開始しない |

完了が不明な場合、失敗確率を新たに割り当てたり、αを変えたり、拒否shotを除外したりしない。
それらは別の実装/確率モデル・予算を要する研究判断なので、必要性だけ報告して終了する。
有効な平均上界でも、guardの1e-12 allowanceより緩ければ完了保証にならない点が判定力の制限。
一方、Model IIの両条件を確認できれば、数値実装誤差が局所条件を妨げないという新しい科学的情報が得られる。
どの結果でも固定PCG64のIID性、全18round/最終energy・総費用は未評価のまま。

## 性能・承認・STOP

支配的処理はbasisを共有したcomponent行列構成と70次元residual/norm包含。
11 basisをcacheし、独立componentの評価をbatch/並列化、q/rの長列をscalar冪へ置き換える。
既存包含の約3.717秒は旧ideal passの記録で、新しい218 residualのruntimeや速度倍率の予測に使わない。
実行前にCPU affinity、空きRAM、GPU可用性、共有負荷を確認し、利用できる資源へworker/BLAS/FLINTを合わせる。
CPU/RAM/worker/wall-timeの人工capは置かず、実quotaと非干渉を守り、自jobだけ調整する。

次の担当：**Codex**が新context/kernel boundの最小adapterと人工検査をまとめ、実行SOURCEを公開固定する（次段の提案）。
その具体物に対し、**ユーザー**が上記m17単一セル・最大call範囲の新科学実行を明示許可する。
許可は会話の指示で足り、承認原文保存・ID・証明書・grant発行は不要。
このpreflightや方針の採択だけでは実H4計算を開始しない。重要科学レビューを行うならその許可は別。
今回の資料をcommit/push・remote照合した後にmandatory STOPし、技術準備や科学実行を自動開始しない。
