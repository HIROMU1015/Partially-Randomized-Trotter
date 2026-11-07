# 案M：新しい誤差とfull費用の境界

2026-11-08。人工correctnessは成立したが、**実H4の新instrumentの数値誤差上界・必要shot数・期待費用はUNKNOWN**。旧engineの認証値や旧CVのrange/variance式を移植しない。

## 誤差

| 項目 | 今回確認したこと | 実科学前に不足すること |
|---|---|---|
| actual finite-grid Q / q | 人工の整数grid質量と群内Fraction比、rの正確な有理和 | 保存N1 lawとのbinding、全正support被覆、群/外側mass保存 |
| target | 共通A/B/V/φ/phaseの構造一致、人工の元controlled信号との照合 | 実selected policyからの厳密なslot抽出。元basisとの近似一致を根拠にしない |
| A† / control / V | 人工全branch行列、global/relative phaseと順序 | 新しい制御gate列、inverseの数値生成・適用・合成の全support保証 |
| r(z) | 与えられたFraction入力では正確。|r|≤1は解析的性質 | 実lawの由来、計算/保存/cacheのidentityと費用 |
| m(z) / 軸 | 小規模で式とdynamic命令が一致。記号的quarter turnを0へ扱う | atan2/hypot/trig、floatへの変換、丸めの一様誤差。未保証項を0にしない |
| system readout | computational measurementの全分岐を人工列挙 | system測定確率、n-bit joint outcome/gridと破壊性の実運用保証 |
| ancilla readout / feed-forward | X/Y符号、Sdg→H、if_elseのモデル上の意味 | backend対応、readout/latency、制御角誤り。実機対応とは未認定 |
| Born / normalization / output | 人工分岐総質量・decoderを照合。隠れたpostselectionなし | 実engineのBorn計算、状態誤差との合成、weight/normalization・出力丸め |
| finite confidence | 数学的にProductの出力絶対値≤1、Rotation m≤1 | binary64での外向range enclosure、各座標の保証、calibration/α配分。clippingで救済しない |

N2の保存ground自体を再solveする必要はないが、その旧state誤差を新しいweighted instrumentへ移す評価は別途必要。旧exact Fraction engineはsystem測定・z依存軸を認証していない。今回のbinary64 prototypeを科学kernelへ正式採択していない。総誤差の形式は、state、target binding、gate、axis/weight、system/ancilla Born、bit-grid、output等を漏れなく含めるものとし、数値の総上界はまだ設定しない。

## 記号的な費用会計

元full-wrapper価格をC_old(η,l)、案MをC_M(η,g,z)と置く。新価格には少なくとも次を含める。

\[
 C_M=C_{H_a}+C_{\mathrm{ctrl0}(A^\dagger)}+C_{\mathrm{ctrl1}(B)}
     +C_V+C_{\mathrm{system\ readout}}
     +C_{\mathrm{feedforward/axis}(z)}+C_{\mathrm{ancilla\ readout}}.
\]

これは説明用の分解であり、各項のcompiled価格を加算すればfull価格になるとは限らない。full circuitでの相殺・routing・synthesis・phaseを含めて測る必要がある。Productのfeed-forward軸はzに依存しないが、Rotationは依存する。現在の人工実装は2^n個のz条件を明示するだけであり、大規模系への効率的なfeed-forward実装を実証していない。

元のcontrolled label gateは取り除けても、A†/Bを別枝へ移す費用、Vの位置変更による境界相殺の損失、system全bit readout、追加軸RZとlatencyで相殺し得る。入力状態は破壊され、attemptごとにfreshに準備する必要がある。状態準備を除外する場合も、それは研究scopeの仮定であり、shot数が異なる候補間で数学的に相殺されるからではない。

群署名・群質量・conditional lawの生成、r(z)のO(|g|)符号和、軸/weight、metadata/serialization/cache、実行管理の古典費を別記する。full-wrapper資源は既存`df_rpe_hadamard_compiled_cost.py`のno-prep接口を参照するが、新しいmid-circuit/feed-forwardへそのprofileが適合するかは未知。実H4のbuild/compile/native価格は今回0。

総資源比較は、同じtarget/精度/失敗率で、全段階のNと期待full価格、価格取得・学習/calibration・古典費まで揃えた後に行う。一回の実現費用と期待費用保証を区別する。今回Nを設定・算出しない。

## 統計的利益の限界とbaseline

元±1測定に対しProductのM2差はE[1−r²]、Rotationはsin²φ E[1−r²]。平均を保持するので同じ条件付きlawと取得数なら分散差もこのM2差になるが、実H4のその値は未知。小角度では後者が二次に小さく、sup rangeが1のままの場合もある。人工例のM2を実H4の母分散や十分Nへ転用しない。

比較対象はlabel別controlled、単独labelの直接測定、共通basis grouping、同じ外側lawの層別である。平均observable grouping自体は既知の機構であり、案Mと同じreadoutなら統計的に同じものになる。A=B†ならancilla-free直接測定/groupingも可能で、今回の人工例では案Mより同等以上だった。この条件を有料の学習で推測する必要はなく、該当する明示構造があればbaselineに含める。

したがって今回の技術的成果は**coherent targetへの接続と適格性・instrument correctness**であり、強い既知法への優位性や独立新規性ではない。次のmetadata-only確認で適格群が見つかっても、それだけで科学的GOや測定campaign開始にしない。
