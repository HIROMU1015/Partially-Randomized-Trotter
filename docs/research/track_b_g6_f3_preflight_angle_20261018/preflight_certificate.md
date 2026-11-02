# P0〜P5の実装と証拠

正本は固定[F1 §§1–5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/preflight_precision_contract.md)。実装は[precision](../../../src/trottertracks/algorithm_codesign/g6_f3_precision.py)、入口/P5実generator接続は[wrapper](../../../src/trottertracks/algorithm_codesign/g6_f3_generator.py)。以下は固定述語のコード対応であり、新しい十分定数/定理/selectorではない。

## 1. 記号・exact比較

提供値から`Δ=2^(-K_root)`、`h=2^(-H)`、`n_o=(m+1)//2`、`p_min=min(p)`、`t_min=x^(m−1)/(m−1)!`、`q_min=t_min/3`、`d_min=t_min*p_min^(m−1)/2`をFraction/integerだけで作る。浮動小数点比較を使わない。x=0ではこれらを評価しない。

R(ζ,q_min)は正確に次を検査する。

```text
0 <= ζ < 1
h/q_min <= (1−ζ)/(1+ζ) − (1−η)(1+ζ)
h/q_min <= (1+η)(1−ζ) − (1+ζ)/(1−ζ)
```

ζ≥1なら分母1−ζを評価せずFAIL。各checkはname、left、operator、right、passedを持つ。不等式`<`と`<=`を区別し、等号は仕様どおり扱う。guardを通すための丸め・clipping変更はない。

| Predicate | 実装する条件・証拠 | 旧guardとの対応 |
|---|---|---|
| P0 | make_requestのstrict域確認、validated exact domain、fresh-IID/exact controlled-Qが**仮定**であること | frozen kernel/full domainを狭く安全に包む。Qのinvolution性を整数labelから認証しない |
| P1 | e=n_oΔ<1、a=Δ/t_min、ζ_o=(a+e)/(1−e)とR(ζ_o,q_min) | order rootsとnormalized区間のsupport/relative guard。e≥1はζ=NoneでFAIL |
| P2 | h≤ηp_min、h≤ηp_min/2 | labelと全valid parentのchild一点区間dyadic guard |
| P3 | Δ<d_min、Δ/2≤ρ(d_min−Δ) | positive root.lo、halfwidth≤ρlo。等号を含む後者とstrict前者を区別 |
| P4 | Δ<1/4、(2−η)Δ+h≤η/4 | clipped acceptance区間とdownward floorの既存relative guard |
| P5 | input/policy/SOURCEと実generator/kernel/law state binding | order/labelsを同じ凍結関数から別途作り、actualと照合。child lawは同じsealed kernel/固定digital_parentとP2/P3/P4に結び付く |

P5で実p/x/m/H/K/η/ρ/arm、kernel p/x/m、実order/label law、kernel係数配列、B/U区間のstate fingerprintを保持する。SOURCE pinとimmutable requestを各callで再照合する。child lawは訪問時に旧packetが作り、全child lawを先に列挙しない。generator constructorが成功しただけでpreflight PASSにはしない。

## 2. 全parentへの保証の範囲

未訪問を含むFullの全到達valid parentの条件は、固定F1が引用するorder/child/d/Aの一様下界とP0〜P5の**論理的接続**に依存する。人工testの有限parent標本から帰納したものではない。L=1では到達するeven reduced parentは空語、Full m=1/x=1も既存域。canonicalには今回適用しない。

`assess_precision()`はP0〜P4の評価だけを返す。P5未接続の証拠は`ready=false`。prepareが同一精度で旧Fullを構築・P5照合して初めて利用可能になる。constructorのU計算は既存どおり**空parentを1回だけ**使うが、未訪問parentのroot辞書は作らない。test spyでnonempty parentへのprepare時アクセスがないことを確認した。

## 3. EvidenceとSTOP

predicateごとのid/status/passed/checks/bounds/reason、固定spec/proof/source、request、input/policy/source identity、Δ/h、dimensionsを保存する。arithmetic不足は`PREFLIGHT_CONDITIONS_FAILED`をconstructor前に返す。P5照合失敗は`ACTUAL_LAW_BINDING_FAILED`。通過後のlocal guardのValueErrorは`LEGACY_GUARD_AFTER_PREFLIGHT`として元stage/位置を保存してterminal STOPする。bit数を変えない。

テストでは不足H、不足Kでe≥1、R/P3/P4の等号/strict境界、root外向き端点、選択parentのlocal guards、P5 law改変を確認した。不整合分類のfault-injectionは意図的に例外を挿入した**人工テスト**であり、F1と旧SOURCEの実際の矛盾を発見したわけではない。今回の無改変旧SOURCE/artificial fixtureには、その矛盾は見つからなかった。

p=(2/5,3/5),x=1/2,m=3の主fixtureにK_root=128/H=96、η=1/1000、ρ=1/10^6を事前に与えた。m=5の選択parent、L=1/m=1/x=1も検査した。これはdefaultsや全入力での成功保証、最小bit値の探索ではない。**F1の十分条件がFAILでも数学的な不可能性とは分類しない。**

fresh IID、exact controlled-Q、Python算術/実行環境、全run完了や物理providerの無故障は別前提。**mandatory STOP。**
