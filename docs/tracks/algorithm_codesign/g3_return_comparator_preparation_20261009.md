# G3 Phase B：条件付きknown return比較の固定準備

2026-10-09。認可元はGPT G2 review §8。Phase A sourceは
`381449daefc7310e22729725f69656dd651b3d20`。
288 profile・6,912有限lawを完了し、固定対照集合でfinite certified T/1Q差とnondominationが残った。
この差が既知returnで吸収されるかはG3判断に直結するため、条件付きPhase Bを実施する。
科学的materiality/new-method GOや全B2最適性の判断を通過したという意味ではない。

## 数学と同じ実装規則

R=3/4 Q0+1/4 Q1、Q0=ZI、Q1=V† IZ V、V=exp(−iπXX/16)。
χ=Σp_i²=5/8。有限P3のexact identityは

`P3=(1−χx²/2)I−i(x−χx³/6)R−(x²/2)(I−ixR/3)(R²−χI)`。

zero-degree部分はa0=1−χx²/2、b0=x−χx³/6のpaired rotationとする。
ratioはx=1/8で3067/24456、x=1/4で763/3012。σ=+1のみ。
同じbasis/conjugator、controlled half-angle lowering、共有exact adjacent cancellationを使用。
off-diagonal O2はword01/10だけを条件付け、元のIR/count/errorをそのまま再利用する。
word conditionalは01/10各1/2、rotationは独立p=(3/4,1/4)。
normalized列の(-iR)^k係数にはlow-degree χ returnを含める。
この条件付き列を、旧O2の同じD2/D3だけの列として採点しない。

## 固定取得・比較集合

新規取得は2 labels×2 x×3 precisionの12 returned event IRと、対応する12 unique RZ synthesis keysのみ。
precision={1e-3,1e-4,1e-6}、pygridsynth2.0.0/runtime package/source identityは旧R1と完全一致を要求。
旧options（seed0,dps100,up_to_phase=false、ε/4 request）とstrict Frobenius error upperを保持。
global phaseのminimizationはしない。basis π/8 keysとoff-diagonal O2は旧保存値を使う。
新しいdictionary/angle grid/precision/CTS取得なし。

returnの2 groupsに同じ3 precisionを与える9 profiles/x。Phase Aと同じproposal生成器、
dyadic denominator2^60、exact rational reweight、mean/bias/m2/L・整数Bernstein shots・capを使う。
全3資源座標とworkspaceを保存する。T/CX readout0、1Q readout合計5/axis pair。
本比較も有限proposal/pure precision集合であり、任意混合・全IS・全samplerのglobal optimalityではない。
toyのχ取得2項と条件付きsampling構成費用を記録し、DF scaleでの古典取得優位は主張しない。

## 資源・停止・検査

one process、wall600s/CPU480s、RSS512MiB/AS1536MiB、per-key wall30s/CPU20s、periodic watchdog0.2s。
max12 synthesis calls、sequence最大20,000 chars、law最大1,000、packet output32MiB。
exclusive consumed marker、runs1/retries0。partial failureならprefixを最終比較に使わずSTOP。
登録外x=1/3の10 focused testsでexact mean、controlled relative phase、IR再利用、有限certificateを確認。
テストでsynthesis/registered-profile採点を行わない。
完了またはfailure後にmandatory STOP。研究採否・必要な次の検証はGPT G3へ戻す。
