# Hamiltonian表現探索の初期検証範囲

2026-10-10 JST。利用者の初期探索依頼による独立系列で、A/B/Cの機構比較を行う。
中心仮説・研究テーマ・新規性は採択しない。Track A/Bのsource、実験、契約、STOPは変更しない。

## 入力と来歴

基点は `b2e1bf65e21893b6c617223b42313623d3186f12`、branchは `representation-exploration-20261010`。
mainのremote確認値は `0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf`。
利用者の[指示全文](representation_exploration_inputs/user_request_20261010.txt)と
[GPT初期レビュー](representation_exploration_inputs/PR_representation_codesign_research_review_2026-10-10.md)
を入力として保存する。mainから最新研究結果を推定せず、Track A/Bの別commitを読み取り専用で監査した。

この依頼は小規模数値の実行を含む。旧研究の次段authorizationを流用しない。
本系列にも大規模分子計算や次研究stageの認可は含まれない。

## 最小の判別実験

Aは正のsquare-factorをラベル空間で混合する。2または3 fermionic modes、JW full Fock space、rank2、
one-body/constantゼロ、L_D=1、delta=0.05/0.2/0.4。
4つの人工入力、5つの固定角度 0、atan(0.1)、±pi/8、pi/4 を比較する。
reviewの2次元例は既知の代数の再現である。別のGram対角入力を用いて、
Frobenius最適性と具体的I/Z/ZZ tail normがずれるか判別する。
元factorの全L_D=1選択、可換対照、回転不変対照、未吸収正係数・符号付き誤変換を併記する。
共有computational frameと正確なPauli残差の構成も一例保存する。

Bは1 system qubit+1 auxiliary qubit、P=aux vacuum、L_D=0、r=1、
H_tilde=0.7 I_aux Z_sys+0.2 Z_aux X_sys+0.4 X_aux X_sys。
K=0/2/4、t=0.05/0.2/-0.2。6項の未統合反射辞書を既存paired-Taylor RTEで全列挙する。
補正平均、平均channel、個々のleakage、全word共役の誤実装、Taylor bias、B²、
有限標本leakageの正確な二次momentを分離する。
N=1024のRMSは列挙momentからの予測であり、MCや量子shotは実行しない。
Hoeffding診断の各軸shotは2B²/epsilon² ln(4/alpha_total)、epsilon=.05、alpha_total=.05。
Taylor biasを予算へ戻した最終shot計画ではない。
既知echoの密行列参照と、1/2/3 auxiliary vacuum reflectionのcompiled primitiveも調べる。

Cは2 system qubits、A=Z0+0.7Z1+0.3Z0Z1、B0=X0、B1=X1、t=.2、alpha=.05/.1/.2。
各混合propagatorをspectatorの2枝のSU(2)回転で厳密に実装する。
既知THRIFT一次式、通常一次PF、混合oracleを通常splitで代用した式を比較する。
Lie closureは補助診断とし、密行列対角化を効率的oracleと呼ばない。

## 資源と解釈

CPU1、worker1、BLAS/OpenMP/Numba各1。wall300秒、CPU240秒、address space4GiB、
1 output file16MiB、compile64件、B各rowイベント10000件、回路最大5 qubitsをguardする。
数値は倍精度。codeのequivalence toleranceは2e-10、identity testは1e-12級。
物理geometry/basisと化学DF rank policyは該当しない。分子NPZ、ground state、GPU、quantum shotsは使用しない。

compileはQiskit1.3.0、opt1、basis rz/sx/x/cx、seed20261010、topologyなし、ordinary control。
ancillaがqubit0なのでcontrolled行列はinterleaved indices。global phaseも同値検査する。
Aはdelta=.2のcontrolled exact-tail S2 wrapperであり、実RTE wrapperや精度一致総資源ではない。
Bはreflection primitive、Cは既知mixed/ordinary wrapper。state preparationと測定は含めない。
モデル保存、PF/Taylor誤差、single-shot cost、shot proxy、総費用を混同しない。

Exact-data開発例であり、未知データholdoutや外部再現ではない。例・角度は結果前に固定し、
全rowを公開する。割合をGO閾値にせず、重要な機構・反例・比較材料が揃ったらSTOPしてGPTへ戻す。

## Sourceと再現入口

- [library](../../src/trottertracks/representation_exploration/mechanisms.py)
- [runner](../../scripts/run_representation_exploration.py)
- [tests](../../tests/test_representation_exploration.py)
- result出力先 `artifacts/representation_exploration/2026-10-10/run2/`

sourceをcommit固定してからrunnerを実行し、実source SHA、依存package版、時間、peak RSS、output SHAを保存する。
旧DF screeningやUWC prose数値は根拠に使用しない。

## 初回technical STOPと修正版の範囲

source451bfa5のrun1はBのcontrolled-X primitive compile同値検査で停止した。
残差sqrt(2)はscalar phaseのみで、独立したRX/RY/CRX/CRY minimal circuitにも再現した。
phase lossの一般的原因や他環境への影響は未確定で、既存研究の証拠に一般化しない。
run1のfailure/auditを保持し、入力・角度・delta・候補・資源上限を変えずrun2へ進む。

修正はtoy compile監査に限定する。built circuitとcompiled matrixの差がunit-modulus scalarだけであることを
行列normで認証した場合に限りcompiled.global_phaseを補償し、補償後と追加control後の絶対同値性を検査する。
non-scalar/relative-branch errorは拒否する。native gate countはこのscalar metadata変更で変わらない。
このdense certificateは小系検証用であり、truth-freeな大規模compiler修正アルゴリズムではない。
