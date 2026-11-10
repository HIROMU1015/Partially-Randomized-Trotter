# H4 DF10価格モデルV1：実行仕様

2026-11-10 JST。利用者が本V1の保存データ構造計数・価格予測・残差評価を明示許可した範囲を実施する。承認の証明書・grant発行は不要。本書はモデルと実行範囲を固定する科学仕様である。

- 親：`10132af4a67ce112f79c539a9552f5ca0e83e516`のREPORT §2–8。既存branchを継続。元price carrier155529e2、provider c31ef424、保存tol-only DF10/Linear H4/1Å/STO-3G/sector(2,2)、P=0、元H/state/snapshotを保持。
- 開発：PF B0/S2/p7、q8/T.8とPR S2/p1/R4/K2、q8/T.8探索9本。留保：同PF q16/T1.6、同PR q8/T.8確認33本。δ=.1、各X/Y、計44 path/circuit×2軸の保存価格。cohortをpoolしない。
- 全RPE条件の化学精度.00159360010199040 Ha、αtotal=.05/方式、βtotal=.4、h=.1、13roundは変更しない。今回N・signal・energy資格・N×Cは再評価しない。
- compiler：Qiskit1.3.0、rz/sx/x/cx、opt1/seed17、元full-wrapper/measurement規約。compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。DのP0/opt2等へ置換しない。

## 一つのsource由来モデル

`DF10_SOURCE_NATIVE_BASIS_CANCEL_ZFOLD_V1`を事前固定する。予測器は保存価格・candidateの観測mean・cohort名を入力に取らない。係数はfitしない。

1. 元providerの`diag_hermitian`、対角primitive関数、元`gaussian_operations`の関数本文を固定Git objectから再利用する。保存snapshotのone-body/最初の7 fragmentから同じhermitize・sortで局所記述子を取り出す。新SCF/DF fit/ground solveは行わない。GaussianのP/XX+YY gate objectと2×2/4×4行列は記述子取得専用で、実QuantumCircuitを作らない。
2. 元headのuncontrolled forwardとdirectional reverseをδ/2で反復。tailは保存eventのapplication順、full registered basis、同basis共有、identityによる共有の終了、rotation/product符号・relative phaseをそのまま会計する。元SOURCEのrequestはbasis planなしの既定full basisである。
3. GaussianのP/XX+YY、対角RZ/RZZ・controlled rotation、scalar P、X/Y wrapperを、固定した解析的rz/sx/x/cx/measureの局所展開で記述する。XX+YYはQiskit1.3の公式定義。P(t)=exp(it/2)RZ(t)、H=exp(iπ/4)RZ(π/2)SX RZ(π/2)、SXdg=X SX、RY(t)=SX RZ(−t)SXdg。CRZは元の2 RZ/2 CX、controlled RZZはそのparity展開。局所phaseを別に保持する。
4. 直に隣接した同じbasisの逆変換対だけを消去する。各wireで他gateに遮られないRZを加算し、合計がbinary64で厳密0のものだけを消す。近接角のsnap・許容誤差prune・CX相殺・一般1Q再synthesis・P0 passは行わない。
5. 再利用する`track_d_native_resources.NativeCounter`でこの展開列を計数する。`structural_rz`は上記限定rule後の候補価格、`expanded_rz_after_basis_cancellation`はRZ融合前の診断値。どちらも元compilerのpost-P0 countと同一認定しない。未知compiler correctionはnull、保存価格との差を数値残差として示す。

これは元回路の部分compile価格の和ではない。全interrogationの連結順序を保持するstream会計で、実QC・DAG・transpile・新native再synthesisは作らない。一般compilerを複製する開発へ広げない。特定の解析的native分解を採ったため、元basis translationの選択差もcompiled残差に含まれる。

## 手順と終了条件

- 最小人工テストで局所分解の位相付き作用、basis対消去の範囲、zero/同wire融合、tail共有・identity、scopeを確認する。人工入力だけで、分子価格を用いない。
- モデル・runner・本仕様を科学実行前にcommitし、**開発と留保で同じSOURCEとfile hash**を使う。開発結果を見た後もこのV1のrule/定数は変えない。留保価格は既知なので盲検とは呼ばない。
- `develop`でPF q8＋PR探索9本の各X/Yを一括評価しDEVELOPMENT.jsonへ保存。その同SOURCEの`holdout`でPF q16＋PR確認33本を評価しHOLDOUT.jsonへ保存。原本不足・binding不一致なら再sampling/buildで代用せず不足を報告する。
- 残差はcompiled−structuralの符号付き整数。各path・軸・方式を保持し、平均と最大絶対残差、0一致数を別に出す。X/Yを独立nとして数えない。残差があっても固定ruleを全対象へ適用して失敗範囲を保存し、後から補正しない。
- 全留保で整数残差0の場合だけ有限保存回路の一致を報告。そうでなければmodel discrepancyとして記録する。原本やSOURCEの不一致は価格モデルの反証と混同しない。
- 元input/raw/価格/cache/SOURCE/grant/ledger/marker/STOPへ書かない。新fit/bootstrap・sampling・実回路build/compile・signal/N/G・H6数値・長q・詳細科学レビューは0。
- REPORTとpath/axis残差・SOURCEを公開し、remote SHAと主要成果物を照合してmandatory STOP。全law・q4096・全13round・CIの資格へ昇格しない。
