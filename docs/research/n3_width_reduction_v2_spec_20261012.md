# N3 幅指向generator・一mode縮約task：技術仕様案 v2

2026-10-12 JST。科学レビュー第10.3–10.4節に従う設計資料。**幅generator／mode selectorは未実装・未実行。科学契約の採択と実行認可は別途。**
今回実装したのは保存証拠の解析的帰結checkerのみ。旧OCC v1 runnerを上書き・再実行しない。

## 入力と出力

Hは実対称one-body＋非負unordered density、ν固定、候補外部mode集合D_setを結果前に指定し、各候補は一modeだけ空へ射影する。
自動orbital rotation、多mode同時／逐次削除、一般二体は含めない。m_active≥νの候補だけを扱う。
入力はexact rational係数と単位、h/J/SHA、ν、D_set、正の射影誤差予算ε_proj、bound policy、有限budget、cache policy。
真のground/gap/stateや事後truth、既存結果を見たcandidate追加は禁止する。
各mode dのh_A,vと、取得・検証したa_d≤U_d、C_d≽c_d Iをbindする。g_d≤0ではBOUND_PRECONDITION_FAILED。

出力は候補ごとの最良発見済みの正の有理保証幅、元係数へbindしたcertificate、取得費用、未判定理由、採用法／origin、縮約H_A、mode順序写像。
全法同じ(U_d,c_d)で比較する。mode間でU/cが異なる場合はそれぞれの取得費用を記録し、同じbound値として混ぜない。
保証可能候補がなければ元Hと射影誤差0を返すfallbackを残す。未判定を「縮約不能」と表示しない。

## 保証族と有限generator

| 層 | 定義 | 主張できること |
|---|---|---|
| 理想族 | 任意実κ,τ,ZによるOCC可行性 | 数学的な十分保証の存在。数値solverの有限coverageとは別 |
| 冷生成 | 過去caseのcertificateを使わず同じ有限proposal/bit budgetを実行 | 当該sourceで発見できた保証幅と取得費用 |
| warm再利用 | 同一identityの保存κ,τ,Zを検証して利用 | 発見済み上端。冷生成の実績に数えない。今回のS1はこの層のdevelopment |
| operational OR | NORM/FIVE/OLDとraw OCCを安全に併用 | coverageを悪化させない。fallback/埋込みをOCC固有gainへ数えない |

現在v1は目標δで証拠が通ると終了するため、通ったwidthの小ささを目的にしていない。
v2案は有限proposalをbudget内で保存し、各証拠のκ,τ,Zを固定したまま安全なwidthを求め、最良発見済み上端を返す。
OCCだけが多くの候補や有利なcacheを無料で使わないよう、method-specific budgetと共通primitiveの実費を分離する。

## 固定δでの候補生成案

D=g+δ、Kκ=κh_A−vvᵀについて

\[
0\le\kappa\le D,\quad Z\succeq0,\quad K_\kappa-\tau I+Z\succeq0,
\quad\kappa U-\nu\tau+\operatorname{tr}Z\le\delta D
\]

はm×m SDP可行性問題。既存concave spectral-score searchまたはSDP solverをproposalとして使う技術選択は可能。
solver success／failureを認証結果にしない。κ,τ,Zを有理数へ丸め、独立checkerで元行列のPSD・scalar・boxを検査する。
PSD修復でZへrIを足す場合はs=ντ−trZも必ず減らして再判定する。修復費用／bit増加を保存する。
OLDの有効証拠はτ=μ,Z=0、NORMはκ=0,τ=0,Z=vvᵀで埋め込めるがrawと区別する。

実数理想条件のδ受理は上方単調：同じ証拠でδが増えればDとδDが増える。
**有限generatorの探索失敗は可行性の否定ではないので、失敗を根拠に二分探索の下端を上げてはいけない。**
新sourceでδproposal順序・bit ladder・iteration数を結果前固定する。未判定後に同じ結果を見てbudgetを増やさない。

## 保証幅と厳密な区間

保存／新生成済みの一つの証拠についてq=κU−sと置くと

\[
\delta_w=\max\{0,\kappa-g,(\sqrt{g^2+4\max(q,0)}-g)/2\}.
\]

128 fractional bitsの外向きroot区間とpolynomial checkerを既存saved-width実装から再利用する案。
κをδ変更に合わせて作り直す場合は新proposalであり、固定証拠再利用とは区別する。
有理highでscalar条件を満たすことを確認し、δ_w=0でも運用出力には正のδ_proofを付ける。
候補poolの[min low,min high]は**有限pool最良幅**の囲み。理想OCC最適幅の囲みではない。
理想最適幅の下界は証明されたものだけを使い、一般入力では0を保持する。高いunknown下端から近似最適性を宣言しない。
OLDについては有理dualにより全κ/μ不可行を示せるが、そのdualをOCC全体の不可行性へ流用しない。

## 同じ証拠の継承とidentity

同じh_A,v,ν,Uとκ,τ,Zでc_new≥c_oldなら、旧widthの有効証拠は強いcでも維持される。
g/δ scalarを再検査し、originを記録する。新しいsourceがsolverに失敗しても継承済み有効証拠を捨てない。
cache keyはcase SHA、active順序、external mode、ν、U/cとその証拠ID、method/source/solver version、precision、κ/τ/Zのexact bytes。
δだけに依存しないspectral PSD primitiveはmatrix SHAで共有できるが、新binding・scalar検査を省略しない。
別Hamiltonian、mode、scale、νや未知のground/gapに依存するreuseは禁止する。

## mode選択と誤差会計

候補ごとにε_projで採用可能か、raw OCCとBASE ORの保証幅、同情報差、冷生成／warm費用を出力する。
選択規則案は「有効候補の中で保証上端が最小、exact同値なら固定mode順序」。区間重なりをstrict superiorityとはしない。
別のcost proxyをselectorに入れるなら、そのproxy・重み・情報を科学契約で結果前固定する。
各候補が一mode縮約なので一mode削減を返すtaskであり、候補数だけから追加削減量を主張しない。
ν-sectorのground energy推定で |E_est−a_d|≤ε_solver なら

\[
E_0(H)\in[E_{est}-\epsilon_{solver}-\epsilon_{proj},\ E_{est}+\epsilon_{solver}].
\]

したがって総energy予算は少なくともε_proj+ε_solverを会計する。未知の射影誤差を推定値から正確に差し引かない。
PF bias、QPE/RPE統計誤差、state preparationの成功条件・費用は別契約であり、現在のcertificateには含まれない。

## bound取得費用と性能設計

U：同じtrialの期待値をexact評価。最小determinantを全列挙する既存経路は一般fillingでcomb(m,ν)。
安いtrial subset／指定Slater trialを使う代案は安全でもboundが弱くなり得るので、同じUと無料同等には扱わない。
c：外部占有の一体shiftに対するσ_{ν−1}と非負densityを落とした下界は、係数m×m spectral proofで取得可能。
より強いdensity boundのoccupation列挙はcomb(m,ν−1)。c強化の採否利益とOCC条件の利益はrowを分ける。
数値m×m eig/SDP proposal、exact PSD O(m³)算術回数、bit長、bound生成、checkerを別々に記録する。
数値flop次数だけで全pipelineのbit complexityや一般fillingでの多項式計算性を主張しない。

準備時inventory：CPU32、RAM約64GiB、available約52GiB。nvidia-smiはdriver通信失敗。現在の小行列有理照合はCPU1、全thread1。
将来の支配費用はcold bound・spectral proposal・有理PSD。独立mode/caseを並列化しmatrix/proof cacheを共有、BLAS/OMP/MKL/Numba/Rayon各1を基本案とする。
worker数はAS/RSS、immutable data量、同時作業を見てsource固定前に決める。GPUの利益・speedupは未測定。
science sourceのm上限、eigh/SDP/proposal/PSD/算術回数、rational/scaled-integer bit、aggregate RSS/AS、output/log capsは固定planに必須。
wall/CPU時間capは設けない。primitive call／memory制限で停止し、partial/失敗を保存する。
これは将来契約のschema要件。今回、新batchの入力数・2-base gate・数値実験上限を採択しない。

## 最小技術検証と実装の順序

1. 現行原理とsaved checkerを保持し、新v2 module/runner/independent checkerを別pathで作る。
2. rational syntheticでNORM/OLD埋込み、ν1同値、縮退・singular PSD、c/δ継承、bit超過、非PSD・oracle拒否を検査。
3. finite failureでlower boundを上げない、cache poisoning拒否、full-H fallback、cold/warm accountingを検査。
4. 目標δでの早期終了を解除しても有限予算を守ることをmock solverで確認。独立referenceと最適化backendを比較してからsource固定。
5. 科学的task・比較・未使用入力・予算が採択された後に、入力／source／planを固定し、別の明示実行認可を得る。

今回は1–4の将来仕様をまとめた段階で、保存帰結checker以外の新generator実装・数値solver起動は行わない。
採択済みsourceの通常技術テストごとにGPTレビューを要求しないが、新しい科学比較の採択をCodexが代行しない。

## 次の検証案の目的と未固定事項

既存12入力・固定8入力・S1二方向はdevelopment。独立入力の係数は今回生成しない。
次の確認的task候補は、結果前に決めたε_projで同一modeの縮約判断が変わるか、保証幅とcold取得費用の両方が意味を持つかを調べること。
S1のδ_OCC<ε_proj<δ_Nに予算を置けば判断差が存在するという解析的帰結はdevelopmentの説明に留め、独立成功のgateに流用しない。
BASE/OCCが同じ元targetで通る8/8を新しい幅taskの勝利へ読み替えない。
実験規模、入力family、誤差予算、強いSDP/RDM対照の採否、費用の評価基準はGPT／ユーザーの科学判断待ち。
新しい条件の採択→source/preflight→実行認可→限定実行→新結果レビューの順序を守る。全結果後Mandatory STOP、retry0。
