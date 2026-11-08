# Track C（O3）研究成果の最終統合報告

2026年11月8日、SERVER 2 / Codex C。対象は `HIROMU1015/Partially-Randomized-Trotter`、最新収集基点は `92495a04d52c65142ed16b3abe977345065f9745`。初期PR改善からClosed P3、3分子のDF比較、Truncated PFの相殺検証P1/P2までを統合した完成版である。

基礎資料には、`100377dac8635fc49074147bfd8b42f52f2b7311`の[旧Track C統合報告][OLD]を再利用した。旧版は対角gate融合までの研究段階の総括として保全する。本書は後続成果を加え、当時の結論がどの比較条件で成立し、何が追加で判明したかを説明する。**今回行ったのは固定文書・SOURCE・保存値の読取、対応照合、文章統合、資料公開だけである。新しい科学計算、数値再解析、回路build/compile、sampling/価格MC、候補探索、資格再判定、詳細科学レビューは行っていない。**

主要な到達点は次のとおりである。PR内部ではDのBCH調整、共同isometry、Closed P3、h/resolution/α選択による改善を確認した。しかし、最適化後の固定H4でPRは強Full PFより高価だった。比較対象をTruncated PFへ広げると、同じ元Hのエネルギー推定で、有限候補内の低RZ方式はH2がPR、H4/H6がTruncated PFとなった。続くP1はH6の指定した5つのstep近傍で資格維持を支持し、P2はH4では個別誤差を保守加算してもTruncated PFが両参照方式より安いことを確認した。H6の保守的評価ではFullより安い一方、PRより高い結果だった。元のcombined評価の条件付き保証は維持される。[OLD]、[OPT]、[DF]、[P12]

本書の構成：

- [第1部　研究目的と、比較する時間発展](#purpose)
- [第2部　初期PR候補から、D内部誤差と回路構造の改善へ](#early)
- [第3部　Closed P3とh/resolution/α共同最適化](#closed-p3)
- [第4部　DF HamiltonianでのH2/H4/H6・3方式比較](#molecules)
- [第5部　Truncated PFの誤差相殺：保証と安定性を分ける](#cancellation)
- [第6部　P1：H6の有限step近傍での局所安定性](#p1)
- [第7部　P2：相殺依存を弱めた個別誤差評価と総資源](#p2)
- [第8部　研究全体で確認できたこと、一般化できないこと](#conclusions)
- [第9部　固定出典、SOURCE、再利用可能な証拠への入口](#sources)

<a id="purpose"></a>

## 第1部　研究目的と、比較する時間発展

### 1. 同じ推定課題で誤差・信号・回路費を結ぶ

目的は、同じ元Hamiltonianの基底エネルギーを同じ精度・成功条件で推定するとき、量子資源の少ない構成を明らかにし、PRの有効性と限界を評価することである。PRの勝利を比較の前提に置かない。高次公式の作用素誤差、RTE normalization、1回路のRZ数のどれか一つを改善しても、全RPE費用が減るとは限らない。

DF表現では、one-bodyと二体DF fragmentをbasis変換・対角演算で実装する。Full PFは全項を決定論的に保持する。PRは `H=H_D+H_R` と分割し、Dを決定論PF、Rを有限Taylor Randomized Time Evolution（RTE）で実装する。Truncated PFは一部のfragmentまたは小係数を除いたHkeepを決定論PFで実装する。Truncatedでも推定対象は元HのE0であり、Hkeepの基底エネルギーへ変更しない。[OLD]、[DF]

PRの動機は、高価なtailを各stepで全て実行する代わりにランダムeventへ置き換えることである。その代償として、有限Taylor誤差、normalization/rangeによる信号減衰、weightの有限算術、十分測定回数が生じる。Truncatedではランダムtailの費用を避けられるが、元HからのHamiltonian近似誤差を負担する。Fullにも同じ回路合成とパラメータ選択の機会を与え、これらを全18roundの資源へ接続した。[P3]、[OPT]、[DF]

### 2. 共通科学条件と保証の意味

| 項目 | 後期の全18round比較で保持した条件 |
|---|---|
| 化学条件 | Linear chain、隣接1.00 Å、STO-3G、RHF積分由来の固定保存Hamiltonian |
| DF入力 | 初期・中期H4は保存DF12/generation order。3分子比較はH2 full XDF3、H4 DF12、H6 full XDF18 |
| 状態・target | 各元Hの厳密ground oracleを仮定。保存stateのresidual/normalizationと厳密groundへの移送を会計。targetは元HのE0 |
| RPE | m=0,…,17、δ0=.02、T_m=.02×2^m、energy精度1.5936e-4 Ha |
| 失敗・位相予算 | 方式1回のfailure上限.05、各roundの位相予算.4 rad |
| 最終compiler | Qiskit1.3.0、Track C reference compiler A、native basis rz/sx/x/cx |
| 費用範囲 | 状態準備除外、理想native gate。RZ/CX/sizeとmodular serial depth上界を別々に評価 |
| 未評価 | hardware noise、routing、誤り訂正、最終改良方式のT合成込み費用、oracle stateの準備費 |

H4のPR LD3はD4 blockとRの218 signed Z/ZZ成分・7 basisを持つ。これはH2/H6へ流用する入力規格ではない。各分子の分割・BCH係数・catalogue・価格は、それぞれの保存入力から検証されたものを使う。旧Pauli-JW B0/B0I研究の探索法は参考にしたが、その価格をDF研究の結果と直接比較していない。[OLD]、[DF]

SOURCE/native/finite-bitの誤差diskを含むスケール済み信号半径下限をρ_m、系統位相予算を引いた統計幅をβ_mとすると、保存研究の十分trial式は

$$
N_{m,b}=\left\lceil\frac{4\log(2/\alpha_{m,b})}{\rho_m^2\sin^2\beta_m}\right\rceil_{\rm upper},
\qquad \sum_{m,b}\alpha_{m,b}\le .05.
$$

各軸αのexact和とdirectedな整数切上げを確認する。alias条件とround間branch条件も必要であり、小さいwrapped終端位相だけで成功としない。既存研究の最終energy幅は `1.52587890625e-4 Ha < 1.5936e-4 Ha`。E0は資格・設計・結果照合に用いるが、decoderには位相列とq列だけを渡し、既知E0による事後補正をしない。[GA]、[OPT]、[DFVALID]、[P12]

資源は各指標jについて `G_j=Σ_m,b N_m,b E[C_j,m,b]`。RZ、CX、sizeを同じ選択済み回路から取得する。depthはmoduleを直列につないだ上界であり、m17巨大回路全体を直接compileした実DAG深さとは区別する。RPE十分trial数は必要最小shot数とは限らず、安い短roundへtrialを増やすα配分が総RZを下げる場合もある。[OLD]、[OPT]、[DF]

<a id="early"></a>

## 第2部　初期PR候補から、D内部誤差と回路構造の改善へ

### 3. 初期仮説を試し、比較基盤を整える

初期研究は、PFの次数・回転合成・processorを変えることで、PRの誤差または単価を減らそうとした。成功しなかった候補と保証不足を残したことが、後のBCH設計と具体的native価格評価の出発点となった。[旧統合報告][OLD]

| 研究 | 目的・実施内容 | 保存された結果 | 次に残った問い |
|---|---|---|---|
| C-N：高次PF・whole合成 | S2、Yoshida4、Morales Y8m10b、h=.02/.01、LD3/12を比較 | 初期opt2価格はnative誤差未会計でSTOP。A導入後は2304 paired S2標本でwhole wrapperと最大D run融合済み強対照のRZ/CX/depth/sizeが一致。q8価格modelの最大点誤差6.0905%は既存目安を不通過 | 次数やwhole化だけで利益を仮定せず、PF誤差の構造を調べる |
| C-S：残差ランダム化 | coarse回転と残差RTE、D+R/Dのみを比較 | 登録π/4 coarse・r_res=1では対象116回転が全てcoarse index0。終端log(range)は通常.55739に対し約386.8/386.2となり、trial負担が極端に増加 | 回転単価とnormalizationを同時に会計する |
| C-P：Processing | 固定YP8m8 kernelと固定processor、h=.02 | PとIの距離約1.270e-12、信号差最大7.664e-15、36軸のuniform shotsは同数 | 固定processorの無改善を保持。processing一般のno-goとはしない |
| C2-A：round依存PF | S2/Y4、h=.02/.01、m=0,8,17の84信号と60短価格セル | 3診断roundではS2を選択しPF族交差は未観測。q4の最大価格点誤差15.3821%、新native envelope未資格で全18roundへ進めずSTOP | 有限診断から高次PF一般の無用性を結論しない |
| C2-B：構造的合成誤差 | Full Y4/LD12/h=.02の共役blockに、時間依存のbasis合成誤差評価を適用 | 強化された両対照の実現総Tは通常493413295481472、構造490876793817392、差.514072%。通常側の緩和下界が構造の実現値を下回り、最適性優越は未確定 | Tの有限policy差を、後期native PRのRZ改善へ転用しない |

出典は[C-N/C-S/C-P原報告][PILOT]、[C-N価格][GB]、[C2-A][A2]、[C2-B強対照][B2]。C-Sの登録policyやC-Pのprocessorに対する負結果、C2の未資格はその範囲に限定する。未実行・screen外・技術STOPを「一般に不可能」と扱わない。初期の欠測価格を0で補わず、短q fitを最長roundへ無条件移送しなかった。

compiler問題は科学的比較の前提を左右した。旧opt2ではCRZ(1e-5)が空回路となる例や、Full Y4 q1でX差約1.4077e-5があった。reference Aは定義に基づくlowering、固定inverse cancellationを使い、微小角cutoffと数値的近似交換を避ける。44人工ケースと全512次元already-controlled operator・X/Y・負時間を照合した。[静的原因調査][STATIC]、[A correctness][ACOR]

保証は局所compiler検査から始まり、SOURCE生成、有限RTE、spectral residual、ground/alias、finite-bit CDFへ順に接続された。128 context・1302 loweringで残余予算が正だった時点でも、未完のSOURCE→mean等を資格済みとはしなかった。後続保証Aと各変更のsupplementにより、固定保存H・exact ground・理想native・fresh独立bits条件で18roundの十分条件を閉じた。許容するのはjoint回路全体の一つの共通global phaseであり、control branch間の相対位相を保持する。[A18][GA]

### 4. 外側の混合公式からD内部BCH調整へ

まず、外側のD/R配置を変える既知5段族

$$U_a(h)=D(ah)R(h/2)D((1-2a)h)R(h/2)D(ah)$$

を検証した。正確なD/Rとexact groundについて、保存代数展開は主要混合biasの係数差を `1/96+(a−1/4)^2/2` と与える。この族のground主要混合biasはa=1/4で最小になり、一般に0にはできない。a=1/4は半刻みS2二回と有限RTEまで同値である。[混合公式の理論・数値][M]、[MTHEORY]

実際のmultiblock PFでは、2MNはH2作用素normを半刻み対照より約56%減らした一方、ground H2は悪化した。双方feasibleな2MN/alpha=0の各576条件で、半刻みS2よりshotsが減った条件は0。96 paired価格標本でも4指標は同一だった。全体partial Y4は負R時間でnormalization負担が増えた。作用素誤差の最小化だけでは目的の基底位相・総費用を改善できないため、D内部PF誤差を利用する設計へ進んだ。[M]

D内部研究は、H4 LD3の4 fragmentを保った24順序、fragment時間再配分、D自己対称S2、D-only Yoshida4、Dの対称3段合成を比較した。D自身のS2をS_Dとし、

$$D_v(t)=S_D(vt)S_D((1-2v)t)S_D(vt),\qquad \eta(v)=2v^3+(1-2v)^3$$

を外側 `D_v(h/2)R(h)D_v(h/2)`へ入れる。保存BCH設計は `⟨H2,outer⟩+η(v)⟨K_D⟩/4=0` を満たすvを元ground期待値から選んだ。D-only Yoshida4のη=0と違い、負の内部biasで正の混合biasを相殺する。中央D時間は負だがR(h)は正一回のままで、負R時間によるrange増を避けた。[D]

H4の調整0123はv=1.4768778287800015、調整0321はv=1.490624664134615。内部と混合のground H2寄与はそれぞれ約−.00303236699、+.00303236699となった。ただしH2作用素normは約.03146/.03177と非ゼロである。消えたのは固定元groundの主要energy補正であり、一般の作用素4次公式を得たという意味ではない。係数はBCHから固定し、長round位相や価格へのfitで選んでいない。[D]、[DFINAL]

| D内部研究の登録主比較 | 全18round shots | 総RZ（1 MC SE） | 総CX（1 MC SE） |
|---|---:|---:|---:|
| 元PR S2 | 8598 | 4.77920e12（1.2893e11） | 1.09324e12（1.4076e10） |
| D調整0123 | 8498 | 3.52729e12（6.5269e10） | 7.57756e11（7.0997e9） |

登録r_mを保持し、h=.02から.16へ移した主比較では、総RZ26.195%、総CX30.687%減。m17のPF位相上界は.01400660→.00232990、shots/軸は626→592となった。[DREG]。一方、有限menuでround別に選ぶ別policyでは改良PR RZ3.34845e12、強Full1.20943e12で、Fullが安かった。[DNR]。登録主比較とmenu選択policyを混ぜず、この段階のPR内部改善とFull比較を分ける。

D-only Yoshida4では混合biasが残り、価格診断は悪化した。順序変更だけの利益も限定的だった。以後の問いは「D誤差をさらに小さくするか」から、「残るRTE basis費と信号減衰のどこを減らすか」へ移った。[D]、[K]

### 5. Singleton、共同isometry、対角gate融合

費用分解では、当時の改良PR RZ3.34845e12のうちDが43.37%、RTEが56.63%、RTE RZの91.44%がbasis変換だった。m17が総RZの72.2%を占めた。これは当該policyの保存分解であり、後続全policyに固定して使う割合ではない。[K]

| 改善 | 実装・比較の単位 | 保存結果と制約 |
|---|---|---|
| Singleton partial QR | 1/2列だけ必要なZ/ZZに既存partial-column QR/Givensを利用。最大same-basis run長1だけを置換 | always-partialは共有を壊して悪化したため棄却。長runはfull共有と実価格fallbackを保持。step/resolution/αの共同選択も加えた最終PRはRZ1.79060e12、強Full9.13529e11。出発点からの46.52%減をsingleton単独の効果とはしない |
| 共同isometry | 同一source-basis runで必要な各列を保つunitary completionを一度作り、run全体で共有 | 4layout、625採用集合、未登録・高rankはfallback。独立paired holdoutで旧singletonに対し総RZ3.355%、CX1.789%、size3.190%減。ただしserial depth上界は.696%増 |
| 対角gate融合 | 効率的な単一Pauli controlを共通基準にし、同一basis内の角度集約・CZ偶奇・parity共有を比較 | 既知control後のRTE run固有の全PR追加RZ利益は約.135%、CX約1.67%。Dも含む共通parity共有の追加CX効果は約8.47%。全変更のRZ18.196%減の大部分は既知controlによる |

出典：[singleton・分解][K]、[KFINAL]、[共同isometry][J]、[JFINAL]、[対角融合][F]、[FFINAL]。各削減率は各研究内の指定対照・holdoutに対する値で、掛け合わせた累積改善率を作らない。

共同isometryは占有部分空間のspanだけでなく、各列projectorを保持する。必要support和集合の列をまとめ、Fock Z/ZZ作用、controlled operator、X/Y、native loweringを別々に検査した。既知QR/Givens・unitary completionの再利用であり、その原理自体の新規発明を主張しない。[J]

対角融合は異なるbasis間の演算を可換扱いせず、binary64角度をexact rationalとして集約し、最後に一度丸める。全候補に効率controlを与え、各module/trajectoryで一つのRZ優先回路を選び、その同じ回路のCX・size・depthを用いた。Fullにも同じ合成機会を与えると、対角融合段階のPR総RZ1.42343e12に対しFull5.66091e11で、Fullがなお安かった。[F]、[FFINAL]

この結果は固定H4と当時の比較対象に関する正しい歴史的結果である。対角RZの追加削減だけで差を埋める証拠は得られず、次にRTE samplingとnormalizationへ改善対象を広げた。

<a id="closed-p3"></a>

## 第3部　Closed P3とh/resolution/α共同最適化

### 6. 同じ有限Taylor平均を、異なるsampling lawで実現する

Track B接続ではSERVER 2と公開SERVER 1調査を照合し、G7 Closed P3をTrack Cの218成分へ接続した。SERVER 1資料はこの段階で取得済みになった。旧localの別partition・324 label・m5条件をCへ移していない。G9/G10/C1の負結果・STOPも保持し、新しい固定H4/P3研究として実施された。[P3]

符号を吸収したinvolution Q_i、正規化確率p_i、`A=Σ_i p_i Q_i`、時間方向σ、`x=λ_R t/r`に対し、Canonical K2とClosed P3の補正平均は同じ有限多項式

$$M_3=I-i\sigma xA-\frac{x^2}{2}A^2+\frac{i\sigma x^3}{6}A^3$$

を保つ。Closed P3は有限P3の形式returnを集約するsampling表現で、Taylor次数を上げたP5ではない。identityは既存scalar側、ordered parentの演算子語と回路時間順の逆関係、Taylor符号、負Q、controlled相対位相を保持する。全12使用cellの70次元平均検査の最大作用素差は1.66534e-15だった。理想同値性の根拠と有限数値検査は区別する。[P3]、[P3STAGE1]

理想のreturn集約normalizerと、実dyadic proposal π_eは別の量である。全使用cellで0<x≤1を確認し、256-bit proposalと、proposal許容パラメータη=ρ=10^-24（信号半径ρ_mとは別）の既存規約を使う。実eventは `w_e=α̃_e/π_e` を持ち、有限bit・root・angle・binary64 weight丸め・全path積の誤差を会計する。測定推定量は**全trialを分母にするweighted平均** `Σ_a A_a(Π_j w_a,j)Z_a,b/N_trials`。認証path range L_mで割ったY/L_m∈[−1,1]へ十分式を適用する。通常の無重みHadamard平均へそのまま同一視しない。[P3]、[P3SOURCE]

このClosed P3では正当なrejection/zero質量は0、acceptance=1で、期待quantum callsと十分trial数が一致する。技術failureや未完走prefixを正当なzeroへ置換しない。有限bit proposal差をweightで補正し、同じ差をmean biasとして二重に課金しない。basis/source/nativeの一様diskと新角度の追加会計は残す。[P3]

### 7. 固定回路条件での利益は主に十分trial数の減少

D、h/resolution、共同isometry、対角合成を固定した独立paired holdoutの結果は次のとおり。

| 固定H4・同じpaired holdout | 十分trial数 | 総RZ | 総CX |
|---|---:|---:|---:|
| Canonical P3、今回再評価 | 17186 | 1.422901381e12 | 1.972696355e11 |
| Closed P3 | 17028 | 1.394938574e12 | 1.933790045e11 |

総RZ削減は1.965196%、差2.796280709e10、paired MC SE2.871016745e8。CX削減は1.972240%。同じcanonical trial数を固定した単価ablationのRZ差は約−6.8629e4で、native単価の大幅改善が主因という証拠はなかった。主要機構はreturn集約によるrange低下と高価格roundのtrial減少である。normalization改善から総資源改善を先取りせず、実whole occurrence価格まで確認した。[P3]、[P3FINAL]

旧対角融合の1.42343e12と、このcohortの1.422901381e12の違いは、価格標本・α配分の違いを含む。これをsamplingによる追加改善と数えず、上表のpaired2方式で比較する。強Fullの保存総RZ5.660906974e11はなお低かった。古典setup/cacheの測定境界も揃っていないため、CPU速度優越や総実行時間優越は認定していない。[P3]

### 8. h/resolution/αを再選択し、信号と単価のtradeoffを評価

sampling変更後は、既存資格と一回限りの局所拡張を利用し、最終183 PR条件を両P3に、51 Full条件をFullへ与えた。価格未検証の残り116拡張条件は、資格あり・価格未検証として残す。screenされた候補を一般的不適格にしない。[OPT]

選択はRZ目的の候補/α交互更新と整数局所交換で、二開始点・同じ停止規則を各方式に与えた。旧training/holdoutを選択用へ明示的に再分類し、最終policy・36軸α・Nを凍結してから、別seedの2176軌道を最終holdoutとした。共通cellの全round係数を合算してcovarianceを戻す。[OPT]、[OPTSOURCE]

| 固定H4・最終の同一holdout | trial合計 | 総RZ | 総CX | RZ MC SE |
|---|---:|---:|---:|---:|
| 旧固定Closedを今回再評価 | 17028 | 1.387608544e12 | 1.925041360e11 | 7.110533e9 |
| 最適化Canonical P3 | 16946 | 1.332071607e12 | 1.863565000e11 | 4.943179e9 |
| 最適化Closed P3 | 16791 | 1.303898609e12 | 1.824266292e11 | 4.834427e9 |
| 強Full PF | 14225 | 5.647934021e11 | 1.012313194e11 | 0 |

旧固定Closed→最適化ClosedのRZ削減は6.03267655%、差8.370993524e10±8.581509829e9（paired MC SE）。再最適化Canonical→Closedは2.11497621%、差2.817299755e10±1.089716414e8。これらは[元REPORT][OPT]と[extended/FINAL][OPTFINAL]の保存値で、異なるcohortの公表値同士を引いた改善率ではない。

最終両P3は全18roundで同じD/h/resolutionを選んだ。この有限menuではClosed固有の選択差や相互作用の追加利益は観測しない。m17はh=.32、resolution64→48、RTE event数8388608→6291456となり、log rangeは.270397071→.360529421へ増加、trialも[253,253]→[303,302]へ増えた。それでもevent数25%減による単価低下との総合で費用が減った。normalization・trial数を最小化すること自体が総RZの最適化ではない。[OPT]、[OPTROUNDS]

強Fullにも同じ効率control・対角合成・整数α改善を与えた結果、Closed/FullのRZ比は保存値2.308629。ここまででPR内部の改善は複数確認したが、固定H4のPR対FullではFullが低コストだった。この比較に、低寄与項を保持する必要があるかという次の問いを加えたのがTruncated PF研究である。[OPT]

<a id="molecules"></a>

## 第4部　DF HamiltonianでのH2/H4/H6・3方式比較

### 9. 比較対象を増やし、各分子の固定DFを使う

| 分子 | 固定入力・順序 | 信号sector / system qubits | 候補構成の特徴 |
|---|---|---|---|
| H2 | 保存full XDF3、元frame順 | Nα=Nβ=1、4次元 / 4 qubits | PR LD0/1/2。H4のLD3や218成分を移さない |
| H4 | 保存DF12、generation order | N=4、70次元 / 8 qubits | 既存の最適化Full/Closed P3を資格・価格対応を確認して再利用 |
| H6 | 保存full XDF18、元frame順 | Nα=Nβ=3、400次元 / 12 qubits | PR LD3/9/14/17。分子・split固有BCHと新しいcatalogueを評価 |

各元Hは保存行列のHermitian部分とし、微小なDF再構成差も元Hへの誤差に含めた。H2/H6は既存保存入力を使い、新SCF・DF fitは0件。元H ground/residualの必要なsector計算は当該研究で実施し、H6の旧rank14 own-DF groundを転用していない。H4のBCH係数やcatalogue、h/resolution、sampling価格を他分子へ無検証でコピーしていない。[DF]、[DFSOURCE]

Truncated候補は、元順DF prefixの保持（末尾identityも除くI0、identityだけ残すI1）と、非identity小係数をabs順にnestedで除く累積L1 budgetから構成した。Fullも必ず含め、全方式にS2/Y4と同じ既存h候補を与えた。Fullを含む重複除去候補数はH2/H4/H6で9/36/48、stand-alone Hkeepの十分pilot screen内は2/19/20。さらにm15..17の先頭最大4因子の全順列と隣接hを一度だけauditし、選択した48点を実価格・directed評価へ接続した。[DF]

screen外・未価格・未評価候補は、その有限探索で採用されなかったという記録であり、相殺を含めたあらゆるPFで不可能という意味ではない。旧Pauli-JW B0/B0Iからはnested探索、constant保持、元H stateの直接信号、bias/N/価格の同時評価を再利用した。旧Pauliや別compiler/13roundの数値は混ぜていない。[DF]

### 10. 同じ元H・精度・成功条件での保存結果

| 分子 | Full PF 総RZ | Truncated PF 総RZ | 改良Closed PR 総RZ | 有限候補内の低RZ方式 |
|---|---:|---:|---:|---|
| H2 | 8.277771739e9 | 7.851934551e9 | 5.009545544e9 | PR |
| H4 | 5.647934021e11 | 3.337772238e11 | 1.303898609e12 | Truncated PF |
| H6 | 2.082026044e12 | 1.222907256e12 | 1.617753939e12 | Truncated PF |

正本は[原REPORT][DF]、[ordering/COMPARISON.csv][DFCMP]。全9行が固定元H・oracle ground・理想native・独立draws条件で18round資格を持つ。failure .05は各方式1回についての条件で、9行同時成功の保証ではない。[DFVALID]

| 分子・方式 | 総CX | 総size | serial depth上界 | 十分trial数 | 総RZのMC SE |
|---|---:|---:|---:|---:|---:|
| H2 Full | 2.176712270e9 | 1.249852239e10 | 6.349359243e9 | 14208 | 0 |
| H2 Truncated | 1.836036870e9 | 1.173200980e10 | 5.767372489e9 | 14207 | 0 |
| H2 PR | 1.714095870e9 | 7.737398221e9 | 4.179183735e9 | 15896 | 7.327e5 |
| H4 Full | 1.012313194e11 | 8.371684838e11 | 2.088520659e11 | 14225 | 0 |
| H4 Truncated | 5.830379151e10 | 4.941682283e11 | 1.188928069e11 | 14629 | 0 |
| H4 PR | 1.824266292e11 | 1.904338949e12 | 4.992174341e11 | 16791 | 4.834e9 |
| H6 Full | 3.657833912e11 | 3.085775668e12 | 5.917635392e11 | 14248 | 0 |
| H6 Truncated | 2.106713327e11 | 1.809978048e12 | 3.347103693e11 | 13539 | 0 |
| H6 PR | 2.708191202e11 | 2.389309688e12 | 4.422261561e11 | 15754 | 2.001e9 |

これも[保存比較表][DFCMP]と原REPORTの表示値である。決定論価格のMC SE=0は実装gate数が確定する意味であり、化学モデル・物理誤差が0という意味ではない。PR価格MCのnominal SEはRPE成功保証とは別で、正規近似SEから分布自由の価格保証を主張しない。H4 Full/PRはc5a3の資格・独立holdoutを再利用した。H2/H6の最終価格holdoutはpolicy凍結後の新3712軌道で、旧training/holdoutは追加auditのselection dataへ再分類している。[DF]、[DFSOURCE]

### 11. 分子ごとに異なった切り捨て余地

H2 m17の非Fullはcoeff_L1_0.01で11個の極小係数を除くが、除去L1は約3.33e-16 Ha、ground差は保存精度で0だった。実fragmentを落とすprefix2は大きなenergy差で十分screen外となった。したがってH2のTruncatedは有意なfragment近似の成功例ではない。この有限menuでは、tailを保持するPRが低コストだった。[DF]

H4 m17はcoeff_L1_2、204係数を除去して236非identity項を保持する。H6 m17はprefix13_I0で18中13 DF fragmentとone-bodyを保持し、末尾5 fragmentをidentityごと除く。H4では小係数除去、H6では低寄与tail除去の余地があった。ただし採択Truncatedは**roundごとにretention/formula/order/hを変えるpolicy**であり、全roundを一つのHkeepで実行する方式ではない。[DF]、[DFMECH]

PRのR部分が総RZに占める保存割合はH2約9.5%、H4約73.6%、H6約8.8%。同じ「PR」でもDF構造・split・basis変換価格・normalization負担が異なる。分子間の絶対費用と分子内の方式順位を分け、3分子の有限結果から漸近スケーリングや一方向のサイズ依存を確定しない。[DF]

この追加比較によってH4の最安方式がTruncatedとなったことは、旧H4の「強FullはPRより安い」という結果を取り消さない。後続研究で比較対象を増やしたため、Fullより安い別方式が見つかったのである。

<a id="cancellation"></a>

## 第5部　Truncated PFの誤差相殺：保証と安定性を分ける

### 12. 保存されたsigned位相とdirected上界

元H groundのHkeep信号をzK、PF信号をzPとすると、保存診断は `φT=Arg(zK exp(+iE0t))`、`φPF=Arg(zP conj(zK))`、`φC=Arg(zP exp(+iE0t))`。加法はmodulo 2πの関係である。以下は保存された点診断と、それとは別にdirected評価された資格欄の引用である。[SURVEY]、[DFMECH]

| m17 | H4 | H6 |
|---|---:|---:|
| retention / formula / h / 先頭順序 | coeff_L1_2 / Y4 / .16 / 1023 | prefix13_I0 / S2 / .08 / 2013 |
| signed Truncation位相 [rad] | −.021852026035 | +.279809991922 |
| signed PF位相 [rad] | +.021344614091 | −.313816019988 |
| signed combined位相 [rad] | −.000507411944 | −.034006028065 |
| combinedのdirected上界欄 [rad] | .000528852867 | .034588467134 |
| 最終β下限 [rad] | .399468960687 | .365371590801 |
| 十分N_X / N_Y | 130 / 128 | 157 / 155 |

signed成分を差し引いて新しい認証上界を作った表ではない。combined上界欄と、SOURCE/native/finite-bit等のsupplement後の最終βも区別する。大きい逆符号成分があること自体は、固定点のdirected成功保証を無効にしない。一方、この保証を別h・順序・stateへそのまま移すことはできない。[SURVEY]

ここでの相殺は**Hamiltonian切り捨て誤差とPF誤差**の相殺である。第2部の**PR内部D誤差とD/R混合誤差**のBCH相殺とは、誤差成分も調整対象も異なる。

### 13. 回路単価低下と、β・Nを介した効果

切り捨てによる構造的な単価低下は、同じformula/h・generation orderの保存middle moduleにも見られる。

| 分子 / formula / h | Full RZ / CX | Truncated RZ / CX |
|---|---:|---:|
| H4 / Y4 / .16 | 68232 / 12267 | 42888 / 7542（coeff_L1_2） |
| H6 / S2 / .08 | 78096 / 13802 | 57854 / 10094（prefix13_I0） |

出典：[既存証拠調査§7][SURVEY]。これらはmodule単価で、wrapper・step数・Nを含む総資源削減率ではない。identity-only blockではGaussianを省略するため、保持block数と実行basis変換数も同じではない。

相殺はcombined上界を小さくしてβを増やし、十分Nへ影響する。総Gでは単価・h/formula/order・retention・α配分が結合する。H4 TruncatedはFullよりtrial総数が多くても低RZであり、H6 m17のN_X/N_Y=157/155はFullの141/141より多い。切り捨てが必ずNやstep数を減らすという説明にはならない。[SURVEY]、[DFCMP]

全18roundも同じ機構ではない。H4はm0..14がprefix7_I0/S2、m15..17がcoeff_L1_2/Y4へ切り替わる。H6はm14だけcoeff_L1_8/Y4で、φT=+.009167664、φPF=+.039922374と同符号でも資格を満たす。相殺の持続性、各roundの誤差予算、alias/branchは別の確認事項であり、m17の小さい値だけから全RPEの安定性を推定しない。[DFMECH]、[SURVEY]、[DFVALID]

### 14. 静的調査で分かったことと、残った問い

静的調査12e74ddは、同じretention/hでも複数の保存順序でround資格が成立することを確認した。H4 Y4では1023/1032/元順、H6 S2では2013/0213/元順の有限証拠があった。順序audit前policyもTruncatedを低コスト方式と記録したが、そのpolicyにも相殺は残るため、相殺なし対照とは呼べない。[SURVEY]

一方、保存したhalf/double hのproxyだけでは小さい近傍のdirected資格を判断できず、相殺に頼らない全18round費用も未取得だった。さらに保存stateの数値residual包囲は、物理的な近似ground・別固有状態・overlap摂動の科学検証ではなかった。この不足からP1（固定総時間の有限h近傍）とP2（個別誤差を保守加算した総費用）が提案され、後の別許可scopeで実施された。静的調査時点の未確定事項と、その後に解決した範囲を区別する。[SURVEY]、[P12]

<a id="p1"></a>

## 第6部　P1：H6の有限step近傍での局所安定性

H6 full XDF18、元H exact ground、prefix13_I0 / S2 / order2013（後続因子は元順）、m17を固定した。総時間を `t=(1/50)×2^17` と先に固定し、nを変えてh=t/nを有理数で定義した。中心n0=32768を再利用し、新規4点だけを評価した。[P12]

| n | exact h=t/n | combined directed上界 [rad] | ρ下限 | β下限 | N_X / N_Y（元α固定） |
|---:|---|---:|---:|---:|---:|
| 32752 | 4096/51175 | .034895025353 | .999388576989 | .365065032661 | 158 / 156 |
| 32767 | 65536/819175 | .034607613841 | .999388399695 | .365352444098 | 158 / 156 |
| 32768（保存中心） | 2/25 | .034588467134 | .999388387775 | .365371590801 | 157 / 155 |
| 32769 | 65536/819225 | .034569322157 | .999388375941 | .365390735773 | 157 / 155 |
| 32784 | 4096/51225 | .034282358070 | .999388198468 | .365677699746 | 157 / 155 |

正本は[P1_RESULTS.csv][P1CSV]と[P1/P2 REPORT][P12]。Truncation点位相は固定Hkeep/state/tの+.279809992 radを保ち、PF点位相は指定点で−.314122767から−.313509722 rad、combined点位相は−.034312775から−.033699730 radだった。これらのsigned点値と上表のdirected上界を区別する。

non-dyadic hのlogical有理時刻とbinary64 hの差、kernel scalarのn h_binと固定総時間wrapperの差を追加diskへ会計した。これは既知Hamiltonian constantの相対位相処理で、E0による補正ではない。新4点の角度/source/native資格、独立expm/Schur・full controlled X/Y、他17roundを固定したdecoder/aliasと十分Nを確認した。全5点で条件付きRPE資格が閉じ、新規不適格点はなかった。[P12]、[P12SOURCE]

**結論は指定有限5点での安定性支持。** n0より小さい2点では整数ceilによりNが1ずつ増えたが、資格の崩壊は観測しない。連続h領域、広いPF構成、別retention・別状態への頑健性までは証明していない。m15/m16の新しいh評価は不要で、追加探索へ広げなかった。[P12]

<a id="p2"></a>

## 第7部　P2：相殺依存を弱めた個別誤差評価と総資源

### 15. 評価A/Bの違いと、directed保証の構成

評価Aは元Hの理想信号に対するcombined信号を直接囲む従来の十分条件。評価BはHkeepに対するTruncation位相と、HkeepからPF/native信号への位相差を別々にdirected評価し、その非負上界を足す。**Bは物理的な誤差相殺を回路から除去した実験ではない。** 同じ回路を、逆符号の相殺に依存しない十分上界で評価する対照である。[P12]

保存実装はzK∈D(cK,eK)、zP∈D(cP,eP)を囲み、積zP conj(zK)のdiskに `|cP|eK+|cK|eP+ePeK+積の丸め` を含める。directed norm下限で原点を除き、中心Argとangular radiusから個別位相上界BT、BPを得る。BTには元E0のintervalも含め、`Bsum=upper(BT+BP)`、`βB=lower(.4−Bsum)`とする。signed点値の絶対値を単純加算した認証ではない。[P12]、[P12SOURCE]

Hkeep spectral/形成丸め、SOURCE generator、ground/state、complete-number-sector、時刻のdiskを含め、PF native/readout等はP側へ残した。S2/Y4のabsolute block時間が正時間Hkeep発展を上から覆うことをexact rationalで確認し、既存SOURCE diskを保守移送した。原点を除くdiskと小さい位相上界に加え、元Hのalias・全18round decoder/branchも確認する。保存nonresonant proxyをdirectedなunwrapped連続時間Floquet energy定理へ格上げしていない。[P12]

### 16. 実行前固定の157点と、Bで閉じなかった点

P1完了後、個別directed評価前に、保存された10514 round点のpoolから157点を固定した。元Truncated資格点を含め、各分子/roundの保守proxy上位4点とcombined proxy上位2点を加える事前規則で、H4が74、H6が83点。P1の新h、新retention、新familyは追加していない。proxyは選択用で、最終保証には用いない。[P12]

154点がBでround資格を満たした。Bで閉じなかったのはH6 m17 / prefix13_I0 / h=.08のS2元順、0213、2013で、個別上界和は保存値.641723561654、.613345045899、.594300003431 rad。これらは**Aでは引き続きclosed**である。Bの十分条件の不成立はAの科学的成功や実際の物理的相殺の否定ではない。欠測・計算失敗を不適格や0費用へ置き換えていない。[P12]、[P12POINTS]

### 17. 全18roundの結果と、採択回路の変化

| 分子 | A：Truncated総RZ | B：Truncated総RZ | 参照Full PF | 参照Closed PR | P2の判定 |
|---|---:|---:|---:|---:|---|
| H4 | 3.337772238e11 | 3.864575491e11 | 5.647934021e11 | 1.303898609e12 | 結果A：Bでも両参照より低コスト |
| H6 | 1.222907256e12 | 1.868265339e12 | 2.082026044e12 | 1.617753939e12 | 結果B：BはFullより安いがPRより高い |

「評価A/B」と「判定名の結果A/B」は異なる。表は固定cohort・選択・今回の保守上界による結果であり、他候補のno-goや大域最適性ではない。元Aの低コスト性と条件付き資格は維持する。[P12]、[P12RESULT]

| B policy | 総CX | 総size | serial depth上界 | 十分trial総数 |
|---|---:|---:|---:|---:|
| H4 | 6.802588264e10 | 5.724967151e11 | 1.385172409e11 | 14825 |
| H6 | 3.249293075e11 | 2.767213795e12 | 5.159556638e11 | 14353 |

Bではretention・formula・hの選択も変わる。H4 m17はcoeff_L1_2 / 元順Y4 / .16、BT=.021852902724、BP=.013082229005、和=.034935131730。H6 m17はprefix15_I0 / Y4 / .16へ移り、BT=.001786505392、BP=.021644598729、和=.023431104121となった。18中15 fragment保持の価格も含めた選択である。全roundの値、ρ/β、α/Nは[P2_ROUNDS.csv][P12ROUNDS]に保存されている。[P12]

policy/α/N凍結後に新規6点を独立expm/Schur・native X/Yで照合し、保存同一点はkeyと実価格を確認して再利用した。P2照合点のunique総数は43、全6 policyの18round mean decoder・十分N・α予算が通過した。決定論価格に新MCは不要で、Full/PR参照は元資格とPR独立holdoutを保持した。[P12]、[P12VALID]

### 18. 同一B回路をAで評価した限定ablation

採択policyの変更と、同じ回路の評価規則による効果を分けるため、`A_on_B_fixed_circuits`を凍結した。次表は原REPORTの保存ablationをそのまま引用する。

| 分子 | A自由選択の総RZ | B回路をAで評価 | 同じB回路をBで評価 | 回路policy変更の経路差 | 同一回路の評価規則差 |
|---|---:|---:|---:|---:|---:|
| H4 | 333777223814 | 344455185076 | 386457549139 | 10677961262 | 42002364063 |
| H6 | 1222907255629 | 1863969347635 | 1868265339430 | 641062092006 | 4295991795 |

出典：[P1/P2 REPORT §7][P12]、[RESULTS.json][P12RESULT]。`A自由選択 → B回路をA評価 → 同じB回路をB評価`の順序付き経路であり、唯一の因果分解ではない。policyとα最適化の相互作用が残り、割合を新たに算出・合成しない。

H6ではBの同じ回路をAで評価しても元Aの価格へ戻らず、Bの追加Nだけで全差を説明できない。保持fragment・formula等のpolicy変更が大きく関わる。H4はnear-zero combined点に強く依存しない保守的policyでも両参照より安い。H6では今回のBでPRに対する低コスト性を確認できなかったが、保守上界の緩さと物理的相殺を分けて解釈する。[P12]

<a id="conclusions"></a>

## 第8部　研究全体で確認できたこと、一般化できないこと

### 19. 成果と成立範囲

| 確認できた成果 | 固定証拠と、その意味 |
|---|---|
| PR内部BCH誤差相殺 | 固定H4/元groundでD内部・混合の主要energy補正を共同調整し、有限RTE・native・18round資源まで改善。作用素4次や他状態への保証ではない [D] |
| 回路資源の追加削減 | Singleton限定、共同isometry、対角合成を実回路へ接続。depth悪化や既知controlの共通効果も分離 [K]、[J]、[F] |
| Closed P3のsampling側改善 | 同じ有限P3平均、weighted/finite-bit/native会計を保持し、指定paired比較で約2%のPR総RZ改善 [P3]、[OPT] |
| h/resolution/α共同選択 | event数・減衰・trialのtradeoffを扱い、旧固定Closedを同一holdoutで再評価した比較でRZ6.0327%減 [OPT] |
| 分子ごとの方式順位 | 固定DF/元Hの同精度・成功条件、有限menuでH2 PR、H4/H6 Truncatedが低RZ [DF] |
| H6有限h近傍の資格維持 | 指定5つの整数stepを固定総時間で検査し、全5点が条件付き資格を維持 [P12] |
| 相殺依存度の限定検証 | H4はBでも両参照より安い。H6はAでTruncatedが低コスト、BではFullより安いがPRより高い [P12] |

初期の負結果、旧compiler由来のSTOP、C2の未資格、always-partialの悪化、共同isometryのdepth増、対角融合のPR固有RZ利益の小ささ、H2の実fragment除去の不採択、H6のBでの高コストも成果の一部である。後続研究が成功しても、過去の未実施条件を遡って資格済みにしない。

この研究が示したのは、誤差の種類、状態への作用、信号半径、必要trial、具体的回路価格を共通の推定課題で接続すると、補助指標だけでは見えない方式順位と改善限界が現れることである。世界的新規性や先行研究全体への優位性を確定する新しい文献調査は行っていない。既知の対称PF、QR/Givens、isometry、parity network、RPE/Hoeffdingを再利用した事実を保持する。[OLD]、[P3]、[DF]

### 20. 支持しない一般化と未解決の限界

PRが常にFull PFより安い、決定論PFが常にPRより安い、Truncated PFが一般に最良、という一般命題は得られていない。相殺の任意の分子・状態・hへの移送、H2/H4/H6からの漸近スケーリング、有限menu最良値の大域最適性、理想native RZ順位からの実機・耐故障総資源順位も支持しない。[OLD]、[DF]、[P12]

保証は固定保存Hamiltonian、その厳密ground oracle、理想native、モデル化した有限bit/独立drawの条件付きである。元の連続積分・実数Hamiltonianへの無条件包含は未資格。保存stateの数値残差を囲んだことは、未知の近似stateへの頑健性を意味しない。状態準備、noise、routing、T合成、別geometry/基底等の費用と成功条件は未評価である。

十分trial式は必要最小shot数の証明ではない。価格MC SEは期待回路価格のnominal不確かさで、RPE failureや価格modelの系統誤差を覆うものではない。独立holdout・独立計算経路は、別研究者による外部再現や詳細科学レビューと同義ではない。今回これらの未解決点を再判定する計算・レビューは実行していない。[OPT]、[DF]、[P12]

### 21. 今後の候補と、判断を戻す範囲

| 未解決事項・研究候補 | 既存資料が示す目的 | 現在の位置付け |
|---|---|---|
| P3：retention/cutoff感度 | 同じPF/h/order/tでdrop集合を少し変え、相殺・資格・価格の持続性を調べる。Hkeep-only pilotと一部round資格は再利用可能 | 静的調査の提案。P2のpolicy比較は、固定条件での一般的cutoff感度を完成させたものではない [SURVEY]、[P12] |
| P4：初期状態摂動 | 元groundと直交方向との有限混合等により、物理的な近似入力の信号・branch・資格を調べる | exact-ground用qualifierへのstate差替えだけでは不十分。保存residual allowanceと区別 [SURVEY] |
| 別geometry・分子・基底 | DF構造と切り捨て余地、PR tail価格の違いが方式順位へ及ぼす影響を検証 | 3分子の結果から移送を認定しない [DF] |
| Ground oracle仮定を外す費用 | state準備・overlapと成功条件を含む同じ課題の総費用を評価 | 現在のno-prep費用の外側 [DF]、[P12] |
| PRの分割・sampling・合成 | 残るbasis価格とrange/normalizationの共同問題を改善できるか調べる | 既存Closed P3の成功から新law・新familyの成功を予告しない [OLD]、[P3]、[OPT] |
| 保守上界と有限menuの限界 | P2の保守diskの緩さ、cohort外・連続hの可能性を区別する | 現在の負結果をno-goとしない。追加探索・新上界は未実行 [P12] |

これは既存の未解決事項の整理であり、新しい研究方針の正式採択ではない。**次の担当はユーザー**で、成果の採否、研究を継続するか、継続する場合の科学条件・有限範囲を判断する。Codexは追加指示なくP3/P4、別分子、H8以上、別PF family、新sampling law、詳細科学レビューへ進まない。

<a id="sources"></a>

## 第9部　固定出典、SOURCE、再利用可能な証拠への入口

### 22. 必須6資料の対応と原本保全

公開snapshotのSHAは閲覧版の識別子で、実loadされた科学SOURCEと一致するとは限らない。以下の固定REPORTとSOURCE表・bundle・保存結果を組にして読む。本書作成のために旧REPORT、科学SOURCE、raw、result、grant、ledger、marker、STOPを変更していない。

| 資料 | 固定公開commit | 正本・SOURCE/結果への入口 |
|---|---|---|
| A：対角融合までの旧統合 | `100377dac8635fc49074147bfd8b42f52f2b7311` | [旧統合報告][OLD]。初期C-N/C-S/C-P、C2、compiler A、D、singleton、共同isometry、融合の原REPORTと完全SOURCE対応を収録 |
| B：Closed P3統合 | `2e38af3b12200c48c4f1b793638d223baff24e76` | [REPORT][P3]、[SOURCE][P3SOURCE]、[FAIR_FINAL][P3FINAL]、[raw索引][P3RAW] |
| C：共同最適化 | `c5a3c102199fe97f4d5c21837c059ae2684f5ef2` | [REPORT][OPT]、[SOURCE][OPTSOURCE]、[extended/FINAL][OPTFINAL]、[ROUNDS][OPTROUNDS]、[raw索引][OPTRAW] |
| D：3分子DF比較 | `da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13` | [REPORT][DF]、[SOURCE][DFSOURCE]、[ordering比較][DFCMP]、[資格][DFVALID]、[機構][DFMECH]、[raw索引][DFRAW]・[復元][DFREPRO] |
| E：相殺の既存証拠調査 | `12e74dd68568f493ed39937f8856f26ce48d33e1` | [REPORT][SURVEY]。既存値と未検証事項を分類しP1〜P4を提案。新しい科学runのSOURCEではない |
| F：P1/P2科学検証 | `92495a04d52c65142ed16b3abe977345065f9745` | [REPORT][P12]、[SOURCE][P12SOURCE]、[RESULTS][P12RESULT]、[VALIDATION][P12VALID]、[raw索引][P12RAW]・[復元][P12REPRO] |

Bの数値は公平性補助確認後のFAIR_FINAL、Cは局所拡張後のextended/FINAL、Dは順序audit後のordering/結果、Fはlogical h追加会計後のP1/FINALとP2/FINALを用いる。各rootの初期FINAL、中間qualification、旧CSV、失敗logを最新正本へ混ぜない。元rawにはこれらの履歴も保全され、詳細は各原REPORTに記録されている。[P3]、[OPT]、[DF]、[P12]

Dの科学code履歴tipはa4e32484254a090eab6d221776a50a1cfd9f0f17、Fはedb013f09de01d6bdb9184cc8da7347eb94baa9aで、それぞれSOURCE_HISTORY.bundleに保全される。FのP1初期資格/独立照合は241944d2182be59112c70a82389d72c7768e93eb、logical h追加会計/P2 planは7101be1597f0798408f1d36324856d6ee4bb2929、P2資格/凍結/独立/finishはbf8c4ded7ff6ffe7e1b050713d9afd4ad09fc890。公開SHAへ原実行SOURCE欄を付け替えない。[DFSOURCE]、[P12SOURCE]

### 23. 保存値とコードを再利用する際の対応

| 既存資産 | 再利用対象と境界 |
|---|---|
| `track_c_d_internal`等、既存BCH/有限RTE証拠 | D係数・case・時刻・分割が合う保存資格。別分子・状態・新hへ無検証で移さない [D]、[OPT] |
| `track_c_joint_*`、`track_c_diagonal_*`、compiler A | 列projector、controlled scalar、same-basis共有、exact角度和、fallback。採用catalogue・新角度のsource/native補足が必要 [J]、[F]、[DF] |
| G7 `CanonicalGenerator`、C `track_c_closed_p3*` | 有限P3平均、実proposal/weight、全trial/range、whole occurrence価格。weighted推定量を無重み平均へ置換しない [P3] |
| `track_c_closed_p3_joint_opt*` | finite menu接続、整数α/N、shared-cell covarianceとデータ分離。値の読取と再最適化を区別 [OPT] |
| `track_c_df_truncation`、`track_c_df_order_audit` | public replay、exact drop集合、元H信号、modular builder、directed資格、expm/Schur/native X/Yとdecoder [DF]、[DFREPRO] |
| `track_c_df_cancellation_validation/conservative/results` | fixed-time adapter、個別複素disk、凍結policy・限定ablation。results/exportも数値処理を含むため今回実行しない [P12]、[P12REPRO] |

公開input codecとbundleはprivate pickleの公開や旧job再開を必要としない。新しい科学条件での実行は、別途定めた許可scope・fresh出力でのみ扱う。本書では科学moduleをimportせず、既存CSV/JSON fieldの引用・文書値との対応とGit object/リンクの技術確認だけを行った。

### 24. 本書の作成条件とSTOP

着手前にrepositoryのAGENTS.md、PROJECT_MAP.md、研究概要と関連契約を確認した。正式共通指示はSERVER 2上で取得できた `research_project_instructions_unified_2026-10-25.md` を参照した。未公開資料や別サーバーの全チャットを取得済みとは扱わない。旧個別契約の実行条件と過去のmandatory STOPを保持する。

数値は原REPORTおよび対応する保存結果の表示値を引用した。新しい誤差上界、資源値、差分、改善率、因果寄与率を計算していない。文書統合の技術確認を新しい科学的資格判定や第三者レビューとは呼ばない。先行研究に対する未確認の新規性・優位性も追加していない。

**本統合報告の専用branch公開、remote SHA/tree・全文・主要参照先・旧資料不変の照合を終えた後、mandatory STOP。** 次の科学研究の採否はユーザーに戻す。Codexの自動後続研究はない。

[OLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/100377dac8635fc49074147bfd8b42f52f2b7311/docs/research/track_c_o3_research_summary_20261108/TRACK_C_O3_RESEARCH_SUMMARY.md
[P12]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/92495a04d52c65142ed16b3abe977345065f9745/docs/research/track_c_df_cancellation_validation_20261108/REPORT.md
[DF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/docs/research/track_c_df_truncation_20261108/REPORT.md
[PILOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a78054dc48737b06b9396e649f53ecb9efd4bd3b/docs/research/track_c_h4_pilots_20261106/REPORT.md
[GB]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/docs/research/track_c_guarantee_cn_price_20261106/B_REPORT.md
[A2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ce84fa8e2d2f1d487df3f05e63d519ebb03f83ad/docs/research/track_c2_a_pilot_20261106/REPORT.md
[B2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/602ca4b67b1037e221e910a2733444630c5add62/docs/research/track_c2_b_strong_baseline_20261106/REPORT.md
[STATIC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05d9c5d2eb984e0f3ce89f44a3f2928685b8d4bf/docs/research/track_c_compiler_static_investigation_20261106/REPORT.md
[ACOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f45e9b07bf730f7272933b28055914cfe76fdaab/docs/research/track_c_compiler_a_correctness_20261106/REPORT.md
[GA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/docs/research/track_c_guarantee_cn_price_20261106/A_REPORT.md
[MTHEORY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/48209b088ca0dffd22a8d3fddcb82888eb29d3a5/artifacts/track_c_mixed_formula/2026-11-08/THEORY_AND_LOCAL.json
[M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/48209b088ca0dffd22a8d3fddcb82888eb29d3a5/docs/research/track_c_mixed_formula_20261108/REPORT.md
[D]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9dacb639eabdd3cf499e89f8807ff50b68e23ef/docs/research/track_c_d_internal_20261108/REPORT.md
[DFINAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9dacb639eabdd3cf499e89f8807ff50b68e23ef/artifacts/track_c_d_internal/2026-11-08/FINAL.json
[DREG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9dacb639eabdd3cf499e89f8807ff50b68e23ef/artifacts/track_c_d_internal/2026-11-08/REGISTERED_COMPARISON.json
[DNR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9dacb639eabdd3cf499e89f8807ff50b68e23ef/artifacts/track_c_d_internal/2026-11-08/NONRESONANT.json
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10bf5a4813559cea807fcd8c50f3b966407747d1/docs/research/track_c_bottleneck_20261108/REPORT.md
[KFINAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10bf5a4813559cea807fcd8c50f3b966407747d1/artifacts/track_c_bottleneck/2026-11-08/FINAL.json
[JFINAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2745d57bea5c28604739bf9c01338e14e649c467/artifacts/track_c_joint_isometry/2026-11-08/FINAL.json
[FFINAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/artifacts/track_c_diagonal_fusion/2026-11-08/FINAL.json
[J]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2745d57bea5c28604739bf9c01338e14e649c467/docs/research/track_c_joint_isometry_20261108/REPORT.md
[P3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e38af3b12200c48c4f1b793638d223baff24e76/docs/research/track_c_closed_p3_20261108/REPORT.md
[P3STAGE1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e38af3b12200c48c4f1b793638d223baff24e76/artifacts/track_c_closed_p3/2026-11-08/STAGE1.json
[P3SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e38af3b12200c48c4f1b793638d223baff24e76/artifacts/track_c_closed_p3/2026-11-08/SOURCE.json
[P3FINAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e38af3b12200c48c4f1b793638d223baff24e76/artifacts/track_c_closed_p3/2026-11-08/FAIR_FINAL.json
[OPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md
[OPTSOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/artifacts/track_c_closed_p3_joint_opt/2026-11-08/SOURCE.json
[OPTFINAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/artifacts/track_c_closed_p3_joint_opt/2026-11-08/extended/FINAL.json
[OPTROUNDS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/artifacts/track_c_closed_p3_joint_opt/2026-11-08/extended/ROUNDS.csv
[DFSOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/SOURCE.json
[DFCMP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/ordering/COMPARISON.csv
[DFVALID]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/ordering/VALIDATION.json
[DFMECH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/ordering/TRUNCATION_MECHANISMS.csv
[SURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/12e74dd68568f493ed39937f8856f26ce48d33e1/docs/research/track_c_df_cancellation_evidence_20261108/REPORT.md
[P1CSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/92495a04d52c65142ed16b3abe977345065f9745/artifacts/track_c_df_cancellation_validation/2026-11-08/P1_RESULTS.csv
[P12SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/92495a04d52c65142ed16b3abe977345065f9745/artifacts/track_c_df_cancellation_validation/2026-11-08/SOURCE.json
[P12POINTS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/92495a04d52c65142ed16b3abe977345065f9745/artifacts/track_c_df_cancellation_validation/2026-11-08/P2_POINTS.csv
[P12RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/92495a04d52c65142ed16b3abe977345065f9745/artifacts/track_c_df_cancellation_validation/2026-11-08/RESULTS.json
[P12ROUNDS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/92495a04d52c65142ed16b3abe977345065f9745/artifacts/track_c_df_cancellation_validation/2026-11-08/P2_ROUNDS.csv
[P12VALID]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/92495a04d52c65142ed16b3abe977345065f9745/artifacts/track_c_df_cancellation_validation/2026-11-08/VALIDATION.json
[F]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/docs/research/track_c_diagonal_fusion_20261108/REPORT.md
[P3RAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e38af3b12200c48c4f1b793638d223baff24e76/artifacts/track_c_closed_p3/2026-11-08/RAW_INDEX.json
[OPTRAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/artifacts/track_c_closed_p3_joint_opt/2026-11-08/RAW_INDEX.json
[DFRAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/RAW_INDEX.json
[DFREPRO]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/docs/research/track_c_df_truncation_20261108/REPRODUCE.md
[P12RAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/92495a04d52c65142ed16b3abe977345065f9745/artifacts/track_c_df_cancellation_validation/2026-11-08/RAW_INDEX.json
[P12REPRO]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/92495a04d52c65142ed16b3abe977345065f9745/docs/research/track_c_df_cancellation_validation_20261108/REPRODUCE.md
