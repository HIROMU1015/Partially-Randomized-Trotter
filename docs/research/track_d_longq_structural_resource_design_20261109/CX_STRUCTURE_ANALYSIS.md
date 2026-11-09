# CX count：構造的に説明できる部分と未閉包のP0界面

2026-11-09 JST。[固定条件・証拠](REPORT.md)。Δ=Reference−Strong。対象はC6a〜C8-1と同じ強いwhole-path最適化policy（C1 head-tail追加融合OFF）。**208−224qは保存88＋8本の有限観測式で、今回全q・全pathの公式として採択しない。**

## 1. SOURCEに沿った発生源

| 要素 | Strong | Whole-path Reference | 計数時の条件 |
|---|---|---|---|
| head | uncontrolled Gaussianを挟み、control modeに応じた対角中心。`head_recipe`では初端uncontrolled、終端directional、step界面にはordinary中心を含む | 同じSOURCEをphysical segmentへ制限 | 全headを一律にordinary CRZ/RZZ、または一律directionalと数えない |
| RTE中心 | 保存signed angle/supportからのcontrolled rotationとparity | 既存`control_one`によるphysical rotation | event32個と、非identity・非零control中心32個が等しいには条件が要る |
| Gaussian | 同rotationをgroup化、異frameはphase補正したrelative transition。tailにも同basis共有 | 同じbasis実装・相対橋をlane内で再利用、後半はactual inverse | frame label一致だけでは同じGaussianと判定しない。実rotation/gaugeと順序を保持 |
| step境界 | q−1の合法half-sweep融合 | 両segment内部でq−2、global cut一境界は分断 | q−1とq−2の差、cut損失を別々に重複加算しない |
| scalar/identity | 一回集約してancilla relative phase | A/B global phase、path χ補正 | 元の位相を無視して固定費を合わせない。phase単体は1qでも周囲のP0再合成へ影響し得る |
| shell | Hadamard/readout/measure | 二CSWAP cascade＋χ＋readout/measure、追加n register | CSWAP単体価格を単純に2n倍してfull shell価格とはしない |
| P0 | 定義展開・translation・1q再合成・inverse cancellation | 同じP0、actual A†の逆向き展開も対象 | Gaussian/parity/端部/CSWAPの境界を跨ぐ簡約を残差として保持 |

`append_directional_diagonal`の非零RZは anc→j CX、RZ(−θ)、同CX の2本。非零RZZは parity CX、anc→j CX、RZ(−θ)、anc→j CX、parity CX の4本。対応physical RZは0本、RZZの標準parity列は2本で、**この孤立展開規約ではpremium各2CX**である。`control_one`はこの命令列のancilla=1作用を取るためX対やglobal phaseも残り、それらがP0で消える条件を別に見る必要がある。ordinary `RZZGate.control(1)`は別の定義展開であり、このpremiumを無条件転用しない。[SOURCE](SOURCE_INDEX.md)。

## 2. 既存反例が排除する主張

以下はC7-2固定報告の読解で、再実行・新価格計数ではない。

| 固定人工観測 | 排除する仮定 |
|---|---|
| zero directional中心はraw2CX→P0 0 | center数だけを見て常に2CXを課金 |
| mixed signed RZ/RZZでphysical2CX、ordinary18CX、directional8CX | ordinaryとdirectionalの同一視 |
| consecutive同ZZでcontrolled8→6CX、physical4→2CX | 孤立primitiveのP0価格の単純加算 |
| 同Zの＋/−でcontrolled4CXが残る一方physical0 | 数学的identityなら既存opt1が全て消すという想定 |
| 単体CSWAP8CXに対しscalar-onlyの4-CSWAP shell全体28CX | shellを4×8=32CXと決定 |
| Aとactual A†の連結がP0でsize0 | compiled chunk価格の和は全列価格と同じという想定 |

必要なのは、このP0が実際に行う簡約である。任意の代数的に合法な簡約を追加すると、同じ作用でも**既存P0と異なる価格policy**となる。既存最適化を弱めることも、P0より強い新compilerを同一価格と呼ぶことも避ける。

## 3. 一般会計と条件付き−224

C7-2/C7-4の一般会計を保持する：

\[
\Delta C_{CX}^{P0}=F_{gadget}+F_{ends}+q d_{step}
 +(q-2)b_R-(q-1)b_S+T(q,\omega)+\epsilon_{P0}(q,\omega).
\]

Fはshell/端部、dはstep bulkの差、bは当該規約での融合界面寄与、Tはevent/basis/angle依存の残り、εはその展開と実際のP0全体とのずれである。これらをデータだけから全て独立にfitできるわけではない。定義した展開を跨ぐ節約を二重計上しない。

より狭い候補は、同じ局所中心premium、Gaussian/parity共通部分の相殺、cut closureの会計を条件として、

\[
\Delta C_{CX}^{P0}=2nf+2G_0-2q(P_{det}+r)+R_{CX}(q,\omega).
\]

ここでnはsystem幅、fは孤立CSWAPのCX、G0は**この分解において独立に確定すべき**端部／cut Gaussian寄与、Rは残りを全て含む。G0を結果に合わせて定義して残差をゼロにする方法は導出にならない。

既存DF対角SOURCEでは、n=8に対してone-bodyには8RZ、各squared blockには最大8RZ＋28RZZがある。LD2の二つのsquared blockについて、prune/group後も全中心が非零で上のpremium会計が成り立つなら、

\[
P_{det}=8+2(8+28)=80,\quad r=32,
\qquad -2(P_{det}+r)=-224.
\]

これは係数の**条件付き構造説明**である。今回実H4のprimitivesを再評価して80を新確認したものではない。identity event、特殊角、同basis融合での角の和、control mode、normalization/scalar移送により、単純中心会計がそのまま価格にならない場合をRへ残す。normalization/attenuationはlaw・信号・shot条件にも属し、量子gateに現れない係数を勝手に新gateとして数えない一方、既存descriptorから削除もしない。

## 4. 208を導出できていない理由

孤立f=8、n=8なら候補shell項2nfは128である。しかしfull shellは既に非加法例を持ち、128が全wrapperの確定固定費という保証もない。**208−128からG0=40を逆算しない。** G0=40、端部との閉包、CSWAP/χ/relative basis界面残差0を独立SOURCEから証明した証拠は未取得である。

したがって、保存pathにΔCXのばらつきがない説明候補は「Gaussian等のpath依存bulkが両方式で相殺し、制御中心の一律premiumと固定cut/shellだけが残る」である。これは有望な仮説だが、P0がその相殺を任意列で保つという橋がない。有限残差0は全lawの分散0、母平均式、q4096公式を意味しない。

## 5. 残差0を主張するための証明義務

1. **中心の完全性**：prune後のnonzero/special angle/identity、符号・support、ordinary/directional/physicalの各展開を分類する。
2. **共通部分の対応**：StrongとA†/Bで、basis-run、relative Gaussian、parity、actual inverseに対応を作る。順序・gauge・phaseの対応まで含める。
3. **stepとcut**：q−1／q−2の界面規則と端部を一回だけ会計し、内部連結に対して規則が閉じることを示す。
4. **shell界面**：CSWAP cascade・共通ancilla・χ/readout・lane先頭末尾がP0で相互作用するケースを含める。
5. **実P0の閉包**：1q再合成とinverse cancellationの反復、translation、特殊角の判定を含め、未表現の跨界面変換が価格を変えないことを示す。有限窓で常に十分とは仮定しない。

これらが同じ展開規約で任意合法列に閉じる場合に限りR_CX=0を主張できる。部分domainだけ閉じれば、そのdomain限定式として扱う。新しいP0人工検証も今回は未実施で、最小ケースを[次段階案](NEXT_EXECUTION_PROPOSAL.md)へ分離した。

## 6. 絶対価格と長qへの限界

ΔCXがqだけの式になっても、C_StrongとC_Referenceの絶対値にはbasis/run/angle依存項が残り得る。全RPE費用には方式別期待価格が必要である。主Aを構造差だけで置換せず、既存 `_checked_price` に未資格式を登録しない。q64以降の追加fixtureやモデルfitは今回行わず、**全P0 bridge未完成**として終了する。
