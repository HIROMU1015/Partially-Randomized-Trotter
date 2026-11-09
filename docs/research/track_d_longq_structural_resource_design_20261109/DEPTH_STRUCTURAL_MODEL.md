# Depth：固定native列の厳密max-plus計数とP0 bridge

2026-11-09 JST。[条件・対象policy](REPORT.md)。**ここで厳密に導出するのは、既に確定した命令列に対するdepthである。入力descriptorから既存P0の出力列をfull compileなしで得る問題は別に残る。** topology-freeの依存depthを扱い、実機のgate時間・routing・T合成の価格とは区別する。

## 1. 固定wordのrecurrence

wire集合Wにはqubitsに加え、測定先classical bitも含める。今回の狭いdomainは、bound命令・明示wire、control flow／条件／reset／変数なし。directiveは対象外とする。最後のreadout・測定は元metricsと同じ規約で含める。

命令oの対象wireをS、各wireの到着時刻をt_iとする。指標gに対し、

\[
a=\max_{j\in S}t_j+w_g(o),\qquad
t'_i=\begin{cases}a&i\in S,\\t_i&i\notin S.\end{cases}
\]

totalでは対象命令の重み1、RZ/CX depthではそのgateだけ1、他は0。最終depthはmax_i t_iである。**対象外のgateもwireを同期する。** 例えばRZ(0), RZ(0), CX(0,1), RZ(1)のRZ depthは3であり、CXを削除した列の2とは異なる。この仕様は[Qiskit1.3のdepth実装](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/circuit/quantumcircuit.py#L3479)を静的確認し、既存C7-2の`weighted_depth`が同じ狭いdomainを実装している。

空列は恒等transfer。global phase自体はgate depthを増やさないが、その位相をancilla P/RZへ移した場合は実際の命令として数える。位相が単独ではdepth0だからといって、周辺P0の再合成への影響が0とはしない。

## 2. 全wire transferの圧縮表現

max-plusでは和をmax、積を通常の加算とし、依存がない項は−∞とする。template Tのtransfer M_Tは、

\[
t_{out,i}=\max_j\{(M_T)_{ij}+t_{in,j}\}
\]

を満たす。恒等行列は対角0、他−∞。一つのgateについて、S×Sの要素をw、それ以外のwireの対角を0にすれば上のrecurrenceになる。複数gateでは合成する。**時間順A→Bなら M_{AB}=M_B\otimes M_A**であり、行列積の順序を逆にしない。異なるregisterへのwire permutationは実際のmappingに従って行・列へ適用する。

各templateにはM_total、M_RZ、M_CXを別に保持する。同じcritical pathが三指標を同時最大化するとは限らない。加えてnative count vector（rz/sx/x/cx/measure/その他）とwire使用情報を持つ。固定wordのcountは加算できるが、compiled chunkを連結後に再最適化した価格は単純加算できない。peak qubitsは同時使用registerの和集合／最大幅で、template個数に比例して増やさない。

**零入力で得たexit vectorだけでは足りない。** X(0)とX(1)からなる独立2gate列と、CX(0,1)一つは、totalの零入力exitがともに(1,1)である。しかし入力(10,0)では前者(11,1)、後者(11,11)。新しいtemplateの途中へ任意arrivalを渡すには依存行列相当の情報が要る。単体scalar depthはさらに情報を落とす。

このmax-plus表現の正確性はgateごとのrecurrenceから帰納的に従う。有限のmock例の一致だけを根拠とした経験式ではない。ただし命令のsupport・重み・順序がP0後と同じことは、この数学的導出では保証していない。

## 3. Whole-path Referenceの合成

既存`reference_gadget(later,earlier,phase)`の実順序は、最初のHadamardの後、system index昇順のCSWAP cascade、system側later.inverse()と別register側earlier、同じ順のCSWAP cascade、ancilla P(χ)、X/Y readout・測定である。V=ABのAはlater、Bはearlierに対応し、時間順ではBの後Aが作用する。短いgadgetのq回反復へ置き換えない。

固定native展開が与えられれば、

\[
M_{ref}=M_{readout/measure}\otimes M_{P(\chi)}\otimes M_{swap2}
 \otimes\mathrm{embed}(M_{A^{\dagger}},M_B,I_{anc})
 \otimes M_{swap1}\otimes M_{H}.
\]

lane部分は互いに独立なのでblock diagonalにembedできる。だが各CSWAPが共通ancillaに触れ、wireごとの入口時刻と出口の待ち合わせを作る。shellの到着時刻を全wire一律のscalarへ潰すとこの依存を失う。CSWAP cascadeの内部native CX/RZ等も展開順のまま扱う。P0が周辺と混ぜる場合は次節のbridgeが必要である。

A†には**actual inverseの命令template**を使う。max-plus transferは一般に可逆でなく、M_Aの行列逆を使ってはならない。同一support・同一重みのgateを単に逆順にする限定表現では転置の関係を使える場合があるが、sxのinverse再展開やP0再合成後にも同じnative列・重みになる保証はない。今回は実際の逆命令を基準とする。

C7-2固定人工反例では、q2 sameのReference全体total162に対しshell42＋max(lane115)=157、差+5で、RZ/CX depthも各+3だった。q2 mixed/q4 same/q4 mixedにも同種の反例がある。これは**scalar shell＋max(lane)での情報損失とP0界面を分離すべき**ことを示す。新しいtransfer設計がこれらの実P0数値を再現した試験は今回行っていない。+5/+3をH4共通補正としない。

## 4. P0後へ対応させるinterface

[Qiskit1.3 builtin pass manager](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/preset_passmanagers/builtin_plugins.py#L535)のopt1では、1q再合成とinverse cancellation等をdepth/size固定点まで反復する。既存phase compilerは、その前のcustom定義展開と、宣言位相を失わないMPFR160 offsetの処理も含む。topology-freeなのでroutingによる新SWAPを価格に加えない一方、将来の実機routing費は未評価である。

P0前のtemplateを独立compileしてcount/transferを足すと、boundaryを跨ぐinverse pair・1q run・再translationが失われる。必要候補は次のとおり。

| interface情報 | 保持する理由 |
|---|---|
| wireごとのordered frontierとcontrol/target方向 | 相殺可能な隣接CX等を、単なるcountで識別できない |
| 未確定1q runの実パラメータ・位相／積 | 再合成でRZ/sx/x数とwire depthが変わる。zero/special-angle判定も関係 |
| basis/frame・relative transitionのactual IR | frame名だけの同一視、隣接Gaussianの二重課金を防ぐ |
| step/cutとlane対応、forward/inverseの区別 | 全pathの順序・分断箇所・actual A†を維持 |
| scalar/identity/χ・readoutの未確定寄与 | 不正なphase移送やancilla依存の欠落を防ぐ |
| 使用したSOURCE／emitter／P0 profile／最適化policy／wire mapping | C1 ON/OFFや別compiler結果をcache混用しない |

これらは**候補情報**で、固定サイズで常に十分と証明したものではない。相殺が連鎖すると遠い命令が隣接し、固定個数のsuffixだけでは閉じない可能性がある。まず閉包の証明または反例分類が必要である。可変長frontierなら厳密な列を保持できる可能性はあるが、worst case O(L) memoryへ戻り得る。

P0の実pass順・数値的1q合成・停止条件を変えたstreaming optimizerは、数学的に等価でも同じP0価格とは限らない。新しい低価格IRを許す場合は、別policyとして測定同値性と費用を検証する。今回は新compilerもP0 pass実行も行っていない。

## 5. 計算量と長qでの見込み

Wはqubit＋必要classical wire数、Lは実際のnative word長、Kはtemplate呼出し数。以下は有界gate arityの理論的演算量であり、P0閉包を前提に含めない。

| 処理 | 時間 | 作業memory | 注意 |
|---|---|---|---|
| 一度だけnative streamを走査 | O(L) | O(W) | countと三種類のarrival、全word格納不要 |
| template transferをgate列から作成 | O(L_T W) | O(W²) | affected rowをmaxする。各metricを別に保持 |
| cached transferをarrivalへ適用 | O(K W²) | O(W²)＋cache | n固定ならqに比例。lane blockのsparse性を使える |
| transfer同士の一般合成 | O(K W³) | O(W²)＋cache | 到着時刻だけ要る呼出しで毎回dense合成しない |
| event/run特徴のstream | O(qr) | 集約O(1)〜O(W²)＋cache | 全event原本の保全storageとは別。逆順lane用に保存descriptorの逆読取等が必要 |
| P0互換frontier | 未確定 | 未確定、最悪O(L) | 有限閉包・反復回数・表現を証明するまで一定memoryとしない |

固定n8/r32のq1024〜4096なら、命令streamや再利用templateのcount/max-plus更新自体は全状態伝播より有望である。しかし角度依存の全templateを無制限にcacheするとevent数に比例してmemoryが増える。数値angleを粗く丸めてcache hitを作らず、同一IRの再利用と構造的template／実パラメータを分ける。実行時間やspeedupの実測値は未取得。

qに依存しない単一transferの累乗は**同一template反復の場合だけ**使える。PRのordered event列は一般に各stepが異なるため、q乗で置換しない。期待depthにはE[max]を含み、平均arrivalや平均run数のmaxで置換するには別の保証が要る。

## 6. 今回の確認と残条件

[mock試験](check_maxplus_mock.py)は、既存counterとの整数一致とtransfer合成の失敗しやすいケースを確認する。H4価格、C7-2保存P0の再計数、Qiskit新compileは対象外。今後P0閉包が示せない場合も、このbackendは**確定native列の厳密計数器**として使え、未解決部分は経験的proxyとして明示できる。P0後の保証へ自動昇格させない。
