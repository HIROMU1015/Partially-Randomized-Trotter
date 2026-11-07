# O2-S2/J1：Return構造とwhole-path Samplingの接続調査

2026-11-08 JST。**`NO_NEW_CHEAP_ACQUISITION_STRUCTURE_IDENTIFIED`**。A2のアクセス不足を記録した後、既存SOURCEで新しく安価に取得できる構造があるかを静的に確認した。一般的な不可能性・科学的NO-GOの判定ではない。

## 1. 今回の具体的な問い

J1に必要なのは新しいISの式ではなく、**return状態から価格依存scoreの部分和と次の条件付きlawを取得でき、全path表を作らずに済む追加性質** である。以下の照合は既報S2の再計算や新proposalの生成を伴わない。

元path係数をaω、既存の正価格scoreを `aω/√Cω` と書けば、必要なのは

\[
 Z=\sum_\omega a_\omega/\sqrt{C_\omega},\qquad
 q(\omega)=a_\omega/(Z\sqrt{C_\omega})
\]

の取得と、実finite-bit q・weight・supportの接続である。この既知whole-path ISを新手法として提案しない。元weight/phaseを別に持つ場合はそれも保持する。

## 2. SOURCEによる接口の切り分け

| 既存資産 | 実際に取得するもの | 取得しないもの／重複判定 |
|---|---|---|
| return_aggregation.py：ReturnKernel | p,x,mからのfirst-passage/return series、指定reduced wordのsigned mass、parent aとchild masses、局所digital packet | native IR、D/wrapper、全parent normalizer、価格tiltの部分和を返さない。G6/Greenを再実装しない |
| g10_generator.py：ClosedP5Tail | 既存P5 closed groupとordinary tailのfinite-bit law、実q/weight | group indexを価格同値類とする保証はない。Closed/Fullの既知関係を新利益にしない |
| parent_allocation.py：digital_proposal/reweight_events | 与えられた完全有限tableと外部costからのroot enclosure、dyadic law、M2/range | cost acquisitionは外部必須。Parent/S4の再探索をしない |
| rte_development/boundary.py | 完全なnative gate keyのinverse照合、stack簡約、pair境界の相殺深さ、保存cacheからのliteral価格 | formal return classだけから価格を取得しない。既存D2 kernelを新設しない |
| s2_multi_event.py：pair_table/normalizer_enclosure | 保存event IRから全ordered pair、価格ごとの係数集約、既存IS | normalizerの同価格groupは全pair表を作った後。表取得の回避ではない |

参照snapshot・blobは[SOURCE.json](SOURCE.json)。科学moduleをimportしていない。

## 3. Return係数と価格normalizerは異なる

G6のword seriesはQ_i²=Iのformal returnを集約する。parentの係数massには `sqrt(a²+s²)`、child rotationには `s/a` が現れる。元のsigned polynomialのreturn和を、そのまま全parentの正係数和や価格tilt Zにしてはいけない。

非加法価格では前後のnative IR・basis・wire・angle・orientationが必要になる。境界kernelは実際に `inverse(left[-1−k]) == right[k]` を検査している。例えば局所接口で、同じformal return情報だけを持たせても、右先頭のnative rotation keyが左末尾のinverseか別keyかで相殺は変わる。これはSOURCEの入力依存関係の指摘であり、実H4にそのような二pathがあると新たに確認した結果ではない。

完全なIR stackを状態へ含めれば、既存の逐次簡約で価格を更新できる。しかしそれはreturn summaryだけによる縮約ではなく、状態数が小さいという保証もない。Qiskitのfull compileをこのliteral stackと同一視もしない。

## 4. 安価な新構造として検討した三つの接続

| 接続案 | 成立に必要な追加性質 | 今回の結論 |
|---|---|---|
| return/groupを価格同値類とする | 同classで全wrapper価格が一定、または誤差つき有効価格境界があり、class質量とclass条件付き生成も取得可能 | 係数のclosed groupはあるが価格不変性の証拠がない。角度・basis・contextを落とす価格oracleを作らない |
| native境界状態のtransfer/DP | 小さい閉じた状態集合、exactな遷移質量と価格、全suffixのscore和 | D2はpairの完全IRを使う既存部品。whole-path用の小状態閉性は未確認。新しい大規模DP/全列挙を開始しない |
| grouped root/Parent scoreをreturn queryから取得 | 全parentを訪問せず価格score部分和を得られる恒等式または既存接口 | 局所queryと有限table再配分は存在するが、この部分和はない。同じParent/ISへ還元される範囲だけが実装済み |

仮に加法/局所価格へ制限しても、`1/√(Σ cost)` は通常はstepごとの積へ分解されない。通常のMarkov samplerをそのまま価格tilt samplerと呼べない。累積cost等を含む別の閉じた状態や、精度つき積分/近似の正当化が必要なら、それを未取得のまま残す。新しい理論・汎用normalizerライブラリを無断で作らない。

Returnとsamplingを併用するだけの効果は、既存Closed/CTSと同情報を与えたISをbaselineにすれば消える可能性がある。新しい取得構造を特定できない現段階では、量子測定・compileや古典benchmarkを始めても今回の問いを判別できない。

## 5. 統計・価格・生成費を省略しない

既存のactual qに対する `w=a/q`、`M2=Σ a²/q`、`L=max|a/q|` を使う。安いpathが増えたことと必要N×full価格の改善は別で、range項・bias余裕・整数Nを残す。実測noiseをpath平均の分散に置換しない。strict nativeとcompiled RZは別座標である。

量子価格に加えて、kernel setup、価格情報の取得、normalizer、bit生成、条件付き取得、cache miss、拒否/zero trial、online weight評価の古典費を記録する必要がある。C1のzero-fillは既存機構で、zero pathを観測しなかった旧結果を新しいzero回避利益へ書き換えない。

## 6. 維持する旧結果と次の条件

[S2結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc42c3a9f77e960f415d0475da2ba03bb56252c7/docs/research/o2_s2_multi_event_validation_v2_20261106/REPORT.md)は3-qubit synthetic D2 Closed P5、p=(1/5,3/10,1/2)、state000、degree5、r2、τ5/7、x5/14、root256/probability160、固定native精度の有限pair表・T主指標での限定負結果である。分子geometry/basis/DF/L_D/δ窓は非該当。期待path単価低下を十分N増加が上回ったという既報結論を維持し、表・law・Nを再評価しない。

[R1選別](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6305ee2e33436ed96d8aefdee15588676242000/docs/research/o2_r1_server_candidate_screening_20261106/REPORT.md)の `NO_SUPPORTED_R1_CANDIDATE` を変更する具体的なreturn/child構造も、今回のJ1読解からは得られなかった。Closed/CTSを除外せず、G6〜G10やC1を再実行しない。

再検討の最小材料は、既存入力上で具体的な **価格に関する十分なsummary、閉じた遷移、score部分和/条件付きlaw** を一組示すこと。そのうえで同情報baseline、finite-bit補正、生成費を比較できる場合だけ最小adapterを作る。本調査ではその組を特定できず、新しい実装・人工suite・科学実行契約を作らなかった。
