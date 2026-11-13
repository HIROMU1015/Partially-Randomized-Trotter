# 全schedule技術拡張：元数値契約を適用する条件

元[NUMERICAL_CONTRACT](../h4_structural_kernel_20261113/NUMERICAL_CONTRACT.md)・run02のSOURCE/結果は変更しない。今回の作業は準備だけで、下記の物理処理は未実行。m0/m6の四値は固定Git objectからcopyし、再計算しない。

## 共通の構造と再検査

| 要素 | preparation共通の部分 | 新R/q/Tで検査・取得する部分 |
|---|---|---|
| support/basis | 同H4/保存hash、324非identity/9 basis、元operations・inverse・順序・to_dictの完全表現 | 保存descriptor/hash/全component係数・確率を照合。既存配列を別basisへ縮約しない |
| head・basis境界 | ΔPF=.1/S2/prefix1のnative headとliteral basis切替、reverse-headのreset構造 | 共通native branchを元Accountingで検査。各Rの有限Z graphでentry/close/outer/terminal対応を確認 |
| pending Z | None/数値0/各binary64 hexを区別し元順加算。reset又はuntouchedの論証 | 新angleで生じる正確なZ値・local reward・閉包・enclosureを毎R確認。旧56等のstate数を強制しない |
| phase | K2非identityではorder2のπ＋負productのπ。zero/positiveの分岐とcontrol barrier | 元π逐次加算・control reset検査を3×maxR=24,480まで拡張。maxRだけが増え、moduloや並べ替えなし。未検証条件ならSTOP |
| event law | 同componentの元probability/order grammar、products→rotation | 元make_rte_configでR4/8/16/32/64/255/510/1020/2040/4080/8160の11lawを取得。確率・angleを今回先計算しない |
| q・scalar | 同じouter出口を明示してrewardを合成、価格counterの途中resetなし | qPF・TはB rowをそのまま使用。scalar −cT一回、XY suffix/finishを実Tごとに作り直してparity |
| 確率・精度 | 同じfinite law、既存MeanKernel/Band、非負matrixの外向き丸め、mass tolerance=1e−10 | 毎RのP^R/unit mass/longdouble/interval、PASS時だけq反復とXYのE[S]。閾値変更・事後再正規化なし |

R2/R128のK0 PASSは他Rの科学PASSではない。同じsupport/native演算を再利用することと、law別kernelの検証完了を区別する。元の小状態論証はreset/untouched・finite support・control barrierが同じなら構成に使えるが、新angleでの局所parity・全状態閉包・確率mass・precision・scalar terminalは新たな物理K0が必要である。

## 最小のcore拡張

`Model.total_time(law)`はexplicit `law.T`を使う。旧2law callerには従来のfallbackを保持し、科学結果を再実行しない。新adapterはcanonical B rowのTを必ず供給する。旧コードの「R2以外はT6.4」という2点専用条件を他roundへ誤用しないための変更である。

`run_K1`へ確率丸めenclosureの評価関数を任意引数として渡せるようにした。central MeanKernel・Z演算・reward・boundaryの数式は不変。旧callerの既定経路は従来のFraction累乗のまま。kernel保存は既存関数のwire key列挙を一般化し、失敗時の完成wireだけも保存できるようにした。新framework/controller/C−S modelは作らない。

## 同じγ16式の効率的な外向き評価

元K1の最終丸め式は、n=qPF×R_outer、γ=16/(2^53−16)について

```text
lower = lower_before_factor × (1−γ)^n
upper = upper_before_factor × (1+γ)^n
relative bound = (1+γ)^n − 1
```

である。この**数式・γ・法則の意味は変更しない**。旧Fractionの厳密な分子/分母のbit長はO(n)となる。新経路は最初の1±γを同じFractionからbinary64の外側へ丸め、非負intervalの乗算をnextafterで外向きにし、二乗法で同じ累乗を囲む。相関した同じbaseを二乗する場合も外包性は保たれる。新intervalは追加の丸め分だけ保守的になり得るが、同じ数学的式を含む。centralやmassを調整してPASSさせる処理ではない。

計算はO(log n)回の固定精度interval乗算・O(1)個のendpointで済み、qR bitの巨大整数を作らない。これは数値実装の高速化で、新しい科学的近似や閾値緩和ではない。人工n=0/2/8192ではFraction厳密値包含、人工大指数4096×8160では100桁Decimal参照との包含と二乗法のoperation数をtestした。実m12のE[S]を評価したtestではない。

## 計算量と失敗時の処理

support size L=324、wire state数dがenclosure内なら、局所表/graphは概ねO(dL)個の遷移と固定native長に依存し、event kernelの合成・反復はO(d³ logR)、outer合成はO(logq)。kernel memoryはwire/R当たりO(d²)。phaseの追加確認はO(maxR)時間・cacheを各stepで解放してO(1)追加memory。全event列のO(qR)保持やorder2のL³ tuple列挙はしない。各Rのd、実CPU/RSS/時間・区間幅は未測定。

将来の一括実行はcontext/preparation・common native字句を1回だけ構成し、11lawを分離して検査する。各roundの9wireをaffinityに応じた9task以内で並列化し、1つのpoolを再利用する。専用processのBLAS/OMP1は多重並列回避の既存方針で、共有環境を変更しない。独自の小さなCPU/RAM上限は設けない。

全体は固定順m1,2,3,4,5,7,8,9,10,11,12。共通条件又はあるroundのK0がUNKNOWN/FAILなら、それまでのlaw/完成wire/完成roundとSTOPを保存し、当該roundのK1・次roundへ進まない。K0 PASSとkernel/状態辞書の保存後にのみそのroundのXY二値を評価する。K1で区間/quantity等が破綻してもSTOPし、既存PASSや部分結果を消さない。m0/m6・旧STOPを再実行/上書きしない。

E[S]のintervalはIEEE/BLAS丸めモデル前提の構造・確率伝播enclosure。E[C]、E[C−S]、実PCG64/CDF対ideal距離、rare/長q systematic、共分散/total CI、N/RMSE/Gは資格しない。残差はCodex Dの担当で、今回Cはfitやcompiled価格を実装しない。
