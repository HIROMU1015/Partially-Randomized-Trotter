# V5 depth境界の導出（固定方式・保存lowering）

各native gateのoperand上のfrontierをmax+1で更新する。barrierはmaxのみ、測定はqubit/classical bitを含む+1、resetは単一wire+1。回路全体の最大値は固定Qiskitの単位gate DAG depthであり、実行時間ではない。既存 `pr_reference_transfer_qualification.frontier_depth` を再利用する。

局所native列のmax-plus行列Mは、入力wire jから出力wire iへの最長gate依存鎖の長さである。依存しない成分を−∞とし、入力frontier dに対し `d'_i=max_j(M_ij+d_j)`。システムwire集合Sについて `δ(C)=min_i∈S max_j∈S M_ij` とする。全system入力がt以上なら、ancilla入力を仮定せず全system出力はt+δ以上。連続する保存済みD/frameのδを加算することは、その依存鎖による**下界**であり、gate数の和や厳密なwhole depthとは異なる。局所span s(C)=max(M)の直列和は上界。2 laneは別々に評価してmaxを取る。

## compiler/cancellationの範囲

旧B′の構造補遺にある全結合・固定equivalence context・SOURCE/final cancellationの論証を再利用する。D/frameの最終pass前CX/X両側fenceとSOURCE RZ/XX+YYを確認する。B′中央の露出parity CXやproduct/phase/wrapperは下界から全て捨てる。D/frameは保存nativeの依存鎖として残る。reference上界はproduct-Zと任意の境界CX/X相殺を無視するため、安全に大きくなる。任意の新compilerや回路への保証ではない。

局所Dの符号反転ではRZ/RZZの角度だけが変わり、固定loweringのwire列は変わらない。Fのactual adjointではblock順序・frame/inverseを反転するが、以下のspan/δの和には同じmultisetが寄与する。最終passが消去するgateを挿入し直した回路のdepthは、削除後回路の上界になる。これらの構造条件を保存context/QPY/cohortと照合する。

## finite-bit期待値と並列max

保存floor-CDFの有理成分確率をp_c、basis集計をp_b、order2確率をp₂とする。r eventを持つ一つのRTE occurrenceの期待application数はA=r(1+2p₂)。basis bのrun期待数は `p_b+(A−1)p_b(1−p_b)`。δ(frame)+δ(inverse)を重みにした和をμ_L、span(frame)+span(inverse)と中央rotation/product/phaseを足した和をμ_Uとする。どちらも**cost boundの期待値**であり、平均event回路を作っていない。

B′のDは `left_all+right_all+(q−1)(left_no0+right_no0+boundary_control)`。それぞれδなら下界、spanなら上界。RTEはq μ_L/q μ_Uを足す。

referenceはq≥2ならn=q/2の独立laneを固定F/Gに対応させる。各laneのDは `2n Σ_k C_k,half +(n−1)(C_0,full−2C_0,half)`。これは融合後の正のblock多重度をまとめた式であり、negative gateや単純比例外挿ではない。q=1は片laneにRTEを置く既存特例を使う。

一つのlaneのRTE上界costは、occurrenceごとの独立なW∈[0,M]の和。`M=r(3 max_b frame_pair_span + max_rotation_span + 4 product_span)`（B′はevent phase allowanceも加算）。これは確率support全体を覆う範囲で、標本最大ではない。Popoviciuの分散上界Var(W)≤M²/4から、同じ分布のn occurrenceを持つ2 laneで

`E max(S_F,S_G) = n E W + E|S_F−S_G|/2 ≤ n μ_U + M sqrt(n/8)`。

独立lane条件はcanonical independent-bit lawに従う。実native lane間の相関をこの式に勝手に仮定せず、各laneを覆う上界cost Wを同じ保存event列から定義する。平方根は128-bit二進格子への有理ceilingを整数算術で作る。下界には `E max ≥ max(E lower_F,E lower_G)` を使う。q=1は同一D上界+片lane μ_Uで十分。

referenceのwrapper上界は、保存8-CX CSWAPを8個接続したcascadeの実frontier spanを2回、H/Sdg、vacuum P、測定を追加する。保存SOURCEはnative compile後、referenceのfresh registerへ8 resetを明示的に付けている。これらは互いに独立でdepth 1、ancilla上の初期Hのnative depth 3と並列なので、cascade前の最大frontierはmax(1,3)=3。初期Hの3にresetを直列加算しない。保存価格と一致する**reset/測定込みnative DAG**を主指標とし、unitary-only depthや実機の初期化時間とは区別する。追加8 qubit/reset準備の代償も別に併記する。

全18roundでは保存N_m,aを掛けて足す。優越はreference上界<B′下界の場合だけ認定。重なればUNDETERMINED。T、実機時間、space-timeへ読み替えない。
