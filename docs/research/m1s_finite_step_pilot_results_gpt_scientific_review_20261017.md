# M1-S有限step複素信号pilot：完了結果の独立科学レビュー

- レビュー日：2026-10-17 JST。
- 対象：`HIROMU1015/Partially-Randomized-Trotter`、固定commit `025c5a90888b02b73d2815fbb10d6bd1d2823602`。
- 開始承認：ユーザー「commit `025c5a90888b02b73d2815fbb10d6bd1d2823602` の有限step pilot結果について、詳細科学レビューを開始してください。visibilityとphase bias、T1bとの整合、shots・回路費用への接続、研究継続の価値を独立に判断してください。新規科学計算・SOURCE変更・研究方針の正式採択は行わず、レビュー結果をMarkdownに保存してください。」
- 区分：Level 1。研究継続・縮小の判断を含む。本セッションでの独立読解・紙上検討であり、別モデルや外部研究者による再レビューの実施記録ではない。
- 適用指示：`research_project_instructions_unified_2026-10-16.md`。SHA256 `8323be3f65856adf0f4d095d8e68546db3326822023c20e101ba5f596605ebfa`。
- 実施：固定GitHub資料の取得・bytes照合、保存結果・CSV・SOURCE・契約・理論の読解、解析的な数式の検討、一次文献の関連箇所確認。
- 未実施：科学runner、保存値の新規科学再集計・fit・ランキング・shot forecast、行列／Pauli／W・v・ρ／signalの再評価、数値テスト、simulation、回路生成・compile、SOURCE変更、正式採択、実行authorization・grant・markerの発行、Git commit・push。

**判定：`FINITE_STEP_VISIBILITY_GAIN_SUPPORTED_WITH_PHASE_TRADEOFF / T1B_COMPATIBLE_NOT_RATE_VALIDATED / END_TO_END_RESOURCE_GO_NOT_ESTABLISHED / NARROW_FEASIBILITY_DESIGN_RECOMMENDED_NOT_ADOPTED / MANDATORY_STOP`。**

## 1. 結論

**今回のpilotは、固定H4におけるshared ±の有限step visibility改善を支持する。ただし、位相推定アルゴリズムとしての優位性を支持する段階には達していない。研究は限定継続に値するが、RPE全体のsimulationや広いnative費用評価への直行は推奨しない。**

G3は明確なvisibility改善と、対照より大きい位相biasが併存する。G6もvisibility差は解像されているが、絶対量は小さく、位相biasの増加が見られる。G9は絶対差を判別できず、このgridで性能positiveの根拠に使えない。数値半径が小さいことは、この位相biasを除去したことを意味しない。

T1bとの関係では、今回の固定時間での信号と、有限減衰予算を保つ長時間secular極限を区別する必要がある。後者の先導rate比だけでは前者の位相を説明し切れない。一step生成子に既に含まれる非secular結合は、固定時間では減衰改善と同じ次数で信号位相へ寄与し得る。今回の傾向はT1bの反証ではなく、**rateのheadroomから有限時間の位相推定利益へ飛躍できない具体例**である。

次に研究資源を投入するなら、まず**位相誤差予算と実装費用の成立可能性を判断する限定契約の設計**を推奨する。共通の精度・confidenceに対して改善余地を残す条件が明確になった場合にだけ、別途承認された最小検証へ進む。成立可能性を示せなければ、T1a/T1bと今回の信号trade-offを限定理論・数値ノートとして保全し、実用資源優位を追う系列を区切る選択が妥当である。これは推奨であり、方針の正式採択ではない。

## 2. 固定証拠とレビューの範囲

### 2.1 実際に取得・照合した公開ファイル

以下6ファイルを指定commitからGitHub connectorで取得し、内容のbase64を正規化してlocal Git objectのbytesと直接比較した。全6件で一致し、Git blob SHA1も一致した。SHA256は同じbytesから算出したファイルidentityであり、科学数値計算ではない。

|資料|bytes|SHA256|
|---|---:|---|
|[完了報告][result-report]|7,535|`9a119228c9ac3deba3ccec27199dd886d1c6716fa2e82ec349dad40a866a7ac8`|
|[試行02 raw result][raw]|454,422|`ff6c5192d8cca769cd61a1478cea01671d058a85a8f6a4cdcdc9baa672a56fe0`|
|[108行 signals.csv][signals]|22,436|`1f0e2173bbb231177e4a48e925286e6c02f1da5501f7b2a6a53e440e543312b4`|
|[48対照 contrasts.csv][contrasts]|4,708|`82b7198cf1563eb9e4c6742a54c605411b7b331116aa94dee65d11f6e1c76bb4`|
|[公開照合receipt][receipt]|46,637|`9d9bcea147bb6e18f0432e4b76dd8fd46d2f9f00cdc901b838ff1e6834aa50ae`|
|[T1b証明原本][t1b-current]|32,550|`e7b621c489405e9c4876a9d2b71a2afd6d38c7c84e0f4bafec53f9b836d2c7fc`|

固定worktreeのHEADは指定commitと一致し、レビュー開始時はcleanだった。公開branchのHEADを本レビューで改めて問い合わせたとは主張しない。レビューの対象identityは固定commitである。

追加で、同commitの正式実行契約、技術修正・実施承認記録、誤差設計、`pilot.py`・`numerics.py`、保存監査、研究概要・研究目的、既存レビューを読んだ。SOURCEをimport・実行していない。raw JSONの欄・行の抽出は閲覧のためだけで、新しい科学量の算出に使っていない。

|provenance|固定commit／資料|
|---|---|
|完了SOURCE S2|`2eac35ff33f100fa96e296b737752addde5af5de`、[SOURCE][source]|
|完了契約 C2|`5afd57ed3feb900b8f97317c38cc62f52959981b`、[machine契約][contract]|
|完了実施authorization A2|`3dfd4470f8755937be1818c24580e8cbb76d3171`、[実施記録][authorization]|
|結果payload|`71ec2fb928cd68d8075e77643ddc4cdd8acdc848`|
|公開receiptを含む固定対象|`025c5a90888b02b73d2815fbb10d6bd1d2823602`|
|元の採択契約 C0|`c6254be7c41d800a8ee417e8dc124f03ba6b382d`|
|前回pilot設計レビュー|[2026-10-16原本][design-review]、SHA256 `fb2ff8695b9bf0355a84fb3a368543e9d36aa6e97dc172a034116efb190432b4`|
|Gate 0独立照合後レビュー|[2026-10-16原本][gate-review]。有限信号の独立検証とは区別する|
|T1b独立受理|[Checkpoint 2][checkpoint2]。固定model/r・共鳴・weight・適用域の限定を維持する|

### 2.2 再試行と証拠の強さ

旧起動前STOP、試行01の`TECHNICAL_STOP`、試行02の完了は別記録として残る。ユーザーの「計算完了まで行ってほしい　修正があれば行い計算完了まで行って」と、[修正・完了承認の記録][workflow]に基づく技術的な別試行である。最初から一試行で成功したとの記述は不適切である。

精度文字列parseの修正は値を変えない。試行02のPhi半径修正も、各unitary二側作用のFrobenius isometryと正規化された正の確率から

\[
\|\Phi(X)-\Phi(Y)\|_F\le\|X-Y\|_F
\]

を使い、入力半径を一度だけ伝播するものとして紙上で正当化できる。実装は中心行列の演算を保ち、算術・係数・三角関数の予算を別に加えている。[修正の導出][radius-repair]とSOURCEはこの意味で整合する。保存監査は、重複16行の中心値が両試行でbit一致したと記録する。本レビューで再計算した証拠ではない。

この経緯は、結果依存の閾値緩和やvisibility選抜の証拠ではない。一方、**同じdevelopment系を見ながら行った技術修正後の完了結果**であり、未使用データによる独立検証にはならない。

初回2×2 physical-word照合と全3分割の256次元twirl probeは有用なcorrectness証拠である。ただし、108信号すべてを別SOURCE・別精度で独立再生成した証拠ではない。旧Gate 0の独立Pauli/W/v/ρ照合PASSを、今回の有限step信号の完全な独立再現へ転用しない。保存監査1,022件・provenance監査124件PASSと公開同一性も、platform certificateではない。

## 3. 比較taskと今回得られた観測

### 3.1 固定scope

linear H4、隣接距離1.00 Å、STO-3G、8 spin orbitals、保存DF rank12。分割は生成順DF prefix G3/G6/G9。tailは保存された非冗長・符号集約済みnonidentity Pauli lawであり、DF-native event一般ではない。identityは契約どおりHD側に置く。

保存数値oracleの元norm²で期待値を割り、全256次元で計算する。認証された厳密基底状態とは呼ばない。\(\delta=1/8,1/16\;\mathrm{Ha}^{-1}\)、\(R=2,4\)、\(T=1,4\;\mathrm{Ha}^{-1}\)、zero control込み108行。非零72行とzero36行は、同一Hamiltonian・stateを使う決定論的coherent平均の評価である。108件の独立した物理instanceやshot標本ではない。

3方式は同じ相対時間、law、概念的独立label数\(sR\)、\(s=T/\delta\)を共有する。sharedの負枝は同じlabel順序で符号反転し、full-word adjointではない。sharedは\(D\Phi^R(DXD)D\)、ordinaryは\(O^s\)、independentは\(V^sV^s\)の期待値であり、\(V^\dagger V\)ではない。

### 3.2 完了・誤差の読み方

[保存summary][summary]の記録は、108/108行完了、全行principal phase resolved、最大signal数値半径`3.8150467991527054e-8`、最大phase数値半径`4.6659565697084285e-8 rad`である。採択した\(2^{-24}\)と\(2^{-18}\;\mathrm{rad}\)の要求内にある。

**このphase要求は「位相をどこまで数値的に分解できるか」であり、「アルゴリズムのbiasが許容されるか」ではない。** `RESOLVED_PRINCIPAL_PHASE`もbiasが0、RPEに適格、unwrap成功、という意味ではない。

条件付きIEEE binary64、complex BLAS、norm、三角関数・atan2・log、underflow/FTZ等の仮定は未認証のままである。以下の数値判断も`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`の範囲に限る。

### 3.3 Visibility改善と位相biasを同時に読む

以下は[signals.csv][signals]に保存された値の引用で、表示桁のみ短くした。新しい差分集計やfitではない。位相\(b\)はraw-H参照に対する符号付きprincipal phase bias [rad]である。表はすべて\(R=2,\delta=1/8\;\mathrm{Ha}^{-1}\)。

|split|T [Ha^-1]|shared visibility|ordinary visibility|independent visibility|shared b|ordinary b|independent b|
|---|---:|---:|---:|---:|---:|---:|---:|
|G3|1|0.9835785267|0.9802726319|0.9802716770|−0.0047520213|+0.00008749285|+0.00005151164|
|G3|4|0.9556370833|0.9233899381|0.9233872290|−0.0078561882|+0.0003549442|+0.0002120888|
|G6|1|0.9999921570|0.9999917566|0.9999917565|−9.56693e−7|+5.35560e−8|+1.77340e−8|
|G6|4|0.9999751894|0.9999670267|0.9999670265|−5.30843e−6|+2.14223e−7|+7.10026e−8|

G3の\(T=4\)では、shared−ordinaryのvisibility差は保存値`0.032247145221058315`、その半径は`2.106209710069508e-8`。これと、sharedのbias`−0.007856188182993353 rad`・数値半径`1.5724468179701225e-8 rad`は同時に成立する。改善と代償の両方が数値分解能を大きく上回る。visibilityだけから「位相推定に有利」とする判断は支持できない。

細かい側の例、G3・\(R=4,\delta=1/16,T=4\)でも、sharedのvisibility`0.9887278497445431`に対してordinaryは`0.9802775510155235`、biasはそれぞれ`−0.001977556858540422`、`+5.769398578503326e-5 rad`である。粗い一点だけのtrade-offではない。

G6・\(R=2,\delta=1/8,T=4\)のvisibility差は保存値`8.16274012682694e-6`、半径`2.130856684350743e-8`。解像はされるが、G3と比べ絶対量は小さい。sharedのbias`−5.308429785218499e-6 rad`は数値半径`1.5572526462595124e-8 rad`より明らかに大きい。これは科学的biasの存在を示すが、taskのbias許容値はまだないため、精度不適格と自動判定もしない。

G6・\(R=4,\delta=1/16,T=4\)ではshared bias`−1.3295854380189094e-6 rad`、ordinary bias`+4.9163176963036576e-8 rad`。対照の微小biasは数値半径と同程度の場合もあり、対照間の細かなbias順位まで断定しない。sharedでのbias増加という主要なtrade-offは、そのような微小順位に依存しない。

[既存対照分類][contrasts]はG3・G6それぞれ16件のhigher visibility、G9は16件すべて絶対差unresolvedである。この件数は既存summaryの引用で、今回新たに位相比較の勝敗件数を作っていない。共有する入力とgridの対照を、独立標本の「32勝」や統計的有意性へ読み替えない。

### 3.4 G9と絶対尺度

Gate 0の保存値引用では\(\lambda_R\)はG3 `0.8137790393725155 Ha`、G6 `0.026294457645549787 Ha`、G9 `5.47382434715216e-7 Ha`である。[旧独立結果レビュー][gate-review]。先導尺度は\(\lambda_R^2\)を含むため、\(\rho<1\)だけで絶対改善の大きさは決まらない。

G9の表示差は数値半径に覆われる。visibility中心が僅かに1を超える行も、球を含めて読むべきで、増幅効果や物理的不正の証拠にしない。G9の結果は**この固定時間・law・精度で改善を解像できない**という限定結論である。一般的なM1-S no-goでも、長時間で必ず利益が出る証拠でもない。λで時刻を拡大してG9を救済する追加計算は本レビューの範囲外である。

## 4. モデル差の会計と位相帰属

\[
H_{\rm full}=\text{保存raw binary64 DF入力からの参照},\qquad
\widetilde H_j=H_{D,j}+H_{R,j}^{\rm saved}.
\]

各方式のtargetは共通の\(\widetilde H_j\)。exact Hermiticityの確認と、\(\Delta H_j=\widetilde H_j-H_{\rm full}\)に対する行和上界\(B_j\)を用い、Duhamelから

\[
\|e^{iT\widetilde H_j}-e^{iTH_{\rm full}}\|_2\le |T|B_j=\eta_j(T)
\]

となる。同じstateの元norm²で割った期待値にも適用でき、固有状態仮定は不要。位相へ移すには参照visibilityの正の下界と、zero・principal branch cutからの分離が別に必要である。

raw resultの\(R=2,\delta=1/8,T=4\)の保存欄を引用すると、次のとおり。

|split|保存されたη signal上界|保存された条件付きmodel phase上界 [rad]|
|---|---:|---:|
|G3|1.79356410763102e−16|6.234456220038177e−16|
|G6|8.925719313256752e−18|4.530149292325497e−16|
|G9|8.794776627122414e−23|4.44089297797855e−16|

これらのmodel phase上界には既存library allowanceも入る。Bやηを本レビューでhex有理数から再評価していない。

したがって、少なくとも引用したG3/G6のbias増加を、このHamiltonianモデル差で説明することはできない。モデル差を独立台帳へ分けた修正は有効だった。`MODEL_TARGET_CONFOUNDED`はη>0で付く保守的な帰属ラベルであり、必ず解釈不能な大きさの交絡があるとの意味ではない。

それでも、\(z_{\widetilde H_j}^{\rm ideal}\)は今回直接評価されていない。raw-Hに対する総差を純アルゴリズム誤差の点値へ改名しない。保存された三角不等式区間と上界によって帰属の限界を述べる。保存oracleの固有状態距離、真の量子化学Hamiltonianへの誤差は別問題である。旧v5に未保存だったfull-H/HDの直接比較も、引き続き`UNDETERMINED`である。

## 5. T1bとの整合：何を説明し、何を説明しないか

### 5.1 先導rateと今回の極限は異なる

T1bは固定有限次元・固定r・固定modelで\(\delta\to0\)、\(u=s\delta^2\)を有界に保つ。非自明な減衰では\(u\)は正の有限値となり、\(T=s\delta\)は大きくなり得る。単一pair-sum block条件(SP)と厳密な対象固有stateの下では

\[
\gamma_{\rm sh}=\beta(W+v),\qquad
\gamma_{\rm ord}=2\beta W,\qquad
\beta=\lambda_R^2/(4R),\qquad
\rho=(W+v)/(2W).
\]

同じ総label数にしたindependent ±は、先導ordinaryと一致する。これはT1bの各枝rをそのまま同一視する比較とは異なる。

今回のpilotはT=1または4を固定しているため、\(u=T\delta\to0\)。T1b主上界がこの小さなuも含むことと、**その上界が減衰改善の先導係数を相対的に分解する精度を持つこと**は別である。非secular誤差項の一部は固定TでO(δ)のままで、観測する減衰差自体と同じ次数になり得る。

全Fock空間の対象固有state、(SP)、pair-sum gap、mode weightの実入力認定は今回もない。二つのδ点から減衰次数・位相次数をfitしたり、有限時刻のlog visibility比をρと等置したりはできない。G3/G6の改善方向はheadroomの期待と整合するが、**T1bのscalar rate式や長時間一様保証を実入力で数値検証したとは言えない**。

### 5.2 固定時間ではO(δ)の位相が生じ得る：紙上の機構説明

以下はT1bの一step展開を固定Tで読むための解析的検討である。H4のスペクトル・係数を新しく算出したものではなく、新たな実験契約やT1c研究を開始するものでもない。

説明のため、\(H=\widetilde H_j\)の正確な固有state\(|k\rangle\)を仮定する。\(A=J_H/2\)、\(J_H(X)=HX+XH\)、T1bの二次散逸をQとすると

\[
\Psi_\delta=I+i\delta A+\delta^2(Q-A^2/2)+O(\delta^3),
\qquad \delta^{-1}\log\Psi_\delta=iA+\delta Q+O(\delta^2).
\]

固定有限T、固定model/R、\(s=T/\delta\)が整数の系列では、Duhamelの一次摂動から

\[
e^{-iTE_k}z_{\rm sh}(T)
=1+\delta\int_0^T\sum_a q_{ka}e^{it(E_a-E_k)}dt+O(\delta^2),
\qquad q_{ka}=\langle k|Q(|a\rangle\langle a|)|k\rangle.
\tag{1}
\]

T1bの\(Q=\beta[-\mathrm{Id}-\mathbb E(P\,\cdot\,P)+J_h^2/2]\)を代入すると

\[
q_{kk}=-\beta(W_k+v_k),\qquad
q_{ka}=\beta\left(|h_{ka}|^2-\sum_\ell p_\ell|(P_\ell)_{ka}|^2\right)\le0\quad(a\ne k).
\tag{2}
\]

最後の不等式は凸性による。\(\Delta_{ak}=E_a-E_k\ne0\)なら、(1)の相対信号の虚部一次項は

\[
\delta\sum_{a\ne k}q_{ka}\frac{1-\cos(T\Delta_{ak})}{\Delta_{ak}}.
\tag{3}
\]

つまり**散逸QがHS上self-adjointでも、固定時間の複素信号には一次の位相補正が出る**。入力Iの他の固有projector成分と対象出力との振動結合が原因である。非縮退全空間最低stateならΔ>0でこの項は非正だが、今回の保存sector oracleがその仮定を満たすとは認定していない。したがって(3)でH4の負biasを定量的に説明したとは主張しない。

対してordinaryは\(\delta^{-1}\log O_\delta=iH+\delta Q_o+O(\delta^2)\)、\(Q_o=-2\beta(I-h^2)\)であり、厳密な\(|k\rangle\)への一次補正は\(\delta T\langle k|Q_o|k\rangle\)という実数である。sharedの過渡的位相と、ordinaryの主要な減衰を同じ単一scalar rateで置換できないことが分かる。

この説明は、T1bの固有modeのlog位相補正\(O(s\delta^3)=O(T\delta^2)\)と矛盾しない。**単一modeの固有値の位相と、入力・出力weightや他modeを含む実信号の位相は異なる。** T1b自身が射影weightと他modeの一様残差を別に評価している。今回のδを細かくしたときの保存biasの傾向はこの機構と整合し得るが、二点から次数や原因を確定しない。

### 5.3 共鳴とJensen floor

Checkpoint 2で受理されたJensen型floorは、非縮退対象エネルギーに対するC-sec参照の正の振幅と理想位相を支える。共鳴を許す参照の下界があることは、実際の有限step写像がその参照と同じ位相を持つことを意味しない。有限step差の控除が必要である。今回のbiasを理由にJensen floorを棄却せず、逆にfloorを理由にbiasを無視もしない。

## 6. Shotsへの接続：visibility単独では決まらない

### 6.1 同じ位相taskに戻す必要がある

Hadamard測定の2軸の平均はRe z、Im zであり、各shotは有界な結果を持つ。IIDにlawを引き直す条件なら、coherent平均zは測定分布の中心を与える。一方、増shotで除けるのは統計誤差であって、\(\operatorname{Arg}z\)と目的の位相との系統差ではない。

RPEに必要なのは、全roundの目的位相とのbias、branch選択、confidence配分、整数shot、最大時間、guiding-state条件である。今回のraw参照信号の位相自体も、認証された\(TE_0\)ではない。T=1,4の主値が解像されたことから、全roundのenergy精度やunwrapを認定しない。RPEの既存理論には訂正があり、初期の定数を無批判に流用できない。[Kimmel–Low–Yoder原論文・2021訂正][rpe][rpe-erratum]。

### 6.2 数値代入なしの設計上の十分条件

将来の同一taskで、方式mの真のvisibility下界を\(A_m>0\)、目的位相との差の上界を\(b_m\)とする。\(b_m\)には、目的に応じたoracle/reference・モデル差・方式biasと検証不確かさを正しく含める。許容位相誤差\(\epsilon_\phi\)に対する残余を\(q_m=\epsilon_\phi-b_m\in(0,\pi/2)\)とする。

各軸M shot、±1結果のHoeffding上界と2軸union boundから、複素標本誤差が\(A_m\sin q_m\)を超えないための単一時刻の十分条件として

\[
M_m\ge \left\lceil\frac{4\log(4/\alpha)}{A_m^2\sin^2 q_m}\right\rceil
\tag{4}
\]

が得られる。導出は、各座標誤差を\(t=A_m\sin q_m/\sqrt2\)以下にすると複素誤差が\(\sqrt2t\)以下となり、失敗確率が\(4e^{-Mt^2/2}\)以下となることによる。半径diskの幾何から、円周距離での角誤差をq以内にする。二つの主値の通常の差を使う場合は、branch cutからの分離も必要である。RPEの多段branch保証・最適shot数・下界ではなく、一つの保守的な設計proxyである。**保存値への代入やshotsの算出は今回行っていない。**

\(b_m\ge\epsilon_\phi\)なら、この無補正推定と予算配分では実行可能性を保証できない。shotを増やすだけで解決しない。ただし、これを任意の推定器・補正・別scheduleに対する不可能性定理とも呼ばない。

G3のvisibility改善を受け取るには、その位相biasを許容するtaskか、実用的に利用可能な情報だけで補正する方式が必要になる。保存真値を差し引く「oracle補正」を無料で許すと、研究上必要な保証を失う。G6のようにvisibilityが既に1に近い場合、減衰の相対改善が大きく見えても、shot削減の絶対余地は小さい可能性がある。実際の削減数は未計算である。

## 7. 回路費用：branch数とnative価格を区別する

### 7.1 倍のbranch演算は倍のnative費用を意味しない

保存契約ではsharedは2sR、ordinary/independentはsRのbranch Pauli指数演算を数える。しかし、同じPを共有する正負branchは、一つのancillaを含むPauli rotationへまとめられる。

\[
|0\rangle\langle0|\otimes e^{-i\theta P}
+|1\rangle\langle1|\otimes e^{i\theta P}
=e^{-i\theta Z_a\otimes P}.
\tag{5}
\]

対して通常のcontrolled Pauli rotationには

\[
\operatorname{diag}(I,e^{iaP})
=e^{ia I_a\otimes P/2}e^{-iaZ_a\otimes P/2}
\tag{6}
\]

という分解がある。これは実回路を生成せず確認できる演算子恒等式である。Clifford parity network、隣接融合、angle精度、native gate setにより価格は変わるため、(6)の因子数を必須最小gate数とも呼ばない。**sharedは必ず2倍高いという棄却も、必ず安いという推薦も根拠不足**である。

このdirectional-controlの原理自体は先行研究で既知である。[Güntherほか、Appendix A.4・Fig.12][pr-paper]。同論文の独立乱択RTE、今回の共有Pauli指数law、決定論部分の対称性は区別する。label共有の非可換word全体を勝手にadjointへ置換して既知の時間半減効果を転用してはいけない。

### 7.2 実際の費用評価に欠けるもの

今回のHD exponentialは数値参照としてのexact-Hermitian targetである。量子回路ではHD内部のDF sweep/PF・合成誤差・controlled/directional wrapperが必要になる。この費用と追加biasはpilotで測っていない。旧PR-RTEのcompiled proxyを、新しいPauli lawとwrapperへそのまま移す根拠もない。

評価すべき費用は、tail回転だけでなく、HD・scalar phase・control・readout・合成精度・workspaceを含む1 shotの全費用である。状態準備を対象外に置く場合も仮定として明示し、shot数が変われば準備費用は通常相殺しない。乱択回路を複数shotで再利用する場合は、今回のIID平均から別の有限標本・reuse契約への橋渡しも必要になる。

単一時刻の連続proxyとして(4)を使い、同じconfidenceで整数丸めを一時省略すれば、

\[
G_m\propto\frac{C_m}{A_m^2\sin^2 q_m},\qquad
\frac{C_{\rm sh}}{C_{\rm cmp}}
<\left(\frac{A_{\rm sh}}{A_{\rm cmp}}\right)^2
\left(\frac{\sin q_{\rm sh}}{\sin q_{\rm cmp}}\right)^2
\tag{7}
\]

がこのproxy内のbreak-even条件になる。実際の最適総資源の必要十分条件や一般no-go規則ではない。位相biasが増えれば残余qが縮むため、visibility改善だけをshot比へ変換できない。\(C_m\)は未取得なので、(7)を実データ評価した資源結果は存在しない。

### 7.3 強いbaselineと新規性

今回のordinary/independentは同一Pauli指数lawでの機構対照として適切である。しかし、既知の部分乱択RTEや決定論endpointまで含むmatched-accuracy最良法ではない。RTEの平均保持・normalizationと、今回のPauli指数平均のbiasは異なる。将来の資源claimには同じtaskでの強い対照が必要になる。[Güntherほか、Appendix A.3][pr-paper]。

部分乱択による量子化学位相推定、RPE資源評価、directional controlそのものは新規claimにできない。今回は同論文の関連節とRPE原論文・訂正を確認した限定照合であり、shared相関の全関連文献を網羅したpriority監査ではない。今回の信号改善が、独立した新しい実用アルゴリズムや原著論文として十分との結論は出せない。

## 8. 科学的に支持する主張・支持しない主張

|主張|判断と範囲|
|---|---|
|保存H4・G3/G6の有限step平均visibilityでsharedが対照を上回る|条件付き数値証拠として支持。固定grid・law・state内|
|G3/G6のvisibility改善は位相bias改善も伴う|支持しない。保存値に明確なtrade-offがある|
|G9に有意な相対資源改善がある|支持しない。絶対contrast unresolved|
|T1bが今回の位相biasで反証された|支持しない。fixed-Tの過渡mode寄与と矛盾しない|
|T1bのρが有限時間の減衰比として実証された|未確立。SP/state/gap/weight・漸近域が未認定|
|Hamiltonianモデル差がG3/G6の主なbias増加原因|引用した保存上界とは整合しない。model差は極小|
|純アルゴリズム誤差を点値として独立測定した|未実施。三角不等式区間のみ|
|correctness PASSは108行の独立再現・platform認証を意味する|意味しない。toy/各split twirlと条件付き予算の証拠|
|同一label数なら同一native費用、またはbranch数からnative費用比が分かる|支持しない。wrapper・合成・融合が必要|
|RPE shots／総native費用の改善|未検証。共通精度・confidence taskも未固定|
|M1-S全体を一般的にNO-GOとできる|できない。未取得の費用と他運用条件が残る|
|限定した理論・信号trade-offの成果として保全する価値|ある。新規性・投稿十分性は別途未確定|

確信度は、固定数値の読解と比較意味論＝高（採択算術モデルに条件付き）、T1bとの論理整合＝高、H4での詳細なbias原因の定量帰属＝未検証、end-to-end利益＝未判定、広い一般化とpriority＝未認定である。

## 9. 継続・縮小・終了の推奨

### 9.1 推奨する限定継続

**M1-Sを直ちに実用資源改善の主線へ昇格させず、位相と費用の成立可能性を一つの小さな判断単位にすることを推奨する。** G3/G6の有限step差は成果として残す。G9の精度向上や長時間化、別分割追加、広いδ/R探索を次の既定作業にしない。

最初の成果物は追加計算ではなく、ユーザーが採否を判断できる次期契約候補とする。最低限、以下の科学的選択を明示する必要がある。

1. **同じtaskを定義する。** raw保存Hamiltonianに対する信号位相か、ground energy推定か。許容bias・最終energy精度・confidence・時間horizonを定め、数値分解能とは分離する。
2. **biasの支払い方法を決める。** 無補正のままか、利用可能情報だけによる補正か。dense真値を無料の実行時情報にせず、oracle/state・モデル差・追加HD近似を会計する。
3. **wrapperと対照を決める。** sharedのdirectional実装、ordinary/independentの最適化機会、HD内部近似、強い既知RTE／deterministic対照を同条件に置く。
4. **大きな計算へ進む条件を定める。** 位相予算を満たす候補に、実装費用の不確かさを支払ってなお利益が残る見込みがあるか。現H4の事後設計であることを明示し、結果依存の最良点を独立held-outと呼ばない。

これは新たな精度・費用・baseline選択を要するので、今回のレビューだけで正式採択しない。採択後も、保存値のshot再解析、native compile、追加signal等は、それぞれ固定scopeと別途明示の実施承認を要する。CodexによるSOURCE実装・実行へ自動移行しない。

### 9.2 判断を変える最小の追加証拠

|将来得たい証拠|変わる判断|必要な扱い|
|---|---|---|
|共通taskでbiasを支払っても利益が残る解析条件|小規模な次期検証に投資する価値|まず非数値の契約設計。閾値は結果への一致で決めない|
|同じ精度でのshotsとwrapper費用の限定照合|信号成果からresource候補へ進めるか|別途採択・数値／compile実施承認後のみ|
|十分に異なる計算経路での少数finite信号照合|pilotの数値証拠の強さ|必要になった場合にscope固定。旧Gate 0独立PASSで代替しない|
|位相機構の定量的識別|改善・biasを予測できる研究へ進めるか|必要なら別の解析／数値契約。現時点で大規模spectrumは不要|

これらを全部行う計画は推奨しない。taskと支払うbiasを決めないまま、新しい数値を増やしても資源判断に近づかない。

### 9.3 区切る場合の妥当な着地点

限定gateで位相・費用の成立可能性が残らない場合は、**「branch相関によるcoherent減衰の変更と有限時間位相trade-off」**を理論・数値ノートの中心にする。T1a/T1bの適用域、共鳴・floor、固定H4でのvisibility positive、位相代償、G9の絶対分解能限界をまとめられる。一般的資源優位の未達を、理論成果の誤りと混同しない。

今回の結果だけで終了を強制する理由も、長RPE・大系へ拡張する理由も十分ではない。継続予算を小さく限定し、判断に必要な条件を先に固定することが、現時点の最も合理的な選択である。正式に継続・縮小・終了を選ぶ主体はユーザーである。

## 10. 前回判断からの更新と停止状態

前回は「有限stepの絶対visibility差が存在するか」が未確認だった。今回はG3/G6でその差を支持する数値が得られたため、その限定claimは前進した。同時に位相biasの代償が明確になり、**visibility positiveだけで資源評価へ進むという弱い読み方は退ける**。前回から保留していたRPE/native総資源のGOは、今回も出さない。

本書は科学的評価・提案の保存である。採択済み契約、raw result、manifestの科学status、SOURCE、旧v4 STOP/null、v5／独立one-shot、各試行のauthorization・消費済marker、親／他worktreeを変更しない。新しい方針が採択された場合に、研究概要・規範文書・dated note・manifestへ正式に反映する。

**レビュー完了。新規科学計算0、科学的数値再解析0、SOURCE変更0、新execution authorization／grant／marker0。研究方針の正式採択なし。Mandatory STOP。**

## 11. 一次文献の確認範囲

- Günther, Witteveen, Schmidhuber, Miller, Christandl, Harrow, *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum 7, 020332 (2026), [arXiv:2503.05647v2][pr-paper]、[DOI](https://doi.org/10.1103/ynxb-p2xq)。今回確認したのは関連するAppendix A.3–A.4、Fig.12、Appendix BのRPE定義・branch条件等。全44ページの定理再証明や網羅的priority監査ではない。
- Kimmel, Low, Yoder, *Robust Calibration of a Universal Single-Qubit Gate-Set via Robust Phase Estimation*, PRA 92, 062315 (2015), [2021改訂arXiv版][rpe]、[Erratum, PRA 104, 069901 (2021)][rpe-erratum]。訂正はround成功条件と旧定数解析に関わる。本レビューは旧\(\pi/2\)のbranch判定や訂正前shot定数を採用せず、RPE全体への数値保証も作らない。

## 12. 保存・保全の技術確認

レビュー文書をUTF-8・LFで新規保存した。Markdown参照ラベル、参照先Git objectの存在、数式display delimiter、末尾空白を静的確認した。親checkoutのHEAD・既存tracked変更状態、固定結果worktreeのHEAD・clean状態、今回snapshotした指示・旧レビュー・SOURCE 2ファイル・両試行raw結果・両試行markerの計9ファイルのSHA256は維持されている。これは参照・保全確認であり、新しい数値correctness検証ではない。

今回保存した成果物は本レビュー1ファイルである。GitHub上の旧資料を変更せず、新しいcommit・pushや契約採択記録を作成していない。レビューの推奨を研究概要やmanifestの採択済statusへ反映していない。

以下のGitHub参照は、特記のない限り今回の固定commitの資料である。

[result-report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/docs/tracks/algorithm_codesign/m1s_finite_signal_v2_completed_results_20261016.md
[raw]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/artifacts/track_b_m1s_finite_signal_v2_result/2026-10-16/completion_attempt_02/result.json
[signals]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/signals.csv
[contrasts]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/contrasts.csv
[receipt]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/publication_receipt.json
[t1b-current]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/docs/tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md
[source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/2eac35ff33f100fa96e296b737752addde5af5de/src/trottertracks/algorithm_codesign/m1s_finite_signal_v2
[contract]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5afd57ed3feb900b8f97317c38cc62f52959981b/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/attempt_02/execution_contract_adopted.json
[authorization]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3dfd4470f8755937be1818c24580e8cbb76d3171/artifacts/track_b_m1s_finite_signal_v2_execution_authorization/2026-10-16/completion_attempt_02/execution_authorization.json
[design-review]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/docs/research/m1s_finite_step_pilot_gpt_independent_review_20261016.md
[gate-review]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/docs/research/track_b_m1s_gate0_v5_independent_result_gpt_scientific_review_20261016.md
[checkpoint2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62554ee127a2bec1206367cf1754df2ff49fa000/docs/research/track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md
[workflow]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/workflow/repair_and_completion_authority.json
[radius-repair]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/workflow/phi_contraction_radius_repair.md
[summary]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/saved_result_summary.json
[pr-paper]: https://arxiv.org/pdf/2503.05647v2
[rpe]: https://arxiv.org/pdf/1502.02677v3
[rpe-erratum]: https://journals.aps.org/pra/pdf/10.1103/PhysRevA.104.069901
