# N3-B0：価格評価前に固定するパラメータ選択規則

2026-11-08 JST。基準`79dfa01e6796d1ecab50e30694f5b3f73b95f4ab`。[PREFLIGHT](PREFLIGHT.md)、[数値scope](NUMERIC_EXECUTION_SCOPE.md)。**規則の静的な定義のみで、b／予算／基本key／coverageをまだ計算していない。**

## 1. 数学的制約と仕様上の選択

今回のcertificateを成立させるには、実binary64 targetを覆い、量子化・基本word和・emitter誤差の安全な上界を最終Frobenius許容epsilon=10^-11へstrictに収める。N3-Aの一様和は保守的十分条件であり、その不成立から実際のoperator近似の不成立や一般的な不可能性は主張しない。

本preflightでは、**同じ内部要求precisionを基本wordへ使い、要求をできるだけ粗くし、その後に辞書サイズを小さくする**辞書仕様の選択規則を定義する。これは唯一の数学的解でも最小Tの最適化でもない。未知の価格・backend成功率・cache coverageを見ずに、内部要求precisionと辞書サイズの優先順を固定するための設計上の選択である。実Tへの優劣を推測しない。

誤差を量子化と合成へ半分ずつ割り当てない。emitterについても未証明の0や恣意的な固定割合を置かず、選んだ仕様が**emitterへ要求する残余上限**を出す。実emitterがそれを満たす資格は独立に残す。

## 2. 保証付きの量子化項

Lは保存CT-0のm7/q128/r1行`absolute_actual_binary_angle_upper`を、実binary64の正確な有理数として読み取る。保存SOURCE／IEEE／remainder条件を変更しない。epsilonはFraction(1,10^11)。

既存の192-bit区間算術規約を使い、sqrt2を閉区間[c_lo,c_hi]で覆う。区間端点は`_mpi_`の整数mantissa・指数からFractionへ直接変換する。floatへ変換してから上端を読む経路は使わない。`0 < c_lo <= c_hi`と`c_lo^2 <= 2 <= c_hi^2`を厳密整数／有理数で検算する。評価precisionは192 bitsで固定し、判定を変えるために後から増やさない。

\[
Q_b^{lo}=c_{lo}2^{-b-2},\qquad Q_b^{up}=c_{hi}2^{-b-2}.
\]

以後の選択はこの**固定された有理上端Q_b^up**に対して行う。真のsqrt2値に対する最小bや最適precisionまで主張しない。下端は包含確認と保守性の記録用で、上端判定を都合よく置換しない。

## 3. 有限候補と辞書サイズ

bは0…1074の整数に限定する。下端はh=2^-bの定義、上端は既存float target interfaceで2^-bを非ゼロに表現できるbinary64指数限界による。CPU／実行時間を理由とする恣意的な探索上限ではない。

各bについて

\[
N_b=\lfloor L2^b+1/2\rfloor,\quad
J_b=\operatorname{bitlength}(N_b)-1-b,\quad M_b=J_b+b+1
\]

を整数で求める。N_b=0なら構成の退化ケースとして扱うが、保存L>1という今回の入力条件も検査する。基本指数は-b…J_b。J_b>1023など、必要基本角を既存binary64で正確に表現できない条件なら資格せずSTOPする。

全bの数値評価前に候補集合をこの規則で固定する。key catalogueは選択したbの指数域についてだけ生成する。候補全体の角度表は生成しない。

## 4. 内部要求precisionと決定的な選択

各bで`G_b = epsilon - Q_b^up`を求める。G_b<=0の候補は、この十分条件では基本word／emitterの正の余裕を確保できないため除外する。実際の全trajectoryが誤ると判定したものではない。

G_b>0でM_b>0なら、次を満たす**最小の正整数p_b**を決める。

\[
M_b10^{-p_b}<G_b.
\]

これは既存acquireの十進整数exponent形式へ適合する、最も粗い一様基本要求precisionである。整数の10倍／除算比較で決め、浮動logを使わない。G_b>0なので有限p_bが存在する。`p_b=1`又は`M_b10^{-(p_b-1)} >= G_b`も検算する。別のprecision候補を価格に応じて追加しない。

本書の「成立候補」は、実emitterが残余条件を満たすことを条件とするパラメータ候補であり、実回路のstrict PASSではない。その候補から、tuple **(p_b, M_b, b)** が辞書式順序で最小のものを主仕様とする。すなわち、まず基本要求を粗くし、同じ要求なら使用可能な基本word数上界を小さくし、最後にbで一意化する。全候補の整数／有理判定と選択理由を保存する。cache照合は選択後に行い、再利用が少なくてもbを選び直さない。

主仕様の各基本wordに対する割当上限は

\[
g_j^{alloc}=10^{-p_*}\quad(-b_*\le j\le J_*).
\]

既存wordの実保存guardがこれより小さければ再利用できる。要求ラベルがより粗いというだけで、その実上界が適合するrecordを排除しない。一方、あるwordに余剰精度があっても、その余りを使って他keyの要求を緩める再最適化は含めない。

## 5. Emitterへ要求する残余

主仕様で

\[
R_{emit}^{lo}=\epsilon-Q_{b_*}^{up}-M_*10^{-p_*}>0,\quad
R_{emit}^{hi}=\epsilon-Q_{b_*}^{lo}-M_*10^{-p_*}
\]

を出力する。これは**実emitter誤差の測定値／保証値ではない**。全基本wordの実Frobenius上界が各割当以下で、実emitterに対して`e_emit < R_emit^lo`を別途資格できる場合に、全体上界がepsilonへstrictに入る。

数学的wordとexactなClifford+T命令の一致を証明した部分だけe_emit=0とできる。現[lower_native](../../../src/trottertracks/track_c2_b_verify.py)はWごとに浮動global_phaseを加算するため、新しい連結wordに対する追加誤差0を今回の前提にしない。数値結果は残余を満たすことを要求する**条件付き辞書仕様**であり、実emitterの資格はUNQUALIFIEDとして保存する。未資格を避けるためにp／b／余裕を事後変更しない。

## 6. 保存wordの照合規則

固定poolは基準commitに保存された、CT-4B `strict_records.tar.xz`、旧C2-B `raw_synthesis.tar.xz`、CT-1 `strict_records.tar.xz`の有限recordだけ。各BUNDLESのhashを保持する。別worktreeの可変cache、新規取得word、再guard、欠損metadataの補作は含めない。

基本word候補の必要条件は、literal正角一致、保存PASS、strict spec／backend／seed、word本体とsequence SHA、W、実adjoint規約、資格済みguard方式、**保存Frobenius上界 <= g_j^alloc**。既存metadata validatorを再利用できるが、method欠損のlegacyを許容する枝は再利用資格へ使わない。actual adjointは既存SOURCEの反転規約と対応させ、古いraw recordに存在しない新しいflagを必須扱いしたり補作したりしない。

pool内に保存された異なる要求precisionも、元のラベルを保持して実上界の適合候補として列挙する。自動的に新precisionへ再ラベル・正式採用しない。複数候補がある場合の参考順はCT-4B→旧C2-B→CT-1、各pool内はrecord key順。同じkeyの科学的内容が食い違う場合はSTOP。選択へT価格・coverage数を使わない。候補表と未取得keyを保存し、N3-B1の正式なword policy実装とは区別する。

旧C2-BのFull Y4/LD12やCT-1のq1／epsilon=10^-9という研究条件・whole成功保証は移送しない。再利用対象は、同じliteral RZへの全2次元空間strict word／保存guardだけである。実上界が今回の内部要求に入ることを個別確認するため、旧要求precisionのラベルを今回の成功保証へ読み替えない。

CT-4Bの最終native epsilon=10^-11採用表C0は別に固定し、D／CSWAP／scalar／中央回転のwordと価格を一切置換しない。内部基本recordへの適合と、最終native targetの既存cache優先規則を混同しない。`2^-61`も同じ規則で照合し、指数域外ならその事実を記録する。key集合をcacheに合わせて変えない。
