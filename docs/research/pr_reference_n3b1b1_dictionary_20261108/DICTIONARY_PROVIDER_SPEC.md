# N3-B1b-1：固定有限辞書provider仕様・証明境界

2026-11-08 JST。[REPORT](REPORT.md)。対象SOURCEは[pr_reference_dyadic_dictionary.py](../../../src/trottertracks/pr_reference_dyadic_dictionary.py)。今回固定したproviderであり、新しい合成法・optimizerではない。

## 1. 入力と固定record

許す科学targetはPython binary64 floatのthetaで、有限かつ|theta|<=保存L。Lのhex・厳密比、b=36、N_b=5668839823158、指数-36…6は[N3-B0正本](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/FIXED_PARAMETERS.json)から読む。NaN/Inf・域外入力をrejectし、自由な角度mod縮約を行わない。

constructorへ渡すのは、再利用したCT-0 validate_recordを通った次の固定データだけ。

1. 今回の全43基本record：同じliteral、e=-13、strict exact cyclotomic／outward Frobenius guard、seed17。
2. 固定CT-4B全299 C0 record：元のe=-11採用sequenceと価格、保存guardにemitter余裕があるもの。

[実際の組立・検証caller](../../../scripts/validate_pr_reference_n3b1b1.py)がmanifest／hash／spec／guard方式を検査してconstructorへ渡す。任意の未検査cacheや、仮のguard=0 recordを受け入れてよいというAPIではない。recordの文字列・価格・guardを保持する内部viewはimmutableで、元recordを変更しない。

## 2. C0優先と正確な丸め

abs(theta).hex()がC0に一致すれば、元positive wordをそのまま選ぶ。負角はその全体adjoint。優先判定はarm／roleに依存せず同じliteral keyで行い、固定K／d0／d2を変えない。現在の全C0 keyが保存角域内にあることも確認済み。

missの場合、abs(theta)=p/dをexact整数比で読み、p*2^36をdで割った商q・余りrについて、

- 2r<d：q
- 2r>d：q+1
- 2r=d：偶数側

を選び、符号を戻してnとする。従ってn(-theta)=-n(theta)、|theta-n2^-36|<=2^-37。浮動theta*2^36をroundしただけの保証ではない。

保存N_bは|n|を覆い、abs(n)のbit lengthは43以下。各bitを高々1回使い、選択指数を**昇順**に固定する。zeroまたは丸めでn=0となる場合、positive written wordは空の恒等wordとなる。

## 3. 連結word・負角・emitter

j昇順の基本sequenceをそのまま連結し、そのwritten matrix productを採用する。近似wordを並べ替えない。

WordCertificateはpositive_sequenceとnegative_whole_adjointを分けて保持する。signed_sequence()はCT-1の文字adjoint対応を再利用し、全体の文字順／chunk順を反転する。T↔t、S→SSS、W→7Wはexact word意味論の対応であり、phase削除やTdaggerを7物理Tとして課金する処理ではない。W_signedも保持する。

emit(theta,lower_native)はfresh 1-qubit placeholder RZから、既存lower_nativeへpositive sequenceを渡す。負側のinverseはlower_native自身が**完成したlocal全体**へ1回行う。既にadjointにした文字列を負角emitterへ再度渡して二重反転させない。

emitter用のsequence-only carrierとallocation token e=-11は、旧interfaceに接続するためのもの。新しいpygridsynth spec／backend strict recordとして保存・偽装しない。基本recordと量子化・composite certificateは別々のprovenanceを持つ。

本番経路ではnamed Clifford+T gate／exact Wを保持し、lower_native内部およびconsumerのjoinはinplace=True。QPY・default copy・to_instruction・再compile・後置controlは追加しない。Operator診断が内部変換を行うことは、本番emitter経路の資格と区別する。

## 4. Composite certificate

missで選ぶ基本word集合をJ(theta)とすると、certificateは

\[
q(\theta)=\frac{\sqrt2_{\rm up}}2|\theta-n2^{-36}|,\quad
g(\theta)=\sum_{j\in J(\theta)}g_j^{saved},\quad
u_{\rm emit}=45/140737488355328
\]

をFractionで保持し、

\[
\|U_{\rm emitted}-R_Z(\theta)\|_F
<q(\theta)+g(\theta)+u_{\rm emit}<10^{-11}
\]

を示す。sqrt2上端と量子化の一様上端はN3-B0正本。各基本guardは実43 recordに対応する。emitter上界はN3-B1b-0で確認した通常IEEE／固定runtime／inplace経路が条件であり、別compilerや任意の後処理へ移さない。

全域の保守的予算はQ_up+43*10^-13+u_emit。実guard和が小さいことを理由にb／precisionを変更しない。理想RZ同士の加法性とunitary telescopingを使うので、近似word自体の可換性は不要。

C0 hitのcertificateはg_C0_saved+64u、u=2^-53を用いる。local Wの定数差と1回の外側inplace joinを含む保守的1-qubit上界で、全299 wordに最終要求の余裕を確認した。先行する別gateの誤差を、このwordへ付け替えたり消したりしない。

zeroの論理wordは恒等で、emissionも空。certificateにはconsumerのphase処理まで覆う保守的emitter項を残せる。global phaseを無害として削除した結果ではない。

これらは局所1-qubit Frobenius保証。jointへはspectral normの埋込み・telescopingで接続し、joint Frobeniusへsqrt2をそのまま使わない。

## 5. 有限停止と価格

offline取得では全43 keyが実際に成功した。online providerはbackendを呼ばず、有限の整数算術・43-bit走査・有限文字列連結だけで終了する。これはpygridsynth一般入力の停止保証ではない。

TはT+Tdagger、Tdaggerは内数。単純連結とactual adjointで

\[
T_{\rm dict}(\theta)\le\sum_jt_j=6072,\quad
T_{\rm C0}(\theta)\le156,\quad B_P=6072.
\]

cache共有を物理gateの共有に数えず、字句のfusionや別合成を加えない。全bit1の43 chunk上界には非到達indexも含み得る。実到達supや期待価格の推定ではない。

## 6. 検査と適用限界

全43基本wordの実guard・hash・正負／actual adjoint、実providerの9固定入力の正負、zero／tie／隣接値／両端／subnormal／reject、C0優先を検査した。全域保証は有限診断の個数ではなく上記の正確な構成とguardに基づく。

43 chunk fixtureの理想all-ones角は保存Lを超えるためpublic providerはrejectする。試験では実43 wordを全て使った任意unitaryの人工written productとして、文字順と全体inverseを確認した。域外の科学target取得やL拡大を行ったのではない。

providerの資格をN2のmissing key取得、全PF回路への接続、全support CT RPE成功、18round総Tへ自動移送しない。C0優先の同一policy・compiler条件C・全role資格が揃うまで、N1との比較はvacuum価格側の条件充足である。

指定scope完了につき**mandatory STOP**。後続の判断と実行承認はGPT／ユーザーへ戻す。
