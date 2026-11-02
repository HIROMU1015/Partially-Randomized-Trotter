# G5 固定辞書の終了認証：結果前scopeと証明規則

利用者が共有・採用したGPT G4 review §13の限定技術作業を実施する。
GPTによる現固定toy・同辞書の実用優位実験主線の区切りを記録し、R0/G1/G4-Aの限定成果を保持する。
Track B全体の終了、次methodの採択、論文化採否はここでは決めない。

## 入力と証拠境界

baseは公開済みG4 commit `a221588f42ef3e58f63373915de95607a4ca36be`。
本worktreeはその直接子となる独立G5 source commitから保存値算術を一回だけ行う。
原G4のsource、contract、authorization、marker、STOP、分類を変更しない。
rootのreviewのみbyte-exactでdocs/researchへ写し、絶対参照元・bytes・SHAを記録する。
元資料の独立性は外部再現を意味しない。G5はG4のstdlib有理区間・sqrt/logとtableのprovenance checkerを共有するが、
G1/G2/G3 evaluator、solver、matrix/native lowering、synthesizerをimport/callしない。

対象は既知development 2-system-qubit distinct_basis controlled、sigma=+1、x=1/4、
R=3/4 ZI+1/4 V† IZ V、V=exp(-i pi XX/16)、finite P3(-ixR)。
分子・geometry・basis/DF rank・split L_D・delta windowはいずれも該当せず、独立held-outではない。
accuracy axis=1/200、alpha axis=1/5280、log argument=10560、workspace=1 beyond2 systems。
登録precisionは1e-3/1e-4/1e-6。新precision/angle/target/seed/gridを取得しない。
CTSはG4 `CTS_selected_complete_laws.json['1/4/1Q']` の同一law、再選択しない。

## 形式係数classと6頂点

順序 O0,O2,P2,P3,A0,A1,A2。各prototypeの(a,b)はdegree k,k+1へ置く。
これはHamiltonian/unitary matrixではなく4×7の有理形式係数系Dである。
D gamma=(1,x,x²/2,x³/6)、gamma>=0の35個の4-column basisを正確Gauss消去で列挙する。
rank4、全column和>0より有界であり、重複を除いた6 BFSで頂点が尽くされる。
mu=(x²+2)/(x²+6)によりgamma=(s,b,mu+(1-mu)s-mu r-b,1-r-b,1-s,1-s,r)。
頂点(s,r,b)はordinary=(1,0,1)、PTSC=(1,0,0)、A=(0,1,0)、J1=(0,0,0)、J2=(0,0,mu)、J3=(1,1,0)。
各頂点のactive group数2,3,3,4,4,3なのでpure precision profilesは252。

任意gammaは頂点混合で表せる。各groupのprecision marginal fを固定すると、
頂点のactive group全体にproduct(f)を割り当てたpure-profile混合が元のevent coefficientを再現する。
zero-mass groupでは条件付きfを任意に選んでよく、寄与は0。inactive massで割らない。
この還元はprototype内の固定IID p-word sharesを保つclassに限る。

## 価格、固定policy下界、digital bridge

event ideal coefficient c>=0、saved native cost C、strict joint error deltaに対し
h=sqrt(C)、d=2delta、K=h^T c、s=1/200-d^T c、Phi=(K/s)²。
group priceは E[sqrt(C)] であり sqrt(E[C]) ではない。
252 profiles×T/CX/native1Qの756価格を有理区間で計算し、元G2の保存最小区間との重なりを確認する。
固定r_T=2150,r_CX=350,r_1Q=3500について各pure-profile K>=r sを確認する。
s>0のprofileでの比の確認は任意の凸混合へ線形不等式として延長する。
s<=0ではK>=0>=r s、実行可能lawではs>0が必要。

任意full-support event proposal q、補正weight c/qについて
m2=sum(c²/q)、EC=sum(q C)、Cauchy-Schwarzよりm2 EC>=K²。
共通Bernstein sufficient-shot policyはn>=2 ln10560 m2/s²。
両axis total native resource G=2n EC>=4 ln10560 (K/s)²>37 r²。
exp(37/4)<10560は60-term有理級数＋幾何tailで確認する。
このpolicyの割当費用の下界であって、真の必要shot数や情報理論的最小費用の下界ではない。
1Qはnative下界を使い、CTS側のtotal1Qはreadout5nを含む。同classのtotal1Q>=nativeなので保守的である。

対応digital classは同一event dictionaryの非負tilde c、あるideal cとのL1距離以上のcharge e、
remaining=1/200-e-d^T tilde c>0を持つもの。全108 saved conditionsについて0<=d<=1かつ
C<=r²(1-d)²を有理数で確認すれば h+r d<=r。
するとh^T tilde c-r remaining=(h^T c-r s)+(h+r d)^T(tilde c-c)+r e>=0。
旧K3のresidual/toleranceだけを満たすclass、別辞書、別confidence規則、stratificationやstate varianceは対象外。

## 同一CTS lawの独立確認

G4の12 saved synthesis sequenceのSHA/T/Tdagger/error metadataを照合するだけで、guardをmatrixで再実行しない。
Pauli scalar coefficientsをhalf-angle radicalの768bit有理区間で独立に導き、保存ideal coefficient区間へ入ることを確認する。
finite P3のreal II/ZZとimag ZI/IZ/XY/YX、負のreal correction、rotation sign、controlled relative phaseを確認する。
固定rational common-angle ratioとideal Lsの差をangle biasへ戻す。coefficient L1、synthesis bias、m2、range、
dyadic qの和・正support・2^60分母、sufficient-shot inequality、workspace、三資源とreadoutを照合する。
新Pauli word/matrix evaluationや新native circuitは作らない。scalar interval algebraのみ。

## 制限と分類

stdlib runtime identityを固定。wall600s、CPU480s、RSS512MiB、artifact output16MiB、
rational serialization50,000digits。35 coefficient systemsはLP callではない。
exclusive G5 markerの作成後はruns1/retries0。既存markerは触らない。
準備中testsはoff-domain x=1/3とsynthetic costsのみ。registered tableをテストfixtureで開かない。

- G5_DIGITAL_SIX_VERTEX_CLASS_EXCLUDED_BY_SAVED_CTS_LAW：idealと全digital bridge成立。
- G5_IDEAL_SIX_VERTEX_CLASS_EXCLUSION_ONLY：ideal成立、digital bridgeは不成立。
- G5_ADDITIONAL_CLASS_CLAIM_NOT_ESTABLISHED：価格区間・source・CTS対応・資源guard等の問題。

追加命題が成立しなくてもrやangleを変えて救済しない。prefixを最終主張に使わない。
G4の既存結果を維持する。全分類でmandatory STOP。
許可はこの一括G5確認・claim/evidence整理・既存source accessの静的監査のみ。
新LP/synthesis/DF/molecule/NPZ/GPU/trajectory/circuit/quantum sampling、全面v4を行わない。
次の科学的仮説や重要GPT reviewの開始判断は利用者/GPTへ戻す。
