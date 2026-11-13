# H4 DF10：DF head＋Pauli tailの限定P1結果

承認されたLD1/LD2×m0/m6の4 Pauli mean、LD1/m0のDF同R対照1件、そのfull256照合1件を、科学SOURCE **05447d1d4f1e3c0e9e335a7085aa58a877c139f2**でfresh run01に一回実行した。**6 meanすべての局所数値gateが通過**した。18complex出力・unique normalizer5組を保存し、旧DF4rowのSOURCE/数値gateを受け入れた。新reference・新DF mean・P0 operator/dictionary再計算・N/MSE/compile/価格/Gは0。

[全結果](run01/result.json)／[mean CSV](MEAN_SUMMARY.csv)／[B mean packet](MEAN_HANDOFF.json)／[HANDOFF](HANDOFF.md)／[SOURCE](SOURCE_INDEX.json)。[実行計画](EXECUTION_PLAN.md)は元75fb0ba1のSCIENCE_PLAN P1と今回の明示承認の範囲だけを具体化したもの。

## 条件・再利用

Linear H4、隣接1.00 Å、STO-3G、保存tol-only1e−8 DF10、同H/state/snapshot、保存weighted-Hermitian completion・fragment順、sector(2,2)/36とfull256、target=保存DF-Hの同sector基底状態、保存数値stateによる条件付き評価。S2、ΔPF=.1、K2、h=.1、M12、P=0、元prior/decoder/shift0/誤差契約を維持した。RMSE予算はmetadataとしてεE=.00159360010199040 Ha、εqpe=εtrot=.00079680005099520 Ha、minkowski/finite_B_diagnosticを保持したが、今回RMSEを計算していない。

共通κは表示近似へ変えず、元のexact有理数
`13112614441153360648422410673503886836581176429267589536059228160/539496659426909122096437389293855970805348475396548904610992201`
を使った。P0のPauli λをbinary64 exact FractionとしてBの既存paper mapperに渡し、r*=κλ²T²→ceil/uniform R_outerを生成した。

- P0結果 **4220f3006d0d8d413e195ed948e250578513b557** の辞書・common block・Pauli_nonidentity行列を再利用し、新しいfull拡張/JW展開は0。保存stateは既存save-only loaderで読み、元H/state/配列hash・sector/liftを保持した。
- headは元保存block、tailはP0のPauli非identity和そのもの。λ_PはLD1=3.3066052942233157、LD2=2.0550998769244275。共通scalarは保存s_R=.4177175120626939/.2223759094107018とc=2.29310124732を維持し、全Tで一回だけ作用させた。s_P−s_Rの微小差はP0残差のまま保持し、追加identity eventや補償phaseを作らない。
- 既存providerのdirect-expm head、paired finite Taylor、独立raw/corrected、exact-tail、matrix power、R-aware empirical allowanceを再利用した。変更は辞書付きB canonical IDの検査と、caller-ownedの同tau/K normalizer共有のみ。full/36で同じnormalizerを再計算していない。
- 保存referenceはbba15083の同H/state full-H独立pairを共有し、新action0。保存meanの比較元は91349e70にあるfefd548e SOURCEのDF結果で、旧spectral meanへ差し替えていない。

## B canonical identity

最新公開B **ce0517de6e7b58cf4bf4c69f99f998d86099c406**のgenerate/_validate/paper_event_mapperを、既存metadata loader経由で使用した。辞書kind/hash/SOURCE・λ/scalar・bit順をsplit/lawに明示し、B自身が以下の新IDを生成した。A独自ID生成器や旧DF IDの再ラベルはない。

| 構成 | B canonical configuration ID |
|---|---|
| Pauli LD1・共通κ | `d905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27` |
| Pauli LD2・共通κ | `d7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3` |
| LD1/m0・DF同R診断 | `81afc1a8c0660b116ac656b6411ea01028a0a56f474aa0b3ba664d0ce5ae3126` |

B native enum `PAPER_TOTAL_ROTATIONS`は元Aのpaper回転数型κと同じ定義へ明示対応させ、元定義名・exact値を保持した。normalization上限型κ_Bではない。fixed-R対照はm0だけR2へoverrideし、そのroundのκ適用=falseを記録した。他roundは共通κのmetadataのみで科学評価していない。[PLAN](run01/PLAN.json)に全schedule・元DF configuration・control区別を保存した。

## 共通κの二時点結果

全候補でqPF=2^m、actualΔ=.1、signed weights=[1]、allocation=[R_outer]。qRPEは別fieldである。以下のbiasは**無次元signal空間のabs(corrected−保存state full-H reference)**であり、phase/energy誤差・RMSEではない。

| Pauli構成 | m/T | qPF | R_outer | event位置数 | log B | attenuation | signal bias | finite-K差 abs(corrected−exact-tail) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LD1 | 0/.1 | 1 | 3 | 3 | .03618365612479528 | .9644631476524569 | 1.3405038383e−5 | 2.5149522279e−9 |
| LD1 | 6/6.4 | 64 | 171 | 10944 | .04092112900511335 | .9599048356116504 | 8.5726826391e−4 | 1.5698160048e−12 |
| LD2 | 0/.1 | 1 | 2 | 2 | .02098516412709399 | .9792334922453302 | 1.3374048678e−5 | 1.2062419070e−8 |
| LD2 | 6/6.4 | 64 | 66 | 4224 | .040954288956605 | .9598730057416063 | 8.5526057695e−4 | 2.1464541804e−11 |

| 構成 | raw mean | corrected mean | exact-tail mean |
|---|---|---|---|
| LD1/m0・κ | .941922041606 + .207296948121i | .976628338676 + .214935063745i | .976628341127 + .214935064307i |
| LD1/m6・κ | .258944612298 + .924318602555i | .269760712407 + .962927332235i | .269760712408 + .962927332237i |
| LD2/m0・κ | .956347163059 + .210471640712i | .976628322695 + .214935091966i | .976628334385 + .214935094940i |
| LD2/m6・κ | .258934170167 + .924288472561i | .269758779149 + .962927873825i | .269758779155 + .962927873845i |

表示は丸め値で、正本は各mean JSONとCSV。referenceはm0=.9766254605337832+.21494815616137702i、m6=.2689351412041076+.9631582890810391iであり、保存stateのfull-H信号をそのまま使用した。referenceを厳密基底固有stateのexp(−iE0T)と置き換えていない。

## DF固定R・full256/sector36の対照

保存DFのRはLD1 m0/m6=2/128、LD2 m0/m6=1/40で、共通κのPauli4候補とはすべて異なる。元4rowのSOURCE・identity・gate・allowanceは保存のまま受け入れ、これら4組については同条件のsignal差を計算せず`DIFFERENT_PHYSICAL_LAYOUT_NO_SAME_CONDITION_COMPARISON`を残した。

別に承認されたLD1/m0・R2対照を評価した。Pauli logB=.053795375562846255、B=1.055268646308115、attenuation=.9476259940997264。raw=.9254783947953189+.2036780521501009i、corrected=.9766283328630632+.2149350623751106i、exact-tail=.9766283411269295+.21493506430740117i。

| 対照 | exact-tail差 | corrected差 | raw差 | 判定 |
|---|---:|---:|---:|---|
| DF対Pauli・LD1/m0/R2・同head/δ/K/配分/scalar | 0 | 2.7755575615628914e−17 | .012693357510252933 | exact-tail/corrected gate通過、rawは辞書依存で相違を許す |
| 同Pauli条件・full256対sector36 | 2.2887833992611187e−16 | 2.7755575615628914e−17 | 3.433175098891678e−16 | 全3 mean gate通過、logB/attenuation差は0 |

rawはcorrected/Bの事後代入ではなく独立伝播で取得した。同じRのcorrected mean一致はP0の同operator数値残差を前提とする局所対照であり、異なる辞書のnormalizer/rawが同じになるという主張ではない。full256照合もこのLD1/m0/R2の1点だけで、未評価roundや実samplerを包括的に資格付けしていない。

full/36のnormalizerは同tau/Kの値を共有したためlogB/attenuationは構成上同一である。3種類のcomplex meanは異なる次元の行列で別々に伝播して比較した。normalizerの独立経路による精度照合を行ったという意味ではない。

## 数値検査・資格

6件のexact-tail norm/unitarity、head unitarity、raw/corrected vector closure、reference pair、raw outcome rangeの既存gateがすべて通過した。gateは変更していない。

- 最大power unitarity残差：1.644016287080198e−13、最大raw/corrected closure：3.3535280294515737e−13（gate1e−9）。
- 保存reference pair差：m0=1.1102230246251565e−16、m6=5.964543396834415e−15（gate1e−10）。
- normalizerは5unique tau/K組がすべて有限でb>1。1ULP log感度は最大4.860093772741936e−12。既存empirical logB marginへ反映し、normalizer精度未解決・underflow/overflowは0。
- R-aware allowanceのuは約1.0000e−9、m6 logB marginはLD1=3.2852394375132916e−10、LD2=1.3161198792270903e−10。いずれも経験的診断でrigorous boundではない。観測norm domainはmacro/power境界であり全event内部ではない。
- 実行後もsnapshot/receipt/contextは元hashと一致。P0/旧DF raw・結果・grant/ledger/marker/KEEP/STOPは変更していない。共有環境・B/C/D worktree/processに変更なし。

新人工6＋既存provider10/planner7の計23件PASS。[TESTS](TESTS.md)。Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1、実科学6.0985秒。SOURCE固定後のscience起動は一回、retry/fallback/閾値緩和/再正規化は0。

## Bへの引渡しと次段階

[MEAN_HANDOFF](MEAN_HANDOFF.json)はB canonicalの各Pauli configurationについて、m0/m6だけmeanを持ち、残る11roundは`NOT_EVALUATED`/nullにしている。fixed-R/full256は診断対照として別枠にし、共通κscheduleのmeanと混在させない。全roundでN/nominal N/十分N/global RMSE/C/G/winnerはnull。保存stateの厳密ground同定・actual sampler/control law・finite-Kの全RPE RMSEへの移送は未資格を維持する。

次は、ユーザーがPauli候補をさらに調べる対象・有限scopeを決定する。Aは承認後だけ不足11round/prefixのsignalへ進み、Bは公開mean/lawを既存risk/統計器へ対応付け、C/Dは同dictionary/scheduleの価格接続条件を別担当範囲で扱う。P1だけで全RPEの精度・十分N・コスト優位・最適候補を認定しない。

成果をcommit/pushしremote SHA/tree/主要ファイル照合後、**mandatory STOP**。全13roundの新signal・K/κ変更・RPE/N/MSE・compile・価格/Gへ自動進行しない。
