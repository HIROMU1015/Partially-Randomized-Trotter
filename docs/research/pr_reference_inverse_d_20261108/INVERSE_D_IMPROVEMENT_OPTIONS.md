# Inverse／D：改善要因を分ける候補

基準06895797e032bc236562da5ee541b33ce8eb5050。[REPORT](REPORT.md)と保存SOURCEに基づく条件整理で、下記の新しい数値評価・構成は未実行、未採択である。

## 1. Record再利用と保証改善の違い

| 案 | 既存資産／最小差分 | 保証・Tへの影響 | 必要な資格 |
|---|---|---|---|
| 保存recordのexact-literal再選択 | S2非劣ID、C0、85 certificate、B1 certificate／carrier | Inverse109／D164ではcurrentも非劣。Guardを下げる別点ならT増加が必要となる可能性。新しい比較はD1/D2待ち | 元spec／guard／W／actual adjoint、selected wordのconsumer経路 |
| 現composite全wordの再保証 | 43基本wordから既に保存されたsequence、exact_word_matrix、strict_guard | 保存Q＋Gは分割加算であり、同じwordの直接保証で鋭くなる可能性。Wordを変えなければlogical Tは同じ。改善は未確認 | exact binary64 targetと全Wに対する新strict guard。Emitterは別 |
| 現independent A/Cのdefect評価 | 保存native列、現C0＋85、既存A1設計 | 新合成／word変更なし。小さいetaが得られるか不明 | Native phase欠損を残したbody評価とfull strict etaを区別 |
| Basis全体actual adjoint化 | 同じAの完全inverse、B1／consumerの逆処理を参考 | Logical eta=0を構成できる。T増減・fence／sharing効果は未取得 | 新SOURCE、全compiler境界、SOURCE→logical、phase、terminal、emitter |
| Support/run上界 | Builderのreuse、identity flush、component順序、SOURCE7 ID | 同じ回路／wordで保守性を減らす候補。Tは変更しないが要求precision／shotsへの効果は未評価 | 全lawの到達性と正しいword-error遷移。期待値で代用しない |

新規合成は、これらの既存record／既存word保証で不足するkeyと要求条件を特定した後に判断する。Exact backend recordがない40 inverse keyにも現compositeは存在する。欠損record数を、新合成必須数と扱わない。

### 既存frontierから分かる分類

1. **現wordより小さいguardの候補**：どのrecordが該当するかは未分類。保存guard文字列はSTATIC_BINDINGSに引用し、新比較はしていない。Nominal exponentだけで実guardを順位付けしない。
2. **高Tでより小さいguardの候補**：現word非劣と両立するtradeoffである。Inverse51 keyには複数word列の非劣候補があるが、その方向はD1で確定する。
3. **Guard低下・T非増加の候補**：現在の保存二目的poolでは109 inverse／164 Dのcurrentが非劣に残るため、strictな支配改善を既存結果に反して存在すると扱わない。構造保証や新しい同word guardは別の証拠範囲。
4. **既存recordだけでは不足し得るkey**：40 inverse keyにはexact-literal backend recordがなく、最大basis31 key中25 keyが該当する。現word直接保証や別構成の余地を残す。Dは全164 keyがC0 recordを持つが、必要なjoint精度を満たすこととは別である。

## 2. 共役式の正しい単位とphase条件

[既存S3式](../pr_reference_s3_s2_reuse_20261108/IMPROVEMENT_OPTIONS.md#1-共役式と資格の段階)を再利用する。Unitary B,D,A,D_tilde,Cに対して

`eps_f=||A-B||_2`, `eps_i=||C-B†||_2`, `eps_D=||D_tilde-D||_2`, `eta=||C-A†||_2`。

D=exp(-itH)、W=diam(spec H)なら

`||A D_tilde C-B D B†||_2 <= eps_D + min(eps_f+eps_i, min(2,|t|W)eps_f+eta)`。

Logical word上で同じAのactual adjointをCとして構成すればeta=0となる。現consumerが各negative RZをinverseにすることだけではこの前提にならない。Compiler後の前後列の再配列・literalが違い、local phaseも欠ける。

既存compile_referenceはalready-controlled jointの相対位相を保持する。一つのcommon global phaseと、control前のscalar／vacuum／event relative phaseは違う。Basis sliceのnative phase不明を0と置いたり、phase-fitした値をstrict etaとして採用したりしない。D supplementのSOURCE global phaseはnative slice phaseの代わりにならない。

Actual-adjoint案では、A中のcommon scalarは同じA†との共役内で相殺できるが、これを理由に外部のrelative phaseを捨てない。Named Clifford+T wordのexact inverseと、Qiskitのfloat phase carrierによるemissionを分ける。Logical eta=0でも、既存emitter／shell丸めのallowanceは消えない。

Deterministic diagonalの小さいtをRTE run全体へ無条件に使わない。RTE centerはproduct Z／ZZとrotationを含み、near-identityでない場合がある。SOURCEではproductが先、identityはactive basisを閉じ、同ID/hashの隣接applicationでbasisを共有する。既に除いたpairを再び削減したと数えない。

## 3. 現構成を保つ案と構成を変える案

**現構成保持**では、D1/D2のrecord再選択、現word再保証、現A/Cのeta評価を別々に扱う。前二者でSOURCE order／compilerは維持できる可能性があるが、新recordのemitter適合や新combined diskまで既存結果を自動移送しない。

**Actual-adjoint構成**では、SOURCEのuncontrolled basis境界を保持し、一度構成した近似Aを対になる戻り側へinverseとして接続する必要がある。現行のjoint lowering後の個別RZ置換だけでは境界を復元できない。新control／再compileで相対位相を変えず、D融合・basis共有・実際のF/G inverseを保持するinsertion方式を先に固定する。旧compiler diskを流用できるという前提は置かない。

保存88 pairは限定書換えでNOT_PROVED。再実行していないし、A/C不一致の証明としても使わない。現word eta評価の結果なしにactual-adjoint構成を必要条件と断定しない。

## 4. D寄与と構造上界の候補

B′のD5 templateとReference forward／actual inverseの各D5 templateを、旧S1の係数で扱う。D1/D2ではそれぞれbaselineのG forward43 mapを固定し、共通5 keyの変更は全roleへ伝播させる。Role別selectorで他項を不自然に固定しない。

D2の寄与は、各templateのguard最大contextでの`n_key × guard_key`に、そのtemplateのSOURCE係数を掛けた**加法上界の分解**とする。Templateごとに別のworst contextを許す現在のS1証明の分解であり、同時到達するtrajectoryや実誤差の寄与ではない。Tはlocal template値までで、係数・shotsを掛けた全RPE価格を作らない。

構造保証では、次の保守性を個別に検討する。

- Dのbasis／diagonal／returnを本当に共役blockとして取り出せるか。Fused template、directional right、controlled phase、SOURCE inverseごとに資格する。
- SOURCEで不可能なcontext組合せを全support証明から除けるか。最大frame basisでは保存4 context全てがtieなので、単にcontextを一つ選ぶだけでは既存template-maxは下がらない。
- Basis reuseとidentity flushを7 basis＋inactiveの状態遷移へ写し、component順序・有限bits到達性を含む安全な遷移集合でworst-case誤差を評価できるか。Max-plus行列の冪を用いるなら全qを展開する必要はないが、これは未実装・未検証の設計案。

新しいsupport quotient、phase条件、D center幅、etaは今回評価しない。既知C2 FullY4の数値、旧native depth資格、期待basis run数をこの条件へ移さない。

## 5. 全体費用への接続条件

局所guard低下→全support logical上界→旧diskとの正しい単位の一度だけの合算→位相・半径→整数shots→各roundの回路価格、という接続が必要。現段階でD／inverse以外のdomain・central・emitter・shell・terminalは残り、626／630 shotsや全18round総Tは再計算しない。

Word再選択とactual-adjoint構成を同時変更しない。D1/D2は各々G forward baselineから独立に比較し、D3は別構成／別保証として扱う。同じrecord poolと情報・資格機会を強B′にも与える。局所Tが低い点、LP下界、未資格価格を実行可能な全体baseline価格として使わない。
