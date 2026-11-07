# N4-A：共通word policyの技術的接続仕様

2026-11-08 JST。[REPORT](REPORT.md)。設計・静的照合のみで、role consumerやproviderの新実行を行っていない。

## 1. 変えない選択規則

既存[DictionaryProvider](../../../src/trottertracks/pr_reference_dyadic_dictionary.py)をそのまま再利用する。

1. 実際のcompiler A後のbinary64 RZ引数を読む。
2. abs(theta).hex()が固定C0にあれば、元のe=-11採用wordを優先。
3. missで保存L以内なら、同じb36・43基本word・j昇順でpositive sequenceを選ぶ。
4. 域外／不明なliteral／不適合contextは未資格として停止。可変cache・別precision・acquireへ黙ってfallbackしない。

arm／role／SOURCE gate名をword選択の分岐にしない。SOURCEの理想角からkeyを作ったり、近接hexを統合したりしない。内部基本e=-13と最終要求e=-11を混同せず、composite certificateをbackend recordへ偽装しない。

同じpositive literalには同じpositive_sequence、負側には同じcomplete wordのactual adjointとなる。固定C0全299採用recordと43基本wordは読み取り専用。今回、両archiveとword hashを照合しただけでproviderをinstantiateしていない。

## 2. SOURCE inverseとword inverseの層

| 操作 | 保持する意味 |
|---|---|
| SOURCE F†／basis inverse／D inverse | 元の順序と各SOURCE gate inverseからcompilerへ渡す |
| Compiler Aのnative lowering | 保存contextのbinary引数を維持。SOURCE inverseを正方向nativeの単純反転へ置換しない |
| 負のnative RZ | positive word全体を実adjointにする |
| Emission | complete positive wordを旧lower_nativeへ渡し、負角でlocal.inverseを1回使う |

signed_sequenceを先にadjoint化してから負角lower_nativeへ渡すと二重反転になるので禁止。W、S/Sdg、T/Tdag、文字順とchunk順を保持し、新controlを合成後へ付けない。

## 3. 最小consumer差分

既存lower_nativeはrz/sx/x/cxを受け、freshなquantum-only circuitを返す。reset／measure／barrierは拒否し、元のclassical bitsを引き継ぐwhole wrapperではない。既存providerは1-qubit RZのword/certificateを返す。この二つを重複実装せず、次の薄い接続が必要。

- 固定archiveからC0と43基本recordを読み、既存metadata validator・manifest/hash・過去資格を対応させる。
- distinct positive native keyからpositive sequenceと別certificateへの読み取り用mapを作る。同一keyのword選択を両armで共有するが、物理gateや誤差出現を共有したと数えない。
- 元nativeのqubit/clbit、reset、measurement、必要barrier順序、global phaseを保持するouter shellと、unitary segmentを分ける。
- unitary segmentだけを旧lower_nativeへ渡し、旧sx/x/cx処理とinplace joinを維持する。
- segment／outerへのphase初期設定とinverseを数え、N3上界のどの項へ対応するか検証する。copy/QPY/to_instructionを無条件に追加しない。

これは本番adapterの仕様案であり今回未実装。旧CT-1 matrices_for等のcompressed word評価も再利用候補だが、その数学的exact word評価だけでfloat global_phaseを持つ実emitter全経路を資格したことにしない。

## 4. Phaseとglobal metadata

SOURCE scalarやcontrolled相対phaseを捨てない。Compiler Aは既にcontrolledなjointをlowerするという既存規約を維持する。局所wordのWをT価格0という理由で削除しない。

保存代表local recordにはglobal phaseが記録されていないものがある。missingを0としてSOURCE→native strict一致を主張しない。局所試験では保存native gate product自体を置換targetとして定義し、同じ未変更prefactorを両側へ保つ。SOURCE unitaryの検証が必要なら別の証拠が要る。

全jointのglobal_phase_hex、qubits、reset/readoutは既存CT-4B native metadataから保持する。独立したphase再初期化やcopyの増分を、既に保存guardへ含まれていると仮定しない。N3-B1b-0のcopy／Operator変換の再正規化の留保も引き継ぐ。

## 5. Compiler条件Cとliteral閉包

[CT-5BのC](../pr_reference_ct5b_20261107/STRUCTURAL_CERTIFICATES.md)を前提に、同じsource gate族・width・parameter数、同じpost-unroll template、同じ同値則・pass列を使う。word置換はcompiler後なので、新T gateをSOURCE contextへ追加しない。

Dはevent非依存の固定SOURCE。frameは同一event列でbasis pair multisetが共通。初段cancellationはD/frameのRZ/XXPlusYYを消さず、最終passはCX/Xだけに作用する。symbolic P templateは引数を変えずRZへ渡す。これらは完了済み静的監査の再利用で、今回の新compile結果ではない。

85候補keyは保存m7台帳からの有限superset。Cを外れるcontextや新しいliteral生成を見つけた場合、既存数字の近似・snapで押し込まず、COMPILER_CONTEXT_UNQUALIFIEDへ戻す。

## 6. N1固定価格との整合

C0優先によりD／中央4 key／scalar／CSWAP／product／wrapperの採用wordは保てる。保存K=160222、d0=106、d2=114、vacuum cap6072<173790は今回再計算しない。

新frameの辞書wordが旧cache wordと同じ価格になるとは言わない。Cと同一literal policyが成立すれば、対応するnative frame multisetに任意の共通価格を代入して差が取消す。SOURCE逆方向をforwardの同価格と仮定して取消すのではない。

product-Z消去はliteral個数を変えても、保存T=0 wordへの価格寄与だけは差から落とせる。B′ event価格の非負性も差の下界で使える。いずれも物理gate／誤差の免除ではなく、全supportの実行資格が別途要る。

## 7. 全位置の誤差会計への条件

同じreset・readoutを保つnative unitaryとword置換unitaryを比較し、各RZのstrict Frobenius上界をspectral normへ移してtelescopingする。frame共通性は誤差取消しに使わない。

量子化＋基本guard＋emitterをword certificateへ一度だけ含める。全jointの追加phase処理がそのcertificateで既に覆われるかを確認し、未包含なら明示的な項を設ける。別位置の誤差をvacuumだけへ付け替えない。

SOURCE→native／vacuum生成の既存diskと、新native→CTのdiskを同じraw信号単位で合算してから位相／半径へ変換する。古いcompilerやπ差を二重計上しない。正規化・attenuation・同じX/Y信号を保ち、平均channel一致をcoherent信号一致へ代用しない。

今回この数値再会計は行っていない。局所存在、実common policy、全joint誤差、RPE成功、pointwise価格条件は別々に判定し、**UNQUALIFIED_FULL_SUPPORT_CT_RPE**を維持する。
