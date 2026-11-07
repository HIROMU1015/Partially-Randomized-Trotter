# N4-B：各RZとconsumer追加位相の会計

2026-11-08 JST。[REPORT](REPORT.md)。**CONSUMER_PHASE_QUALIFIED** は本書の固定経路・C_phi条件・分離上界を意味する。Whole RPE成功の数値資格ではない。

## 1. 再利用する証明

[N3-B1aの位相証明](../pr_reference_n3b1a_emitter_static_20261108/PHASE_ERROR_BOUND.md)、[N3-B1b-0実runtime対応](../pr_reference_n3b1b0_runtime_20261108/EMITTER_BOUND_ASSESSMENT.md)、[N3-B1b-1 provider](../pr_reference_n3b1b1_dictionary_20261108/DICTIONARY_PROVIDER_SPEC.md)を再利用する。u=2^-53、Pi0=`0x1.921fb54442d18p+1`、P=Pi0/4、tau=2Pi0。

通常IEEE f64、固定Qiskit1.3.0/wheel、named ideal Clifford+Tの意味論が条件である。Local W loopのphaseはW個数を推測せず `(K mod8)P` に閉じる。負側は連結済みword全体のinverseで `(-K mod8)P`。数学的W=e^(iπ/4)とbinary P／tauとの差も保存上界に含む。Operatorのnumpy診断をその証明へ代用しない。

## 2. 実consumerへの操作対応

| 実操作 | 上界と帰属 |
|---|---|
| Native→outerの初期numeric phase設定1回 | Consumer shellの追加項 `<16u`。Phase0 constructorはexact |
| Native segment作成 | Phase0。Gate appendにglobal phase操作なし |
| RZのlocal W loopと負角inverse | その物理RZの既存certificate内。Positive/negativeの8状態閉包を使用 |
| Local word→segment内outへのjoin | そのRZのemitter項内。既に累積したsegment phaseとの非自明joinも既存C_phiの対象 |
| Lowered segment→outerへのjoin S回 | **別の**consumer shell追加項、各`<64u` |
| Reset／measure／barrierのappend | Numeric global phaseを操作しない |
| SOURCE/compilerの元global phase・vacuum P生成 | 既存SOURCE→native／vacuum disk側。新word emitterへ付け替えない |

C0の各certificateは保存guard＋64u。Local W＋segmentへのjoinを一度覆う。Dictionaryの各certificateは量子化＋選んだ基本guard合計＋45/140737488355328。このemitter項は旧43-carrier等の保守的上界をそのまま保持する。今回のcomplete-word emissionはその経路上限以下で、各RZの内部処理とsegmentへのjoinへ対応させる。未使用の保守的余裕を再配分・最適化していない。

**Segment→outer joinはRZ内部joinとは別の実操作**なので、どれかのRZへ黙って押し込めずshellへ1回記録する。既存certificateを減額／再定義したり、同じ内部joinを別途加えたりしない。巨大wholeへ旧局所emitter上界を1回だけ適用していない。

一般のcarrier phase同士は[0,tau]内で、和は2tau以下。N3の一般compose上界`<64u`と設定上界`<16u`から、追加scalarの局所1-qubit Frobenius上界は記号的に

`e_shell < (16 + 64S)u`。

これはfull joint Frobeniusのdimension-free上界ではない。Scalar誤差または局所誤差をspectral normへ移すときに保守的に使う。途中reset等を含む一般shellでは同じCPTP fenceのcontractivityを用い、固定RPE形ではprefix reset＋unitary body＋末尾readoutとして扱う。保存8 native回路の非empty unitary segmentは各1だが、この値を用いた新RPE disk計算は行っていない。

## 3. 実runtime・正負・境界

[環境とruntime観測](../../../artifacts/pr_reference_n4b_consumer/2026-11-08/ENVIRONMENT.json)に、旧SOURCE byte、wheel RECORD fingerprint、通常丸め、Pi0/P/tau、setter境界の対応を保存した。旧runtime helperのQPY不要という欄はN3時点のprobe説明として別名に残し、N4-Bの保存QPY入力8件とは区別した。

各局所caseで、実emissionのgate列とinteger W residueから構成した独立binary phase参照が一致した。初期phase、各segmentのphase、outer join前後をhexで保存した。Whole nativeの元phaseを保持するとは、wordのWが加わってもfinal phaseが元hexと同じという意味ではなく、元prefactorを残してWを正しく加えるという意味である。

Tauを負の最小subnormalから生成する境界も試験した。再設定でtau→0となる差をexact真2πと同一視せず、16u項で覆う。Qiskit Operatorはto_instruction時にもphaseを再設定するため、境界でOperator差0を観測してもそのscalar誤差が数学的に0とは主張しない。

保存QPY入力は全8件でphaseが旧hexと一致し、正規範囲[0,tau)内でsetterは恒等。新しいQPY往復はない。任意QPY／copying compose／symbolic phase／新しいpostpassへ本資格を延長しない。

## 4. JointとRPEへ残る接続

Compiler条件CとSOURCEのkey閉包下で、各物理RZ位置iにstrict certificate e_iがある。Spectral normでは局所embeddingにdimension因子を要さず、unitary telescopingで

`epsilon_syn <= sum_i e_i + e_shell`

という保守的な形式へ接続できる。Frameのword価格取消し、期待多重度、signed fusion補正をこの誤差和の免除に用いない。Actual SOURCE inverseと負native wordのinverseも混同しない。

旧CT-0の規約は、同じreset/readoutの複素raw信号diskに `d_syn <= 2 sqrt(2) epsilon_syn` を移し、**d_old+d_synを同じ単位で1回だけ加えてから**半径／位相へ変換する。D_oldには既存compiler・vacuum算術・SOURCEの誤差を残し、新側へ重ねて計上しない。

今回数値化したのは固定85局所certificateと指定局所診断のみ。全物理位置の和、shellのRPE diskへの寄与、合算半径・位相、保存199-shot条件の新しい数値検算は未実行。従って**FULL_SUPPORT_CT_RPE_UNQUALIFIED**を維持し、[次scope](NEXT_EXECUTION_SCOPE.md)をGPT／ユーザーへ返してmandatory STOP。
