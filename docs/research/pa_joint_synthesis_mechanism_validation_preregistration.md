# P-A非退化mechanism validation事前登録

最終更新：2026-09-25 JST

## 目的

完成済みP-A blind evidenceでは全54 recordが一つのsource-basis runにつき一つのselected segmentを
使い、holdout 256 eventは全てTaylor order 0だった。このため、既存RZ改善は
`one_segment_per_source_run` baselineでも説明でき、interval subdivision固有の寄与は未識別である。

本検証では結果を見る前に、次だけを判定する。

> 同一source-basis run内でsupportが非退化に変化し、Taylor order 2のproduct applicationを含むとき、
> interval DPは明示的一区間baselineと異なる分割を未使用basisへ移送し、operator同値性を保ちながら
> compiled RZを追加で減らすか。

H12、長RPE、full wrapper、自然発生頻度、backend/noiseまたは最終総costは対象にしない。

## 固定入力

- molecule：H4 linear chain
- geometry：1.0 Å
- basis：STO-3G
- qubit：8
- DF rank：12
- `L_D=3`
- `delta_time=0.02`
- 全event：Taylor order 2
- compiler：Qiskit 1.3.0、basis `rz,sx,x,cx`、optimization level 1、seed 17
- coupling map、layout、routing：なし
- identity policy：`extract_identity_phase`
- event順序：保存
- 隣接同一basis cancellation：両policyで有効

入力snapshotは既存H4 connected-cluster snapshotを再利用する。新しい分子、geometry、Hamiltonianまたは
状態計算は行わない。

## 比較policy

### baseline

`one_segment_per_source_run`

各最大source-basis runを必ず一つのsegmentとして扱い、そのrun全体について次の2候補だけを同じ
4段lexicographic proxyで比較する。

1. registered full basis
2. run全体のsupport unionを保存するdeterministic completion

### candidate

`interval_union_dp`

同じrunを一つ以上の連続segmentへ分け、各segmentでfull/support-unionを選ぶ。目的順序は

1. twice basis-operation count
2. segment count
3. support-union size sum
4. full-mode penalty

で固定する。candidateだけがrun内部を分割できる。

## training／blind分離

同じH4 snapshot内でsource DF fragmentを分離する。

- training diagnostic：fragment 3、5、7
- blind holdout：fragment 4、6、8
- operator probe：blind側fragment 4

training結果を見てprofile、threshold、blind fragmentを変更しない。

## 固定support profile

各eventは実在DF component 3個からなるorder-2 eventで、記載順はproduct、product、rotationである。

1. `singleton_far_blocks`
   - `[(0),(0),(0)]`
   - `[(7),(7),(7)]`
2. `singleton_pair_far_blocks`
   - `[(0),(1),(0)]`
   - `[(6),(7),(6)]`
3. `zz_far_blocks`
   - `[(0,1),(0,1),(0,1)]`
   - `[(6,7),(6,7),(6,7)]`
4. `mixed_local_far_blocks`
   - `[(0),(0,1),(1)]`
   - `[(6),(6,7),(7)]`
5. `three_separated_zz_blocks`
   - `[(0,1),(0,1),(0,1)]`
   - `[(3,4),(3,4),(3,4)]`
   - `[(6,7),(6,7),(6,7)]`

各fragmentと5 profileの直積を使うため、training 15 task、blind 15 task、合計30 taskである。

## primary gate

次を全て満たした場合だけ、P-Aを非退化mechanism検証通過とする。

1. blindの全eventがTaylor order 2。
2. run内分割がblindの2 source basis以上へ移送。
3. run内分割がblindの2 profile以上へ移送。
4. candidate planがblind rowの25%以上でbaselineと異なる。
5. blind pooled RZがbaseline比2%以上改善。
6. blind個別rowの最大RZ悪化が5%以下。
7. 全operator probeでbaseline/candidateとも最大差$10^{-10}$以下かつrelative ancilla phase一致。

CX、depth、circuit sizeは補助指標とし、primary RZ gateの代用にしない。

## 固定decision rule

- 全gate通過：
  `advance_pa_after_nondegenerate_mechanism_validation`
- 1つでも不通過：
  `stop_pa_interval_dp_as_primary_and_return_to_pc`

結果後にthresholdまたはprofileを変更して同じ結果を採用しない。追加の探索を行う場合は新しい
training/blindを事前登録し、本結果と分離する。

## scope

本streamは有限RTE分布のsupport内にある合法なorder-2 eventから構成するが、自然サンプリングでの
発生頻度や平均資源改善は評価しない。通過しても、literature novelty、Gaussian circuit全体の
global optimum、routed/noisy性能、full-wrapper/RPE総cost、H12または科学的優位性は確立しない。

## 実行規約


## 実行後の履歴注記

上記条件を変更せず30/30 taskを完了した。run内分割、plan変更、追加RZ改善はいずれも0で、固定7 gateの
うち4 gateが不通過となったため、事前規則どおり
`stop_pa_interval_dp_as_primary_and_return_to_pc`を採用した。数値と現行判断は
[結果文書](../research_direction_joint_synthesis_mechanism_validation.md)を参照する。本節は事前登録内容を
書き換えるものではなく、実行先への履歴リンクである。
runnerの`--dry-run`を先に実行し、30 taskのevent digestと固定gateを含むexpected-task artifactを
生成・fingerprintする。その後はmodule、runner、expected-task artifactを変更せず本compileを行う。

- implementation：
  `src/trotterlib/research_direction_joint_synthesis_mechanism_validation.py`
- runner：
  `scripts/run_research_direction_joint_synthesis_mechanism_validation.py`
- expected task：
  `artifacts/research_direction_joint_synthesis_mechanism_validation/2026-09-25/pa_forced_support_order2_expected_tasks_v1.json`
- final artifact：
  `artifacts/research_direction_joint_synthesis_mechanism_validation/2026-09-25/pa_forced_support_order2_mechanism_validation_v1.json`

## compile前固定記録

`--dry-run`は本compile前に完了した。

- expected task count：30（training 15、blind 15）
- 全event order：2
- 全task source run count：1
- expected-task content fingerprint：
  `e8b064e9821fae5c2e7a44d0c98d3f9a0ed973a4cd87945fb051151e446a96fc`
- expected-task file SHA-256：
  `437d905358419e284ed5ad4621be91e412e6d3939c4e213b6c10c4dd816fb2b4`
- implementation SHA-256：`3c5c939862b63805fc9103c2f16649687c5c6a4a5c108ab28089c749f928513e`
- runner SHA-256：`94606639ed1e5762fd2885b30193b3a73e6e00b0b1429d4072f155fed0ffe7ef`

## 実行後の履歴注記

上記条件を変更せず30/30 taskを完了した。run内分割、plan変更、追加RZ改善はいずれも0で、固定7 gateの
うち4 gateが不通過となったため、事前規則どおり
`stop_pa_interval_dp_as_primary_and_return_to_pc`を採用した。数値と現行判断は
[結果文書](../research_direction_joint_synthesis_mechanism_validation.md)を参照する。本節は事前登録内容を
書き換えるものではなく、実行先への履歴リンクである。
