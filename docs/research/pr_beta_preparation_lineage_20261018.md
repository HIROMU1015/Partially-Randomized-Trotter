# B08：WP04物理gridとWP11会計のpreparation来歴

2026-10-18。判定：**MODEL_MISMATCHを維持、技術的な追跡は完了**。新しいpreparation・state・signal・operator norm・overlapの計算は行わない。

科学snapshot `f29eafbf4412193e8e8660d9f051889c2cba684b`、前段監査 `3fd8fca966e6b5a6a972aa71de68b31a0129f5ab`を参照。[固定blobと調査metadata](../../artifacts/resource_applicability/pr_beta_contract_followup/2026-10-18/b08_investigation.json)、[入力blobのcommit・bytes・SHA256](../../artifacts/resource_applicability/pr_beta_contract_followup/2026-10-18/provenance.json)。

## 三つの生成経路

共通の対象は保存linear H4、1 Å、STO-3G、DF rank12、L_D3。物理gridは4電子sector dimension70。β・α・compilerを含む全契約の同一性を、この共通条件だけから推論しない。

|経路|SOURCE上の生成契約|保存metadataと限界|
|---|---|---|
|WP04物理grid|`run_research_direction_ablation.py`が`validate_finite_rte_signals`を呼び、閾値を省略。同関数の既定値`1e-12`をpreparationへ明示的に渡す|requestにも1e-12。108項drop、operator error宣言`4.8294649659693557e-14`。preparation hash `89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919`|
|WP04会計／費用dataset|ablationと`_wp01s`は`prepare_df_partial_s2(..., identity_policy="extract_identity_phase")`で閾値を省略。preparation既定値0|L_D3の7組datasetはpreparation `50c77f5455cc1a3c0710ed3c7588efc957b9cead7436bd0a6d851af61738a4bc`。benchmarkはこの欄に実requestのpreparation hashを記録。opt1費用系列|
|WP11会計／費用worker|`evaluate_coherent_extension_analysis`と`_worker_preparation`も閾値を省略し、SOURCE既定値0。workerはsnapshot SHAとcompiler文脈を照合する|実preparation hash／canonical payloadは、確認した公開最終JSON・aggregateに保存されていない。opt2費用系列。旧opt1のhashを推定で転記しない|

WP04 runnerは同一実行内でも「会計preparation」と「行列検証preparation」を別々に作る。その二つの入口が異なる既定値を持つことが、具体的に確認できた設定差である。hashには他の項目も入るため、この差だけがhash差の唯一の原因だと完全に証明したものではない。

記録されたgrid生成commit `efa90e0b14a95c296b305bfeea37867e56c83a0d`でも、finite signal関数既定値は1e-12、preparation関数既定値は0。WP11初期 `26aa95c7b3f5ecd1f358280e58073ca2ee44ea5c`、追加 `7753eb71844e429facdb6f32c2def6e37379541f`のpreparation既定値も0だった。ただし、当該依存fileの実行時hashはそれぞれの保存source集合に含まれない。記録git commitのSOURCEと、dirty生成時の全依存SOURCEを同一と認定しない。

## 閾値・fragment・λ・恒等成分

`split_df_hamiltonian_by_ld`は既定`lambda_frobenius_squared`でrankしたfragmentのprefixを分ける。gridの保存順はdeterministic `[0,1,2]`、randomized `[3,4,5,6,7,8,9,10,11]`。旧費用datasetとはHamiltonian hash `56e4df83…`、partition hash `9034397f…`が一致する。

`df_rte_tail.py`はpre-threshold componentをfragment ID／Pauli supportで並べ、`coefficient_abs > coefficient_atol`を残す。続いて`extract_identity_phase`の恒等成分を決定論側へ分離する。λは残るrandomized coefficientの絶対値和であり、fragmentをrankするproxy λとは別。

gridにはrandomized component218、λ=`0.5832704246339503`、extracted identity coefficient=`0.09261516589219077`が保存されている。これらはgrid側の観測metadataであり、WP11の値として採用しない。WP11の保存τからλを逆算したり、閾値0なら何項残るかを新しく計算したりしていない。

閾値は単なるlabelではなく、tail項、λ、恒等成分、sampling lawの生成へ関わる。108項dropの保存誤差が小さいことだけで、両preparationのsignalや位相・費用への影響をゼロとはしない。

## hashの意味と、保存されていない証拠

|識別子・数値|意味|これだけでは示せないこと|
|---|---|---|
|Hamiltonian hash|保存DF Hamiltonianの識別|tail閾値、preparation、入力stateの同一性|
|partition hash|fragment集合・rank・順序等の識別|閾値後tailや固有状態の同一性|
|preparation hash|H／partition、deterministic block payload、tail hash、恒等成分政策、coefficient_atol、basis構成、diagonal sort、PF政策をhash化|hash不一致時の数値差の大きさ、異なるruntimeでの科学的等価性|
|tail hash|threshold前component・basis metadata、閾値とnormalization/identity政策を含む|stateや時間発展signalの同一性|
|state fingerprint|SOURCE上はcomplex128配列bytesのSHA256。gridはphysical full-DF-H基底状態とPF-step control固有状態を別に記録|実state配列の可用性、別preparationのcontrol固有状態、全体の位相枝保証|
|circuit fingerprint／保存operator norm|回路の識別または特定比較の数値観測|dense operator配列の保存・同一性証明や厳密な移送上界|

gridのphysical stateはfull保存Hのsector最低固有vector、controlはPF-stepの固有vectorからphysical overlap最大のものを選ぶ。後者はpreparationを変えると同一とは限らない。physical stateも、保存H／sector／basis／固有状態の選択規約と実配列bytesの結合なしに、hash名だけから再利用済みとはしない。

旧gridの高いoverlap、正の半径、位相誤差、閾値誤差は`SAVED_NUMERICAL_OBSERVATION`として保持する。任意の新preparationへのtransfer certificateではない。今回、overlap・norm・signalを再評価していない。

## 18 roundと他の既存検証の扱い

δ=.02、K2、q1〜131072の18組は前段のID対応表を転記した。r scheduleは`[1×8,2×2,4×1,8×2,16×2,32×3]`。パラメータID対応は有用だがpreparation一致の判定とは別。

確認対象は前段inventoryのWP04 grid・費用dataset、WP11公開provenance／aggregate、既知の`artifacts/finite_rte_signal_validation/`の9 JSON、および旧長round schedule JSONに限定した。9 gridはいずれも閾値1e-12で、他のL_D／δ／短いqの条件を含む。長round保存projectionにはfinite signal fingerprintがあるが、preparation hash自体は保存されていない。この確認範囲で同一WP11 preparationの物理検証は特定できなかった。未公開archive全体を調べたとの主張はしない。

WP11公開aggregateの36／15 recordはtask・result fingerprintとmetricsのprojectionである。公開checkpoint pathに元worker payloadはなく、完全なpreparation stampを回収できない。`RAW_WORKER_PREPARATION_STAMP_MISSING_IN_CHECKED_PUBLIC_PATHS`として記録する。SOURCE既定値を実生成stampの代わりにしない。

## 六つの疑問への技術的回答

1. **どの差か**：二つの入口の既定閾値1e-12と0、保存された108項drop、異なるpreparation hash。完全なhash payload差分と旧runtime closureは未確定。
2. **同一条件の検証はあるか**：上記の固定保存範囲では特定できない。これは未公開archiveに存在しないとの断定ではない。
3. **再利用条件は何か**：同じH／partitionに加え、実preparation/tail/basis payload・λ・identity政策・SOURCE/runtime・state配列・sector・operator/phase規約を結合する。同じq/r/K/δだけでは不足。
4. **保存値でできることは何か**：hash／宣言／IDと既存証拠の分類。不一致によるsignalや費用差は今回計算しない。
5. **最低限何が不足か**：WP11実生成preparationのcanonical payload/hash、およびそのpreparationでの物理signal・半径・位相証拠、または正当な移送上界。再利用する場合は実state/operator配列とhash結合も必要。
6. **何に影響するか**：条件付き費用点推定は元の仮定と外挿の下で保持する。今回その数値を誤りと認定しない。一方、同一入力での物理検証済み・RPE最終成功保証という主張へ昇格できない。

## 未承認の最小後続候補

まず既に生成されたarchiveのmetadata／state／operatorの可用性を確認する案を残す。正確なpreparationの保存物が回収できれば、不要な新計算を避けられる。今回private runtime・分子NPZは開いていない。

必要な証拠が回収できない場合に限り、対象をWP11の固定H4 preparationと採用18組へ絞った物理signal検証が候補となる。SOURCE/runtime、入力SHA、実preparation hash、tail、λ、state/sector/basis、各組(q,r,K,δ)、signal規約、停止条件を先に固定する。状態を再利用できなければ、その生成も別承認の対象にする。

これは未承認案であり、科学call・実行grantは発行していない。追加のsampling／compile／β最適化／shot・費用再会計／18 round測定分枝シミュレーションを組み込まない。B-1/B-3等の研究判断へ自動拡張せず、ユーザーへ戻して停止する。
