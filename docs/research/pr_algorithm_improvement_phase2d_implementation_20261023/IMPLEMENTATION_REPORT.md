# 実装内容と未実行状態

実装したのはF1向け小系prototype。空class／TODOだけの実装ではなく、計算手順・型・例外・登録case実行経路を定義した。ただし全関数は動作未検証で、静的解析だけを実施した。

| module | 実装した責務 |
|---|---|
| fixture.py | canonical exact fraction、PlaneRotation/FragmentSpec/FixtureSpec、FrozenTaskIdentity、EvolutionContext、CompilerProfile、canonical JSON／serializer。P0候補は関数の内部に記述し、呼ばない |
| algebra.py | 有理数の実部・虚部によるexact Pauli積、JW二次／四次係数代数、frameの有理数積、S/P alternativesのmaterialize関数。dense trace oracleを使わず、exact zeroだけ除外 |
| prototype.py | MixedAtom/FragmentAlternatives/MixedTailView、P fragment間collect、S origin保持、RTEComponent／RTETailLike接続、既存law/config/event構造の再利用。lambda=0はexact空tailだけdeterministic分岐 |
| circuits.py | frame／任意長Pauli／identityのtyped中央制御、actual inverse、同basis run共有、singleton support／long-run full、phase集約、D半step融合、MixedEvolutionResult、prep込みX/Y wrapper |
| reference.py | CARのbit作用、元rotation仕様からのorbital matrix、exterior minorのGamma(V)、元H、独立paired law／polynomial／trajectory／controlled operator／signal。候補algebra/router/momentをimportしない |
| pricing.py | 固定literal gate grammarのRZ tokens、direct run scan、EMPTY/SINGLE/MULTI期待価格、two-endpoint一pass selector、同情報四候補全列挙、価格未取得時の明示fallback |
| intervals.py | mpmath1.3.0のscalar intervalを利用するB/N/Phi/Jのenclosure関数。整数境界を跨ぐceilは停止。providerのruntime資格は別途必要 |
| gates.py | rtol0のH／operator／signal gate、control0/1とunitarity、actual component由来mean、90/128digits比較、固定compiler設定、full-IR一致だけのcompile cache |
| cases.py | P0のI/u/s/v規則、8motif、別4label catalogの定義。今回生成・列挙していない |
| authorization.py | stdlibだけのread-only packet検査、SOURCE/input/env/case/承認文のhash、F1 scope／件数／依存DAG、fresh output／one-shotチェック。発行APIなし |
| execution.py | 外部でbindingされた登録caseだけを処理するF1 dispatcher、explicit prerequisites、coverage、失敗STOP、既存SOURCE再照合、保存・last-use memory解放。F2経路なし |
| scripts/run_pr_dictionary_f1.py | gate→atomic consumption→科学import／registered case処理→成否を問わずSTOPの順序。今回起動しない |

## 保持した意味論

full Fock registerで操作し、中間sector投影しない。R内Iを外部scalarへ移さない。productsを時間順に保持し最後にchild。SOURCEのunsigned_rotation_angleがsignを既に含むことを考慮する。負product、paired phase、identity rotation、D内scalar、外部cをancilla相対位相へ残す。Pauliは任意長supportで扱い、DF用Z/ZZ metadataに偽装しない。

basisは非制御、中央と相対phaseだけ制御。singleton support completionは元columnsを保持する限定port、長runはfull basis。Dは一blockのため隣接half-stepを融合できる。mixed結果は独立の型を持ち、既存strict DF resultへcastしない。state prep X0X1をwrapper内へ明示的に入れ、Y readoutをSdg→Hとして+Im規約を保持する。

## 実行を拒否する条件

runnerはP0/Iの承認、draft contract、未固定SOURCE/input/env、未登録case／件数、unknown handler、重複case ID、path traversal、使用済みauthorization、既存outputを科学import前に拒否する。将来のauthorizationはユーザーが明示承認した別packetが必要。SOURCEだけから承認を発行する関数は作らない。

各caseはregistered prerequisitesだけを参照し、自動pilot／retry／増精度をしない。正確なcase ID集合はB後に確定するため、今回は発行しない。新moduleのimport時にはfixtureや科学resultを生成しない設計である。

## 技術的な限界

runtimeの型／compiler／Gaussian convention／mpmath scalar intervalを検証していない。特にnative GaussianのU.T、support completion、Y符号、relative phaseはF1で確認する。既存Gaussian helperはbasis global phaseをoperation列へ保持しないが、basisとactual inverseによる共役内で相殺する設計であり、event/scalar phaseは別に保持する。ここに実行済み認証はない。

literal RZ grammarは明示recipeの計数であり、compiled cost予測・上界ではない。interval providerは必要な資格recordが無ければUNPRICED_FALLBACK。sourceの存在だけでcertificateを発行しない。score区間のoverlapは現候補維持、ceilの不確定はSTOP。

F1 executorは登録順DAGを一件ずつ処理し、初回失敗で後続caseへ進まない。full-IR cacheとlast-use解放を用意した。独立compileのprocess並列実行を開始するschedulerは追加していない。最終B/F1契約で実空きCPU/RAM／memory見積もり／非干渉に基づき並列構成を検討し、必要なら新SOURCE版を固定する。この未実施の性能準備をspeedup実績としない。CPU/RAM/GPU/worker/wall-timeの人工capは設定していない。
