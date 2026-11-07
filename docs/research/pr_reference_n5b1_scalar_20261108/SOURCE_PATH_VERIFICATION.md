# N5-B1：固定SOURCEとruntimeへの対応

2026-11-08 JST。[REPORT](REPORT.md)。基準commit **624646b9a4782a65e90561ea29df6beccea2a42c**。旧SOURCEは変更していない。

## 保存文字列から最終測定まで

| 段階 | SOURCE・既存証拠 | 確認した対応 |
|---|---|---|
| 保存word | C0 archiveの固定8 member、N4-B ROLE_WORD_CERTIFICATESの固定2 composite | Record key／member／precision／sequence hash／Wを固定JSONと照合。Compositeは保存positive_sequenceを直接読む |
| Exact written product | [exact_word_matrix](../../../src/trottertracks/track_c_h4_synthesis.py) | Exact cyclotomic ringでW込み行列を評価。N3 helperのmidpoint変換だけを局所診断へ再利用。RZ target比較・strict_guardは実行しない |
| 正負・written→chronological | [lower_native](../../../src/trottertracks/track_c2_b_verify.py) | 文字列を逆順にemit。Wはphase欄、他文字はnamed gate。負側はcomplete local.inverseを一度使う |
| Logical negative参照 | [CT-1 word_matrices](../../../src/trottertracks/pr_reference_ct1.py)、N4-B compressed | Positive wordの行列全体をconjugate transpose。実行にはguardを含まないN4-B compressedの既存処理だけを使う |
| Native segment | [consume_native](../../../src/trottertracks/pr_reference_common_word_consumer.py) | Bound numeric input、固定rz/sx/x/cx、同じwireへappend。Phase0 segmentのunitaryだけをlower_nativeへ渡す |
| Segment／outer phase | 同consumer、Qiskit compose／inverse | Phase欄と命令列が分離。Inplace join、同じwire写像。New control／compile／QPYなし |
| Fencesとclbit | Consumer、N4-B check_consumer | Reset／measure／barrierを同じ順・wire／clbitで保存。今回reset(1)、measure(1,2)、measure(0,0)を直接検査 |
| Final signal | [wrapper](../../../src/trottertracks/pr_reference_interference_pilot.py) | 元SOURCEのH/core/任意Y用Sdg/H/readoutを変えない。今回の小fixtureではbody上のX(q1)／Y(q1)を局所診断 |
| 先行control | 同SOURCE ordinary_core／reference_core、[CT-4 prepare_one](../../../src/trottertracks/pr_reference_ct4.py) | Control／CSWAP等をSOURCEで構成しcompiler Aを通すのが開始点。Consumer後のcontrolを正当化していない |

Exact wordのt→T^7は行列評価のためだけで、実emitterのTdaggerを7物理Tへ置換していない。価格は計算していない。

## Qiskit 1.3.0の対応

[N3-B1aの一次SOURCE監査](../pr_reference_n3b1a_emitter_static_20261108/EMITTER_SOURCE_AUDIT.md)を再利用した。公式commit **896d1c8b72888a52a46081a6d1f4de8c56db8aea**に対応するinstalled quantumcircuit.pyを再読した。

- `inverse`は新circuitのphaseを否定し、命令列を逆順にして各named operation.inverseを置く。
- `compose(..., inplace=True)`の固定経路は `dest.global_phase += other.global_phase` と命令のwire写像を分ける。
- `global_phase` setterはcontrol-flow scopeがなければCircuitDataへ委譲する。固定consumerはcontrol-flow／条件付き命令を対象にしていない。
- Float normalizationがtau境界で再設定されても、命令引数やwireを変える経路ではない。今回そのscalarをstored phaseから直接導出した。

今回使用したPython7 SOURCEのGit blobとaccelerate binary SHA256は、保存N3 runtimeと一致した。[ENVIRONMENT](../../../artifacts/pr_reference_n5b1_scalar/2026-11-08/ENVIRONMENT.json)を参照。Rust binary全命令の形式的同値証明ではない。旧C1〜C6の同一probe・copy/QPY境界は再実行せず引用し、今回の固定prefixに必要な実consumer経路だけを実行した。

## 再利用方法と新wrapperの範囲

[Wrapper](../../../scripts/verify_pr_reference_n5b1_scalar.py)は次の関数bodyを固定commitからAST抽出し、変更せず再利用した。

- `lower_native`、`consume_native`。
- `exact_word_matrix`、N3 helperの`matrix`／`phase_value`。
- N4-Bの`ins`／`expected_stream`／`check_consumer`／`word_matrix`／`compressed`。
- CT-1の`apply_batch`。4x4の小さい評価なのでJIT decoratorを外した同じbodyを使い、既存word matrix cacheは戻した。

旧runner.main、科学module全体、`strict_guard`、`acquire`、production DictionaryProviderは起動していない。新たなliteral lookupは**固定10件の保存文字列をconsumerへ渡すfixture用binding**である。Word生成、quantization、precision選択、certificate計算、域拡張がなく、科学的providerの実装／採択ではない。

追加部分は2-qubit fixtureの組立て、保存literal binding、実phaseとexact Wからのscalar式、比較結果保存だけ。関数body／入力hashは[SOURCE_INPUTS](../../../artifacts/pr_reference_n5b1_scalar/2026-11-08/SOURCE_INPUTS.json)。最終SOURCE hashはこの採用実行と一致する。

## 旧検査の再利用と新しい組合せ

N3のW／carrier／written-word、N4-Bのstandalone composite、途中reset／barrier／readoutと保存8 whole projectionは保存PASSを引用し、再実行していない。特にidentityと2 compositeの既存単独検査を新規実行として数えていない。

今回の固定templateは `SX(0), SX(1), RZ(theta,1)とCXの前後, SX(1)` とprefix／reset／clbitの組合せで、SOURCEから予測したscalarとの比較が旧fixtureにはない。10×2 signs×4 prefix×2位置の160組を完了し、同じnamed streamのOperatorは37通りだけ評価した。Whole PF build／compileや新trajectoryではない。

初回wrapperには再利用関数の名前空間へjsonを渡し忘れた技術エラーがあり、最初のconsumer実行の最終serializationで停止した。数値行列・信号診断の完了行は0、partial consumer callは1。初回SOURCE・入力・出力・STOPを[INITIAL_WRAPPER_ERROR](../../../artifacts/pr_reference_n5b1_scalar/INITIAL_WRAPPER_ERROR_20261108.tar.xz)へ保全した。変更はwrapperのnamespaceへjsonを加える1点のみで、同じ固定入力をfresh領域で完了した。旧SOURCE・strict certificateの不一致による再試行ではない。

## 本番経路へ残る条件

SOURCEの因子分離は、許容alphabet、numeric phase、固定inplace consumer、同じnamed ideal gate、後置controlなしという入力クラスの命題である。m17四条件のactual native全入力がそのクラスへ入ること、全必要literalへqualified logical wordを供給できること、production sinkまで同じ経路を保つことは、今回の2-qubit診断だけでは資格されない。

Compiler条件Cは旧SOURCE→native対応の条件のまま残る。統括兼Dの別adapter／native経路、q64等を再検証せず、未公開の内容も推測していない。Mandatory STOP。
