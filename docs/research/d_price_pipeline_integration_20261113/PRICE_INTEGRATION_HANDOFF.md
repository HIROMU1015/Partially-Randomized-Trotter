# A/B/C/D価格接続

## 呼出し

```python
from trottertracks import rpe_optimization_pipeline as I
from trottertracks import rpe_d_price_connection as P

# requestは既存I.planで作った全round configuration。
pending = I.price_requests(root, request,
    input_source=fixed_input_descriptor,
    execution_source=fixed_D_execution_source,
    axes=('X', 'Y'), preparation_hashes=known_hash_by_round_or_none)

# path_resultsはDが別途許可範囲で保存した出力。
# external_means/bindingsは外部producerの平均価格証拠。空でもよい。
bound = P.bind_results(root, request, pending, paths=path_results,
    prices=external_means, bindings=external_mean_bindings,
    interval_contract=external_simultaneous_contract_or_none)

out = I.compare(root, request, qualified_signal_packet,
    allocations, external_N_evaluations, bound['prices'],
    statistical_contract=statistical_contract, cohorts=bound['cohorts'],
    price_extension=bound['price_extension'],
    price_bindings=bound['price_bindings'],
    interval_contract=bound['interval_contract'])
```

上のrequest/mappingは受動処理。実入力のmean/N/G算術は別の科学実行scopeが必要。CはNativeRuntime.context/sample、Adapter.acquire、cohort統計を呼ばない。

## SOURCEと取込み

- C基準 `2f76b5750`：既存A/C signal、B shot/cost/selectとbindingを保全。
- D `0c6e16342`：candidate_price_adapter、compiled_price_fast、efficient_controlled_zz、p1_efficient_zzを固定Git objectからprivate namespaceへ読む。科学的な関数本体は無変更。
- A `a9abf8de7`：初期LD1/LD2の固定全round configuration。Cは値を再解釈せず、Bのsource-bound adapterを追加読込して扱う。A signal SOURCE `fefd548ec`、B `b9b9ffa62`、C baseline `86f37d25f` は元のまま。

新価格mapping moduleと既存integration moduleの小差分・test・本文を選択取込みする。branch全体のmergeは不要。他担当のbranch/worktreeを変更しない。

## requestの同一性

`input_source/execution_source`を省略した旧 `I.price_requests` 呼出しは元の一般pendingを返し、D requestがあると偽装しない。D形式を使うには両SOURCEを明示する。H4 PR以外はOUTSIDE_DOMAIN。D未対応構成を探索候補から除外しない。

各pendingにはDの原 `d_request`、元 `price_profile`、Cの `physical_context`、lowering extensionを保存する。Dが持つB identity/lawとCのphysical candidateを混同しない。binary64時刻とB exact型の一致は元Dに委ね、丸め直しで橋渡ししない。preparation hash不明はnullで、生成しない。同schedule内の異なるinput/execution/preparation/profile、異なるrequest/軸/lawは拒否する。

実runtimeで新たにresolvedとなるpreparation/tail hash、λ、scalar、fragment順は元Dが検証する。C mappingでは保存されたresolved metadataの一致を検査するだけで、H/component/basisを復元しない。実cacheは読まない。fingerprint、cache reference、event digest、seed/law診断はpath出力のまま保持する。

## path結果と平均価格を分離

`paths`にはD `CANDIDATE_PATH`形状を渡す。quantityはPR_CANDIDATE_PAIRED_PATH_PRICE、各軸はPATH_COMPILED/PATH_STRUCTURAL RZ。request、input/execution SOURCE、conditional/cohort/index、resolved preparationを照合する。軸が不足しても別軸で補わず、meanはnull。`mean_compiled_price`や`qualified_expectation`をpathに後付けして価格を昇格させる入力は拒否する。

path結果はidentity検査を受けても全law/IID/正常sampling資格を得ない。固定seed・希少event・law差・有限cohortの不確かさを0にしない。通常cohortとforced診断は別roleで保持し、Cはpoolもn集計も行わない。

## 外部mean/区間のbinding

外部価格は元Bのquantity/measurement/axis/identity/law/cohort/classification/expectation/mean/source/model/uncertaintyを維持し、`price_profile`には対象D requestの完全なprice profileを付ける。source不明やbinding欠測なら元recordを保存し、その軸はpendingのまま。

元B/D sidecarに次の2キーを追加する。

```python
binding = dict(
    # 元B仕様：identity/law_binding/axis/cohort/source/price_source、
    # physical_context、profile_extension、mean_interval（又はNone）
    ...,
    d_request_id=d_request['request_id'],
    d_context_key=d_request['context_key'],
)
```

`physical_context`はC `I.price_context(request,m)`。`price_source`は親meanのSOURCE、`profile_extension`は `P.extension(d_request)` と一致させる。別のlowering/definitionへSOURCE名だけ付け替えない。平均producerは、どのinput/request/law/cohort/profileへ適用する証拠かを自分のSOURCEで記録する。Cが古いrecordへこのbindingを後付けして資格を作る操作はしない。

外部平均が必要な全軸にない場合は未取得軸をD pendingから保持する。mean不確かさ・model資格・ideal/sampler law・finite cohort・同時区間の最終資格検査は元B。適格flagを供給しただけで科学的な被覆証明ができたわけではない。

`bind_results`の戻り値はB入力5欄に加え、path/供給mean/供給sidecarとunmapped理由を保存する。SOURCE/identity/law/profileの不一致はValueError、情報不足はpending。共分散・CI・未知systematicは元recordのnullを維持し、Cは新しい区間を作らない。有限cohortは既定policyでG UNKNOWN、明示scenarioなら条件付き費用のみ。

## 後続D成果

DのLD1/LD2 domain適合性成果は公開SOURCEと差分を確認して参照する。Cは同診断を重複しない。実path取得はDの完備provider環境で実施し、Cの最小metadata namespaceから科学backendを自動起動しない。

Dがmean/区間producerを公開した後、その出力を上の形へ接続する。新しいsampling/compile/推定/fit、未資格のpriceを使う実Gは別のユーザー承認が必要。新しいscience契約・grantはCで発行しない。
