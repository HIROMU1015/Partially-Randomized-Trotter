# A/B/D向け接続

## 取込み

新規2 module `baseline_signal_pipeline.py` / `baseline_signal_numerics.py`、metadata CLI `prepare_baseline_signals.py`、人工test `check_baseline_signal_pipeline.py` とこの文書一式を選択取込みする。共通B controller、AのPR provider、D compiler、旧20セルrunnerは無変更。branch全体をmergeする必要はない。

固定B `437ba4ac` / D `e7dca5f7` と数値helper SOURCEは [SOURCE_INDEX.json](SOURCE_INDEX.json)。Git objectを読めるrepositoryが必要。B統合環境では `api=SimpleNamespace(B=既存controller,D=既存full_baseline_v2_requests)` を渡せる。ただし別版で同じ意味を確認せずSOURCE名だけ付け替えない。

```python
from trottertracks import baseline_signal_pipeline as P

# 全て公開metadata。spec/layouts/target/reference descriptorsを先に固定する。
config = P.configure(root, spec, layouts=layouts_or_none)
plan = P.prepare(root, config, target=original_full_H_target,
                 references=reference_descriptors)
# ここまではmetadataのみ。以下は別途承認された有限科学実行でだけ呼ぶ。
result = P.execute(root, plan, fresh_output,
                   source={"commit": fixed_execution_source},
                   context_factory=shared_context_factory,
                   reference_reader=saved_reference_reader,
                   saved=matching_saved_signal_records)
```

CLIは `python -B scripts/prepare_baseline_signals.py --config CONFIG.json --target TARGET.json --references REFERENCES.json --out PLAN.json`。`--execute` はなく、同じ出力への上書きも拒否する。

## 入出力と担当

|担当|接続・必要事項|
|---|---|
|A PR signal|PRの入口を追加していない。Aの同一H/state/fragment行列contextを `N.Context(root,saved,binding,metadata)` へ渡せる。savedは `matrices,state,ham.constant`。prefix scalarはmetadata.prefixes、欠ける場合は元extractorが未来の科学実行内で評価する。共通matrix/stateを再生成しない|
|C baseline|Full PF/Truncated PF/Full-Hのみ。global formula/Δ/prefix/Kと全round layoutをBで生成。baselineのtargetは元full DF-H。Full-Hは元full-H preparationをSOURCE付きで供給、PR prefix0やB3は拒否|
|B statistics/N|`result.means` を既存 `rpe_decoder_statistics.prepare(config,means,shots,target=...,shot_policy=...)` へ渡す。新testで元関数へ人工結合済み。今回shots空ではMISSING_N、常にPHYSICAL_MEAN_UNQUALIFIED。新N solver/MCを呼んでいない|
|D price|requestのphysical_candidate/structure/domain.profile/split/preparation/law、B identity/law_bindingを利用。q_RPEとq_PF/pure q、q/R grouping、K、signed weights、scalar、control/lowering/wrapperを区別。N側meanと価格の再利用資格は別。C/価格値はこの実装から供給しない|

reference descriptorは `{m,T,binding,source}`。readerは同descriptorのbinding/T/sourceと、`reference`・`independent_reference`（各real/imag）、元 `signal_difference` を返す。referenceを新計算するfallbackはない。descriptorなしならMISSING_REFERENCEのままでsnapshotすら読まない。独立pairのないreferenceを同じ値の二重copyで代用しない。

`result.records` はproducerのsignal/数値診断とphysical/sourceを保存する。`result.means` はB canonical identity/law_binding・同full-H target・raw/corrected・normalization.log_B/B/attenuation・bias・uncertaintyを保持する。数値失敗recordは保存してもB rawへ供給しない。全体の数値statusと科学的資格は別field。

共有contextはimmutable入力ごとのinstanceでcacheを所有し、system名だけのglobal cacheはない。Full-H spectrumはpreparation hash/scalar、polynomialはdt/K、normalizerはtau/K、PF macro-stepはprefix/formulaとbinary64 signed time tupleで分ける。profile・q/R・law・split/sourceの異なる保存recordはdomain照合で流用しない。同sの価格共有やwrapper省略はしない。

Dの `24b0a6eb` HANDOFFに合わせ、price側にはlowering_id・control・wrapper・measurementを含む明示profile extensionも必要。既存B profileにlowering_idがなければCが旧profileを書き換えず、D/Bが元configuration参照と価格evidence側で対応を保持する。signalが計算可能でもcompiled価格domainが一致したことにはならない。

## 資格・欠測と保全

availableでも `qualification=UNQUALIFIED`。state/数値error bound、経験的allowance、law差の未取得量はnull。Full-Hはideal finite-K polynomialの期待値であり、PCG64有限cohortや全compiled価格とは別。元finite-K law調査 `d72dc4e6` のUNKNOWNを維持する。新backendの経験的allowanceをCが無承認でfit/採択していない。

PFのgateは元direct-expm診断のnorm/unitarity1e-9、reference pair1e-10。失敗・欠測を物理FAIL/0誤差に変換せず、fresh出力に保存してSTOP。元run/raw/markerを上書きしない。実環境のdefault snapshot入口は今回は未実行なので、最初の科学pilotでbinding/locatorを確認し、違えば新H/state/別SOURCEへ自動fallbackしない。

次は [NEXT_SCIENCE_PLAN](NEXT_SCIENCE_PLAN.md)。Cは本技術納品のremote照合後STOPし、実signal・保存値受入再解析・N/価格/Gは開始しない。
