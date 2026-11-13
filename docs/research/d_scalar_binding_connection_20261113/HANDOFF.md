# D・Bへのscalar binding HANDOFF

## 呼出しと取込み

取込み対象は `src/trottertracks/rpe_optimization_pipeline.py`、`src/trottertracks/rpe_d_price_connection.py`、`tests/check_rpe_d_price_connection.py` の小差分。元A/B/C numerical provider・D runtime/compiler・B会計は変更しない。D adapterの固定SOURCEは `748be6875b5007707d864e16aa7a4fc4250f3714`。branch全体のmergeは不要。

```python
# fixed_D_requestはDのld12_price_requests.request_for等から得た固定metadata。
# Cでscalarを算出・推測・変更せず、input/tail SOURCEと一緒に引き継ぐ。
pc = fixed_D_request['physical_context']
pending = I.price_requests(root, C_request,
    input_source=pc['input_source'],
    execution_source=pc['execution_source'],
    axes=fixed_D_request['axes'],
    expected_scalar=pc.get('expected_scalar'))
bound = P.bind_results(root, C_request, pending,
    paths=externally_supplied_paths,
    prices=externally_supplied_means, bindings=externally_supplied_bindings)
```

Cの入口は元どおり全schedule分のpendingを返す。元B configuration・A physical row・SOURCE・axes・preparation_hashが同じなら、対象roundのD request/context IDは元D constructorの生成値と一致する。既知のpreparation hashを使う場合は、既存`preparation_hashes`引数へ同preparationの全round分を渡し、元D側とも一致させる。未知hashはNoneで、復元しない。

`expected_scalar`はschedule共通のoptional引数。R/q/timeが変わってもこのscheduleのpreparation定義は同じである。外部pendingに異scalar又は指定/未指定が混ざると停止する。scalar指定時にはpathのresolved scalarも元D同様のbinary64 `float.hex()`一致を要する。元Dがexpected_scalarをcanonical binary64値に検査する処理をそのまま使用する。

scalarを使わない既存呼出しは引数省略又はNoneのままでよい。その場合、D requestの内容/IDは旧0c6と同一。新SOURCEのlowering extensionは `P.extension(d_request)`から取得する。既存外部mean/sidecarのSOURCEを無断で新SOURCEへ付け替えず、対象requestの`d_request_id/d_context_key`、完全profile、元B identity/law・cohort・physical contextをそのまま照合する。

pathだけならmeanはnull、有限cohortや未資格lawは元の未資格状態を保持する。scalar指定が変わるとrequest/context IDも変わるため、旧keyのmean sidecarは拒否される。Cはmean推定・科学的資格付け・N/Gを実行しない。

## 今回確認していないこと

値の科学的正当性は固定D input/tail descriptor側の証拠に依存する。本修正の人工値を実H4へ流用しない。Cは実preparation、component/tail inventory、scalar差の計算を行っていない。T2の`Prepared scalar differs from the requested total scalar`を解消したという成果ではない。scalar値の変更、tolerance化、runtime gateの無効化、sampling/compileの先行は行わない。

旧D T2 SOURCE/結果/STOP、C1/C2/S0a/kernel、ベースライン計画とその未承認STOPを保全する。Dのpreparation-only診断とは独立したAPI修正だけを今回完了した。

## 次の担当

- **D**：別途承認されたpreparation-only scopeで実scalarと定義を診断する。今回のC修正をT2再実行の許可と扱わない。
- **ユーザー**：ClaudeレビューとD診断後、必要な科学実行範囲を判断する。
- **C**：本技術成果のremote照合後mandatory STOPを維持。新しい科学計算を開始しない。
