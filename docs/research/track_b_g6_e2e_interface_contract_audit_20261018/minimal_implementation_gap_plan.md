# 最小の実装gap：依存順の技術計画

**実装・テスト・実行は未承認。** 以下はcorrectness上の依存関係であり、Codex Aの理論課題・科学的優先順位を選び直したものではない。新アルゴリズム、実験契約C、grant/marker/claimは作らない。

| 順序 | 最小作業案 | 再利用できる部分 | 開始前に必要な条件／担当 |
|---|---|---|---|
| G0 対象の固定 | 入力model層、既存D/R、identity、h/m、保証scope、精度・coverage・confidenceを明示 | 非Pauli調査、M1〜M12、既存DF metadata | **ユーザー**が未採択事項を決める。新保証・重要科学判断を要する場合だけ、別途開始承認したGPTへ戻す。Aの理論候補へ介入しない |
| G1 入力binding | label integer↔component/fragment/basis/hashとexact p/x・Λ_R・σ・元入力誤差の証拠を結ぶ | DF抽出型、registry、DFRTEEventPreparation、厳格integer utility | G0後、**Codex**が明示的な実装承認を受けて対応可能。保存floatを対象にするか元raw DFを対象にするかは勝手に選ばない |
| G2 Finite-bit仕様 | 自動H/K_root又は事前固定bit policyの出力・証拠・fail-closedを接続 | root/dyadic、M10の既存下界、G7 budget | policy/誤差条件の採択が前提。新selectorの理論改善を今回研究しない。実装と人工testは別承認 |
| G3 Eventとtrial情報 | accepted/事前zero/technical failure、実π/c̃/W、ratio、σ、word/phase、input identityをlosslessに保持 | G7 event/None、G8 counters | **Codex**が実装可能な接続作業だが未承認。既存RTEEventへのsidecar、別型等の選択は保留。Noneや例外を同じzeroにしない |
| G4 Controlled binding | label解決、sign、reverse word、child回転、parent scalar、actual inverseを既存回路primitiveへ渡す | DF direct conjugated-Z/ZZ、又はG7 helper IR | **どちらの既存routeを使うか未採択**。workspace/provider誤差・角度証拠が異なる。ordinary StepRequest guardを無断緩和しない。logical試験とnative認証は別 |
| G5 測定とaccumulator | already-controlled wrapper、backend bit、axis、実W、全trial分母、失敗・完了判定を接続 | Hadamard bit/signの既存仕様、G8 prequantum zero | weighted全trial estimatorが必要。fresh-IID、backend failure、state preparation、confidence scopeを固定した後、別途実装・人工test承認。実shotsは科学実行承認後 |
| G6 誤差・費用統合 | 九つの誤差項目、U/digital bound、accepted quantum calls、取得/cache、state、wrapper費用を別保存 | G7/G8、ordinary PF/RPE metadata、限定D系列会計 | 元H・native・multi-stepの未認証要素を0にしない。十分N／cap／factor規約を新taskへ接続する採択が必要。新しい費用数値を出す計算は別承認 |
| G7 最小接続試験 | V1〜V5後、既存人工定義で入力→event→controlled→outcome／集計→証拠のlossless性を検証する案 | 既存synthetic SOURCEとnegative testの仕様 | **ユーザー**が具体的な実装・test範囲を承認した場合だけ。今回人工testも未実行。分子・native・実backendへ自動拡張しない |

## 実装だけで扱えるもの

targetと政策が固定されれば、identityの伝搬、正確な有理値の保存、型検証、trial状態分離、phase/order/signのbinding、全trial counterと保存schema、旧ordinary pathとの分離はCodexが担当できる。通常の接続bugを解決するたびに科学レビューを要求する必要はない。

一方、元Hの保証scope・誤差coverage、許容精度/confidence、provider routeと前提、multi-stepを含む主張の範囲は未採択。既存保証だけで閉じないものをAPI仕様として事実上採択しない。採択する科学条件が必要なら具体的な選択事項だけユーザーへ返す。

## 今回の技術的結論

最初に新しい分子計算や合成を始める必要はない。既存SOURCEにある三つの境界、すなわち**DF float→G6 rational、G7 event→ordinary専用request、accepted circuit→weighted全trial estimator**を、誤差・phase・confidenceを落とさない接続仕様として閉じることが先決である。これは資源改善の見込み・理論深化の順位・論文化GOの判定ではない。

CPU/RAM/worker/wall-timeの新しいcapは設けない。将来の実行では実際の空き資源と共有非干渉、採択済み個別契約を守り、SOURCE最適化が必要なら新しい固定版と許可範囲を用意する。旧SOURCE・結果・caps・claimは今回変更しない。
