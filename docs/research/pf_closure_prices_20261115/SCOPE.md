# 採択済みPF閉包価格60軸の有限実行

元の採否案は `c3f761f0af5b21207e64051c9c619858064450fe:docs/research/pf_closure_price_recommendation_20261115/`。今回の利用者指示による採択を、この有限domainの条件付き利用に適用する。旧推奨書は未採択時点の履歴として保全する。

Full PF/B0I p7 × S2/Yoshida4/M5 × X/Yの12 group、m7/m8/m10/m11/m12の60行。元の `12372fa6…:…/FROZEN_MODELS.json` の各group自身の係数と各行のqを使い、既存 `price_model_validation.predict` でexactにa+bqを60件評価する。整数でない結果は丸めず停止する。SOURCEと本preflightをcommit後、fresh `run01`へ一度だけ保存する。

PREFLIGHTは公開Dの10価格document、Cのm3受入12軸と保存M5/m4の4軸を元locatorで参照する。既知直接価格96軸と対象60軸は互いに素。公開C/D ref、同等成果の関連履歴と同一ユーザーの関連runner名も確認し、確認範囲の限界を記載する。private raw/QPYを開かず、直接価格を再集計・再認定しない。

出力は `run01/PRICES.json#rows` とCSV。結合キーは `(configuration_id, physical_id, m, axis)`。MODEL_PREDICTION、未観測、条件付き、固定compiler/障壁/反復/終端の持続という採択条件を保持する。元モデル資格flagは保全し、無条件誤差上界・2%保証・CIは与えない。

新compile/QPY再監査/refit/N/Gは0。既存96直接価格を増やさず、96直接＋60条件付き導出という内訳を維持する。Bの同一物理候補・固定shot配分による全13round条件付き比較のための価格引渡しであり、その会計やwinnerを今回実行しない。結果公開・remote照合後mandatory STOP。
