# BF-1 execution gate amendment v2

日付: 2026-10-05 JST。利用者review対象: `2d0ddafa95aefb272a36a13f259ee0b45bcc80ef`。
受領判定: **`PROCEED_BF1_AFTER_MINIMAL_EXECUTION_GATE_REVISION`**。
status: `MINIMAL_REVISION_CONTENT_FROZEN_FINAL_REVIEW_REQUIRED`。
**BF-1科学実行のauthorizationではない。**

本書と[結果前契約v1](bf1_preregistration_v1.md)を併読する。
v2が優先する変更は、以下のpost-search診断とauthorization publication方式の二点だけ。
研究方針の全面再設計はBF-1終了後に行う。O/L/FのRQ、family、input、32探索評価/arm、
q/R/K policy、allocation、primary 5%、数値guard、baseline、上限、全case mandatory STOPは維持する。
BF-0四文書、旧v1準備artifact、Aのsource/result/contract/status/runtimeは変更しない。

## 1. Cross-objective診断の結果前仕様

\[
\mathcal W=W_O\cup W_L\cup W_F\cup W_{\rm fixed}
\]

三armそれぞれ32点の探索が完了し、全係数の**primary epsilon=0.01によるfinite再採点**が
完了した後だけ、全union係数をO/L/Fの三設計objectiveでcross-scoreする。
bridge epsilon=0.05はその後。cross-scoreをrefinement、K4 trigger、選択集合、primary分類へ戻さない。
各Jはv1の各armのobjectiveと同じq/R/K最小化であり、J_Fだけを共通finite scoreへ読み替える。
異なるobjectiveの数値を互いに比較して順位を決めない。

保存する項目は、係数identityとO/L/F/fixedのorigin、J_O/J_L/J_F、各objectiveの適格性と理由、
最小化したq/R/K、bias・u・log B・tail bound・work・shots、union内順位、
同じobjectiveのunion最小値に対するregret、各armの自身の探索集合内最小値。
F winnerについても三objectiveの詳細を保存し、Lがその係数をどう評価するか確認できるようにする。
順位はpoint-estimateのcompetition rank（strictに小さい係数数+1）、exactな同scoreは同順位。

sourceの`cross_score()`は完了済み32点×3集合と固定4参照だけを受け取り、
**cache-only**で実行する。ideal/finite cellが不足すれば新たにsignalを計算せずSTOPする。
新candidate生成、optimizer呼出し、追加search evaluations、追加ideal/finite cellsは0。
同一係数を一回だけcross-scoreし、logical diagnostic scoresは3×union件数（上限300）。
既存400/4000 cell cap、CPU/RSS/wall/output cap内に含める。

`cross_objectives.rows`と`objective_attribution`をresultへ、係数ごとの記録を`cells.jsonl`へ保存する。
途中失敗でも既存checkpointを保ち、再探索・追加条件を行わない。

## 2. Objective deltaと探索到達差の解釈

primaryのBF-A/B/C/INCONCLUSIVE分類は変更しない。診断で新しいBF-C routeを作らない。
全case、特にBF-C候補は以下を区別して研究方針reviewへ渡す。

| 保存済みcross-scoreによるpoint-estimate比較 | descriptive interpretation |
|---|---|
| J_L(w_F)≤min J_L(W_L) | `SEARCH_REACHABILITY_OR_BUDGET_EXPLANATION_NOT_EXCLUDED`。Lでも好まれるF候補へLが32点で到達しなかった可能性が残る |
| Lには適格な設計参照があり、F winnerをLがそれより好まない（またはLで不適格） | `L_DID_NOT_PREFER_F_WINNER_POINT_ESTIMATE`。objective orderingの差を調べる候補 |
| Lの適格な設計参照がない | `NO_FEASIBLE_L_DESIGN_REFERENCE` |
| F winnerなし／診断未完了 | attributionを主張しない |

F winnerの有限metricが改善していても、Lでも好まれる場合はfinite固有の設計原理を成立させたとは解釈しない。
Lが好まない場合も、有限B、bias、bound保守性、integer allocation、native workの分解と数値marginのreviewが必要。
cross-scoreの順位だけで因果証明、global optimality、新method成立を宣言しない。
`establishes_design_principle=false`を保存し、全結果後の全面再評価へ戻る。

## 3. Authorization publicationはB案

固定方式: **`source_plus_single_authorization_only_child_v1`**。

1. 修正版source、規範文書、v2 source plan/test reportを一つのsource commit Sへ固定する。
   `2d0ddaf`は修正前のreview対象であり、修正後sourceのSHAとして再使用しない。
2. 別の最終reviewと明示的利用者実行承認を受領した場合だけ、正式authorization JSONへ
   `source_commit=S`、plan/domain/test fingerprint、承認scope・実行指示を記録する。
3. Sを唯一の親とするauthorization-only commit Aを作る。HEAD=Aで将来の一回実行を行う。
   authorization JSONへA自身のSHAを書く必要はない。AのSHAはlaunch時にGitから読み記録する。

正式authorization-only commitで**追加**できるpathは次の二つに固定する。

- 必須: `artifacts/track_b_bf1_authorization/2026-10-05/authorization_v1.json`
- 任意: `docs/tracks/algorithm_codesign/bf1_execution_authorization_v1.md`

既存fileの変更・削除、source/plan/test/thresholdの変更、他pathの追加、merge commit、
複数の後続authorization commit、commit外JSONのままの実行は許可しない。
修正が必要ならこの実行契約をreviewし直す。source SHAをHEADへ付け替えてgateを迂回しない。

`verify_launch()`は科学input操作より前に、full source SHA、HEADの唯一の親がSであること、
S→Aの変更が上記追加だけであること、working authorization bytesがAのJSON blobと完全一致することを検査する。
source plan、既存domain、限定test reportがSに収録されていること、全sealed sourceのworking/S blobの
hash一致、環境一致、authorization指示・fingerprint一致も検査する。
source_commitとauthorization_commitを別fieldでresultとB one-shot registryへ保存する。
一回実行、science retry禁止、全case STOPの規則は維持する。

## 4. 修正版packetと最終review

v2準備packet: `artifacts/track_b_bf1_preparation/2026-10-05/v2/`。
source_plan、synthetic_semantic_report、authorization_draft、preparation_auditを新規作成する。
domainはv1の既存JSONを参照し、raw SHA
`caece92bd2d9f827b0f1f17281865869420bf923273ff294fd30cabd24b51efe`を維持する。
domainを再生成・更新・再探索しない。旧v1 artifactは履歴として保持する。
今回も`science_execution_authorized=false`、`BF1_executed=false`、draft source_commit=null。

最終reviewはv1+v2、修正版code、限定test reportとsource-content inventoryを対象にする。
cross-scoreが探索へ逆流しないこと、未評価cellを増やさないこと、B publicationの成功・拒否条件を確認する。
synthetic Git fixtureの承認値はtest専用であり、実プロジェクトの科学実行承認ではない。
数値guardの導出を含む既存source review義務も維持する。内容hash固定とcommit固定を区別する。
commit/pushと実行authorizationは別の指示に従い、自動的に科学実行を開始しない。
