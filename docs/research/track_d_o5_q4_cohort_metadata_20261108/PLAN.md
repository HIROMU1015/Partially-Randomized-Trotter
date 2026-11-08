# O5：C6a保存q4・16 pathの固定cut構造調査PLAN

2026-11-08 JST。利用者の明示指示に基づく一回のmetadata再解析。基点2f316fb2、新しい専用branch/worktree/outputを使う。元C6a・SERVER1結果・q64のSOURCE/input/raw/result/STOPやprocessへ変更・介入しない。

## 対象・固定規則

Linear H4/隣接1.00Å/STO-3G/Nα=Nβ=2、元H/number-CDF8/元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1、q4の保存calibration cohort `track_d_c6a_q2_q4_paired_price_pilot_v1`、sample00..15の全16本。元H厳密ground仮定・準備費除外という元条件を保つが、今回state/Hamiltonianを使う計算はない。

C6a実行SOURCE6b666702、公開索引e9d4e2f6、各archive/member events hash・保存SOURCE/seed/endpoint/input/law・4×32 K0 eventを照合する。原本はSERVER2の既存ユーザー専用領域からread-onlyで利用し、欠損や不一致は補完せずSTOP。追加sampling・再生成・旧run resumeは0。

selectorはa54d0405の`track_d_cut_selection.py`、record bindingは2f316fb2の`track_d_cut_h4_saved.py`、特徴は64072ba8の`track_d_c7_4_models.py:event_features`をそのまま使う。feature moduleは前回と同じimmutable Git bytes loaderで読み、他の価格/model関数は呼ばない。`track_d_cut_artificial.py`も固定版のbytesを維持するが、人工回路の実行入口は使用しない。

各stepの負荷l=1+basis_runs+support_work、cut候補1/2/3、score=max(prefix,suffix)+1、tie-breakは中央への距離→小さいcutで不変。係数/候補/score/意味論を変えず、compiled価格やdepthをfeatureへ渡さない。

## 最小実装・一回性

新規コードはevent reader・固定関数への接続・集計の薄いrunnerと最小人工テストだけ。既存`checked_payload`は全QPY/NPZ bytesまで返すため、この用途ではeventsだけをextractする薄い入口を用いる。archive全体のhash/sizeとmember集合、events memberのhash/sizeを照合し、既存`checked_record`へ渡す。QPY/NPZのdeserialize・回路/価格の読込は0。

sample00は既存decisionのevent hash/descriptor hash/SOURCE・規則を照合して再利用し、cutを選び直さない。旧decisionに別保存されていない各stepのr_i/w_iだけを固定特徴関数で一度取得し、既存loadsとの一致を確認する。sample01..15は各一回だけselectorを呼び、その内部で呼ばれる特徴関数の結果をcaptureして別計算を避ける。予定はmetadata16本、旧decision再利用1、新cut選択15、step特徴64回。

各pathのr_i/w_i/l_i、3 cut score、選択cut/中央との差、minimum-score tieと単なる非最小score tieを区別して記録する。全16のcut分布、非中央割合、best非中央score−中央score、step負荷range/左右負荷差を集計する。X/Yでn32にしない。

## 分岐・資格・STOP

Case A（全中央）なら、固定q4 cohortでは使用機会0と報告し、Bの追加H4 compile保留・候補C接口融合を次の技術候補とする案を提示する。Case B（非中央あり）なら、sample ID昇順で最初の非中央pathだけを次候補にし、必要な再利用/全入力correctness/選択Reference X/Yの追加compile等を未実行案として整理する。価格/改善幅で候補を選ばない。Case C（identity/descriptor不適格）は科学条件や規則を変更せずSTOPする。

calibration cohortの有限記述であり、独立holdout/全law support/全q一般性/compiled depth効果/新規性の認定ではない。score差をdepth差にしない。今回のH4 build/raw simulation/P0/postcheck/保存価格再解析/モデルfit/N/全RPE更新はすべて0。B後続やC実装へ自動進行しない。

処理は小さなmetadata・archive照合で、共有CPU/RAMの空きを確認して必要十分に実行する。低い資源上限を新設せず、共有環境/venv/cache/OSや他jobを変更しない。archive hashは実行前後に照合し、元データを書き換えない。

## 出力・公開

namespace `artifacts/track_d_o5_q4_cohort_metadata/2026-11-08/run01`。実行前SOURCEとinput hashを固定し、path別科学scalar、集計、呼出数、原本前後照合、STOPを保存する。詳細descriptorはprivate `local/`へ保全。原event/ordered payload/archive/QPY/NPZ/private input/内部絶対pathはGitHubへ追加公開しない。

報告・index・研究概要・dated note・manifestを更新し、専用branchへcommit/push、remote SHA/tree/public bytes一致を確認した後mandatory STOP。次の研究判断はユーザー/GPTへ戻す。
