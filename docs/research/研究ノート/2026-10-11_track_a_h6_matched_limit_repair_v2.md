# 2026-10-11 Track A H6精度一致比較 v2 limit継承修正

利用者がv1のRLIMIT_AS継承修正・実子process synthetic検査・新source/manifestの公開までを指示した。科学実行の認可は含まない。[新実行準備契約](../track_a_h6_matched_accuracy_execution_contract_v2.md)を追加し、旧source057ba97・v1 STOP274b984・消費済みgrant・保存入力・旧科学結果を保全する。

coordinatorの割当上限soft2 GiBと、子へ継承するhard ceiling8 GiBを分ける。signal/cost workerはsoft/hard8 GiB、live RSS/output/log/call/instruction guardは旧planのまま。coordinatorが制限を設定してから実子をspawnし、production worker gateで設定・receiptを検証した。17新規と159関連回帰、計176 local PASS。旧hard2での失敗、soft2の実効割当拒否、上位hard不足・古いgrant・改変planの拒否も確認した。

source54986a9、runtime216・保存入力33・環境・新出力をseal。科学planは旧公開12183dbとdigest一致、92候補・4精度・shot会計・最大1708 wrapperは不変。sector/compact/cache・費用2 process・wall cap=nullを維持し、実H6科学処理・sampling・compile0、新grant未発行、mandatory STOP。H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION / next_stage_authorized=false、H8・追加探索未認可。
