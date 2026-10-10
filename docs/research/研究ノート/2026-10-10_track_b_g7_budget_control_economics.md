# 2026-10-10 Track B G7

## Source preparation

利用者がGPT G6 review §12の方針を採用。Codexは同§12.3の技術裁量で結果前契約を固定。
研究条件・主要対照・T-primaryを変えず、2 known development inputs/4 arms/24 keys。
U予算と三次対照を独立に導出、18 focused testsを通過。runtime新規導入なし。
GPT selfcheck同梱directoryは未取得、期待値コピーなし。source freeze後一束、retry0、STOP。

## Source固定と一束結果

source ab2549f41b3546fb3940342a2162dd9ee93699c4 をcommit/pushしremote SHA一致、clean HEADからrunner一回。
全24keys strict guard PASS/8 rows complete、marker消費済み、mandatory STOP。
P5のconditional期待Tの差は全登録対照に負、P3では三次closed-form対照に正。
試行hard capとCPUが増える。新規性・主method採択・次scopeはGPTへ戻す。
保存値11項目と旧893paths/source52hashを照合、追加scienceなし。
